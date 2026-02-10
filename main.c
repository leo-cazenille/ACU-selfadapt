/**
 * @file phototaxis_optim.c
 * @brief Phototaxis with a configurable optimizer (HIT / ES / SPSA / PGPE / SEP-CMA-ES / SL)
 *        on an int8 MLP policy (swarm setting).
 *
 * Each robot:
 *  - Runs a small int8 Q0.7 MLP controller (implemented in MLP_int8.h).
 *  - The MLP weights are encoded as a float genome of dimension D.
 *  - The genome is optimized online using the unified optimizer interface (optim.h),
 *    with decentralized exchanges via IR (HIT-compatible messages).
 *
 * Objective:
 *  - Minimize an instantaneous phototaxis cost:
 *        cost = 1 - (avg_light / PHOTO_MAX),
 *    so lower cost = stronger light on the robot.
 *
 * Communication:
 *  - Robots advertise (parts of) their genome and a sliding-window score
 *    using the optimizer infrastructure (opt_get_last_advert, opt_observe_remote*).
 *  - If HIT_BLOCK_SIZE > 0, only contiguous blocks of the genome are exchanged
 *    (HIT block mode); otherwise, the full genome is sent.
 *
 * Optimizer configuration:
 *  - In SIMULATOR builds, the optimizer algorithm and its hyperparameters
 *    are configured from YAML, using the same keys as ACU-selfadapt.c:
 *
 *      optimizer:
 *        algorithm: "hit"          # "hit", "es1p1", "spsa", "pgpe", "sep-cmaes", "sl"
 *        hit:
 *          alpha: 0.35
 *          sigma: 0.15
 *          eval_T: 5
 *          evolve_alpha: true
 *          alpha_sigma: 1.0e-3
 *          alpha_min: 0.0
 *          alpha_max: 0.9
 *          auto_sigma: false
 *          loss_mut_gain: 0.5
 *          loss_mut_clip: 1.0
 *
 *    and analogous sections for other algorithms:
 *      optimizer.es1p1.*, optimizer.spsa.*, optimizer.pgpe.*,
 *      optimizer.sep_cmaes.*, optimizer.sl.*
 *
 *  - On real robots (non-SIMULATOR), we fall back to the previous behavior:
 *      - Algorithm: HIT
 *      - Config:    opt_default_cfg(OPT_HIT, D)
 */

#include "pogobase.h"
#include "pogo-utils/tiny_alloc.h"
#include "pogo-utils/optim.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

/* ------------------------- MLP configuration --------------------------- */
/* You can override these at compile time, e.g.:
 *   -DMLP_INT8_INPUT_DIM=3 -DMLP_INT8_HIDDEN_DIM=8 -DMLP_INT8_OUTPUT_DIM=2
 *
 * Default: 3 light sensors -> 8 hidden -> 2 motor commands.
 */
#ifndef MLP_INT8_INPUT_DIM
#define MLP_INT8_INPUT_DIM   3
#endif

#ifndef MLP_INT8_HIDDEN_DIM
#define MLP_INT8_HIDDEN_DIM  8
#endif

#ifndef MLP_INT8_OUTPUT_DIM
#define MLP_INT8_OUTPUT_DIM  2
#endif

/* Optional: make the output also hard-tanh; otherwise it's linear Q0.7. */
/* #define MLP_INT8_OUTPUT_HARD_TANH */

/* Total number of trainable parameters in the MLP. */
#define MLP_PARAM_COUNT ( \
    (MLP_INT8_HIDDEN_DIM * MLP_INT8_INPUT_DIM) + \
    (MLP_INT8_HIDDEN_DIM) + \
    (MLP_INT8_OUTPUT_DIM * MLP_INT8_HIDDEN_DIM) + \
    (MLP_INT8_OUTPUT_DIM) \
)

/* Dimension of the optimizer genome.
 * By default it equals the full MLP parameter count, but you may override D
 * at compile time (with -DD=...) to optimize only a prefix.
 */
#ifndef D
#define D MLP_PARAM_COUNT
#endif

#if (D > MLP_PARAM_COUNT)
#error "D cannot exceed the total number of MLP parameters (MLP_PARAM_COUNT)."
#endif

#include "pogo-utils/MLP_int8.h"

/* HIT block size for decentralized exchange.
 *  - 0 => full genome (legacy HIT)
 *  - >0 => block mode: only contiguous chunks of length HIT_BLOCK_SIZE are sent.
 */
#ifndef HIT_BLOCK_SIZE
#define HIT_BLOCK_SIZE 16
#endif

/* Default algorithm (for non-SIMULATOR builds or when not configured). */
#define OPT_PHOTO_ALGO OPT_HIT

/* Maximum light level used for normalization in the cost. */
#ifndef PHOTO_MAX
#define PHOTO_MAX 1023.0f
#endif

/* ---------------------------- Optim config ----------------------------- */

/* Global optimizer configuration (SIMULATOR: from YAML; else: unused). */
static opt_cfg_t  g_opt_cfg;
/* Global optimizer algorithm (SIMULATOR: from YAML; else: defaults to HIT). */
static opt_algo_t g_opt_algo = OPT_PHOTO_ALGO;

#ifndef OPT_CLAMP
#define OPT_CLAMP(v,a,b) ((v)<(a)?(a):((v)>(b)?(b):(v)))
#endif

#ifdef SIMULATOR
/**
 * @brief Build an optimizer configuration from the YAML configuration tree.
 *
 * Keys and semantics mirror ACU-selfadapt.c:
 *   optimizer.algorithm: "es1p1", "spsa", "pgpe", "sep-cmaes", "sl", "hit"
 *   optimizer.es1p1.*   : sigma0, sigma_min, sigma_max, s_target, s_alpha, c_sigma
 *   optimizer.spsa.*    : a, c, A, alpha, gamma, g_clip
 *   optimizer.pgpe.*    : eta_mu, eta_sigma, sigma_min, sigma_max,
 *                         baseline_alpha, normalize_pair
 *   optimizer.sep_cmaes.*: sigma0, sigma_min, sigma_max
 *   optimizer.sl.*      : roulette_random_prob, loss_mut_gain,
 *                         loss_mut_clip, dup_eps, repo_capacity
 *   optimizer.hit.*     : alpha, sigma, eval_T, evolve_alpha,
 *                         alpha_sigma, alpha_min, alpha_max,
 *                         auto_sigma, loss_mut_gain, loss_mut_clip
 *
 * @return Filled opt_cfg_t; also sets global g_opt_algo.
 */
static opt_cfg_t create_opt_cfg(void) {
    /* Algorithm selection from configuration. */
    opt_algo_t algo;
    char algo_name[32] = "hit";
    init_string_from_configuration(algo_name, "optimizer.algorithm", 32);

    if (strcasecmp(algo_name, "es1p1") == 0) {
        algo = OPT_ES1P1;
    } else if (strcasecmp(algo_name, "spsa") == 0) {
        algo = OPT_SPSA;
    } else if (strcasecmp(algo_name, "pgpe") == 0) {
        algo = OPT_PGPE;
    } else if (strcasecmp(algo_name, "sep-cmaes") == 0) {
        algo = OPT_SEP_CMAES;
    } else if (strcasecmp(algo_name, "sl") == 0) {
        algo = OPT_SOCIAL_LEARNING;
    } else if (strcasecmp(algo_name, "hit") == 0) {
        algo = OPT_HIT;
    } else {
        printf("Unknown value of configuration entry 'optimizer.algorithm'. "
               "Can only be 'es1p1', 'spsa', 'pgpe', 'sep-cmaes', 'sl' or 'hit'.\n");
        exit(1);
    }

    g_opt_algo = algo;

    int n = D; /* problem dimension */

    opt_cfg_t c;
    memset(&c, 0, sizeof(c));
    c.use_defaults = 0; /* we'll fill everything ourselves */

    switch (algo) {
    default:
    case OPT_ES1P1: {
        c.P.es1p1.mode = ES1P1_MINIMIZE;
        init_float_from_configuration(&c.P.es1p1.sigma0   , "optimizer.es1p1.sigma0",    0.2f );
        init_float_from_configuration(&c.P.es1p1.sigma_min, "optimizer.es1p1.sigma_min", 1e-5f);
        init_float_from_configuration(&c.P.es1p1.sigma_max, "optimizer.es1p1.sigma_max", 0.8f );
        init_float_from_configuration(&c.P.es1p1.s_target , "optimizer.es1p1.s_target",  0.2f );
        init_float_from_configuration(&c.P.es1p1.s_alpha  , "optimizer.es1p1.s_alpha",   0.2f );
        init_float_from_configuration(&c.P.es1p1.c_sigma  , "optimizer.es1p1.c_sigma",   0.0f );
    } break;

    case OPT_SPSA: {
        init_float_from_configuration(&c.P.spsa.a     , "optimizer.spsa.a",      0.3f  );
        init_float_from_configuration(&c.P.spsa.c     , "optimizer.spsa.c",      0.15f );
        init_float_from_configuration(&c.P.spsa.A     , "optimizer.spsa.A",      20.0f );
        init_float_from_configuration(&c.P.spsa.alpha , "optimizer.spsa.alpha",  0.602f);
        init_float_from_configuration(&c.P.spsa.gamma , "optimizer.spsa.gamma",  0.101f);
        init_float_from_configuration(&c.P.spsa.g_clip, "optimizer.spsa.g_clip", 1.0f  );
    } break;

    case OPT_PGPE: {
        c.P.pgpe.mode = PGPE_MINIMIZE;
        init_float_from_configuration(&c.P.pgpe.eta_mu        , "optimizer.pgpe.eta_mu",         0.05f);
        init_float_from_configuration(&c.P.pgpe.eta_sigma     , "optimizer.pgpe.eta_sigma",      0.10f);
        init_float_from_configuration(&c.P.pgpe.sigma_min     , "optimizer.pgpe.sigma_min",      1e-5f);
        init_float_from_configuration(&c.P.pgpe.sigma_max     , "optimizer.pgpe.sigma_max",      0.8f );
        init_float_from_configuration(&c.P.pgpe.baseline_alpha, "optimizer.pgpe.baseline_alpha", 0.10f);
        init_bool_from_configuration( &c.P.pgpe.normalize_pair, "optimizer.pgpe.normalize_pair", true );
    } break;

    case OPT_SEP_CMAES: {
        c.P.sep.mode = SEP_MINIMIZE;
        init_float_from_configuration(&c.P.sep.sigma0   , "optimizer.sep_cmaes.sigma0",    0.3f );
        init_float_from_configuration(&c.P.sep.sigma_min, "optimizer.sep_cmaes.sigma_min", 1e-6f);
        init_float_from_configuration(&c.P.sep.sigma_max, "optimizer.sep_cmaes.sigma_max", 2.0f );
        c.P.sep.weights = NULL; /* default log-weights inside backend */
        c.P.sep.cc = c.P.sep.cs = c.P.sep.c1 = c.P.sep.cmu = c.P.sep.damps = 0.0f;
        c.sz.lambda = (uint16_t)OPT_CLAMP(2*n, 4, 64); /* small default popsize */
        c.sz.mu     = (uint16_t)OPT_CLAMP(c.sz.lambda/2, 2, c.sz.lambda);
    } break;

    case OPT_SOCIAL_LEARNING: {
        memset(&c.P.sl, 0, sizeof(c.P.sl));
        c.P.sl.mode = SL_MINIMIZE;
        init_float_from_configuration(&c.P.sl.roulette_random_prob, "optimizer.sl.roulette_random_prob", 0.10f);
        init_float_from_configuration(&c.P.sl.loss_mut_gain       , "optimizer.sl.loss_mut_gain",        0.01f);
        init_float_from_configuration(&c.P.sl.loss_mut_clip       , "optimizer.sl.loss_mut_clip",        1.0f );
        init_float_from_configuration(&c.P.sl.dup_eps             , "optimizer.sl.dup_eps",              1e-3f);
        c.P.sl.repo_capacity = 0; /* will be set from sz.repo_capacity if provided */
        init_uint16_from_configuration(&c.sz.repo_capacity        , "optimizer.sl.repo_capacity",        16   );
    } break;

    case OPT_HIT: {
        memset(&c.P.hit, 0, sizeof(c.P.hit));
        c.P.hit.mode = HIT_MINIMIZE;
        init_float_from_configuration(&c.P.hit.alpha      , "optimizer.hit.alpha",       0.35f); /* initial α */
        init_float_from_configuration(&c.P.hit.sigma      , "optimizer.hit.sigma",       0.15f); /* genome mutation std */
        init_int32_from_configuration(&c.P.hit.eval_T     , "optimizer.hit.eval_T",      5);     /* sliding window length */
        init_bool_from_configuration( &c.P.hit.evolve_alpha, "optimizer.hit.evolve_alpha", true);
        init_float_from_configuration(&c.P.hit.alpha_sigma, "optimizer.hit.alpha_sigma", 1e-3f); /* α mutation std */
        init_float_from_configuration(&c.P.hit.alpha_min  , "optimizer.hit.alpha_min",   0.0f);  /* α clamp min */
        init_float_from_configuration(&c.P.hit.alpha_max  , "optimizer.hit.alpha_max",   0.9f);  /* α clamp max */
        init_bool_from_configuration( &c.P.hit.auto_sigma , "optimizer.hit.auto_sigma",  false); /* adapt σ from loss? */
        init_float_from_configuration(&c.P.hit.loss_mut_gain, "optimizer.hit.loss_mut_gain", 0.5f);
        init_float_from_configuration(&c.P.hit.loss_mut_clip, "optimizer.hit.loss_mut_clip", 1.0f);
    } break;
    }

    return c;
}

/**
 * @brief Global simulator setup: read optimizer config from YAML.
 */
void global_setup(void) {
    g_opt_cfg = create_opt_cfg();
}
#endif /* SIMULATOR */


/* ========================= Small helpers ================================ */

static inline float clampf(float x, float lo, float hi){
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Convert a float in roughly [-1, 1] to int8 Q0.7. */
static inline int8_t float_to_q07(float v){
    v = clampf(v, -1.0f, 1.0f);
    float scaled = v * 127.0f;
    int32_t tmp = (int32_t)(scaled);
    if (tmp > 127)  tmp = 127;
    if (tmp < -128) tmp = -128;
    return (int8_t)tmp;
}


/* ========================= Genome <-> MLP =============================== */

/**
 * @brief Fill an MLP_INT8 network from a float genome.
 *
 * @param x         Genome array of length n_genome (D).
 * @param n_genome  Number of usable entries in x (<= MLP_PARAM_COUNT).
 * @param net       Network to fill (int8 Q0.7 weights).
 *
 * If n_genome < MLP_PARAM_COUNT, remaining weights are set to 0.
 * Genome layout:
 *   [ W1 (hidden x input) | b1 (hidden) |
 *     W2 (output x hidden) | b2 (output) ]
 */
static void genome_to_mlp(const float *x, int n_genome, MLP_INT8 *net){
    int idx = 0;

    /* W1: [hidden][input] */
    for (int i = 0; i < MLP_INT8_HIDDEN_DIM; ++i){
        for (int j = 0; j < MLP_INT8_INPUT_DIM; ++j){
            float v = (idx < n_genome) ? x[idx++] : 0.0f;
            net->W1[i][j] = float_to_q07(v);
        }
    }

    /* b1: [hidden] */
    for (int i = 0; i < MLP_INT8_HIDDEN_DIM; ++i){
        float v = (idx < n_genome) ? x[idx++] : 0.0f;
        net->b1[i] = float_to_q07(v);
    }

    /* W2: [output][hidden] */
    for (int i = 0; i < MLP_INT8_OUTPUT_DIM; ++i){
        for (int j = 0; j < MLP_INT8_HIDDEN_DIM; ++j){
            float v = (idx < n_genome) ? x[idx++] : 0.0f;
            net->W2[i][j] = float_to_q07(v);
        }
    }

    /* b2: [output] */
    for (int i = 0; i < MLP_INT8_OUTPUT_DIM; ++i){
        float v = (idx < n_genome) ? x[idx++] : 0.0f;
        net->b2[i] = float_to_q07(v);
    }
}


/* ====================== Objective: phototaxis cost ====================== */

/**
 * @brief Instantaneous phototaxis cost in [0, 1].
 *
 * @param p0 Photodiode 0 reading.
 * @param p1 Photodiode 1 reading.
 * @param p2 Photodiode 2 reading.
 *
 * Cost is 1 - normalized_average_light, so minimizing cost corresponds to
 * maximizing average light intensity.
 */
static float phototaxis_cost(int16_t p0, int16_t p1, int16_t p2){
    float avg = ((float)p0 + (float)p1 + (float)p2) / 3.0f;
    float norm = avg / PHOTO_MAX;
    norm = clampf(norm, 0.0f, 1.0f);
    return 1.0f - norm;
}


/* ============================ USERDATA ================================== */

typedef struct {
    opt_t *opt;

    /* Search bounds for the genome. We keep weights in [-1, 1]. */
    float lo[D];
    float hi[D];

    tiny_alloc_t ta;
    uint8_t heap[8192];

    MLP_INT8 policy;              /* int8 MLP controller */

    uint32_t last_print_ms;
    uint32_t last_tx_ms;
    uint32_t last_epoch_to_mlp;   /* To avoid rebuilding MLP at every step */
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA)


/* ============================ Messaging ================================= */

/* Similar to ACU/HIT: block vs full genome adverts. */

#if (HIT_BLOCK_SIZE > 0)
/* HIT + block mode: send only a contiguous chunk of the genome. */
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    uint32_t epoch;
    float    alpha;       /* sender's current transfer rate α */

    uint16_t offset;      /* first index in [0, D-1] */
    uint16_t len;         /* number of valid entries in block[] */

    float    block[HIT_BLOCK_SIZE];  /* contiguous chunk x[offset:offset+len) */
    float    f_adv;                  /* advertised sliding-window score */
} photo_msg_t;

#else
/* Full-genome advert. */
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    uint32_t epoch;
    float    alpha;
    float    x[D];
    float    f_adv;
} photo_msg_t;
#endif

#define PHOTO_MSG_BYTES ((uint16_t)sizeof(photo_msg_t))


/* RX callback: integrate neighbours' adverts into optimizer (typically HIT). */
static void on_rx(message_t *mr){
    if (!mydata->opt) return;
    if (mr->header.payload_length < PHOTO_MSG_BYTES) return;

    photo_msg_t msg;
    memcpy(&msg, mr->payload, sizeof(photo_msg_t));

    if (msg.sender_id == pogobot_helper_getid()) return;

#if (HIT_BLOCK_SIZE > 0)
    /* Block-based observe: copy to aligned buffer. */
    int len = (int)msg.len;
    if (len < 0) len = 0;
    if (len > HIT_BLOCK_SIZE) len = HIT_BLOCK_SIZE;

    float block_aligned[HIT_BLOCK_SIZE];
    for (int i = 0; i < len; ++i){
        block_aligned[i] = msg.block[i];
    }

    opt_observe_remote_block(mydata->opt,
                             msg.sender_id,
                             msg.epoch,
                             block_aligned,
                             (int)msg.offset,
                             len,
                             msg.f_adv,
                             msg.alpha);
#else
    /* Full-genome observe. */
    float x_buf[D];
    memcpy(x_buf, msg.x, sizeof(x_buf));

    opt_observe_remote(mydata->opt,
                       msg.sender_id,
                       msg.epoch,
                       x_buf,
                       msg.f_adv,
                       msg.alpha);
#endif
}


/* TX callback: periodically advertise genome + optimizer score. */
static bool on_tx(void){
    if (!mydata->opt) return false;

    uint32_t now = current_time_milliseconds();
    const uint32_t period_ms = 200; /* 5 Hz beacons */
    if (now - mydata->last_tx_ms < period_ms) return false;

    /* For HIT-like algorithms, opt_ready() can gate transmission (maturation). */
    if (!opt_ready(mydata->opt)) return false;

    photo_msg_t m;
    m.sender_id = pogobot_helper_getid();
    m.epoch     = opt_iterations(mydata->opt);
    m.f_adv     = opt_get_last_advert(mydata->opt);
    m.alpha     = opt_get_alpha(mydata->opt);

#if (HIT_BLOCK_SIZE > 0)
    const float *x = opt_get_x(mydata->opt);
    if (!x) return false;

    int n    = D;
    int Bmax = HIT_BLOCK_SIZE;
    if (Bmax > n) Bmax = n;

    int len = Bmax;
    int start_max = n - len;
    int offset = (start_max > 0) ? (rand() % (start_max + 1)) : 0;

    m.offset = (uint16_t)offset;
    m.len    = (uint16_t)len;

    for (int i = 0; i < len; ++i){
        m.block[i] = x[offset + i];
    }
#else
    const float *x = opt_get_x(mydata->opt);
    if (!x) return false;
    memcpy(m.x, x, sizeof(float)*D);
#endif

    mydata->last_tx_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t *)&m, PHOTO_MSG_BYTES);
}


/* ============================== INIT ==================================== */

void user_init(void){
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    srand(pogobot_helper_getRandSeed());

    main_loop_hz = 60;
    max_nb_processed_msg_per_tick = 100;
    msg_rx_fn = on_rx;
    msg_tx_fn = on_tx;
    error_codes_led_idx = 3;

    /* Tiny allocator setup (same style as example_optim.c). */
    const uint16_t classes[] = { 32, 48, 64, 128, 512, 4096 };
    tiny_alloc_init(&mydata->ta,
                    mydata->heap, sizeof(mydata->heap),
                    classes, 5);

    /* Search bounds: weights roughly in [-1, 1]. */
    for (int i = 0; i < D; ++i){
        mydata->lo[i] = -1.0f;
        mydata->hi[i] =  1.0f;
    }

    /* --- Optimizer creation (configurable in SIMULATOR) --- */
    opt_algo_t algo;
    opt_cfg_t  cfg_local;

#ifdef SIMULATOR
    /* Use algorithm + parameters chosen in global_setup() from YAML. */
    algo = g_opt_algo;
    cfg_local = g_opt_cfg;
#else
    /* Non-simulator: keep previous behavior (HIT + default config). */
    algo = OPT_PHOTO_ALGO;
    cfg_local = opt_default_cfg(algo, D);
#endif

    int ok = opt_create(&mydata->opt,
                        &mydata->ta,
                        D,
                        algo,
                        OPT_MINIMIZE,
                        mydata->lo,
                        mydata->hi,
                        &cfg_local);
    if (!ok || !mydata->opt) {
        printf("[PHOTO] opt_create failed (heap=%uB). "
               "Try bigger heap or smaller lambda/repo.\n",
               (unsigned)sizeof(mydata->heap));
    }

    /* Random initial genome. */
    uint32_t seed = pogobot_helper_getRandSeed()
        ^ ( (uint32_t)pogobot_helper_getid() * 0x9e3779b9u );
    opt_randomize_x(mydata->opt, seed);

    /* For HIT, we typically do NOT call opt_tell_initial: let it fill its sliding window. */
    mydata->last_print_ms     = current_time_milliseconds();
    mydata->last_tx_ms        = mydata->last_print_ms;
    mydata->last_epoch_to_mlp = (uint32_t)(-1);

    /* Initial MLP build from the initial genome. */
    const float *x0 = opt_get_x(mydata->opt);
    if (x0){
        genome_to_mlp(x0, D, &mydata->policy);
        mydata->last_epoch_to_mlp = opt_iterations(mydata->opt);
    }

    printf("[PHOTO] phototaxis controller ready. algo=%d  n_params=%d  MLP_PARAM_COUNT=%d\n",
           (int)algo, D, MLP_PARAM_COUNT);
}


/* =============================== STEP =================================== */

void user_step(void){
    if (!mydata->opt){
        /* Safety: if optimizer isn't available, just stop. */
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    /* 1) Read light sensors. */
    int16_t p0 = pogobot_photosensors_read(0);
    int16_t p1 = pogobot_photosensors_read(1);
    int16_t p2 = pogobot_photosensors_read(2);

    /* 2) Get current genome & rebuild MLP if iteration changed. */
    const float *x = opt_get_x(mydata->opt);
    uint32_t epoch = opt_iterations(mydata->opt);
    if (x && epoch != mydata->last_epoch_to_mlp){
        genome_to_mlp(x, D, &mydata->policy);
        mydata->last_epoch_to_mlp = epoch;
    }

    /* 3) Build MLP input vector from light sensors (normalized to [-1,1]). */
    int8_t in_vec[MLP_INT8_INPUT_DIM];
    float norm_p0 = clampf(((float)p0 / PHOTO_MAX) * 2.0f - 1.0f, -1.0f, 1.0f);
    float norm_p1 = clampf(((float)p1 / PHOTO_MAX) * 2.0f - 1.0f, -1.0f, 1.0f);
    float norm_p2 = clampf(((float)p2 / PHOTO_MAX) * 2.0f - 1.0f, -1.0f, 1.0f);

    /* If the input dimension is >3, we repeat sensors cyclically. */
    for (int i = 0; i < MLP_INT8_INPUT_DIM; ++i){
        float v;
        switch (i % 3){
        default:
        case 0: v = norm_p0; break;
        case 1: v = norm_p1; break;
        case 2: v = norm_p2; break;
        }
        in_vec[i] = float_to_q07(v);
    }

    /* 4) Forward pass: get motor commands in int8 Q0.7. */
    int8_t out_vec[MLP_INT8_OUTPUT_DIM];
    mlp_int8_forward(&mydata->policy, in_vec, out_vec);

    /* 5) Map outputs to left/right motor commands.
     * Here we use a simple binary mapping: output > 0 => motorFull, else motorStop.
     * If OUTPUT_DIM > 2, we ignore extra outputs.
     */
    int8_t yL = (MLP_INT8_OUTPUT_DIM >= 1) ? out_vec[0] : 0;
    int8_t yR = (MLP_INT8_OUTPUT_DIM >= 2) ? out_vec[1] : 0;

    pogobot_motor_set(motorL, (yL > 0) ? motorFull : motorStop);
    pogobot_motor_set(motorR, (yR > 0) ? motorFull : motorStop);

    /* 6) Compute instantaneous cost and feed it into optimizer. */
    float f_inst = phototaxis_cost(p0, p1, p2);
    opt_tell(mydata->opt, f_inst);

    /* 7) LED feedback + optional debug. Use sliding-window advert as quality. */
    float f_adv = opt_get_last_advert(mydata->opt);
    if (!isfinite(f_adv)) f_adv = f_inst;

    /* Map cost in [0,1] to LED brightness: bright when near light source. */
    float g = clampf(1.0f - f_adv, 0.0f, 1.0f);  /* higher g => better */
    uint8_t val = (uint8_t)(g * 255.0f);
    pogobot_led_setColors(val, 0, val, 0);

    uint32_t now = current_time_milliseconds();
    if (now - mydata->last_print_ms > 5000) {
        printf("[PHOTO] it=%u  f_inst=%.4f  f_adv=%.4f  alpha=%.3f  "
               "p=[%d,%d,%d]  y=[%d,%d]\n",
               opt_iterations(mydata->opt),
               f_inst,
               f_adv,
               opt_get_alpha(mydata->opt),
               (int)p0, (int)p1, (int)p2,
               (int)yL, (int)yR);
        mydata->last_print_ms = now;
    }
}


/* ============================== MAIN ==================================== */

int main(void){
    pogobot_init();
#ifndef SIMULATOR
    printf("init ok\n");
#endif

    pogobot_start(user_init, user_step);
    SET_CALLBACK(callback_global_setup, global_setup);

    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
