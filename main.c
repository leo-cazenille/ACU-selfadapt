/**
 * @file main.c
 * @brief Self-adaptive ACU controller with social learning and fast transmission.
 *
 * This file implements the self-adaptive controller used in the ACU
 * (Alignment with Crowding and U-turns) motility model for Pogobot
 * robots and the Pogosim simulator.
 *
 * Each robot:
 *   - Follows the ACU motility model: Vicsek-style alignment with angular
 *     noise, density-dependent speed modulation (crowding), and propagating
 *     collective U-turns triggered by wall avoidance.
 *   - Continuously estimates local phenotypic statistics (polarization,
 *     wall/U-turn ratio, neighbor persistence, neighbor count) and converts
 *     them into a bounded multi-objective loss in [0, 1].
 *   - Runs an online optimizer (1+1-ES, HIT, SPSA, PGPE, SEP-CMA-ES or
 *     social learning) on a small genotype
 *       g = (beta, sigma, speed, phi_norm, crowd_depth_norm) in [0,1]^5
 *     to minimize this local loss.
 *   - Broadcasts its genotype, advertised loss and optimizer metadata in
 *     its regular messages, and observes neighbors to drive social learning.
 *
 * On top of the slow optimizer dynamics, a fast transmission (FT) mechanism
 * implements thresholded probabilistic cloning of clearly better neighbors:
 * if a neighbor advertises a significantly lower loss, the robot can clone
 * its genotype immediately with some probability and re-seed its optimizer
 * at this new point. This creates fast information fronts while the optimizer
 * maintains slower exploratory dynamics.
 *
 * The overall behavior corresponds to the self-adaptive controller described
 * in the accompanying ANTS 2026 paper on “Self-adaptive phase control in
 * robotic swarms using social learning with fast transmission”.
 */

#include "pogobase.h"
#include "utils.h"

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pogo-utils/version.h"
#include "pogo-utils/kinematics.h"
#include "pogo-utils/tiny_alloc.h"
#include "pogo-utils/optim.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** Maximum number of neighbors that can be stored locally. */
#define MAX_NEIGHBORS  20u
/** Percentage of ticks during which a message is sent (when selected). */
#define PERCENT_MSG_SENT 35
/** Main control loop frequency in Hz. */
#define MAIN_LOOP_HZ 60
/** Max number of incoming messages processed per control tick. */
#define MAX_NB_MSGS_PROCESSED_PER_TICK 100

/** Dimension of the optimized genotype: (beta, sigma, speed, phi_norm, crowd_depth_norm). */
#define OPT_D 5 /* beta, sigma, speed, phi_norm, crowd_depth_norm */


/* ---------- Tunables (YAML) ----------
 * All of these are configurable via the YAML configuration file when running
 * in the simulator. On the real robots they can be compiled-in defaults.
 */

/* Core dynamics */
/** Neighbor timeout in ms; neighbors older than this are purged. */
uint32_t max_age                   = 600;      /* ms neighbor timeout */
/** Maximum allowed continuous integration step (s) for numerical stability. */
double   cont_max_dt_s             = 0.05;     /* integration clamp */
/** Heading-detection geometry parameter (deg) for the photodiode arrangement. */
double   alpha_deg                 = 40.0;     /* heading detection geometry */
/** Robot radius in meters (26.5 mm for Pogobot). */
double   robot_radius              = 0.0265;
/** Chirality convention used by the heading detection (clockwise / counter-clockwise). */
heading_chirality_t heading_chiralty_enum = HEADING_CW;
/** Enable PID control in the differential-drive controller. */
bool     enable_pid                = true;

/* Bounds for genotype (all mutated in normalized space [0,1]) */
/** Lower/upper bound for Vicsek alignment gain β (rad/s). */
float    lo_beta   = 0.0f,   hi_beta = 15.0f;      /* rad/s */
/** Lower/upper bound for angular noise σ (rad/sqrt(s)). */
float    lo_sigma  = 0.00f,  hi_sigma = 0.80f;     /* rad/sqrt(s) */
/** Lower/upper bound for base speed v0 ∈ [0,1]. */
float    lo_speed  = 0.10f,  hi_speed = 1.00f;     /* [0..1] */
/**
 * Normalized genes (default domain is [0,1]) for the phase of the U-turn
 * and the depth of the crowding slowdown; they are usually left to the
 * optimizer but can be fixed for debugging.
 */
float lo_phi_norm = 0.0f, hi_phi_norm = 1.0f;
float lo_crowd_depth_norm = 0.0f, hi_crowd_depth_norm = 1.0f;

/* Crowding speed modulation (YAML tunable, not optimized directly) */
/**
 * Maximum multiplicative slowdown depth used when the neighbor count
 * is close to the target density Ntgt (dcrowd in the paper).
 */
double   crowd_factor_max          = 1.00;   /* ±max multiplicative change at target degree */
/**
 * Width (in neighbors) around the target neighbor count where crowding
 * effectively reduces the speed; outside this band the robot runs at v0.
 */
double   crowd_width_n             = 2.0;    /* neighbors above target_nb needed for full braking */

/* Cluster U-turn */
/**
 * Duration (ms) during which a U-turn heading override is active after
 * wall-avoidance is triggered and propagated.
 */
uint32_t cluster_u_turn_duration_ms = 1500;   /* duration of coordinated U-turn */

/* Local windows */
/**
 * Time scale (ms) used to normalize neighbor age when computing the
 * neighbor persistence metric Πi.
 */
uint32_t neighbor_persist_norm_ms  = 10000;   /* ms to normalize neighbor age */
/**
 * Maximum effective neighbor count used to scale the neighbor count
 * error in the loss (avoids over-weighting dense regions).
 */
uint8_t  neighbor_norm_max         = 12;      /* for neighbor count error scaling */

/* Targets (user-provided)
 * These correspond to the desired values of the local statistics
 * (polarization, wall/U-turn ratio, neighbor persistence and neighbor count)
 * used by the bounded multi-objective loss.
 */
/** Target local polarization (Pi) in [0,1]. */
double   target_pol   = 0.90;
/** Target fraction of time spent in wall avoidance or U-turn modes. */
double   target_wall  = 0.10;
//double   target_pers  = 0.15;
/** Lower bound of desired neighbor persistence (gas-like regime). */
double   target_pers_lo = 0.05;  // gas-like persistence
/** Upper bound of desired neighbor persistence (solid-like regime). */
double   target_pers_hi = 0.90;  // solid-like persistence
/** Target neighbor count Ntgt (robots) around which crowding slows down motion. */
double   target_nb    = 6.0;

/* Loss weights */
/** Weight of polarization term in the multi-objective loss. */
double   w_pol  = 1.0;
/** Weight of wall/U-turn ratio term in the loss. */
double   w_wall = 0.5;
/** Weight of neighbor persistence term in the loss. */
double   w_pers = 0.5;
/** Weight of neighbor count term in the loss. */
double   w_nb   = 0.25;

/* --- Learning --- */
/**
 * Duration (ms) of one evaluation window Teval during which loss is
 * accumulated before being given to the optimizer.
 */
uint32_t  eval_window_ms   = 5000;    /* ms per evaluation */
/**
 * Quiet guard time (ms) after a parameter change before loss accumulation
 * starts; this discards transient behavior.
 */
uint32_t  eval_quiet_ms    = 200;     /* guard before sampling loss */
/**
 * Target loss threshold L_target; once the EWMA loss is below this value
 * the robot considers the problem solved and stops asking for new genotypes.
 */
double    target_loss      = 0.010;   /* Stop learning if loss is below this value */
/** Smoothing factor for the EWMA of window loss. */
double    ewma_alpha_loss  = 0.05;
/** Global optimizer configuration produced from YAML settings. */
opt_cfg_t opt_cfg;                    // Configuration of optimizers

/**
 * @enum fast_transmission_strategy_t
 * @brief Strategy controlling when fast transmission cloning can occur.
 *
 * The FT mechanism compares the robot's current loss with a neighbor's
 * advertised loss and, if the neighbor is clearly better, may clone its
 * genotype immediately according to the chosen strategy.
 */
typedef enum {
    /** Always accept FT when the neighbor is strictly better. */
    ALWAYS_TRANSMIT,
    /** Never accept FT; only slow optimizer dynamics are used. */
    NEVER_TRANSMIT,
    /** Accept FT with a fixed probability when the neighbor is better. */
    TRANSMIT_WITH_PROB,
} fast_transmission_strategy_t;

/** Current fast transmission strategy mode. */
fast_transmission_strategy_t fast_transmission_strategy_mode = ALWAYS_TRANSMIT;
/** Base acceptance probability p0 used in TRANSMIT_WITH_PROB mode. */
float prob_fast_transmit = 0.20f;
/**
 * Minimal improvement in loss required to consider a neighbor for FT.
 * In absolute or relative terms depending on ::fast_transmit_use_relative.
 */
float fast_transmit_improvement_threshold = 1e-9f;
/**
 * If true, FT uses a relative improvement criterion I = ΔL / max(L_i, ε),
 * otherwise it uses absolute difference ΔL.
 */
bool fast_transmit_use_relative = true;  /* true: use relative improvement; false: absolute */

/**
 * @enum main_led_display_type_t
 * @brief LED encoding mode for monitoring the system on each robot.
 */
typedef enum {
    /** LED color encodes the current loss (blue = solved, red = poor). */
    SHOW_LOSS,
    /** LED color encodes the robot heading. */
    SHOW_ANGLE,
    /** LED color encodes the local polarization. */
    SHOW_POLARIZATION
} main_led_display_type_t;

/** Current LED feedback display mode. */
main_led_display_type_t main_led_display_mode = SHOW_LOSS;


/* ------------ Neighbors ------------- */
/**
 * @struct neighbor_t
 * @brief Local representation of a neighboring robot.
 *
 * Each entry stores minimal data necessary for local polarization,
 * neighbor persistence and the ACU alignment rule.
 */
typedef struct {
    uint16_t id;            /**< Unique robot identifier of the neighbor. */
    uint32_t last_seen_ms;  /**< Timestamp (ms) when a message was last received. */
    uint32_t first_seen_ms; /**< Timestamp (ms) when this neighbor was first observed (without interruption). */
    int16_t theta_mrad;     /**< Neighbor heading estimate in milliradians. */
    uint8_t dir;            /**< IR face index that last detected this neighbor. */
} neighbor_t;

/**
 * @brief Insert or update a neighbor entry in the neighbor array.
 *
 * If a neighbor with the given @p id already exists in @p arr, the existing
 * entry is returned. Otherwise a new neighbor is appended if there is free
 * space, initialized with current timestamps.
 *
 * @param arr Pointer to neighbor array.
 * @param nbn Pointer to the current number of stored neighbors (updated in-place).
 * @param id  Neighbor robot identifier to upsert.
 * @return Pointer to the updated or newly created neighbor, or NULL if the
 *         array is full.
 */
static neighbor_t* upsert_neighbor(neighbor_t* arr, uint8_t* nbn, uint16_t id){
    uint32_t now = current_time_milliseconds();
    for(uint8_t i=0;i<*nbn;++i) if(arr[i].id==id) return &arr[i];
    if(*nbn>=MAX_NEIGHBORS) return NULL;
    neighbor_t* n=&arr[(*nbn)++];
    n->id=id;
    n->theta_mrad=0;
    n->dir = 0;
    n->last_seen_ms=now;
    n->first_seen_ms=now;
    return n;
}

/**
 * @brief Remove stale neighbors that have not been seen for longer than ::max_age.
 *
 * This function scans the neighbor array and removes entries for which
 * (@c now - last_seen_ms) exceeds the neighbor timeout. The array is kept
 * compact by swapping removed entries with the last one.
 *
 * @param arr Neighbor array.
 * @param nbn Pointer to the number of active neighbors (updated).
 */
static void purge_old_neighbors(neighbor_t* arr, uint8_t* nbn){
    uint32_t now=current_time_milliseconds();
    for(int i=(int)*nbn-1;i>=0;--i) {
        if(now - arr[i].last_seen_ms > (int32_t)max_age){ arr[i]=arr[*nbn-1]; (*nbn)--; }
    }
}

/* -------- Genotype ------ */
/**
 * @struct genotype_t
 * @brief Optimized ACU parameters for a single robot.
 *
 * Each field lies in [0, 1] after normalization, but is interpreted in
 * physical units inside the controller (e.g., β and σ are scaled by 10).
 * These parameters are exchanged between robots, mutated by the optimizer,
 * and broadcast in messages.
 */
typedef struct {
    float beta;             /**< Alignment gain β controlling the Vicsek torque. */
    float sigma;            /**< Angular noise σ (diffusion strength). */
    float speed;            /**< Base linear speed v0 in [0, 1]. */
    float phi_norm;         /**< Normalized U-turn phase ϕ_norm ∈ [0,1], ϕ = 2πϕ_norm. */
    /**
     * Normalized slowdown depth dcrowd ∈ [0,1].
     * 0 → no slowdown at target density, 1 → maximal slowdown; when combined
     * with ::crowd_factor_max this can reduce speed to zero near Ntgt.
     */
    float crowd_depth_norm;
} genotype_t;


/* === Optimizer mapping === */
/**
 * @brief Convert an optimizer vector into a genotype_t structure.
 *
 * @param x Optimizer vector of dimension ::OPT_D.
 * @param g Output genotype structure to fill.
 */
static inline void x_to_genotype(const float *x, genotype_t *g){
    g->beta              = x[0];
    g->sigma             = x[1];
    g->speed             = x[2];
    g->phi_norm          = x[3];
    g->crowd_depth_norm  = x[4];
}

/**
 * @brief Convert a genotype_t structure into an optimizer vector.
 *
 * @param g Genotype structure.
 * @param x Output vector of length ::OPT_D.
 */
static inline void genotype_to_x(const genotype_t *g, float *x){
    x[0] = g->beta;
    x[1] = g->sigma;
    x[2] = g->speed;
    x[3] = g->phi_norm;
    x[4] = g->crowd_depth_norm;
}


/* -------------- Messaging -------------- */
/**
 * @name Message flags
 * @{
 */
/** Flag indicating that the message carries a U-turn broadcast/relay. */
enum : uint8_t { VMSGF_CLUSTER_UTURN = 0x01 };
/** @} */

/**
 * @struct acu_msg_t
 * @brief Infrared message payload exchanged between robots.
 *
 * Carries:
 *   - Heading and cluster U-turn broadcast information.
 *   - Social-learning / optimization metadata: genome, epoch and advertised loss.
 *   - HIT-specific transfer rate α.
 *
 * This structure is packed to minimize bandwidth in the infrared messages.
 */
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;            /**< Sender robot identifier. */
    int16_t  theta_mrad;           /**< Sender heading in milliradians. */
    uint8_t  flags;                /**< Bitfield of message flags (::VMSGF_CLUSTER_UTURN). */
    /* Cluster */
    int16_t  cluster_target_mrad;  /**< Target heading for U-turn, in milliradians. */
    uint32_t cluster_wall_t0_ms;   /**< Time when the wall was hit (origin of U-turn). */
    uint16_t cluster_msg_uid;      /**< Unique U-turn identifier for relay deduplication. */
    /* Social learning broadcast */
    uint32_t par_epoch;            /**< Optimizer epoch/iteration index. */
    float    par_beta;             /**< Broadcast alignment gain β. */
    float    par_sigma;            /**< Broadcast angular noise σ. */
    float    par_speed;            /**< Broadcast base speed v0. */
    float    par_phi_norm;         /**< Broadcast U-turn phase ϕ_norm. */
    float    par_loss_adv;         /**< Advertised local window loss. */
    float    par_crowd_depth_norm; /**< Broadcast crowding depth dcrowd. */
    float    hit_alpha;            /**< Transfer rate α for HIT (ignored by other optimizers). */
} acu_msg_t;

/** Compile-time sanity check of message size. */
#define MSG_SIZE ((uint16_t)sizeof(acu_msg_t))

/* -------------- USERDATA -------------- */
/**
 * @struct USERDATA
 * @brief Per-robot state used by the main controller and the simulator.
 *
 * This structure is instantiated once per robot and accessed via the
 * ::mydata pointer (see DECLARE_USERDATA/REGISTER_USERDATA).
 * It holds controller state, neighbor cache, statistics accumulators,
 * optimizer and allocator state, and some simulator-only data.
 */
typedef struct {
    /* DDK + photostart */
    ddk_t        ddk;      /**< Differential-drive kinematics and heading detection. */
    photostart_t ps;       /**< Photostart mechanism to gate experiment start. */

    /* Timing */
    uint32_t last_beacon_ms; /**< Time of last emitted message (ms). */
    uint32_t last_update_ms; /**< Time of last control step (ms). */

    /* Neighbors */
    neighbor_t neighbors[MAX_NEIGHBORS]; /**< Local neighbor cache. */
    uint8_t    nb_neighbors;             /**< Number of active neighbors. */

    /* Vicsek */
    double         theta_cmd_rad;  /**< Last commanded heading θ* (rad). */
    ddk_behavior_t prev_behavior;  /**< Previous DDK behavior (for U-turn trigger detection). */

    /* Cluster U-turn state */
    bool     cluster_turn_active;     /**< True if the robot is currently in U-turn mode. */
    double   cluster_target_rad;      /**< Target heading for U-turn (rad). */
    uint32_t cluster_wall_t0_ms;      /**< Time (ms) when local wall avoidance triggered U-turn. */
    uint32_t cluster_active_until_ms; /**< Time until which U-turn override remains active. */
    uint16_t cluster_msg_uid;         /**< U-turn unique identifier for messages. */
    uint16_t last_seen_cluster_uid;   /**< Last seen U-turn identifier. */
    bool     have_seen_cluster_uid;   /**< Whether we have received any U-turn yet. */

    /* Short-window U-turn / wall-avoidance statistics (for export) */
    double win_uturn_ms;            /**< Accumulated ms spent in U-turn mode since last export. */
    double win_wall_avoid_ms;       /**< Accumulated ms spent in wall avoidance since last export. */
    bool   uturn_active_now;        /**< True if U-turn is active at the current tick. */
    uint32_t last_export_ms;        /**< Timestamp (ms) of last simulator data export. */

    /* Local stats (phenotype) */
    double ls_pol_norm;             /**< Rayleigh-corrected local polarization Pi ∈ [0,1]. */
    double ls_wall_ratio;           /**< Wall/U-turn time ratio Wi ∈ [0,1]. */
    double ls_neighbor_persist;     /**< Neighbor persistence Πi ∈ [0,1]. */
    double ls_neighbor_count;       /**< Raw neighbor count di (not normalized). */
    double accum_total_ms;          /**< Total time accumulated for wall ratio. */
    double accum_wc_ms;             /**< Time spent in wall / U-turn behavior. */

    /* Evaluation (loss) */
    uint32_t eval_window_t0_ms;     /**< Start timestamp of current evaluation window. */
    double   accum_loss;            /**< Accumulated loss · weight over current window. */
    double   accum_w;               /**< Accumulated weight (usually time) for loss. */
    double   last_loss_for_led;     /**< Latest instantaneous or window loss shown on LED. */
    bool     reached_loss_target;   /**< True if EWMA loss is below ::target_loss. */
    double   loss_ewma;             /**< Exponential moving average of window losses. */
    int32_t  accepted_fast_transmission_from_ID;  /**< Last neighbor ID from which FT was accepted (-1 = none). */
    uint32_t accepted_fast_transmission_counter;  /**< Total number of FT events accepted since boot. */
    uint32_t accepted_fast_transmission_window_counter; /**< FT events accepted since last log print. */

    /* Genotype currently applied */
    genotype_t cur_g;               /**< Current ACU genotype used by the controller. */

    /* --- Generic optimizer --- */
    opt_t *opt;                     /**< Pointer to optimizer instance (HIT, ES1+1, etc.). */
    float  opt_lo[OPT_D];           /**< Lower bounds passed to optimizer. */
    float  opt_hi[OPT_D];           /**< Upper bounds passed to optimizer. */

    // Heap and allocator
    tiny_alloc_t ta;                /**< Tiny allocator state used by optimizer. */
    uint8_t      opt_heap[4096];    /**< Static heap backing tiny allocator. */
} USERDATA;

/** Declare and register the per-robot USERDATA structure. */
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);


/* --- Polarization (Rayleigh-corrected) --- */
/**
 * @brief Compute the Rayleigh-corrected local polarization around the robot.
 *
 * The polarization is computed over the robot and its visible neighbors,
 * then corrected for finite-sample bias using the Rayleigh correction.
 * The result is clamped to [0,1].
 *
 * @param self_heading Current heading of the robot (rad).
 * @param N_eff_out Optional pointer to store the number of neighbors used.
 * @return Local polarization Pi ∈ [0, 1] (0 = isotropic, 1 = perfectly aligned).
 */
static double compute_local_polarization_norm(double self_heading, uint32_t *N_eff_out){
    double sx = cos(self_heading), sy = sin(self_heading);
    uint32_t N = 0;

    for (uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th);
        sy += sin(th);
        ++N;
    }

    if (N_eff_out) *N_eff_out = N;
    if (N == 0) return 0.0;

    double R_bar = sqrt(sx*sx + sy*sy) / (double)N;
    double R0    = sqrt(M_PI) / (2.0 * sqrt((double)N));
    if (R0 > 0.999) R0 = 0.999;
    double P = (R_bar - R0) / (1.0 - R0);
    if (P < 0.0) P = 0.0;
    if (P > 1.0) P = 1.0;
    return P;
}

/* --- Loss (targets vs consensus) --- */

/**
 * @brief Clamp an absolute error to the [0,1] range.
 *
 * @param e Error value.
 * @return |e| clipped to [0,1].
 */
static inline double unit_err01(double e){           // absolute error in [0,1]
    double a = fabs(e);
    return (a > 1.0) ? 1.0 : a;
}

/**
 * @brief Error to a pair of target values, taking the minimal distance.
 *
 * Used for the neighbor persistence, where either a gas-like or solid-like
 * regime is acceptable.
 *
 * @param x  Observed value.
 * @param t1 First target.
 * @param t2 Second target.
 * @return Minimal error between x and t1 or t2, clipped to [0,1].
 */
static inline double dual_err01(double x, double t1, double t2){
    double d1 = unit_err01(x - t1);
    double d2 = unit_err01(x - t2);
    return fmin(d1, d2);
}

/**
 * @brief Compute the bounded multi-objective loss L ∈ [0,1].
 *
 * This function maps normalized local statistics (polarization, wall ratio,
 * neighbor persistence, neighbor count) and their target goals into a
 * bounded loss according to the formulation in the paper:
 *
 *  - Each objective is turned into a goodness score s_x ∈ [-1,1].
 *  - The weighted utility U ∈ [-1,1] is a weighted average of s_x.
 *  - The loss is L = (1 - U) / 2 ∈ [0,1], to be minimized.
 *
 * @param pol   Local polarization Pi in [0,1].
 * @param wall  Wall/U-turn ratio Wi in [0,1].
 * @param pers  Neighbor persistence Πi in [0,1].
 * @param nb    Neighbor count (not normalized).
 * @return Bounded loss L in [0,1] (0 best, 1 worst).
 */
static double compute_loss_unit01(double pol, double wall, double pers, double nb) {
    double nb_scale = (neighbor_norm_max > 0) ? (double)neighbor_norm_max : MAX_NEIGHBORS;

    double d_pol  = unit_err01(pol  - target_pol);
    double d_wall = unit_err01(wall - target_wall);
    //double d_pers = unit_err01(pers - target_pers);
    double d_pers = dual_err01(pers, target_pers_lo, target_pers_hi);
    double d_nb   = unit_err01((nb  - target_nb) / nb_scale);

    // goodness in [-1,1]
    double s_pol  = 1.0 - 2.0*d_pol;
    double s_wall = 1.0 - 2.0*d_wall;
    double s_pers = 1.0 - 2.0*d_pers;
    double s_nb   = 1.0 - 2.0*d_nb;

    double num =  w_pol*s_pol + w_wall*s_wall + w_pers*s_pers + w_nb*s_nb;
    double den = fabs(w_pol) + fabs(w_wall) + fabs(w_pers) + fabs(w_nb);
    if (den <= 0.0) return 1.0; // degenerate: treat as worst loss

    double U = num / den;       // in [-1,1]
    double L = (1.0 - U) * 0.5; // in [0,1]
    return L;
}


/**
 * @brief Helper that returns the current local loss based on instantaneous statistics.
 *
 * This function is used inside an evaluation window to accumulate loss over time.
 *
 * @return Current local loss L ∈ [0,1].
 */
static inline double compute_current_loss(void){
    return compute_loss_unit01(
        mydata->ls_pol_norm, mydata->ls_wall_ratio, mydata->ls_neighbor_persist, mydata->ls_neighbor_count);
}


/* --- Messaging TX --- */
/**
 * @brief Compose and send an ACU message on the infrared channel.
 *
 * The message encodes:
 *   - Sender ID and heading.
 *   - Optional U-turn broadcast information (if a U-turn is active).
 *   - Current genotype, optimizer epoch and advertised loss.
 *   - HIT transfer rate α as metadata for social learning.
 *
 * @note The HIT maturation disabling (no TX during immature phase) can be
 *       re-enabled via the commented condition if desired.
 *
 * @return true if the message was successfully queued for transmission.
 */
static bool send_message(void){
    uint32_t now=current_time_milliseconds();

//    /* For HIT, do not transmit during maturation (before window is full) */
//    if (!opt_ready(mydata->opt)) return false;

    double heading_now = heading_detection_estimate(&mydata->ddk.hd);
    acu_msg_t m = {
        .sender_id = pogobot_helper_getid(), .theta_mrad= rad_to_mrad(heading_now),
        .flags = 0,
        .cluster_target_mrad = 0, .cluster_wall_t0_ms = 0u, .cluster_msg_uid = 0u,
        //.par_epoch     = mydata->par_epoch_local,
        .par_epoch     = opt_iterations(mydata->opt),
        .par_beta      = mydata->cur_g.beta,
        .par_sigma     = mydata->cur_g.sigma,
        .par_speed     = mydata->cur_g.speed,
        .par_phi_norm  = mydata->cur_g.phi_norm,
        .par_loss_adv  = (float)mydata->last_loss_for_led,
        //.par_loss_adv  = opt_get_last_advert(mydata->opt),
        .par_crowd_depth_norm  = mydata->cur_g.crowd_depth_norm,
        .hit_alpha  = opt_get_alpha(mydata->opt),
    };

    if (mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms)){
        m.flags               |= VMSGF_CLUSTER_UTURN;
        m.cluster_target_mrad  = rad_to_mrad(mydata->cluster_target_rad);
        m.cluster_wall_t0_ms   = mydata->cluster_wall_t0_ms;
        m.cluster_msg_uid      = mydata->cluster_msg_uid;
    }

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t*)&m, MSG_SIZE);
}

/**
 * @brief Apply the fast transmission (FT) rule to an incoming message.
 *
 * This function implements the thresholded probabilistic cloning mechanism:
 *   - Compare the robot's own loss with the neighbor's advertised loss.
 *   - Compute an improvement I in either relative or absolute terms.
 *   - Gate cloning based on a threshold and the configured FT strategy:
 *       * ALWAYS_TRANSMIT: clone whenever neighbor is strictly better.
 *       * NEVER_TRANSMIT: FT is disabled.
 *       * TRANSMIT_WITH_PROB: clone with probability ::prob_fast_transmit.
 *   - If accepted and the local loss is above ::target_loss, copy the
 *     neighbor's genotype and re-seed the optimizer at that point.
 *
 * @param m Pointer to decoded incoming ACU message.
 */
void fast_transmission(acu_msg_t const* m) {
    // Fast transmission of genotypes
    const float myL     = (float)mydata->last_loss_for_led;
    const float theirL  = m->par_loss_adv;
    const float deltaL  = myL - theirL;  /* positive if neighbor is better (lower loss) */
    const float eps    = 1e-9f;

    /* Choose improvement domain: absolute or relative */
    float I = fast_transmit_use_relative
              ? (deltaL / fmaxf(myL, eps))
              : deltaL;

    /* Threshold in same domain as I */
    float thr = fast_transmit_use_relative
                ? (fast_transmit_improvement_threshold / fmaxf(myL, eps))
                :  fast_transmit_improvement_threshold;

    /* Gate: only consider if strictly better than threshold and we aren't already "good enough". */
    const bool strictly_better = (I > thr);
    //const bool strictly_better = (deltaL > fast_transmit_improvement_threshold);

    float p_accept = 0.0f;
    if (fast_transmission_strategy_mode == ALWAYS_TRANSMIT) {
        p_accept = 1.0f;
    } else if (fast_transmission_strategy_mode == NEVER_TRANSMIT) {
        p_accept = 0.0f;
    } else if (fast_transmission_strategy_mode == TRANSMIT_WITH_PROB) {
        p_accept = prob_fast_transmit;
    }

    /* Safety clamps */
    if (p_accept < 0.0f) p_accept = 0.0f;
    if (p_accept > 1.0f) p_accept = 1.0f;

    /* Stochastic accept */
    bool do_fast_transmit = (rand_uniform(0.0, 1.0) <= (double)p_accept);

    if (do_fast_transmit && strictly_better && !mydata->reached_loss_target) {
        mydata->accepted_fast_transmission_from_ID = m->sender_id;
        mydata->accepted_fast_transmission_counter += 1;
        mydata->accepted_fast_transmission_window_counter += 1;
        /* Accept the genome */
        mydata->cur_g.beta  = m->par_beta;
        mydata->cur_g.sigma = m->par_sigma;
        mydata->cur_g.speed = m->par_speed;
        mydata->cur_g.phi_norm = m->par_phi_norm;
        mydata->cur_g.crowd_depth_norm  = m->par_crowd_depth_norm;
        float new_x[OPT_D];
        genotype_to_x(&mydata->cur_g, new_x);
        opt_set_x(mydata->opt, new_x);
    }
}

/* --- Messaging RX --- */
/**
 * @brief Process an incoming message from the infrared stack.
 *
 * This function:
 *   - Lets the differential-drive kinematics module process its own messages.
 *   - Discards messages that are too short or from ourselves.
 *   - Updates or inserts neighbor entries and their estimated heading.
 *   - Relays U-turn broadcasts when appropriate.
 *   - Applies the fast transmission rule.
 *   - Feeds the remote genome and advertised loss to the optimizer for
 *     social learning (HIT / SL).
 *
 * @param mr Pointer to the received message wrapper.
 */
static void process_message(message_t* mr){
    if (diff_drive_kin_process_message(&mydata->ddk, mr)) return;
    if (mr->header.payload_length < MSG_SIZE) return;
    acu_msg_t const* m=(acu_msg_t const*)mr->payload; if (m->sender_id == pogobot_helper_getid()) return;

    /* Which IR face saw this packet? */
    uint8_t rx_face = mr->header._receiver_ir_index;
    if (rx_face >= IR_RX_COUNT) return;

    /* Neighbor & headings */
    neighbor_t* n = upsert_neighbor(mydata->neighbors, &mydata->nb_neighbors, m->sender_id);
    if(!n) return;
    n->theta_mrad = m->theta_mrad;
    n->last_seen_ms  = current_time_milliseconds();
    n->dir = rx_face;

    /* Cluster relay */
    if (m->flags & VMSGF_CLUSTER_UTURN){
        bool newer = (!mydata->cluster_turn_active) || (m->cluster_wall_t0_ms > mydata->cluster_wall_t0_ms) || (!mydata->have_seen_cluster_uid) || (m->cluster_msg_uid != mydata->last_seen_cluster_uid);
        if(newer) {
            mydata->cluster_target_rad = mrad_to_rad(m->cluster_target_mrad);
            mydata->cluster_wall_t0_ms = m->cluster_wall_t0_ms;
            mydata->cluster_active_until_ms = m->cluster_wall_t0_ms + cluster_u_turn_duration_ms;
            mydata->cluster_turn_active = true;
            mydata->last_seen_cluster_uid = m->cluster_msg_uid;
            mydata->have_seen_cluster_uid = true;
        }
    }

    fast_transmission(m);

    // Pass the received genome to the optimizer, if needed
    float x_remote[OPT_D] = {
        m->par_beta, m->par_sigma, m->par_speed,
        m->par_phi_norm, m->par_crowd_depth_norm};
    opt_observe_remote(mydata->opt, m->sender_id, m->par_epoch, x_remote, m->par_loss_adv, m->hit_alpha);
}

/* --- LED --- */
/**
 * @brief Update the RGB LED color based on the selected display mode.
 *
 * Modes:
 *  - SHOW_LOSS: color encodes current loss (red = high loss, green = low),
 *    with blue when the target loss has been reached.
 *  - SHOW_ANGLE: color encodes the robot heading on the HSV circle.
 *  - SHOW_POLARIZATION: color encodes local polarization.
 *
 * @param heading Current heading (rad) used for SHOW_ANGLE mode.
 */
static void led_update(double heading){
    if (main_led_display_mode == SHOW_LOSS){
        if (mydata->reached_loss_target) {
            pogobot_led_setColor(0, 0, 25);
            return;
        }
        double L = 1.0 - exp(-10. * mydata->last_loss_for_led);
        if (L < 0.0) L = 0.0;
        if (L > 1.0) L = 1.0;
        float hue = (float)((1.0 - L) * 120.0);
        uint8_t r8,g8,b8; hsv_to_rgb(hue, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8 = SCALE_0_255_TO_0_25(g8); b8 = SCALE_0_255_TO_0_25(b8);
        if (r8==0 && g8==0 && b8==0) r8 = 1;
        pogobot_led_setColor(r8,g8,b8);
        return;
    }

    if (main_led_display_mode == SHOW_ANGLE){
        if (heading < 0.0) heading += 2.0*M_PI;
        float hue_deg = (float)(heading * 180.0/M_PI);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8 = SCALE_0_255_TO_0_25(g8); b8 = SCALE_0_255_TO_0_25(b8);
        if (r8==0 && g8==0 && b8==0) r8 = 1;
        pogobot_led_setColor(r8,g8,b8);
        return;
    }

    if (main_led_display_mode == SHOW_POLARIZATION){
        float hue_deg = (float)(mydata->ls_pol_norm * 360.0);
        uint8_t r8,g8,b8; hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8,&g8,&b8);
        r8 = SCALE_0_255_TO_0_25(r8); g8 = SCALE_0_255_TO_0_25(g8); b8 = SCALE_0_255_TO_0_25(b8);
        if (r8==0 && g8==0 && b8==0) r8 = 1;
        pogobot_led_setColor(r8,g8,b8);
        return;
    }
}

/* --- Init --- */
/**
 * @brief Initialize and create the optimizer, and seed the initial genotype.
 *
 * The function:
 *   - Fills the optimizer lower/upper bounds from YAML tunables.
 *   - Creates the optimizer instance (algorithm selected in ::opt_cfg).
 *   - Randomizes an initial genotype within bounds.
 *   - Calls ::opt_tell_initial with a neutral loss and obtains the first
 *     candidate via ::opt_ask, which is mapped into ::mydata->cur_g.
 */
static void optimization_init(void) {
    /* Bounds for each coordinate (reuse YAML tunables) */
    mydata->opt_lo[0] = (float)lo_beta;             mydata->opt_hi[0] = (float)hi_beta;
    mydata->opt_lo[1] = (float)lo_sigma;            mydata->opt_hi[1] = (float)hi_sigma;
    mydata->opt_lo[2] = (float)lo_speed;            mydata->opt_hi[2] = (float)hi_speed;
    mydata->opt_lo[3] = (float)lo_phi_norm;         mydata->opt_hi[3] = (float)hi_phi_norm;
    mydata->opt_lo[4] = (float)lo_crowd_depth_norm; mydata->opt_hi[4] = (float)hi_crowd_depth_norm;

    int ok = opt_create(&mydata->opt, &mydata->ta, OPT_D,
                        OPT_HIT, OPT_MINIMIZE,
                        mydata->opt_lo, mydata->opt_hi, &opt_cfg);
    if (!ok || !mydata->opt){
#ifdef SIMULATOR
        printf("[OPT] create failed.\n");
#endif
    }

    /* Seed the optimizer around a random feasible genotype (keeps old behavior) */
    opt_randomize_x(mydata->opt, /*seed*/ rand() % 10000 + pogobot_helper_getid());

    /* Prime algorithms that need an initial fitness (SL/HIT also accept this) */
    /* Start with a neutral number; the true loss will be fed next step. */
    const float f_parent = 1000.0f;
    opt_tell_initial(mydata->opt, f_parent);

    /* Use optimizer's new x as controller genotype (authoritative) */
    const float *x_try = opt_ask(mydata->opt, f_parent);
    if (x_try) {
        x_to_genotype(x_try, &mydata->cur_g);
    } else {
        printf("Error creating first individual!\n");
        exit(1);
    }
}


/**
 * @brief Per-robot initialization hook called once at startup.
 *
 * This function is registered as the main user initialization callback in
 * ::main and performs:
 *   - Seeding of the random number generator.
 *   - Initialization of global Pogobase parameters (loop rate, message hooks).
 *   - Tiny allocator setup for the optimizer.
 *   - Differential-drive kinematics and photostart setup.
 *   - Initialization of U-turn and stats accumulators.
 *   - Creation and seeding of the optimizer and genotype.
 */
static void user_init(void){
    srand(pogobot_helper_getRandSeed());
    memset(mydata, 0, sizeof(*mydata));

    // General initialization
    main_loop_hz = MAIN_LOOP_HZ;
    max_nb_processed_msg_per_tick = MAX_NB_MSGS_PROCESSED_PER_TICK;
    percent_msgs_sent_per_ticks = PERCENT_MSG_SENT;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;
    error_codes_led_idx = 3;

    // Heap initialization
    const uint16_t k_classes[] = { 32, 48, 64, 128, 512 };
    tiny_alloc_init(&mydata->ta, mydata->opt_heap, sizeof(mydata->opt_heap),
                    k_classes, (uint16_t)(sizeof(k_classes)/sizeof(k_classes[0])));

    /* DDK */
    diff_drive_kin_init_default(&mydata->ddk);
    photostart_init(&mydata->ps);
    photostart_set_ewma_alpha(&mydata->ps, 0.30);
    diff_drive_kin_set_photostart(&mydata->ddk, &mydata->ps);
    heading_detection_set_geometry(&mydata->ddk.hd, alpha_deg, robot_radius);
    heading_detection_set_chirality(&mydata->ddk.hd, heading_chiralty_enum);
    mydata->prev_behavior = diff_drive_kin_get_behavior(&mydata->ddk);
    diff_drive_kin_set_pid_enabled(&mydata->ddk, enable_pid);

    /* Cluster defaults */
    mydata->cluster_turn_active=false;
    mydata->cluster_target_rad=0.0;
    mydata->cluster_wall_t0_ms=0u;
    mydata->cluster_active_until_ms=0u;
    mydata->have_seen_cluster_uid=false;

    mydata->win_uturn_ms = 0.0;
    mydata->win_wall_avoid_ms = 0.0;
    mydata->uturn_active_now = false;
    mydata->last_export_ms = current_time_milliseconds();

    /* Stats init */
    mydata->accum_total_ms = 1e-9;
    mydata->accum_wc_ms = 0.0;
    mydata->reached_loss_target = false;
    mydata->loss_ewma = 0.5;
    mydata->last_loss_for_led = 1.0;
    mydata->accepted_fast_transmission_from_ID = -1; // -1 = No accepted transmission
    mydata->accepted_fast_transmission_counter = 0;
    mydata->accepted_fast_transmission_window_counter = 0;

    /* --- Optimizer init --- */
    optimization_init();

#ifdef SIMULATOR
    printf("Self-adaptive ACU\n");
#endif
}

/**
 * @brief Update local statistics (phenotype) and U-turn triggers.
 *
 * This function:
 *   - Detects the rising edge of wall-avoidance behavior to trigger a
 *     new U-turn broadcast using the current genotype phase ϕ_norm.
 *   - Updates short-window counters for U-turn and wall-avoidance times.
 *   - Computes local polarization, wall ratio, neighbor persistence and
 *     neighbor count used by the loss function.
 *
 * @param heading Current robot heading (rad).
 * @param dt_s    Time step since last update (seconds).
 * @param now     Current time (ms).
 * @return Effective number of neighbors used in polarization (for diagnostics).
 */
static uint32_t compute_local_stats(double heading, double dt_s, uint32_t now) {
    /* Cluster origin (rising edge of avoidance): fixed phi from genotype */
    ddk_behavior_t beh = diff_drive_kin_get_behavior(&mydata->ddk);
    if (mydata->prev_behavior != DDK_BEHAVIOR_AVOIDANCE && beh == DDK_BEHAVIOR_AVOIDANCE){
        double phi = 2.0*M_PI * (double)mydata->cur_g.phi_norm;
        mydata->cluster_target_rad      = wrap_pi(heading + phi);
        mydata->cluster_wall_t0_ms      = now;
        mydata->cluster_active_until_ms = now + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active     = true;
        mydata->cluster_msg_uid         = (uint16_t)(rand() & 0xFFFF);
        mydata->last_seen_cluster_uid   = mydata->cluster_msg_uid;
        mydata->have_seen_cluster_uid   = true;
    }
    mydata->prev_behavior = beh;

    // per-tick instantaneous flags
    bool avoid_now = (beh == DDK_BEHAVIOR_AVOIDANCE);
    bool uturn_now = (mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms));
    mydata->uturn_active_now = uturn_now;
    // per-window accumulators (ms) that we will flush at export
    double dt_ms = dt_s * 1000.0;
    if (avoid_now) mydata->win_wall_avoid_ms += dt_ms;
    if (uturn_now) mydata->win_uturn_ms      += dt_ms;

    /* Local stats */
    uint32_t N_eff = 0;
    mydata->ls_pol_norm = compute_local_polarization_norm(heading, &N_eff);
    mydata->accum_total_ms += dt_s * 1000.0;
    if (beh == DDK_BEHAVIOR_AVOIDANCE || (mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms))){
        mydata->accum_wc_ms += dt_s * 1000.0;
    }
    mydata->ls_wall_ratio = clamp01(mydata->accum_wc_ms / mydata->accum_total_ms);

    double sum_norm_age = 0.0;
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double age_ms = (double)(now - mydata->neighbors[i].first_seen_ms);
        double norm   = (neighbor_persist_norm_ms>0) ? fmin(age_ms / (double)neighbor_persist_norm_ms, 1.0) : 0.0;
        sum_norm_age += norm;
    }
    mydata->ls_neighbor_persist = (mydata->nb_neighbors>0) ? (sum_norm_age / (double)mydata->nb_neighbors) : 0.0;
    mydata->ls_neighbor_count   = (double)mydata->nb_neighbors;
    return N_eff;
}


/**
 * @brief Single ACU model step: compute commanded heading and drive the DDK.
 *
 * Implements:
 *   - Vicsek-style alignment with neighbors and angular diffusion (β, σ).
 *   - U-turn override of the commanded heading during an active U-turn.
 *   - V-shaped crowding-based speed modulation as a function of neighbor count.
 *   - Normalization and clamping of effective speed before sending to the
 *     differential-drive kinematics.
 *
 * @param heading Current heading θ (rad).
 * @param dt_s    Time step (seconds).
 * @param now     Current time (ms).
 */
static void acu_step(double heading, double dt_s, uint32_t now) {
    /* Vicsek mean */
    double sx = cos(heading), sy = sin(heading);
    for (uint8_t i=0; i<mydata->nb_neighbors; ++i){
        double th = mrad_to_rad(mydata->neighbors[i].theta_mrad);
        sx += cos(th);
        sy += sin(th);
    }
    double theta_mean = (sx==0.0 && sy==0.0) ? heading : atan2(sy, sx);
    double theta_cmd  = (mydata->cluster_turn_active && (now < mydata->cluster_active_until_ms))
                        ? mydata->cluster_target_rad : theta_mean;
    mydata->theta_cmd_rad = theta_cmd;

    /* dtheta from Vicsek */
    double err = wrap_pi(theta_cmd - heading);
    double dtheta = ((double)mydata->cur_g.beta * 10.0) * sin(err) * dt_s;

    /* Angular diffusion */
    if (mydata->cur_g.sigma > 0.0){
        double z  = randn_box_muller();
        dtheta += ((double)mydata->cur_g.sigma * 10.0) * sqrt(dt_s) * z;
    }

    /* --- V-shaped crowding: slowest at target_nb, faster as you deviate --- */
    double deg_local = (double)mydata->nb_neighbors;

    /* avoid zero/neg width; “distance” normalization */
    double width_n = (crowd_width_n > 1.0 ? crowd_width_n : 1.0);

    /* normalized deviation |e| in [0,1] (≥1 means “far” from target) */
    double e_abs = fabs(deg_local - target_nb);
    double u = e_abs / width_n;
    if (u > 1.0) u = 1.0;

    /* depth of slowdown at the target (k in [0,1]) */
    double k = crowd_factor_max * (double)mydata->cur_g.crowd_depth_norm;

    /* multiplicative speed factor:
       - at e=0: factor = 1 - k  (minimum speed)
       - at |e|≥width_n: factor = 1 (back to base speed) */
    double factor = 1.0 - k * (1.0 - u);
    if (factor < 0.0) factor = 0.0;     // safety (shouldn’t be needed if k≤1)

    /* apply to base speed */
    double speed_eff = mydata->cur_g.speed * factor;
    if (!isfinite(speed_eff) || speed_eff < 0.0) speed_eff = 0.0;
    if (speed_eff > 1.0) speed_eff = 1.0;

    /* Drive DDK with *effective* speed */
    diff_drive_kin_step(&mydata->ddk, (float)speed_eff, dtheta, heading);
}

/**
 * @brief Perform a single optimization step given the elapsed time.
 *
 * Within each evaluation window:
 *   - Accumulate weighted loss over time (excluding the quiet guard).
 *   - When the window ends, compute the average loss and feed it to
 *     the optimizer via ::opt_tell.
 *   - If the EWMA loss is above the target, request a new genotype
 *     via ::opt_ask and update ::mydata->cur_g.
 *   - Otherwise mark that the loss target has been reached.
 *
 * @param dt_s Elapsed time since last call (seconds).
 * @param now  Current time (ms).
 */
static void perform_optim_step(double dt_s, double now) {
    uint32_t t0 = mydata->eval_window_t0_ms;
    if (now > t0 + eval_quiet_ms){
        double loss = compute_current_loss();
        double step_ms = dt_s * 1000.0;
        mydata->accum_loss += loss * step_ms;
        mydata->accum_w    += step_ms;
        mydata->last_loss_for_led = loss;
    }
    if (now - t0 >= eval_window_ms){
        float window_loss = (mydata->accum_w > 1e-3)
                            ? (float)(mydata->accum_loss / mydata->accum_w)
                            : (float)mydata->last_loss_for_led;
        window_loss = (float)clamp01(window_loss); /* Ensure bounded losses */
        mydata->loss_ewma = (1.0 - ewma_alpha_loss) * mydata->loss_ewma + ewma_alpha_loss * window_loss;
        mydata->loss_ewma = clamp01(mydata->loss_ewma); /* Clamp EWMA too, to be safe */
        //mydata->last_loss_for_led = mydata->loss_ewma; // window_loss;

        // Tell the optimizer the achieved score
        (void)opt_tell(mydata->opt, window_loss);

        // Ask for a new individual
        if (mydata->loss_ewma > target_loss) {
            mydata->reached_loss_target = false;
            const float *x_try = opt_ask(mydata->opt, window_loss);
            if (x_try) {
                x_to_genotype(x_try, &mydata->cur_g);
            }
        } else {
            mydata->reached_loss_target = true;
        }

        /* Reset window stats */
        mydata->eval_window_t0_ms = now;
        mydata->accum_loss = 0.0;
        mydata->accum_w    = 0.0;
    }
}


/**
 * @brief Per-tick main control step for each robot.
 *
 * This function is called at ::MAIN_LOOP_HZ and implements the full
 * control pipeline:
 *   - Photostart gate and safe stop before experiment start.
 *   - Time-step computation and clamping.
 *   - Heading estimation and neighbor maintenance.
 *   - Local statistic computation.
 *   - ACU model step (alignment, noise, crowding, U-turns).
 *   - Optimizer window update.
 *   - LED feedback update.
 *
 * In simulator mode, it also periodically prints a log line.
 */
static void user_step(void){
    // Photostart gate
    if (!photostart_step(&mydata->ps)){
        pogobot_led_setColors(20,0,20,0);
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    // Time management
    uint32_t now  = current_time_milliseconds();
    double dt_s = (now - mydata->last_update_ms) * 1e-3;
    if (dt_s < 0.0) dt_s = 0.0;
    if (dt_s > cont_max_dt_s) dt_s = cont_max_dt_s;
    mydata->last_update_ms = now;

    // Current heading
    double heading = heading_detection_estimate(&mydata->ddk.hd);

    // Neighbor maintenance
    purge_old_neighbors(mydata->neighbors, &mydata->nb_neighbors);

    // Compute local stats
    uint32_t N_eff = compute_local_stats(heading, dt_s, now);
    (void)N_eff;
    mydata->accepted_fast_transmission_from_ID = -1; // Set to a neighbor ID this step if we accept a fast transmission event

    // ACU model
    acu_step(heading, dt_s, now);

    // Accumulate & step window
    perform_optim_step(dt_s, now);

    // LED
    led_update(heading);

#ifdef SIMULATOR
    // Print log message
    if (pogobot_ticks % 1200 == 0){
        printf("[ID %u] pol=%.2f wall=%.2f pers=%.2f nb=%.2f  | beta=%.2f sigma=%.2f v=%.2f phiN=%.2f cs=%.2f  | L_inst~%.3f L_ewma=%.3f FT_nb=%u\n",
               pogobot_helper_getid(),
               mydata->ls_pol_norm, mydata->ls_wall_ratio, mydata->ls_neighbor_persist, mydata->ls_neighbor_count,
               mydata->cur_g.beta, mydata->cur_g.sigma, mydata->cur_g.speed, mydata->cur_g.phi_norm,
               mydata->cur_g.crowd_depth_norm,
               mydata->last_loss_for_led, mydata->loss_ewma, mydata->accepted_fast_transmission_window_counter);
        mydata->accepted_fast_transmission_window_counter = 0;
    }
#endif
}


/* --- Simulator hooks & main --- */
#ifdef SIMULATOR
/**
 * @brief Declare all the columns used in simulator data exports.
 *
 * Columns include genotype parameters, local statistics and U-turn/wall
 * durations over the last export window.
 */
static void create_data_schema(void){
    data_add_column_float16("loss_led");
    data_add_column_float16("beta");
    data_add_column_float16("sigma");
    data_add_column_float16("speed");
    data_add_column_float16("phi_norm");
    data_add_column_float16("crowd_depth_norm");
    data_add_column_float16("pol_local");
    data_add_column_float16("wall_local");
    data_add_column_float16("pers_local");
    data_add_column_float16("nb_local");

    data_add_column_float16("uturn_active_ms_since_last");
    data_add_column_float16("wall_avoid_ms_since_last");
}

/**
 * @brief Export current per-robot data to the simulator logging system.
 *
 * This function is called periodically by Pogosim and writes both genotype
 * and phenotype information, as well as U-turn and wall-avoid durations
 * since the last export. Short-window accumulators are reset afterwards.
 */
static void export_data(void){
    data_set_value_float16("loss_led", mydata->last_loss_for_led);
    data_set_value_float16("beta",  (float)mydata->cur_g.beta);
    data_set_value_float16("sigma", (float)mydata->cur_g.sigma);
    data_set_value_float16("speed", (float)mydata->cur_g.speed);
    data_set_value_float16("phi_norm", (float)mydata->cur_g.phi_norm);
    data_set_value_float16("crowd_depth_norm",  (float)mydata->cur_g.crowd_depth_norm);
    data_set_value_float16("pol_local",  mydata->ls_pol_norm);
    data_set_value_float16("wall_local", mydata->ls_wall_ratio);
    data_set_value_float16("pers_local", mydata->ls_neighbor_persist);
    data_set_value_float16("nb_local",   mydata->ls_neighbor_count);

    uint32_t now_ms = current_time_milliseconds();
    data_set_value_float16("uturn_active_ms_since_last", mydata->win_uturn_ms);
    data_set_value_float16("wall_avoid_ms_since_last",   mydata->win_wall_avoid_ms);

    // flush window accumulators so next export starts fresh
    mydata->win_uturn_ms = 0.0;
    mydata->win_wall_avoid_ms = 0.0;
    mydata->last_export_ms = now_ms;

}


/**
 * @brief Optional global-step hook called each simulator tick.
 *
 * Currently unused, but can be used to implement time-dependent global
 * changes (e.g., switching target goals or arena parameters).
 */
static void global_step(void) {
//    uint32_t now  = current_time_milliseconds();
//    if (now >= 100000) {
//        target_pol = 0.0f;
//        //printf("DEBUG global step: %lu\n", now);
//    }
}


#ifndef OPT_CLAMP
#define OPT_CLAMP(v,a,b) ((v)<(a)?(a):((v)>(b)?(b):(v)))
#endif

/**
 * @brief Build an optimizer configuration from the YAML configuration tree.
 *
 * This function:
 *   - Reads the @c optimizer.algorithm string and selects the corresponding
 *     optimizer type (1+1-ES, SPSA, PGPE, SEP-CMA-ES, SL, HIT).
 *   - Fills the relevant sub-config fields from YAML entries, with sensible
 *     defaults when missing.
 *   - Leaves ::opt_cfg.sz fields (e.g., SEP-CMA-ES population) consistent
 *     with the problem dimension ::OPT_D.
 *
 * @return Fully filled ::opt_cfg_t structure.
 */
static opt_cfg_t create_opt_cfg() {
    // Set algorithm from configuration
    opt_algo_t algo;
    char algo_name[32] = "hit";
    init_string_from_configuration(algo_name, "optimizer.algorithm", 32);
    if (strcasecmp(algo_name, "es1p1")==0 ) { algo = OPT_ES1P1; }
    else if (strcasecmp(algo_name, "spsa")==0 ) { algo = OPT_SPSA; }
    else if (strcasecmp(algo_name, "pgpe")==0 ) { algo = OPT_PGPE; }
    else if (strcasecmp(algo_name, "sep-cmaes")==0 ) { algo = OPT_SEP_CMAES; }
    else if (strcasecmp(algo_name, "sl")==0 ) { algo = OPT_SOCIAL_LEARNING; }
    else if (strcasecmp(algo_name, "hit")==0 ) { algo = OPT_HIT; }
    else {
        printf("Unknown value of configuration entry 'optimizer.algorithm'. Can only be 'es1p1', 'spsa', 'pgpe', 'sep-cmaes', 'sl' or 'hit'.\n");
        exit(1);
    }

    // Set dimension
    int n = OPT_D;
    (void)n;

    // Create config
    opt_cfg_t c;
    memset(&c, 0, sizeof(c));
    c.use_defaults = 0;

    switch (algo){
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
            c.P.sep.weights = NULL; /* default log weights inside backend */
            c.P.sep.cc = c.P.sep.cs = c.P.sep.c1 = c.P.sep.cmu = c.P.sep.damps = 0.0f;
            c.sz.lambda = (uint16_t)OPT_CLAMP(2*n, 4, 64); /* small by default */
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
            init_float_from_configuration(&c.P.hit.alpha      , "optimizer.hit.alpha",       0.35f); /* initial transfer rate α */
            init_float_from_configuration(&c.P.hit.sigma      , "optimizer.hit.sigma",       0.15f); /* mutation stddev on genome coords  */
            init_int32_from_configuration(&c.P.hit.eval_T     , "optimizer.hit.eval_T",      5);     /* sliding-window length / maturation */
            init_bool_from_configuration(&c.P.hit.evolve_alpha, "optimizer.hit.evolve_alpha",true);  /* α is evolvable */
            init_float_from_configuration(&c.P.hit.alpha_sigma, "optimizer.hit.alpha_sigma", 1e-3f); /* mutation stddev on α */
            init_float_from_configuration(&c.P.hit.alpha_min  , "optimizer.hit.alpha_min",   0.0f);  /* clamp α to [0, 0.9] */
            init_float_from_configuration(&c.P.hit.alpha_max  , "optimizer.hit.alpha_max",   0.9f);  /* clamp α to [0, 0.9] */
        } break;
    }
    return c;
}

/**
 * @brief Global setup hook for the simulator.
 *
 * This function:
 *   - Imports all YAML-exposed tunables (ACU parameters, loss goals and
 *     weights, learning schedule).
 *   - Optionally clamps the genotype to a fixed set of parameters
 *     (for non-adaptive baselines).
 *   - Builds the optimizer configuration via ::create_opt_cfg.
 *   - Configures LED display mode and fast transmission strategy.
 */
static void global_setup(void){
    /* Import tunables */
    init_from_configuration(max_age);
    init_from_configuration(cont_max_dt_s);
    init_from_configuration(alpha_deg);
    init_from_configuration(robot_radius);
    char heading_chiralty[32] = "cw";
    init_array_from_configuration(heading_chiralty);
    heading_chiralty_enum = (strcasecmp(heading_chiralty,"ccw")==0)?HEADING_CCW:HEADING_CW;
    init_from_configuration(enable_pid);

    /* Bounds */
    init_from_configuration(lo_beta);
    init_from_configuration(hi_beta);
    init_from_configuration(lo_sigma);
    init_from_configuration(hi_sigma);
    init_from_configuration(lo_speed);
    init_from_configuration(hi_speed);
    init_from_configuration(crowd_factor_max);
    init_from_configuration(crowd_width_n);

    // Fixed genotype? Useful to just check model behavior
    bool fixed_genotype;
    init_from_configuration(fixed_genotype);
    if (fixed_genotype) {
        float fixed_beta, fixed_sigma, fixed_speed, fixed_phi_norm, fixed_crowd_depth_norm;
        init_from_configuration(fixed_beta);
        init_from_configuration(fixed_sigma);
        init_from_configuration(fixed_speed);
        init_from_configuration(fixed_phi_norm);
        init_from_configuration(fixed_crowd_depth_norm);
        lo_beta=hi_beta=fixed_beta;
        lo_sigma=hi_sigma=fixed_sigma;
        lo_speed=hi_speed=fixed_speed;
        lo_phi_norm=hi_phi_norm = fminf(1.0f, fmaxf(0.0f, fixed_phi_norm));
        lo_crowd_depth_norm=hi_crowd_depth_norm = fminf(1.0f, fmaxf(0.0f, fixed_crowd_depth_norm));
    }

    /* Cluster */
    init_from_configuration(cluster_u_turn_duration_ms);

    /* Targets + weights */
    init_from_configuration(target_pol);
    init_from_configuration(target_wall);
    //init_from_configuration(target_pers);
    init_from_configuration(target_pers_lo);
    init_from_configuration(target_pers_hi);
    init_from_configuration(target_nb);
    init_from_configuration(w_pol);
    init_from_configuration(w_wall);
    init_from_configuration(w_pers);
    init_from_configuration(w_nb);

    /* Learning schedule */
    init_from_configuration(eval_window_ms);
    init_from_configuration(eval_quiet_ms);
    init_from_configuration(target_loss);
    init_from_configuration(ewma_alpha_loss);
    opt_cfg = create_opt_cfg();

    /* LEDs */
    char main_led_display[32] = "loss";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display,"loss")==0 ) { main_led_display_mode = SHOW_LOSS; }
    else if (strcasecmp(main_led_display,"angle")==0){ main_led_display_mode = SHOW_ANGLE; }
    else if (strcasecmp(main_led_display,"polarization")==0){ main_led_display_mode = SHOW_POLARIZATION; }
    else {
        printf("Unknown value of configuration entry 'main_led_display'. Can only be 'loss', 'angle' or 'polarization'.\n");
        exit(1);
    }

    // Fast transmission strategy
    char fast_transmission_strategy[32] = "always";
    init_array_from_configuration(fast_transmission_strategy);
    if (strcasecmp(fast_transmission_strategy, "always") == 0)      { fast_transmission_strategy_mode = ALWAYS_TRANSMIT; }
    else if (strcasecmp(fast_transmission_strategy, "never") == 0)  { fast_transmission_strategy_mode = NEVER_TRANSMIT; }
    else if (strcasecmp(fast_transmission_strategy, "prob") == 0)   { fast_transmission_strategy_mode = TRANSMIT_WITH_PROB; }
    else {
        printf("Unknown value of configuration entry 'fast_transmission_strategy'. Can only be 'always', 'never', or 'prob'.\n");
        exit(1);
    }
    init_from_configuration(prob_fast_transmit);
    init_from_configuration(fast_transmit_improvement_threshold);
    init_from_configuration(fast_transmit_use_relative);
}
#endif

/**
 * @brief Program entry point.
 *
 * Initializes the Pogobot runtime, registers user callbacks (init/step)
 * and, in simulator mode, the global setup and data export hooks.
 *
 * @return 0 on normal termination (should not normally return).
 */
int main(void){
    pogobot_init();
    pogobot_start(user_init, user_step);
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");

    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_global_step,        global_step);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker

