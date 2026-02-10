# ACU-selfadapt

This project implements a **self-adaptive Vicsek-style controller** with **crowding** and **collective U-turns** (ACU), combined with **online decentralized optimization** and **social learning with fast transmission**.

Each robot:
- Follows the ACU motility model (alignment + crowding + U-turns).
- Measures local statistics (polarization, wall/U-turn ratio, neighbor persistence, neighbor count).
- Minimizes a local loss using a lightweight optimizer (HIT, 1+1-ES, etc.).
- Exchanges its genotype and loss with neighbors and can **clone** clearly better neighbors via a **fast transmission** rule.

The simulator runs swarms of such robots in different arena geometries (e.g. torus, disk, star) and shows how they **self-adapt** their microscopic control parameters to reach target macroscopic behaviors (e.g. flocking).

Here is an example for a swarm in a disk-shaped arena self-adapting to a flocking behavior with wall avoidance

![Flocking demo](figs/selfadapt/flocking.gif)

The robot LED colors correspond to the local loss score: green/yellow: low loss, orange/red: high loss, blue: reached loss target (very low loss), so the robot stops learning and always keep the same parameters.

---

## Build

Requirements:
- The Pogosim simulator [here](https://github.com/Adacoma/pogosim.git)
- The pogo-utils toolkit for Pogobots [here](https://github.com/Adacoma/pogo-utils.git)
- (Optional) The Pogobot SDK, to compile for real Pogobots [here](https://github.com/nekonaute/pogobot-sdk.git)

To compile the project:
```bash
export POGO_SDK=PATH_TO_pogobot_sdk_BASE_DIR/
export POGOSIM_INCLUDE_DIR=PATH_TO_POGOSIM_BASE_DIR/src/
export POGOUTILS_INCLUDE_DIR=PATH_TO_POGO-UTILS_BASE_DIR/src/
./build.sh
```

## Build using Apptainer/Singularity (on Linux/WSL)

Ensure first that you have apptainer/singularity installed.

To create the image:
```bash
sudo apptainer build -F apptainer_ACU-selfadapt.sif apptainer_ACU-selfadapt.def
```

To compile the project:
```bash
apptainer exec apptainer_ACU-selfadapt.sif ./build.sh
```


## Running a demo
To launch a demo simulation where a swarm self-adapts toward a **flocking** behavior in a mix of torus/disk/star arenas:
```bash
./ACU-selfadapt -c conf/selfadapt/selfadapt-ACU-torus_disk_star.yaml
```

Or, using apptainer/singularity:
```bash
apptainer exec apptainer_ACU-selfadapt.sif ./ACU-selfadapt -c conf/selfadapt/selfadapt-ACU-torus_disk_star.yaml
```
(Just prepend "apptainer exec apptainer\_ACU-selfadapt.sif" to all commands if you use apptainer/singularity)

The configuration file "selfadapt-ACU-torus\_disk\_star.yaml" sets:
- The target behavior (flocking)
- The arenas (torus, disk, star)
- The optimizer and its parameters
- The loss targets and weights

You can duplicate and modify this .yaml configuration to explore other target behaviors (e.g. disordered gas, clusters, MIPS-like states) or different optimizers.


## Launch large-scale simulation to retrieve all results from the original article

Make use your have the Python pogosim package installed and up-to-date:
```bash
pip install -U pogosim
```

Then you can just use pogobatch to launch 32 runs of all cases:
```bash
nice -n 20 pogobatch -c conf/selfadapt/selfadapt-ACU-torus_disk_star.yaml -S ./ACU-selfadapt -r 32 -t results/selfadapt-ACU-torus_disk_star-32run/tmp -o results/selfadapt-ACU-torus_disk_star-32run
```
(faster to run on a 32-core computer with at least 32GB of free RAM)

To create all plots from these results, ensure you have numpy/matplotlib/seaborn/etc installed:
```bash
pip install -r requirements.txt
```
Then, use the following scripts:
```bash
./scripts/optim_tables.py -i results/selfadapt-ACU-torus_disk_star-32run -o results/selfadapt-ACU-torus_disk_star-32run/optim_tables --final-window-s 100

for i in $(cd results/selfadapt-ACU-torus_disk_star-32run; ls *feather); do echo $i; ./scripts/local_stats_plot.py -i results/selfadapt-ACU-torus_disk_star-32run/$i -o results/selfadapt-ACU-torus_disk_star-32run/`echo $i | sed 's/.feather//'`/local_stats --final-window-s 100  ; done

./scripts/traces.py -i results/selfadapt-ACU-torus_disk_star-32run/result_hit_prob.feather -o results/selfadapt-ACU-torus_disk_star-32run/result_hit_prob/traces --gif --gif-fps 20 --jobs 8 --arenas-dir /usr/local/share/pogosim/arenas --kSteps 20      # Replace --arenas-dir ARG with the directory where pogosim/arenas is installed on your computer
```


---

## Self-Adaptive Controller Parameters

This repository contains the implementation of the self-adaptive swarm controller
described in the accompanying ANTS paper.  
Below we summarize the main parameters, hyper-parameters, and target statistics
used in the experiments. Full details can be found in the source code.

---

## ACU Motility Model Parameters

The table below lists the parameters of the **ACU (Alignment with Crowding and U-turns)** motility model used in the experiments.  
Parameters marked as *in genotype* are optimized online by the self-adaptive controller; the others are fixed hyper-parameters.

| Symbol / Name | In genotype? | Domain / Units | Description |
|---|---|---|---|
| `v₀` (base linear speed) | Yes | `[0, 1]` mm·s⁻¹ | Baseline commanded speed before crowding adjustment, in normalized units (`1.0` = maximum robot speed). |
| `β` (alignment gain) | Yes | `[0, 1]` s⁻¹ | Strength of the Vicsek alignment torque toward the commanded heading `θ*`. |
| `σ` (angular noise) | Yes | `[0, 1]` s⁻¹ | Standard deviation of angular diffusion. |
| `φ_norm` (normalized turn) | Yes | `[0, 1]` | Normalized phase offset; the absolute angular offset is `φ = 2π · φ_norm` (rad). |
| `d_crowd` (slowdown depth) | Yes | `[0, 1]` | Maximum relative slowdown when the local neighbor count is close to `N_tgt`. |
| `N_tgt` (target neighbors) | No | `[0, 20]` | Desired neighbor count around which the speed policy is neutral. |
| `w_n` (width in neighbors) | No | `4.0` | Width controlling how quickly speed recovers as `|d_i − N_tgt|` increases. |
| `τ_ut` (U-turn duration) | No | `1.5 s` | Fixed activation window for the U-turn heading override. |
| Neighbor timeout | No | `1.0 s` | Time after which a stored neighbor is considered stale and discarded. |
| Integration clamp | No | `0.05 s` | Maximum internal integration time step used for numerical stability. |

**Note:**  
All continuous parameters are normalized to `[0,1]` when included in the genotype, simplifying decentralized online optimization.

---

### Controller Parameters and Hyper-Parameters

The table below lists the parameters exposed by the decentralized self-adaptive
controller, including optimization settings, social learning parameters, and
fast-transmission (FT) parameters.

| Name | Domain / Value | Meaning |
|---|---|---|
| Goals `G = (P̂, Ŵ, Π̂, N̂)` | `[0, 1]` | Desired statistic values. |
| Weights `Ω = (ω_P, …, ω_N)` | `ℝ≥0` | Objective weights. |
| `L` | `[0, 1]` | Loss to minimize on each robot, weighted sum of goals `G`. |
| `L_target` | `0.15` | A robot stops optimizing iff `L ≤ L_target`. |
| `N_max` | `≤ 20` | Maximum effective neighbors used for scaling. |
| `T_eval` | `15 s` | Duration of an optimization evaluation window. |
| `t_quiet` | `5 s` | Guard time before sampling the loss. |
| Message sending frequency | `60 Hz` with `35%` chance | Frequency of robot message broadcast with probabilistic sending. |
| **1+1-ES `σ₀`** | `0.2` | Initial global mutation step size. |
| **1+1-ES `σ`** | `[1e-5, 0.8]` | Allowed range for the mutation step size. |
| **1+1-ES `s_target`** | `0.2` | Target success rate for step-size adaptation. |
| **1+1-ES `c_σ`** | `0.1` | Learning rate for step-size adaptation. |
| **HIT `α_HIT`** | `0.35` | Initial transfer rate (fraction of parameters copied from a better neighbor). |
| **HIT `σ_HIT`** | `0.10` | Std. dev. of Gaussian mutation applied after transfer. |
| **HIT `eval_T`** | `5.0 s` | Maturation window for HIT reward computation; communication disabled during this period. |
| **HIT `α_σ_HIT`** | `1e-3` | Std. dev. of Gaussian mutation applied to `α_HIT`. |
| **HIT `α_min_HIT`** | `0.0` | Minimum value of the evolved transfer rate. |
| **HIT `α_max_HIT`** | `0.9` | Maximum value of the evolved transfer rate. |
| **FT probability `p₀`** | `10%` | Probability to directly accept a better neighbor’s genome. |
| **FT improvement threshold `τ`** | `5%` better than own loss | Minimum relative improvement required for direct cloning. |
| **FT evaluation time `t_FT`** | `5 s` | Duration of a fast-transmission evaluation. |

---

### Swarm-Level Statistics and Target Goals

The self-adaptive controller optimizes local statistics computed on each robot
to achieve different macroscopic swarm behaviors.  
The table below lists the target values and weights used for each goal.

| Symbol | Name | Flock + follow walls (Value / Weight) | Disordered gas (Value / Weight) | Clusters (Value / Weight) | MIPS (Value / Weight) | Description |
|---|---|---|---|---|---|---|
| `P̂` | Polarization | `1.00 / 1.0` | `0.10 / 0.1` | `– / 0.0` | `0.10 / 0.2` | Mean local polarization across robots. |
| `Ŵ` | Wall-avoidance + U-turns | `0.20 / 1.0` | `– / 0.0` | `– / 0.0` | `– / 0.0` | Fraction of time spent detecting walls or performing collective U-turns. |
| `Π̂` | Neighbor persistence | `– / 0.0` | `0.00 / 1.0` | `1.00 / 1.0` | `0.70 (solid), 0.00 (gas) / 1.0` | Mean normalized time neighbors are preserved. |
| `N̂` | Neighbor count | `– / 0.0` | `– / 0.0` | `5.00 / 20 / 0.2` | `4.00 / 20 / 0.2` | Mean neighbor count normalized to `[0,1]`. |

---

**Note:**  
These values correspond to the experimental setup reported in the ANTS paper.
They can be modified to explore alternative swarm behaviors or optimization regimes.

## Simulation Results: Time to Reach Loss Targets

The table below summarizes simulation results over **32 independent runs**.
For each configuration, we report the **mean ± std** time (in simulation seconds)
to reach different loss thresholds (`L ≤ 0.30`, `0.25`, `0.20`, `0.15`, `0.10`),
as well as the number of runs `n` that successfully reached each target.

Only configurations with **at least 5 runs** reaching a given target are considered
when identifying minimal convergence times (shown in **bold**).

---

### Convergence Times Across Algorithms and Conditions

| Algorithm | FT | Arena | Biases | t(L≤0.30) ± std | n | t(L≤0.25) ± std | n | t(L≤0.20) ± std | n | t(L≤0.15) ± std | n | t(L≤0.10) ± std | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1+1-ES | No | torus | No | 377.3 ± 366.9 | 28 | – | – | – | – | – | – | – | – |
| 1+1-ES | No | torus | Yes | 877.6 ± 415.5 | 7 | – | – | – | – | – | – | – | – |
| 1+1-ES | No | disk | No | 152.6 ± 98.9 | 32 | – | – | – | – | – | – | – | – |
| 1+1-ES | No | disk | Yes | 560.3 ± 322.1 | 22 | – | – | – | – | – | – | – | – |
| 1+1-ES | No | star | No | 313.1 ± 262.2 | 31 | – | – | – | – | – | – | – | – |
| 1+1-ES | No | star | Yes | 799.1 ± 289.3 | 16 | – | – | – | – | – | – | – | – |
| 1+1-ES | Yes | torus | No | 45.6 ± 13.7 | 32 | 90.7 ± 63.5 | 32 | 197.3 ± 114.5 | 29 | **328.9 ± 266.9** | 16 | – | – |
| 1+1-ES | Yes | torus | Yes | 73.2 ± 43.6 | 32 | 202.9 ± 127.8 | 29 | 507.4 ± 330.8 | 14 | – | 3 | – | – |
| 1+1-ES | Yes | disk | No | 44.4 ± 13.2 | 32 | **92.5 ± 49.5** | 32 | 244.8 ± 185.3 | 30 | **455.9 ± 357.9** | 14 | – | 2 |
| 1+1-ES | Yes | disk | Yes | **76.9 ± 41.0** | 32 | 276.8 ± 198.1 | 30 | **523.3 ± 285.4** | 13 | – | 2 | – | – |
| 1+1-ES | Yes | star | No | 60.0 ± 36.7 | 32 | 174.3 ± 211.4 | 31 | 386.3 ± 264.4 | 26 | **486.9 ± 349.8** | 15 | – | – |
| 1+1-ES | Yes | star | Yes | **110.1 ± 51.1** | 32 | 451.9 ± 270.0 | 29 | 876.4 ± 383.6 | 10 | – | – | – | – |
| HIT | No | torus | No | 323.3 ± 203.7 | 32 | – | – | – | – | – | – | – | – |
| HIT | No | torus | Yes | 770.7 ± 378.3 | 29 | – | 1 | – | – | – | – | – | – |
| HIT | No | disk | No | 158.8 ± 128.6 | 32 | 908.0 ± 294.7 | 8 | – | – | – | – | – | – |
| HIT | No | disk | Yes | 532.1 ± 246.4 | 32 | – | 4 | – | – | – | – | – | – |
| HIT | No | star | No | 477.7 ± 339.5 | 32 | – | 2 | – | – | – | – | – | – |
| HIT | No | star | Yes | 670.3 ± 294.7 | 30 | – | 2 | – | – | – | – | – | – |
| HIT | Yes | torus | No | **45.0 ± 13.4** | 32 | **85.0 ± 26.9** | 32 | **174.3 ± 93.2** | 31 | 414.9 ± 242.0 | 30 | – | 1 |
| HIT | Yes | torus | Yes | **70.0 ± 43.1** | 32 | **171.3 ± 89.6** | 32 | **497.7 ± 288.5** | 31 | **878.3 ± 284.9** | 19 | – | – |
| HIT | Yes | disk | No | **42.5 ± 15.0** | 32 | 115.1 ± 94.1 | 32 | **205.3 ± 118.2** | 31 | 538.1 ± 351.1 | 28 | **885.4 ± 256.7** | 12 |
| HIT | Yes | disk | Yes | 77.5 ± 37.3 | 32 | **202.0 ± 116.4** | 32 | 595.6 ± 340.4 | 30 | **849.8 ± 280.6** | 15 | – | – |
| HIT | Yes | star | No | **58.2 ± 32.7** | 32 | **131.3 ± 69.5** | 32 | **333.3 ± 238.5** | 32 | 658.7 ± 371.4 | 25 | – | 1 |
| HIT | Yes | star | Yes | 117.6 ± 83.6 | 32 | **304.7 ± 189.0** | 31 | **648.9 ± 319.1** | 28 | – | 3 | – | – |

---

**Legend**
- `FT`: Fast Transmission enabled (`Yes` / `No`)
- `Biases`: presence of robot hardware biases
- `n`: number of runs (out of 32) reaching the corresponding loss target
- `–`: fewer than 5 runs reached the target
- **Bold values** indicate minimal convergence times among eligible configurations


