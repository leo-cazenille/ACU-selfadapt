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


## Self-Adaptive Controller Parameters

This repository contains the implementation of the self-adaptive swarm controller
described in the accompanying ANTS paper.  
Below we summarize the main parameters, hyper-parameters, and target statistics
used in the experiments. Full details can be found in the source code.

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

