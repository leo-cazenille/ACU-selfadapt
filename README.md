# ACU-selfadapt

This project implements a **self-adaptive Vicsek-style controller** with **crowding** and **collective U-turns** (ACU), combined with **online decentralized optimization** and **social learning with fast transmission**.

Each robot:
- Follows the ACU motility model (alignment + crowding + U-turns).
- Measures local statistics (polarization, wall/U-turn ratio, neighbor persistence, neighbor count).
- Minimizes a local loss using a lightweight optimizer (HIT, 1+1-ES, etc.).
- Exchanges its genotype and loss with neighbors and can **clone** clearly better neighbors via a **fast transmission** rule.

The simulator runs swarms of such robots in different arena geometries (e.g. torus, disk, star) and shows how they **self-adapt** their microscopic control parameters to reach target macroscopic behaviors (e.g. flocking).

Here is an example for a swarm in a disk-shaped arena self-adapting to a flocking behavior with wall avoidance:
![Flocking demo](figs/selfadapt/flocking.gif).

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

