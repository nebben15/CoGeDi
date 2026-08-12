# Coupled Geometry Distributions

**Coupled Geometry Distributions (CoGeDi)** extends [Geometry Distributions](https://1zb.github.io/GeomDist/) to shape collections using the [UniDiffuser](https://github.com/thu-ml/unidiffuser) formulation for multi-modal diffusion. CoGeDi captures geometry and correspondence in one model.

<img src="assets/CoGeDiOverview.png" alt="drawing" width="800"/>

**Paper**: [TODO]() 

CoGeDi enables joint sampling

<img src="assets/faust_five.png" alt="drawing" width="800"/>


and conditional sampling, where points are generated based on a given point on one shape:

<img src="assets/non-iso-full-fixed-background.png" alt="drawing" width="800"/>

CoGeDi works for points of any dimension, allowing for applications like **texture swapping** by using *6D* points

<p align="center">
  <img src="assets/TexturesBoth.png" alt="CoGeDi overview" width="400">
</p>

or **dynamic modelling** using *4D* points:

<img src="assets/Dynamic.png" alt="drawing" width="800"/>

## Quick Start

### Setup:

From the `CoGeDi` folder (the main repo folder):

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Train/Sample/Eval:

Each experiment has its own `.yaml` configuration file. In it, we set all data paths, hyperparams, and the mode: train, sample, eval. 

Run with a config:

```bash
python -m cogedi.main --config configs/<config>
```

## Repository Layout

Top level:
- `CoGeDi/configs/`: experiment configurations.
- `CoGeDi/checkpoints/`: trained model checkpoints.
- `CoGeDi/samples/`: generated outputs.
- `CoGeDi/train.sbatch`: SLURM launcher.
- `CoGeDi/requirements.txt`: Python dependencies.

Package layout (`CoGeDi/cogedi/`):
- `main.py`: CLI entrypoint, config loading, mode dispatch.
- `registry.py`: component registries and registration.
- `build.py`: object construction from config.
- `dtypes.py`: shared datatypes (batch/state containers).
- `data/`: datasets and dataloading.
- `forward/`: forward/noise process definitions.
- `models/`: diffusion model and model subcomponents
	- `backbones/`, `embedders/`, `parameterizations/`, `precond/`.
- `conditioning/`: conditioning policies (observed/unobserved handling).
- `losses/`: training losses.
- `schedules/`: sigma/noise schedules.
- `solvers/`: sampling/ODE/SDE solver implementations.
- `optim/`: optimizer and LR scheduling helpers.
- `metrics/`: evaluation metrics.
- `orch/`: train/sample/eval orchestration.
- `utils/`: utility helpers.


## Datasets
The following datasets contain shape collections with full (vertex level) correspondence between shapes.

- [FAUST](https://faust-leaderboard.is.tuebingen.mpg.de): Humans in different poses.
- [DFAUST](https://dfaust.is.tue.mpg.de/): Dynamic human pose sequences.
- [SMAL](https://smal.is.tue.mpg.de): Animal Shapes
- [SMALR](https://smalr.is.tue.mpg.de/): Animal Shapes + Textures

## Visualizing Results
Sampling produces `.ply` point cloud files that can be viewed in tools like [Cloud Compare](https://www.cloudcompare.org/).

For viewing the dynamic point clouds sampled from a D-FAUST model, we provide a simple script at `visualization/dynamic.py`. 

## Citation
TODO
