# Coupled Geometry Distributions

**Coupled Geometry Distributions (CoGeDi)** extends [Geometry Distributions](https://1zb.github.io/GeomDist/) to shape collections using the [UniDiffuser](https://github.com/thu-ml/unidiffuser) formulation for multi-modal diffusion. CoGeDi captures geometry and correspondence in one model.

<img src="assets/faust_five.png" alt="drawing" width="800"/>

Paper: [TODO]() 


## Quick Start

From the `CoGeDi` folder:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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

## Citation