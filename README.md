# vop_nav

[简体中文](README_ZH.md)

An unofficial IsaacLab reproduction of
[Learning Agile Navigation in Crowded Environments for Quadruped Robots](https://arxiv.org/abs/2607.15036)
for the Unitree Go2. The BasicNav stage adapts
[ABS — Agile But Safe](https://github.com/LeCAR-Lab/ABS) from Isaac Gym to IsaacLab.
This project does not claim to match the paper or official implementation's results.

## Workflow

1. Train a BasicNav policy.
2. Collect trajectories and prepare analytic velocity-obstacle labels.
3. Train VOP-Net offline.
4. Train a VOP-Nav policy with the frozen VOP-Net.

The source release contains code and configuration files. Pretrained weights,
datasets, local robot URDF/mesh files, and the original development repository's
Git history are not included. Robot configuration is imported from IsaacLab.

## Setup

Prepare an NVIDIA GPU, IsaacLab, Isaac Sim, PyTorch, NumPy, and PyYAML in an
existing Conda environment. IsaacLab must provide `MultiMeshRayCasterCfg` and
`MultiMeshRayCasterCameraCfg`. A complete tested version matrix is not yet
provided. Start with a small run in your environment before a full training run.

```bash
export LOCAL_CONDA_ROOT="$HOME/miniconda3"
export LOCAL_CONDA_ENV="$LOCAL_CONDA_ROOT/envs/isaaclab"
source env.sh
install_env
```

`install_env` installs only the four local project packages using `--no-deps`;
prepare third-party dependencies separately. Some playback and evaluation scenes
also require Python-RVO2; see the pinned installation instructions in the
[Chinese guide](README_ZH.md). Offline VOP-Net training does not start Isaac Sim.

Edit the YAML files in `configs/` before launching tasks. Later stages require
the checkpoints or datasets produced by earlier stages.

```bash
train --config configs/nav_basic.yaml --num-envs 16 --max-iterations 2
```

This command is an installation smoke run, not a trained policy or a benchmark.
For the complete workflow, configuration fields, playback, evaluation, and task
management, see [README_ZH.md](README_ZH.md). Remote runtime configuration is
described in [scripts/env/servers/README.md](scripts/env/servers/README.md).

## Attribution and license

See [LICENSE](LICENSE) and [third-party notices](licenses/THIRD_PARTY_LICENSES)
for the applicable terms, upstream attribution, and source correspondence.
The repository uses CC BY-NC 4.0 except for third-party material and separately
licensed components. Because of its noncommercial restriction, the repository
as a whole is not open source under the OSI definition. Third-party licenses
continue to apply; the root license does not grant redistribution rights for
Isaac Sim, external assets, weights, or datasets.
