# vop_nav

[English](README.md) | 简体中文

VOP-Nav 非官方复现。

本项目使用 IsaacLab 训练 Go2 四足机器人，使其能够在动态障碍物中导航，目标是复现 [Learning Agile Navigation in Crowded Environments for Quadruped Robots](https://arxiv.org/abs/2607.15036) 的训练流程。

基础导航阶段参考 [ABS — Agile But Safe](https://agile-but-safe.github.io/)，基于其[官方代码](https://github.com/LeCAR-Lab/ABS)将 Isaac Gym 实现迁移到 IsaacLab。项目包含仿真适配和工具改动，目前不声称达到论文或官方实现的实验结果。

## 训练流程

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| BasicNav | 导航环境与 PPO 配置 | 导航策略 checkpoint |
| 采集数据 | 冻结的 BasicNav checkpoint | 校验后的原始轨迹、五帧样本、解析 VO 标签和训练分片 |
| VOP-Net | 转换后的数据集 | best / last checkpoint，以及每轮验证指标 |
| VOP-Nav | 冻结的 VOP-Net checkpoint | 新的导航策略 checkpoint |

## 环境准备与服务器选择

先手动准备 Docker 或 Conda 中的 IsaacLab、Isaac Sim、PyTorch 及其他第三方依赖。仿真需要 NVIDIA GPU，以及提供 `MultiMeshRayCasterCfg` 和 `MultiMeshRayCasterCameraCfg` 的 IsaacLab 版本。VOP-Net 离线训练和对比导出不启动 Isaac Sim，需要 PyTorch、NumPy 和 PyYAML；开启 TensorBoard 前需安装对应包。Forest、Office、Square 和 Slow 的播放、评估还需要 Python-RVO2；三个阶段的训练不需要它。

发布目录不包含预训练权重或数据集，也不附带原开发仓库的 Git 历史。当前没有提供完整的已验证环境版本矩阵，请在自己的环境中先进行小规模验证。

默认使用本地 Conda 环境。按实际安装位置设置路径：

```bash
export LOCAL_CONDA_ROOT="$HOME/miniconda3"
export LOCAL_CONDA_ENV="$LOCAL_CONDA_ROOT/envs/isaaclab"
source env.sh --server local
install_env
```

服务器配置保存在 `scripts/env/servers/`，支持 Docker、Conda 和多服务器切换。远程 Conda 示例与 Docker 配置项见 [配置说明](scripts/env/servers/README.md)。个人服务器配置不包含在发布目录中。

只有 `sync_code` 执行代码同步，包含 YAML，并删除远端代码目录中本地已经移除的文件，保留日志和 checkpoint。启动任务与 `install_env` 都使用所选服务器现有代码。`install_env` 仅使用 `--no-deps` 安装四个项目包的可编辑版本，第三方依赖完全手动准备。

### 安装播放与评估依赖 Python-RVO2

先在实际执行播放、评估的服务器上进入对应 Docker 容器或激活 Conda 环境，再执行下面的命令。使用与任务相同的 Python 解释器；如果使用 Isaac Sim 自带解释器，将命令中的 `python` 替换为对应安装目录中的 `python.sh`。也可先在项目管理终端执行 `enter` 进入所选环境。

```bash
python -m pip install "Cython==3.3.1" "cmake<4" setuptools wheel
python -m pip install --no-deps --no-build-isolation \
  "git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d"
python -c "import rvo2; print('RVO2 is ready')"
```

## 配置文件

启动前编辑 YAML。相对路径以项目根目录为基准；Docker 模式会将宿主机项目内的绝对路径转换为容器项目挂载路径。

| 文件 | 用途 |
| --- | --- |
| `configs/nav_basic.yaml` | BasicNav 环境、PPO、续训 checkpoint、训练轮数及可选截止时间 |
| `configs/nav_vop.yaml` | VOP-Nav 训练与冻结的 VOP-Net checkpoint |
| `configs/collect.yaml` | BasicNav checkpoint、采集规模、等级、标签与分片参数 |
| `configs/vop_train.yaml` | 数据集、网络规模、优化器、训练轮数与验证参数 |
| `configs/play.yaml` | 策略 checkpoint、任务类型、需要时使用的 VOP-Net checkpoint 及播放参数 |
| `configs/evaluate.yaml` | 策略 checkpoint、任务类型、随机种子及每环境评估回合数 |
| `configs/scenes.yaml` | 播放与评估共用的场景、等级和场景参数 |
| `configs/compare.yaml` | 原始轨迹、可选的 VOP-Net checkpoint、环境编号及导出参数 |

`checkpoint: null`、`dataset: null`、`input: null` 是待填写项，请在使用前设置必需输入。VOP-Nav 还必须设置 `vop_checkpoint`；BasicNav 则保持该项为空。播放、评估配置中的任务分别选择 `Unitree-Go2-VOP-BasicNav[-Play]` 或 `Unitree-Go2-VOP-Nav[-Play]`，播放使用带 `-Play` 的名称。

导航的 `agent` 包含 PPO 参数，`environment` 可覆盖 IsaacLab 原生环境配置，例如 `environment.episode_length_s`。训练轮数统一以 `agent.runner.max_iterations` 为准。导航的 `checkpoint` 用于续训；VOP-Net 每次从头训练，仍保存 `best.pt` 和 `last.pt`，不提供续训入口。

命令行仅保留常用覆盖：`--config`、`--device`、`--run-dir`、`--name`、`--log`。仿真任务支持 `--num-envs`，导航训练支持 `--max-iterations`，VOP-Net 支持 `--epochs`。其余参数统一编辑 YAML，使用 `train --help` 等查看支持项。

每个任务自动创建带时间戳的新目录，`--name` 添加名称后缀，`--run-dir` 可指定项目内的新目录。目录中的 `resolved.yaml` 记录实际任务配置。运维启动命令拒绝复用已有运行目录。

## 使用命令

以下示例需要先填写 YAML 的必需路径。远程运行时需手动同步所选服务器上的代码和配置；本地运行无需同步。

```bash
train --config configs/nav_basic.yaml --num-envs 1024
collect --config configs/collect.yaml
train_vop --config configs/vop_train.yaml --epochs 50
train --config configs/nav_vop.yaml --max-iterations 8000
```

`collect` 一次完成仿真采集、原始数据及连续性检查、标签和分片生成。默认结果目录包含 `raw/`、`prepared/`。`vop_train.yaml` 的 `dataset` 可以指向整个采集目录，也可以直接指向 `prepared/`。独立校验、转换命令已移除；采集失败或被停止时不会继续转换。

```bash
play --config configs/play.yaml
evaluate --config configs/evaluate.yaml
```

BasicNav、VOP-Nav 共用这两个入口。本地播放打开桌面窗口，远程播放使用 WebRTC，启动前编辑 `scenes.yaml` 选择场景。

第二阶段对比通过远端导出流程使用：

```bash
compare --config configs/compare.yaml
download
# 也可以指定新的本地目录：
download exports/my_comparison
```

等待导出完成后下载完整目录，在浏览器中直接打开 `index.html`。页面保留原始轨迹回放和解析 VO，配置 `checkpoint` 后同时比较网络预测。

## 任务管理

```bash
status
log train
log collect
log train_vop
log compare
stop train
stop collect
stop train_vop
stop compare
stop play
stop evaluate
tb
stop_tb
```

同一服务器、同一项目内，同类型最多一个任务，不同类型可并行。BasicNav 和 VOP-Nav 训练都属于 `train`。`log <类型>` 查看最新日志，Ctrl+C 只结束查看；启动时加 `--log` 效果相同。`stop <类型>` 作用于该类型的全部项目进程，并保留各类型停止行为：采集、VOP-Net、播放和评估允许清理保存，导航训练超过停止超时可能被强制结束。VOP-Net 只保证完整训练轮次的 checkpoint；导航训练的可选截止时间模式保留分块保存和课程快照。

`status` 显示任务状态、PID、耗时、可用进度、日志路径与 GPU 信息。`tb` 自动启动 TensorBoard 并管理 SSH 隧道，`stop_tb` 同时停止两者。`enter` 进入当前运行环境，`env_help` 查看命令。


## 上游来源与许可

- VOP-Nav 论文：[Learning Agile Navigation in Crowded Environments for Quadruped Robots](https://arxiv.org/abs/2607.15036)。本项目为非官方复现。
- ABS：[项目主页](https://agile-but-safe.github.io/)与[官方代码](https://github.com/LeCAR-Lab/ABS)。本项目将其导航环境从 Isaac Gym 迁移到 IsaacLab，沿用并修改了官方仓库所包含的 RSL-RL 训练代码，并参考、改写了部分导航奖励。
- IsaacLab：[官方代码](https://github.com/isaac-sim/IsaacLab)，提供仿真框架、机器人资产配置及原生 MDP 接口。
- RL 代码中保留了 RSL-RL / Legged Gym 相关版权和许可声明，使用与分发时应遵循对应条款。

本仓库采用与 ABS 一致的 **CC BY-NC 4.0（署名—非商业性使用）**许可，适用于本项目有权许可的代码、文档及修改，但第三方材料和单独声明的内容除外。分发时需保留署名与许可信息、注明修改；受该许可覆盖的材料不得用于商业用途。完整条款见 [LICENSE](LICENSE)，详细来源对应关系、适用范围和第三方声明见 [THIRD_PARTY_LICENSES](licenses/THIRD_PARTY_LICENSES)。

Legged Gym / RSL-RL 与 IsaacLab 框架代码的 BSD 条款继续适用；可选评估依赖 Python-RVO2 使用 [Apache-2.0](licenses/Apache-2.0.txt)。Isaac Sim、机器人 USD 资产及其他依赖遵循各自条款，本仓库许可不授予它们的再分发权。本发布目录不包含预训练模型、数据集或本地机器人 URDF/网格资产。后续发布这些材料前需单独核查来源与再分发权限。

由于包含非商业限制，本仓库整体属于源码公开的研究项目，不属于 OSI 定义的开源软件。
