# Sampler autoresearch — CIFAR-10 第一版

让 Codex 在一个固定的 CIFAR-10 模型上观察采样过程，提出假设，并在自由
playground 中实现和验证 sampler。当前只支持单 GPU（或 CPU smoke），
不包含多机调度、自动论文生成或复杂实验平台。

**本仓库在本机只做无新增依赖的检查。模型加载、CUDA、完整 FID 的验证必须
在服务器运行，不能把仓库创建完成理解为已经复现论文分数。** 当前状态见
[`documents/state.md`](documents/state.md)。

## 内容

- `AGENTS.md`：Codex 自动读取的项目入口与修改边界。
- `programs.md`：研究目标与观测驱动的工作方式。
- `skills/cifar10-eval/SKILL.md`：固定评测的使用方式，由 AGENTS.md 明确要求读取。
  它没有安装到全局环境，也不依赖 `.agents/skills` 的自动发现。
- `sampler.py`：可修改的候选入口，初始等同 DPM-Solver++(2M)。
- `playground/`：自由分析、算法模块、假设验证；已有轨迹分析示例。
- `harness/`：模型加载、计数、评测、实验记录；研究时保持稳定。
- `baselines/`：Heun（EDM，S_churn=0）与 DPM-Solver++(2M，midpoint)。
- `experiments/`：小型机器可读记录，提交 Git。
- `documents/`：状态、发现与解释性报告，提交 Git。
- `cache/` 和 `artifacts/`：模型、大型轨迹与输出，不提交 Git。

## 服务器准备

以下命令只在你指定的研究服务器执行。推荐 Python 3.11，支持 3.10–3.12。
从 Git 克隆后，在仓库根目录运行；依赖安装到项目虚拟环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m harness.prepare
python -m harness.doctor
python -m unittest discover -s tests -v
python -m harness.verify_baselines
```

按服务器驱动选择合适的 PyTorch CUDA wheel；`requirements.txt` 固定了直接
依赖版本，每次实验还会保存实际 `pip freeze`。无需下载 CIFAR-10 原始图片：
第一版使用官方预训练网络、官方 Inception 和 CIFAR-10 参考统计。
首次准备会下载约数百 MB 文件，失败最多重试三次。

模型为 NVIDIA `edm-cifar10-32x32-uncond-vp.pkl`。资产来源固定在
`configs/cifar10.json`，下载后记录 SHA-256，之后运行前校验。上游没有随本
仓库提供权威 checksum；这些校验用于后续运行一致性，不声称验证首次下载
的独立真实性。NVIDIA 的 pickle 会执行其模型定义，只使用固定官方来源。
来源和各自许可证见 [`third_party/SOURCES.md`](third_party/SOURCES.md)。

已有共享资产目录时，可给 prepare、doctor、verify_baselines 和 evaluate
统一传入 `--assets /your/shared/path`。文件已有但没有对应 manifest 时，
程序会拒绝静默接纳，以免使用了不同的模型或参考统计。

## 第一次实验

先检查基线与候选路径，再收集少量轨迹：

```bash
python -m harness.evaluate --run baseline-smoke --sampler dpmpp --steps 10 \
  --preset smoke --hypothesis "Validate the baseline generation path"
python -m harness.evaluate --run candidate-trace --sampler candidate --steps 10 \
  --preset smoke --trace 4 --hypothesis "Inspect velocity changes across noise levels"
python -m playground.analyze_trace artifacts/candidate-trace/trace.npz \
  --output experiments/candidate-trace/analysis.json
```

基线完整性通过后建立筛选分数：

```bash
python -m harness.evaluate --run baseline-screen-10 --sampler dpmpp --steps 10 \
  --preset screen --hypothesis "Establish the 10-NFE screening baseline"
```

smoke 生成 16 张，不计算 FID；screen 为 2048 张，full 为 50000 张。
每次会保存第一批的图片网格，完整图像默认不落盘；FID 在线累计特征统计。
只有候选值得进一步验证时才跑 full，例如：

```bash
python -m harness.evaluate --run candidate-full-10 --sampler candidate --steps 10 \
  --preset full --seed 100000 --hypothesis "Confirm the screening improvement on fresh seeds"
```

还必须运行同协议的 baseline full 才能比较。保留独立种子重复，不能只报告
最优一次。不要将 2048 张与 50000 张的 FID 或不同库的 FID 直接比较。

## 修改 sampler 与做观测

接口是 `sample(denoise, x, sigmas, **params)`。`x` 已经乘过初始 sigma，
是 `[B,3,32,32]` float64；网络内部使用 float32。返回同形状、同设备、
同 dtype 的有限张量。评测统一负责 clamp/uint8 转换，sampler 不做输出裁剪。

`denoise(x, sigma)` 返回 D(x,sigma)。实际 ODE 是：

```text
dx/dsigma = (x - D(x, sigma)) / sigma
```

基准网格为 rho=7、sigma 从 80 到 0.002，最后接到 0。不能在 sigma=0
调用模型。候选可以在函数内部增加中间评估点、改变量或更新规则，所有
网络计算经过同一 callback。可使用 `--params '{"key": 1}'` 传递算法参数。
也可以写 `playground/new_method.py` 并传 `--sampler playground.new_method:sample`。

trace 记录的是实际模型调用处的 x、sigma、D，以及最终输出；可以自行计算
速度。它不会将 Heun 的预测/校正点误标成严格的已接受轨迹。初始提供的
分析只是示例，agent 可以自行扩展 perturbation、局部拟合、误差传播等实验。

**steps 不等于 NFE**：DPM++ N 步为 N NFE，Heun N 步为 2N-1。
使用 `metrics.json` 中的实测调用量。默认每个 batch 最多 128 次调用，
可通过 `--max-nfe` 显式配置。辅助模型、JVP/反向计算等不能仅以主模型 NFE
表示，必须额外报告成本；本版自动计数只覆盖 denoise forward。

## 长时间运行

```bash
tmux new -s sampler-research
source .venv/bin/activate
codex
```

在 Codex CLI 中建立 Goal，明确本轮预算，例如：

> /goal 按 AGENTS.md 和 programs.md 在本仓库开展 CIFAR-10 sampler research。
> 先验收环境和基线，再用观测提出并验证动力学假设。可以自由使用 playground。
> 本轮 GPU 实验累计预算为 4 小时，单次调用使用适当的 --max-seconds。
> 在预算内完成有证据的研究报告，记录改进、失败和下一步；不要求必然超过基线。

这是启动示例，不会自动创建 Goal。Goal 不随 Git 克隆迁移；服务器需使用支持
`/goal` 的 Codex CLI。SSH 断开后可 `tmux attach -t sampler-research`。
若会话丢失，从 `documents/state.md` 与实验记录恢复。

第一版不实现复杂调度或断点续采样：一个实验失败后保留记录，用新 ID 重跑。
`status.json` 写有 PID、主机和进度；`running` 不代表进程一定仍活着。
`--max-seconds` 在网络调用/批次边界检查，不会强制杀掉一个卡住的 kernel；
服务器需要硬墙钟上限时，使用作业调度器或 Linux `timeout` 包住命令。
总研究预算需要 agent 记账，runner 不充当跨实验资源调度器。

## Git 与可复现性

重要实验前提交代码；运行期间不修改源文件。每次运行保存 Git 状态、实际
源码 hashes、源码 zip、完整配置、资产 hashes、运行环境与指标。结束时检测
源文件是否改变，改变则将实验标为失败。保留 `artifacts/ID/source.zip` 才能
独立恢复当时未提交的代码；Git 只保存它的路径和 checksum。

`experiments/ID/` 是原始证据，`documents/experiments/ID.md` 是解释。
失败实验同样保留；新的候选用新的提交，不用 reset 抹掉失败历史。
Git 默认只在本地提交，本仓库不配置远端，也不会自动 push。

## 检查

无需安装任何依赖的检查：

```bash
python3 -B -m unittest tests.test_core -v
```

服务器安装依赖后，完整测试还检查 tensor 与上游实现的一致性、FID 的
特征统计、种子分批一致性、调用预算和 trace 分析。`verify_baselines` 再在
真实 checkpoint 上检查两种基线。之后用 smoke/screen/full 分别验证生成、
筛选和正式 FID。没有通过这些服务器步骤之前，不声称基线已复现。
