# CIFAR-10 sampler autoresearch

固定 EDM 模型与 FID 工具，开放 sampler 和实验 playground。

- `programs.md`：研究目标；`AGENTS.md`：简短工作约定。
- `sampler.py`：候选算法，初始等同 DPM-Solver++(2M)。
- `playground/`：自由实验区，附一个轨迹分析示例。
- `harness/`：模型加载、采样、FID；`baselines/`：DPM++ 与 Heun。
- `experiments/`：自动保存每次运行的结果、预览和可选轨迹。
- `documents/notes.md`：自由记录发现；Git 保存代码。

## 在服务器运行

推荐 Python 3.11。只在研究服务器安装依赖和下载资产：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m harness.prepare
```

采样数量自选，只有加 `--fid` 才计算 FID：

```bash
python -m harness.evaluate --sampler dpmpp --steps 10 --num 2048 --fid
python -m harness.evaluate --steps 40 --num 16 --trace 4
python -m playground.analyze_trace experiments/RUN/trace.npz
python -m harness.evaluate --steps 10 --num 50000 --fid --seed 100000
```

`--sampler` 可选 `candidate`（默认）、`dpmpp`、`heun`，或 `模块:函数`。
可传 `--params '{"key": 1}'` 给自定义算法；`--run 名称` 可指定输出目录名。
模型在 `cache/assets/`，无需下载 CIFAR-10 原始图片；共享路径用 `--assets`。

## Sampler 接口

`sample(denoise, x, sigmas, **params)` 返回与 x 同形状/设备/dtype 的样本。
x 是已经乘过初始 sigma 的 `[B,3,32,32]` float64 张量；网络内部为 float32。
`denoise(x, sigma)` 返回 D，ODE 速度为 `(x-D)/sigma`，不能在 sigma=0 调模型。
默认 sigma 网格从 80 到 0.002，再接 0；算法可以自行选择内部评估点。

DPM++ N 步用 N NFE；Heun N 步用 2N-1。比较时看实际 NFE 和相同样本数的 FID。
计时包含 sampler，排除预热、加载与 FID；开启 trace 的耗时只供诊断。
trace 保存前几个样本的模型评估点，Heun 包含预测/校正点；完整轨迹含未来信息。
结果 JSON 含参数和 Git 状态，不自动保存源码；未提交的改动需自行保留。
大文件不进 Git。算法实验、记录方式和研究节奏由 agent 决定。

在 tmux 中启动 Codex，用 `/goal` 指定本轮目标和计算预算即可。
检查：`python -m unittest discover -s tests -v`。当前尚未在服务器验证真实模型/FID。
上游来源与许可证见 [third_party/SOURCES.md](third_party/SOURCES.md)。
