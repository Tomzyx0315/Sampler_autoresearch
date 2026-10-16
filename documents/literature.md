# 研究起点

下列结论来自文献，尚未在本仓库的模型与评测设置中重新验证。
它们帮助构造问题，不是限定 agent 必须搜索这些方向。

| 工作 | 值得借鉴的研究问题 |
|---|---|
| [EDM](https://arxiv.org/abs/2206.00364) | 参数化、噪声网格和求解器各自怎样影响采样？本仓库使用其 checkpoint、Heun 和 FID 协议。 |
| [DPM-Solver++](https://arxiv.org/abs/2211.01095) | 换一个积分对象，以及利用历史 denoiser，为什么会改变低 NFE 表现？ |
| [DPM-Solver-v3](https://arxiv.org/abs/2310.13268) | 预训练模型的经验导数统计，能否帮助选出更容易积分的残差？ |
| [AMED](https://arxiv.org/abs/2312.00094) | 观察到的单轨迹低维结构，如何转化为平均方向的预测？ |
| [GITS](https://arxiv.org/abs/2405.11326) | 轨迹几何规律怎样指导时间步的分配？ |
| [Gaussian Score Approximation](https://arxiv.org/abs/2412.09726) | 高噪声阶段能否用简单的解析动力学替代部分网络计算？ |
| [FlowTurbo](https://arxiv.org/abs/2409.18128) | 速度变化的稳定性是否支持便宜的增量预测？注意这是 flow 模型上的观察，不自动适用于这里的 diffusion 模型。 |

研究链条可以是：实际观测 → 可证伪的规律 → 区分性实验 → 利用规律的算法
→ 成本和质量验证。发现共享规律同样有价值，不必为了 model dependent
而人为追求只对单个 checkpoint 有效。
