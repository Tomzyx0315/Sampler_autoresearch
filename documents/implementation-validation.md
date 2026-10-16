# 第一版实现验收 — 2026-10-16

## 已执行

- `python3 -B -m unittest discover -s tests -v`：13 项通过，4 项运行时测试跳过。
- 两种采样器在常数 denoiser、常速度场上的解析解与 NFE 检查。
- 高斯速度场上，增加步数后误差下降；该检查使用独立解析参考。
- 实验成功/失败状态、重复 run ID 拒绝、失败证据保留。
- 资产修改检测、原子 JSON 写入、NaN 指标拒绝、源码快照一致性。
- 固定上游文件 checksum 校验、全部 Python/JSON 的语法解析。
- prepare / evaluate / verify_baselines 的 CLI 帮助入口。
- README 内部链接与简短 skill 元数据检查。

技能包官方 quick_validate 依赖 PyYAML，本机没有该依赖，因此没有执行完成。
没有为它安装依赖；仅用标准库检查了当前简单 frontmatter 的字段和命名。

## 未执行，必须在服务器验收

- 安装 requirements、下载真实 checkpoint / Inception / reference statistics。
- PyTorch tensor 与固定上游实现的一致性测试。
- 真实 checkpoint 下 Heun / DPM++ parity。
- GPU 生成、轨迹导出、Inception 特征与完整 FID 流程。
- 任何质量改进或性能结论。

本机不具备已准备的项目运行时，且用户明确要求不修改本机环境。
创建过程中项目内的临时 .venv 已移除，依赖安装未成功；没有保留安装产物。
验证范围是仓库结构、可运行的无依赖逻辑与服务器验收入口，不能推断 CUDA
或 FID 已经跑通。服务器启动步骤在 README，初始研究状态明确保持未验收。
