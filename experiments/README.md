# 实验记录

`python -m harness.evaluate` 为每次运行创建新的 ID 目录。记录在 Git 中保留，
同一 ID 不会被重用。`manifest.json` 绑定假设、参数、代码和资产；
`status.json` 记录状态；成功后写 `metrics.json`，失败写 `error.txt`。
大文件位于 `artifacts/ID/`，自行保存；不要误删有价值实验的 source.zip。

纯诊断实验也可在此创建独立目录，保存命令、代码版本、数据出处和分析结果。
诊断脚本可以放 playground，文字结论放 documents/experiments。
