# 本仓库维护边界

先读README、docs/METHOD.md、docs/RESULTS.md、docs/REPRODUCE.md与provenance/origin.json。此仓库是用户明确选定的方法发布库，日常proposal/消融继续在ayumurephael/CausalTSAD中进行。只有用户再次明确授权晋升时，才向这里引入下一条方法或更新发布结论。

source-v1对应stat_temporal，不能把未使用的typed/calibrator或TimeRCD接口加入后继续引用69.07%/64.58%。保留三seed、对照、负例、输入/权重SHA和方法来源。不要删除/覆盖原始权重、数据与已发布记录；新复现实验单独保存。

只有observed和正常DAG是目标模型输入。正常孪生、标签、根因、真实系数/滞后/噪声只可作来源监督或审计。不得在目标反传或选checkpoint/阈值。条件RCA不是端到端能力；来源结果不叫目标SOTA。保留MIT许可证和原作者归属。

大文件走Release，Git主分支保存源码、小权重和人可读/可审计结果。不得上传服务器凭据、私有SSH管理脚本、第三方论文全文或整个实验工作区。纯打包验证可CPU运行；新训练与全量前向应在GPU服务器执行。
