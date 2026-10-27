# 复现：从已有数字到来源训练

所有命令从仓库根目录执行。输出目录必须是新的；不要覆盖提供的权重或记录。大数据/缓存/新训练输出被.gitignore忽略。

## A. 不跑模型，核对已保存成绩

```bash
python scripts/verify_results.py
```

仅Python标准库即可。读取3×512逐视图指标，每系统先平均视图与seed，再平均32系统；核验AP0.69068435065、条件Top10.64583333333及权重SHA。正常case只计算误报，不把无正例AP填0。

## B. 核对固定权重的真实前向

```bash
python -m pip install -r requirements.txt
python tests/check_packaged_model.py --device cuda
python tests/check_training_plan.py
```

包含一条原始来源系统和原GPU分数参照。测试生成器数值重现、推理入口分数/节点证据等价、旧TimeRCD缓存字段不参与前向。模型前向跨CPU/GPU/库版本要求1e−7误差；生成器观测重建另要求1e−12，图和标签相同。实测值见[VALIDATION](VALIDATION.md)，不承诺跨运行环境所有浮点字节完全相同。

## C. 重评全部32来源确认系统

```bash
python scripts/download_assets.py --asset source --extract
CUDA_VISIBLE_DEVICES=0 python scripts/source_pipeline.py cache --source data/source_pool --cache runs/cache --device cuda
CUDA_VISIBLE_DEVICES=0 python scripts/source_pipeline.py evaluate --source data/source_pool --cache runs/cache --seed 348 --output runs/evaluate_s348
CUDA_VISIBLE_DEVICES=1 python scripts/source_pipeline.py evaluate --source data/source_pool --cache runs/cache --seed 349 --output runs/evaluate_s349
CUDA_VISIBLE_DEVICES=2 python scripts/source_pipeline.py evaluate --source data/source_pool --cache runs/cache --seed 350 --output runs/evaluate_s350
```

缓存可用多个GPU并行分摊：分别给4个cache进程`--rank 0/1/2/3 --world-size 4`，并绑定不同CUDA_VISIBLE_DEVICES；等四个进程都完成后再运行训练/评估。每系统只由一个进程负责，不同时让不同配置写同一缓存。

`evaluate`使用提供权重的原来源阈值，并额外记录重算阈值用于核验；不用确认标签重新选择阈值。来源确认仅用于固定终点评估。逐点参考数组可用`python scripts/download_assets.py --asset scores --extract`下载至data/scores。

## D. 从随机初始化重新训练统计量头

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/source_pipeline.py train --source data/source_pool --cache runs/cache --seed 348 --output runs/train_s348
CUDA_VISIBLE_DEVICES=1 python scripts/source_pipeline.py train --source data/source_pool --cache runs/cache --seed 349 --output runs/train_s349
CUDA_VISIBLE_DEVICES=2 python scripts/source_pipeline.py train --source data/source_pool --cache runs/cache --seed 350 --output runs/train_s350
```

以上可以分别后台执行。每seed固定384步，观测曝光11,010,048，来源正常答案5,505,024；三seed合计33,030,144和16,515,072。仅保存固定final，不按确认指标挑best。默认6小时技术期限，失败不自动续步。缓存前向、评估、预检计算另计；原72组总792,723,456曝光属于完整消融项目，不能说本单方法也训练了全部72组。

若要重新生成相同来源机制而非下载原始文件：

```bash
python scripts/source_pipeline.py generate --source data/regenerated_source
```

比较完整数组SHA；不相同时再报告最大数值误差，不能静默把近似相等叫字节相等。不同zip序列化和跨平台浮点末位都会造成文件SHA差。`provenance/source_manifest.json`是当时原始数据清单，原结果重评优先下载原文件。

## E. 可选：正常神经基函数也从头训练

正常基函数先前在独立32来源训练系统上训练128步，7,340,032观测曝光。本方法默认复用其固定s348 final，文件已经提供，通常无需重训。

如需完整追溯：

```bash
python scripts/download_assets.py --asset basis --extract
python tools/prepare_syndrome_source.py --source data/basis_parent --output data/basis_source
python scripts/prepare_basis_training.py --source data/basis_source --output runs/basis_plan.json
CUDA_VISIBLE_DEVICES=0 python tools/train_meta_basis.py --root runs/basis_retrain --plan runs/basis_plan.json --arm normal_only --seed 348
```

这些原始训练/构建文件保留其科学实现与数据hash检查。`normal_only`的神经基函数只从正常NLL学习；历史温度/偏置仍可接收关系BCE，但它们不在当前components路径中使用，不能泛称历史全部保存参数完全无异常监督。

如果基函数来源manifest与原hash不同，默认命令会停止；先查看[打包验证中的环境差异](VALIDATION.md)。需要在新环境开展独立重训时，使用`python scripts/prepare_basis_training.py --source data/basis_source --output runs/basis_plan.json --allow-regenerated-source`，它核验原父数据、80系统图和前缀、系统划分，记录完整数组不同的系统及新输入hash，不静默更换原数据。该分支不能称完全相同数据的原权重重放。

新训basis的SHA通常不等于发布权重。要把它用于新的统计量训练，必须作为**另一个复现实验配置**显式登记新SHA并创建新缓存；不能覆盖本仓库权重后还引用原69.07%。重训差异需归因于硬件/库版本或实现，不能默默更新已发布结果。

## 来源与目标的隔离

这里下载的都是来源数据。250M生成基准、其中150M锁定测试、原25亿TimeRCD训练语料均未包含。原大基准的对应432系统迁移未完成，因此没有该方法的目标VUS-PR表，勿把来源AP当成目标指标。
