# Expanded 7740：在 Colab 完成特征并训练

## 当前计算安排

2026-10-07 用户确认：**MMAudio 特征收尾与训练使用 Colab，Myriad 留给之后的 perception embedding 训练。** 现有 Myriad MMAudio 作业作为备选保留，不是后续 embedding 训练代码。Embedding 的模型、监督标签、训练目标、数据切分与评估协议须在该阶段另外明确。

**VERIFIED FACT**：7740 条音视频特征完成收据已经核验（6203/780/757）；Caption 人审尚未完成。因此仍需先按[特征准备说明第 1–2 节](MYRIAD_MIGRATION.md#1-caption-审核与冻结)完成人审、冻结、仅文本提取、五模态合并及 CPU 全量回读。原四模态特征可复用。上述完成前，不能开始正式训练。

入口：[Colab notebook](../notebooks/colab_official_training.ipynb)、[配置模板](../config/colab.example.json)、`python -m fine_tune.colab_train`。Notebook 是操作入口；没有执行真实 Colab GPU 工作。所有阶段默认 preview，`--execute` 才会写入、暂存数据或训练。

## 数据与训练状态放置

| 内容 | 位置与作用 |
|---|---|
| 冻结的最终 feature bundle | Drive 持久副本，使用独立记录的 BUNDLE_READY SHA256 |
| 训练读取的 memmap | 校验后复制到本地 `/content/mmaudio-features/…` |
| 代码、固定 upstream、encoder 和初始权重 | 本地 `/content`，每次新 VM 恢复相同路径与文件哈希 |
| 活跃 run、两个轮换 checkpoint、EMA、日志 | 本地 `/content/mmaudio-runs/…`，使用本地文件锁 |
| 恢复快照 | Drive 独立 smoke/train backup 目录，包含所有恢复所需状态 |

[Colab FAQ](https://research.google.com/colaboratory/faq.html)说明 VM 存在生命周期限制、GPU 和资源供应会变化，并建议减少 Drive 小文件读写。因此训练从本地 memmap 读取，checkpoint 保存时同步发布校验过的 Drive 快照。这里不承诺固定 GPU、运行时长或断线后的零步数损失。

## 1. 准备环境与配置

在 Colab 选择 GPU，记录实际 GPU/CUDA/Python/内存/磁盘。GPU 型号和数值环境必须与自己的 smoke 一致；BF16 不受支持时会停止，需要明确选择 FP32 并重新 smoke。batch 2 / validation batch 2 是候选值，未证明能在每种 Colab GPU 上运行。正式步数默认 null，先由 smoke 确定可用资源与预算。

恢复同一 run 时需使用相同代码 commit、Python/包版本、GPU 型号及固定路径；不能在新 VM 上直接跟随 main 更新代码。固定 upstream 为 `974010a026c731054592d8f777218bd9d85a6c24`。安装真实训练依赖（包括 av-benchmark）、FFmpeg < 7 共享库，并准备 encoder、官方 `mmaudio_small_44k.pth` 和 upstream `ext_weights/empty_string.pth`。现有 [Colab media lock](../environments/colab-media-20260928.lock)只是参考，未证明完整训练依赖可用。`official_preflight` 会检查实际 import/decoder；不会自动安装或下载。

复制 `config/colab.example.json` 为私有配置，建议保存在 Drive 项目目录，或被忽略的 `config/colab.local.json`。补全 HF snapshot 路径、Drive bundle 路径、BUNDLE_READY SHA256、独立备份目录；`python` 设为实际 `sys.executable`。固定所有本地路径，以便恢复后的 receipt 字节和 recipe 身份保持一致。不得把真实配置、Caption、数据或凭据提交 Git。

五模态张量本身约 9.21 GiB；还需容纳模型、优化器、EMA、两个 checkpoint，以及 smoke 恢复测试保留的本地副本。暂存时只检查 bundle 加 2 GiB 的最低余量，**这不是完整训练磁盘需求**。先测实际 peak memory、checkpoint/EMA 体积和 Drive 配额再决定步数。

```bash
# 预览不会复制数据或启动 GPU。
python -m fine_tune.colab_train --config "$CONFIG" --stage prepare
# 核实配置后添加 --execute：环境预检 → Drive 包校验 → 本地复制校验 → 运行绑定。
```

目标包若已存在，只允许完整哈希匹配后复用；不完整暂存保留现场，必须检查后选择新的目标或处理自己的未完成副本，不能直接覆盖。运行绑定位于包和 run 之外；重连时仅允许复用绑定到同一包的文件。

## 2. Colab smoke：验证 Drive 恢复

```bash
python -m fine_tune.colab_train --config "$CONFIG" --stage smoke
# 正式执行这个小测试需显式加 --execute。
```

执行流程：本地新进程训练 20 步 → 保存 checkpoint 和 Drive 校验快照 → 将本地 smoke run 保留为 `.before-restore` → 从 Drive 快照恢复到相同本地路径 → 第二个进程恢复并推进至 24 步。任何一步失败，正式训练都被阻止。

需要 `smoke_report.json` 的 PASSED / `checkpoint_roundtrip_verified=true`，以及 Drive 中的 `COLAB_SMOKE_READY.json`。检验内容包括 model、optimizer、scheduler、EMA、随机数状态恢复、有限 loss/gradient、五模态输入和至少 90% 的可训练 tensor 更新。Smoke/正式 run 分开；旧 smoke copy 不会自动删除。

## 3. 分段训练与快照

确定正式总 `steps`（能整除 `ema_every`）后：

```bash
python -m fine_tune.colab_train --config "$CONFIG" --stage train
python -m fine_tune.colab_train --config "$CONFIG" --stage resume
```

各命令默认预览，加 `--execute` 才训练。`train` 用于全新 run，`resume` 用于已检查过的同一 run。默认每次最多推进 250 步，每 100 步保存一次，阶段结束再次保存并暂停；达到总步数时完成最终 raw/EMA 导出。不会自动循环启动下一段。`session_updates` 可随预算改变，但既有 run 的 recipe、总步数和保存/验证/EMA 策略不可暗改。

快照采用按 SHA256 存储的对象，未变化的 metadata/EMA 文件复用。每份版本收据覆盖 checkpoint、sidecar、训练配置、完整 EMA 历史、best/final 权重、日志和 smoke 证明。写入并回读验证全部成功后，才更新 Drive `LATEST.json`。失败的临时对象不会成为恢复点，训练会报告备份失败。

备份只追加、不自动删除。新的 checkpoint 仍可能很大，**每 100 步备份不是零成本**；用 smoke 测量写入时间和新增空间，再在正式 run 开始前调整保存间隔并记录。只允许一个 Colab VM 操作一个 run/backup，避免两个会话同时推进。Drive/FUSE 持久性和真实 GPU 行为仍需上述实机 smoke 验证。

## 4. VM 重置后的恢复

先挂载 Drive、恢复同一代码/环境/固定 encoder，再执行 `prepare --execute` 重新暂存 features 与相同路径绑定。随后将 smoke 和训练 run 分别从其最新有效快照恢复：

```bash
# 默认仅验证与预览。相同路径必须不存在，避免覆盖仍可恢复的本地 run。
python -m fine_tune.run_snapshot --backup-root "$SMOKE_BACKUP" --output "$SMOKE_RUN"
python -m fine_tune.run_snapshot --backup-root "$TRAIN_BACKUP" --output "$TRAIN_RUN"
# 核验后给各命令加 --execute，再执行 colab_train --stage resume --execute。
```

恢复会校验完成收据及其所有对象；不复制 `.training.lock` 和 `STOP_REQUESTED`。本地已有 run 时直接检查 checkpoint 并 resume，不能再做覆盖式 restore。代码、环境、硬件或 hash 不匹配会阻止继续，不能改写旧收据来绕过。

有计划地暂停可在本地 run 创建 `STOP_REQUESTED`，等待 `phase=paused` 且 Drive LATEST 对应步数通过验证。在同一 VM 继续前，先检查并明确移走停止标记。若 VM 被硬终止，只能恢复最后一个完整快照；未保存的 updates 可能丢失。恢复不是自动重连，也不规避 Colab 的资源限制。

## 验证范围与后续阶段

本地 CPU 测试覆盖真实文件 snapshot/restore、损坏与未完成文件拒绝、EMA 目录恢复、路径保护、Colab preview 和分段命令。Notebook 的默认无副作用代码可在本地执行检查；实际 `prepare`/`smoke`/`train` 需要 Colab、挂载的 Drive、冻结训练包、权重和训练环境，尚未运行。

当前主要待办仍是 Caption 人审和文本特征。Myriad SSH 可用性不再阻塞 MMAudio 的 Colab 路线；后续 perception embedding 训练需使用独立配置、checkpoint 和实验记录，避免将两种训练的结果混为一谈。
