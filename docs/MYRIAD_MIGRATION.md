# Expanded 7740：Colab 收尾与 Myriad 训练准备

## 当前事实与边界（2026-10-07）

**VERIFIED FACT**：扩展集使用官方 `small_44k` 路线；7740 条音视频特征已由 Colab 完成收据确认，分为 train 6203 / val 780 / test 757，来自 85 个 source recording。本次准备重新校验了完成收据的 SHA256、完整 ID 顺序、split、media plan 和 run 绑定，没有在本机重新下载或遍历 7740 条张量。

| 固定输入 | 身份 |
|---|---|
| MMAudio upstream commit | `974010a026c731054592d8f777218bd9d85a6c24` |
| media run ID | `1ba37d9552b3e071b722d2f9a7fb84d94a90c4ec3019af560cef87adf8cb08b7` |
| media plan ID | `9dfc9e66508fcd334fad0daaad98d97332953144cc6a807e0707be7ee01e81b9` |
| media plan SHA256 | `b30cfd34430e210ba762d731c91c686664f767bfef95d24a0b1407bba6e35d13` |
| MEDIA_COMPLETE SHA256 | `37a5764317ffe4685b9ac6bfffa495bfd3097b07dcb146e478b21cbb14f9dfe3` |
| CLIP revision | `01b771ed0d1395ca5ffdd279897d665ebe00dfd2` |
| 44k vocoder revision | `95a9d1dcb12906c03edd938d77b9333d6ded7dfb` |

完成收据为 `MEDIA_7740_CPU_VERIFIED`，时间为 2026-10-07 13:41:20 UTC。此收据仍明确 `training_ready=false`。音视频四项特征是 `mean`、`std`、`clip_features`、`sync_features`；Caption 人审尚未完成（用户已确认），文本特征、五模态最终包、Myriad smoke 和正式训练均未完成。不要将历史 6941 条 `small_16k` pilot 的 checkpoint 或指标用于此路线。

**PLANNED**：Caption 人审 → 冻结 → Colab 仅提取文本 → Colab CPU 合并/逐元素回读 → Drive 持久化 → Myriad 直接拉取 → 校验 → 两进程 smoke → 正式训练。原媒体 cache 和完成收据保持原字节。所有 Python 新入口默认只读，写入/计算需显式 `--execute`；渲染作业不提交作业。

## 1. Caption 审核与冻结

使用独立审核表，保留所有规范 ID、行顺序、split、来源和媒体哈希。原候选文件的 `training_id` 是完整规范 ID；不能用其较短的 `clip_id` 替代。`included`/decoder `pass` 不代表 Caption 或内容审核通过。

以下环境变量都由操作者填写实际路径；命令在本项目代码根目录运行。审核模板已在本地私有 `outputs/myriad_preparation_20261007/` 生成，重复生成请换新版本名。任何 manifest、原始 Caption、凭据或私有 Drive 链接都不提交 Git。

```bash
python -m fine_tune.official_feature_bundle review-template \
  --plan "$PLAN" --candidates "$CANDIDATES" --output "$REVIEW" --execute
```

逐条核实声音描述与 clip 内容后，由实际审核人填写 `caption_final`、`caption_review_status=approved`、`content_review_status=approved`、`reviewer`、含时区的 `reviewed_at` 和 `acoustic_only_reviewed=true`。这些值不可批量伪造；不合格行会阻止冻结。需要排除媒体时，应先制定新的 manifest/plan 分支，不能删除旧 cache 或在本表暗改 split。最终 Caption 必须非空、单行，且后续 tokenizer 会拒绝超过 77 个 token（含起止 token）的文本。

```bash
# 先预检；通过后在同一命令末尾添加 --execute，生成新冻结文件。
python -m fine_tune.official_feature_bundle freeze \
  --plan "$PLAN" --review "$REVIEW" --output "$FROZEN"
sha256sum "$FROZEN"  # 仅在执行冻结之后；将结果记为 FROZEN_SHA
```

冻结绑定完整 Caption 内容和审核记录。Caption 更改后必须重新冻结并提取相应文本的新版本。音频生成的 Caption 属于音频辅助训练元数据；测试时不得把它表述为 video-only 推理输入。

## 2. Colab 仅文本提取与最终包

继续使用干净、固定 revision 的 upstream checkout，以及旧媒体提取时相同的 encoder 文件。`OFFICIAL` 是 upstream checkout；当前项目代码与它分开。`STATE` 是原 `media_state.json`，`COMPLETE` 是已核验的完整媒体完成收据。`WEIGHTS` 含 `v1-44.pth`、`synchformer_state_dict.pth`；HF snapshot 必须保留 revision 目录名及实际文件，不能留下断开的符号链接。

```bash
COMPLETE_SHA=37a5764317ffe4685b9ac6bfffa495bfd3097b07dcb146e478b21cbb14f9dfe3
CLIP_REV=01b771ed0d1395ca5ffdd279897d665ebe00dfd2
VOCODER_REV=95a9d1dcb12906c03edd938d77b9333d6ded7dfb

# 默认只做来源/权重/冻结绑定预检，不加载模型、不验证 token 长度。
# 审核冻结后、明确授权 Colab GPU 工作时添加 --execute。
python -m fine_tune.official_feature_bundle text \
  --plan "$PLAN" --state "$STATE" --complete "$COMPLETE" --complete-sha256 "$COMPLETE_SHA" \
  --frozen "$FROZEN" --frozen-sha256 "$FROZEN_SHA" \
  --official-repo "$OFFICIAL" --weights-dir "$WEIGHTS" \
  --clip-snapshot "$CLIP_SNAPSHOT" --clip-revision "$CLIP_REV" \
  --vocoder-snapshot "$VOCODER_SNAPSHOT" --vocoder-revision "$VOCODER_REV" \
  --output "$TEXT_ROOT"
```

执行阶段只调用官方 `encode_text`，不重新编码音视频；记录实际 Python/包/GPU/CUDA/cuDNN、FP32/TF32、代码哈希和 batch size 1。先检查全部 Caption 的未截断 BPE 长度，再写张量。缓存逐行写入并回读，完整后发布 `TEXT_COMPLETE.json`。中断后仅在完全相同身份下显式加 `--resume --execute`；孤立文件、临时文件或身份变化会停下保留现场，不自动删除或覆盖。

```bash
# CPU 合并：先预检，再显式添加 --execute。BUNDLE 必须是全新目录。
python -m fine_tune.official_feature_bundle assemble \
  --plan "$PLAN" --state "$STATE" --complete "$COMPLETE" --complete-sha256 "$COMPLETE_SHA" \
  --frozen "$FROZEN" --frozen-sha256 "$FROZEN_SHA" \
  --media-root "$MEDIA_ROOT" --text-root "$TEXT_ROOT" --output "$BUNDLE"
```

合并流式读取每条原媒体缓存和文本缓存，检查各自收据，再写 split 独立的五模态 TensorDict。全部逐元素回读通过才发布 `BUNDLE_READY.json`，其中每个文件都有 SHA256。合并失败保留未完成目录；换一个新版本目录重做，不能将其认作完成包。最终训练包需整体保存到 Drive 并在持久副本上再次执行：

```bash
sha256sum "$BUNDLE/BUNDLE_READY.json"  # 独立保存此哈希为 BUNDLE_SHA
python -m fine_tune.official_feature_bundle verify \
  --bundle "$BUNDLE/BUNDLE_READY.json" --bundle-sha256 "$BUNDLE_SHA"
```

目录内只放收据覆盖的文件；日志、作业、README、运行绑定和 checkpoint 放在包外。移动时保留 `provenance/` 内的原收据原字节，不能替换其 Colab 路径。Myriad 使用新绑定文件解析本地路径。

## 3. 迁移范围与容量

| 迁移对象 | 目标要求 |
|---|---|
| 最终 BUNDLE 整个目录 | 保留 split memmap、TSV、逐行 manifest、provenance、BUNDLE_READY |
| 本项目代码 | 使用合并/审定后的 Git commit，记录 `git rev-parse HEAD` |
| upstream MMAudio | 独立、干净、固定在上述 commit |
| Encoder 与 HF snapshots | 必须与媒体收据中的文件 SHA256 和 revision 一致 |
| `mmaudio_small_44k.pth` | 官方初始权重，启动器核对 upstream 发布的 MD5 并记录 SHA256 |
| `ext_weights/empty_string.pth` | 放在 upstream checkout 中，SHA256 绑定 smoke/正式训练 |
| 训练依赖、环境记录 | 在 Myriad 独立准备并验证，计算节点运行时离线 |

五类 FP32 原始张量合计 **9,889,862,400 bytes，约 9.21 GiB**，不含元数据、encoder、预训练权重、环境和 checkpoint。Colab 合并阶段需要同时容纳旧缓存、文本缓存和新包；Myriad Scratch 还需为 optimizer、两个轮换 checkpoint、best/final 权重及 EMA 历史预留空间。容量和 inode 必须按实际文件与目标步数测量，不能把 9.21 GiB 当总配额需求。

在 Myriad 已认证的 rclone remote 上直接拉取 Drive，避免经 Mac 中转大文件。先按 [UCL rclone 文档](https://github-pages.arc.ucl.ac.uk/mkdocs-rc-docs/Walkthroughs/rclone/)配置 Google Drive，凭据留在账号私有配置中；模块名以当前 `module avail` 为准。

```bash
# 在 Myriad 上运行，REMOTE_BUNDLE 指向冻结包文件夹，DEST 为独立的新版本目录。
bash scripts/transfer_feature_bundle.sh "$REMOTE_BUNDLE" "$DEST" "$BUNDLE_SHA"
# 确认源和目标后，实际复制需在同一命令末尾添加 --execute。
```

该脚本采用 [rclone copy](https://rclone.org/commands/rclone_copy/) 和 `--immutable`，默认 `--dry-run`。复制后先核验完成收据 SHA256，完整文件校验由下一节 CPU verify 作业完成。中断可以重试同一复制；文件冲突或多余文件会阻止最终验证。原始媒体、旧 row cache、历史 pilot 无需随最终训练包迁移，但应在 Drive 保留作为独立来源档案。

## 4. Myriad 环境与作业

**UNKNOWN / REQUIRES CONFIRMATION**：账号已存在；本次只读 SSH 尝试超时，实际 Scratch 路径、配额、软件模块、Python 环境与可分配 GPU 尚未登录验证。

公开 [Myriad 页面](https://github-pages.arc.ucl.ac.uk/mkdocs-rc-docs/Clusters/Myriad/)和[作业示例](https://github-pages.arc.ucl.ac.uk/mkdocs-rc-docs/Example_Jobscripts/)目前使用 SGE `qsub`。因迁移计划和现场配置可能变化，登录后先确认 `command -v qsub`、`qstat`、`module avail`、实际配额与存储路径，再将配置里的 `scheduler_verified` 设为 true。若现场为 Slurm，此 renderer 会停用，需另外适配。2026-10-07 查阅的[状态页](https://github-pages.arc.ucl.ac.uk/mkdocs-rc-docs/Status_page/#latest-on-myriad)已记载 9 月 25 日网络维护结束；这不证明个人账号当前可连通。

`mem` 是**每核**内存：例子 4 核 × 8G = 32G，不是总共 8G。1 GPU、batch 2、1 小时只是 smoke 候选配置，尚无实机显存/吞吐证据。BF16 需要相应 GPU 支持；若分到不支持的设备，程序明确停止，选择合适设备或显式 FP32 并重跑相同 recipe 的 smoke。不要假设能获得 Colab 的 L4。

[存储文档](https://github-pages.arc.ucl.ac.uk/mkdocs-rc-docs/Background/Data_Storage/)说明 `$TMPDIR` 随作业清理，ACFS 在计算节点只读。训练状态、运行绑定、日志和 checkpoint 应放 Scratch 持久路径，不能只放 `$TMPDIR`，也不能写入只读 ACFS。需要保留的 checkpoint 在作业结束后另行备份并核对 SHA256。

环境准备要求：

1. Python 3.11 和 Colab 基线版本可作为兼容起点；CUDA wheel/模块须适配 Myriad 驱动。不要在现有环境无差别升级。
2. [Colab media lock](../environments/colab-media-20260928.lock) 是实际媒体提取环境的参考快照，SHA256 为 `a0c1044ef54a022ad17dcf1a5b249035bab3b41215d4ada473601594a3de2b0f`，不是已经验证的 Myriad 训练 lock。关键版本 torch/torchaudio 2.5.1+cu121、torchvision 0.20.1+cu121、tensordict 0.6.2、open-clip-torch 2.29.0、nitrous-ema 0.0.1。
3. 安装实际的 [av-benchmark](https://github.com/hkchengrex/av-benchmark)，记录其 Git commit 和依赖快照。媒体提取环境没有证明这组训练 import 可用；不能以空模块替代。固定 upstream 的 [训练说明](https://github.com/hkchengrex/MMAudio/blob/974010a026c731054592d8f777218bd9d85a6c24/docs/TRAINING.md)也要求该依赖。
4. 准备 FFmpeg < 7 的 CLI 和共享库、全部固定 encoder、初始模型与 empty-string 文件。预检会检查 torio 的共享 decoder，而不只检查 `ffmpeg -version`。
5. 先运行 `python -m fine_tune.official_preflight --official-repo "$OFFICIAL" --report "$REPORT"`。其通过只说明导入和文件/decoder 条件可用，GPU/数值正确性仍需 smoke。所有完整哈希校验和训练由计算作业执行。

将 [配置模板](../config/myriad.example.json)复制为被 Git 忽略的 `config/myriad.local.json`。填入全部绝对路径、包 SHA256、真实模块名。`binding`、`smoke_run`、`train_run`、`job_output` 必须位于包外，smoke 和正式 run 目录分开。先创建 `job_output` 父路径；Python 必须来自所选环境。正式 `steps` 默认 null，需根据 smoke 吞吐、有效 epoch 和存储评估后明确填写，并能整除 `ema_every`；不会继承历史 pilot 的 15000 步。

```bash
# 仅校验配置，再添加 --execute 写出四个 qsub 文件；不会提交。
python -m fine_tune.myriad render --config config/myriad.local.json --output "$JOBS"

# 在已生成并检查脚本之后，按以下顺序手动提交。
# 每阶段成功后才进行下一阶段，不将失败的 verify 当作可继续条件。
cd "$JOB_OUTPUT"
qsub "$JOBS/verify.qsub"
# 检查 verify 的 FINISHED.json、preflight.json、runtime binding，然后：
qsub "$JOBS/smoke.qsub"
# 检查 smoke_report.json 的 PASSED 和 checkpoint_roundtrip_verified=true，然后：
qsub "$JOBS/train.qsub"
```

配置 SHA256 会写进每个脚本；修改配置需生成新版本脚本。verify 阶段遍历全部文件哈希并绑定目标 encoder/source；不改包内文件。每项作业保留模块、Python、pip freeze、GPU/驱动、日志与成功/失败收据。

smoke 使用同一候选训练 recipe：第一进程运行到第 20 步并保存暂停状态；第二进程从校验过的 checkpoint 恢复，再前进至第 24 步。核对 model、optimizer、scheduler、EMA、Runner/Python/NumPy/torch/CUDA RNG 的恢复状态，并验证有限 loss/gradient、全部模态和至少 90% 的可训练 tensor 实际更新。正式训练检查 smoke 的数据、代码、权重、empty-string、recipe 与实际 GPU/软件身份；改变这些条件需重跑 smoke。

正式训练用官方 backbone/Runner、AdamW、flow-matching loss、官方 latent normalization 和 EMA；自定义部分是视频数据选择、日志、保护性 checkpoint 和固定 RNG validation。仅在 train 上更新，val 用于 raw validation 和 best checkpoint，test 不自动读取或评估。需要单独设计并授权最终测试和感知评估。

## 5. 暂停、恢复与验收

在 run 目录创建 `STOP_REQUESTED` 可在 batch 边界暂停并写 checkpoint；`paused` 不会写成 `completed`。正式恢复前，检查 `training_status.json`、`latest_checkpoint.json`、对应 checkpoint SHA256；确认继续后移走停止标记，再提交 `resume.qsub`。恢复使用相同 recipe/目标步数/环境，拒绝从其他 run 的 checkpoint 或已完成 run 继续。硬终止只能恢复最后一次完整 checkpoint，不能保证保住尚未保存的步数。

首次 GPU smoke 后才确定正式资源和步数，记录 GPU 型号/显存峰值、秒/step、validation 时间、checkpoint 体积、EMA 增长、实际模块与 pip freeze。当前 CPU synthetic tests 验证数据绑定、损坏拒绝、真实 TensorDict 文件迁移和作业保护逻辑，**没有执行实际 GPU smoke、迁移或训练**。

当且仅当 Caption 冻结、文本收据、最终包收据、目的端 verify 和匹配 GPU smoke 均通过后，才满足开始正式训练的准备条件。
