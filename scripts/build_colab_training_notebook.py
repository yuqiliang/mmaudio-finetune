"""Build the dependency-free Colab handoff notebook; no training is executed."""
import json
from pathlib import Path
import textwrap


def build():
    cells = []

    def cell(kind, source):
        value = dict(cell_type=kind, id=f"cell-{len(cells):02d}", metadata={},
                     source=textwrap.dedent(source).strip() + "\n")
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """
        # Expanded MMAudio · Colab 训练
        ## Goal
        当前安排：MMAudio 在 Colab 完成特征收尾和训练；Myriad 留给后续 perception embedding 训练。
        7740 条音视频特征已完成，Caption 人审仍待完成。先冻结 Caption、补齐文本特征并生成校验过的 BUNDLE。
        本 notebook 是运行入口，未运行真实 GPU 训练。默认参数不会安装、复制数据或启动 GPU。
        完整说明：[Colab 操作指南](https://github.com/yuqiliang/mmaudio-finetune/blob/codex/myriad-training-prep-20261007/docs/COLAB_OFFICIAL_TRAINING.md)。
    """)
    cell("markdown", """
        ## Setup
        选择 GPU，检查实际显存/内存/磁盘和 Drive 配额。恢复同一 run 必须使用同一代码 commit、环境、GPU 型号及本地路径。
        填写审定的 40 位代码 commit 和私有配置文件路径。配置模板为 `config/colab.example.json`。
        代码 checkout 和固定 upstream 分开；训练环境、encoder、初始权重和 empty_string 需按指南提前准备。
    """)
    cell("code", """
        from pathlib import Path
        import json, os, re, subprocess, sys

        CODE_ROOT = Path("/content/mmaudio-finetune")
        CODE_COMMIT = ""  # 填写本 PR 审定的完整 Git commit；重连时保持不变
        CONFIG_PATH = None  # 例如挂载 Drive 后的私有 colab.local.json 绝对路径
        SETUP_CHECKOUTS = False
        MOUNT_DRIVE = False
        ACTION = "preview"  # preview / prepare / smoke / train / resume / restore-smoke / restore-train
        EXECUTE = False     # 逐阶段预览后明确设 True；不自动循环
    """)
    cell("markdown", """
        ### 挂载 Drive 与检出固定代码
        只在需要时单独启用相应开关。此步骤不会安装包、下载权重或启动训练。
        现有非空或不匹配 checkout 会停止，避免覆盖其他会话的代码。
    """)
    cell("code", """
        if MOUNT_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
        if SETUP_CHECKOUTS:
            assert sys.platform == "linux" and Path("/content").is_dir(), "此步骤需要 Colab"
            assert re.fullmatch(r"[0-9a-f]{40}", CODE_COMMIT), "先填写完整代码 commit"
            repositories = [
                (CODE_ROOT, "https://github.com/yuqiliang/mmaudio-finetune.git", CODE_COMMIT),
                (Path("/content/MMAudio-official"), "https://github.com/hkchengrex/MMAudio.git", "974010a026c731054592d8f777218bd9d85a6c24"),
            ]
            for root, remote, commit in repositories:
                if not root.exists():
                    subprocess.run(["git", "clone", "--no-checkout", remote, str(root)], check=True)
                    subprocess.run(["git", "-C", str(root), "checkout", "--detach", commit], check=True)
                actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                assert actual == commit, "现有 checkout 不匹配，先检查，不自动修改"
                assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True).strip(), "代码存在修改"
            print("固定代码已检查；继续核对依赖和权重。")
        else:
            print("安全默认：未检出代码、安装环境或访问 Drive。")
    """)
    cell("markdown", """
        ## Steps
        先完成人审/冻结与五模态 BUNDLE，然后按 `prepare → smoke → train` 执行，每阶段先保持 EXECUTE=False 预览。
        Smoke 会训练 20 步，验证 Drive 快照，保留本地原副本并从 Drive 恢复，再由新进程推进至 24 步。
        正式 steps 需在 smoke 后明确设置；默认每次最多推进 250 步后保存暂停。后续用 resume。
        VM 重置后，恢复相同环境与路径，重新 prepare，再分别 restore-smoke / restore-train，最后 resume。
    """)
    cell("code", """
        allowed = {"preview", "prepare", "smoke", "train", "resume", "restore-smoke", "restore-train"}
        assert ACTION in allowed, "无效 ACTION"
        if CONFIG_PATH is None:
            print("请先填写私有 CONFIG_PATH；当前没有复制数据或执行训练。")
        else:
            config_path = Path(CONFIG_PATH)
            config = json.loads(config_path.read_text())
            assert re.fullmatch(r"[0-9a-f]{40}", CODE_COMMIT), "先固定本次代码 commit"
            actual = subprocess.check_output(["git", "-C", config["code_root"], "rev-parse", "HEAD"], text=True).strip()
            assert actual == CODE_COMMIT, "代码 commit 与本次配置不一致"
            if ACTION.startswith("restore-"):
                kind = ACTION.split("-", 1)[1]
                command = [config["python"], "-m", "fine_tune.run_snapshot", "--backup-root", config[kind + "_backup"], "--output", config[kind + "_run"]]
            else:
                stage = "prepare" if ACTION == "preview" else ACTION
                command = [config["python"], "-m", "fine_tune.colab_train", "--config", str(config_path), "--stage", stage]
            if EXECUTE and ACTION != "preview":
                command.append("--execute")
            subprocess.run(command, check=True, cwd=config["code_root"], env={**os.environ, "PYTHONPATH": config["code_root"]})
    """)
    cell("markdown", """
        ## Checks
        读取精简状态，确认 completed_updates 与 Drive 最新完整快照一致。正式 run 需要匹配的 smoke 和 Drive 恢复证明。
        `paused` 是已保存暂停，不是训练完成；只有达到目标步数并写出最终权重后才是 complete。
    """)
    cell("code", """
        if CONFIG_PATH is not None:
            config = json.loads(Path(CONFIG_PATH).read_text())
            for kind in ("smoke", "train"):
                status_path = Path(config[kind + "_run"]) / "training_status.json"
                latest_path = Path(config[kind + "_backup"]) / "LATEST.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                latest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
                print(kind, {"phase": status.get("phase"), "local_updates": status.get("completed_updates"), "saved_updates": latest.get("completed_updates")})
        else:
            print("无训练状态：尚未指定配置。")
    """)
    cell("markdown", """
        ## Next Steps
        当前阻塞是 Caption 人审与文本特征；此 notebook 不替代人审，也没有运行真实 Colab GPU smoke。
        硬中断后只能恢复最后一个完整快照。Drive 备份只追加、不自动清理，需要根据实际 checkpoint/EMA 体积预留配额。
        一个 run/backup 同时只允许一个 VM 操作。未来 perception embedding 使用独立数据约定、配置和 Myriad 作业。
        参考：[Colab 运行时与存储限制](https://research.google.com/colaboratory/faq.html)。
    """)
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "colab_official_training.ipynb"}})


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "notebooks/colab_official_training.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
