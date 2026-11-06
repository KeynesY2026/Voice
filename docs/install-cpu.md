# CPU 版安装

先完成[公共准备](install-common.md)中的 Python/uv 安装。这个指南只安装 CPU 运行依赖，不安装开发测试依赖。

## 快速安装与预取模型

在项目根目录的 PowerShell 中整段复制执行：

```powershell
Set-Location C:\src\Voice2
$Extra = "cpu"
uv sync --extra $Extra --no-dev

$env:HF_ENDPOINT = "https://huggingface.co"
$env:HF_HUB_DISABLE_XET = "1"
uv run --no-sync --no-dev --extra $Extra hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints --max-workers 4
uv run --no-sync --no-dev --extra $Extra python -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available('checkpoints')"
```

然后运行[公共准备中的模型文件检查](install-common.md#检查模型文件)。必须看到“模型文件齐全”，否则不要启动服务。

## 启动并验证

保持在同一个 PowerShell 会话中，设置离线模式再启动：

```powershell
$env:USE_MODELSCOPE = "false"
$env:HF_HUB_OFFLINE = "1"
uv run --no-sync --no-dev --extra $Extra voice-reader-web --device cpu
```

打开 `http://127.0.0.1:7860`，上传参考录音并创建声音档案，然后生成语音。模型和辅助资源已预取；离线模式会阻止生成期间下载，若资源缺失会直接报错。

该安装只使用 CPU。首次模型加载和语音生成可能需要较长时间。
