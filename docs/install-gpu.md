# GPU 版安装

先完成[公共准备](install-common.md)中的 Python/uv 安装。这个指南只安装 NVIDIA CUDA 运行依赖，不安装开发测试依赖。

## 检查设备

在 PowerShell 运行 `nvidia-smi`，确认系统能识别 NVIDIA GPU 和驱动。项目使用 CUDA 12.8 版 PyTorch wheel；需要兼容的 NVIDIA 驱动，不需要另装本机 CUDA Toolkit、DeepSpeed 或 FlashAttention。

## 快速安装与预取模型

在项目根目录的 PowerShell 中整段复制执行：

```powershell
Set-Location C:\src\Voice2
$Extra = "gpu"
uv sync --extra $Extra --no-dev

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$env:USE_MODELSCOPE = "true"
uv run --no-sync --no-dev --extra $Extra hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints --max-workers 4
uv run --no-sync --no-dev --extra $Extra python -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available('checkpoints')"
```

然后运行[公共准备中的模型文件检查](install-common.md#检查模型文件)。必须看到“模型文件齐全”，否则不要启动服务。主模型约 5.5 GB，辅助模型约 5.3 GB。

## 验证 CUDA 并启动

保持在同一个 PowerShell 会话中。以下命令先验证 CUDA，再以离线模式启动，强制使用 GPU：

```powershell
$env:USE_MODELSCOPE = "false"
$env:HF_HUB_OFFLINE = "1"
uv run --no-sync --no-dev --extra $Extra python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
uv run --no-sync --no-dev --extra $Extra voice-reader-web --device cuda
```

打开 `http://127.0.0.1:7860`，上传参考录音并创建声音档案，然后生成语音。模型和辅助资源已预取；离线模式会阻止生成期间下载，若资源缺失会直接报错。

显存不足会导致模型加载或推理失败；本指南不会静默切换设备。
