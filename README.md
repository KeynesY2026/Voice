# Local Voice Reader（本地语音生成器）

使用本机模型生成英语语音，支持声音档案、零样本声音克隆和自动/手动情绪控制。创建档案只保存参考录音，不会训练模型；文本和录音不会上传。

## 安装与首次运行

请先按[公共准备说明](docs/install-common.md)安装 Python/uv、下载主模型和预取辅助模型，再选一份与你的设备对应的指南：

- [CPU 版安装](docs/install-cpu.md)
- [GPU 版安装](docs/install-gpu.md)

两种安装共用同一组模型文件，但运行依赖分别安装。平台指南提供最小依赖和可复制的完整命令。生成前必须完成公共说明中的模型文件检查；`start-auto.bat` 会以离线模式启动，不会在生成时补下载模型。

## 使用 Web 界面

启动完成后打开 `http://127.0.0.1:7860`。在 **Voice profiles** 中上传录音、裁剪到 3–30 秒并创建档案，然后在 **Generate speech** 输入文本并生成。声音档案和生成结果保存在被 Git 忽略的 `private-data/` 中。

录音支持 WAV、MP3、M4A 和 FLAC，每个文件最多 100 MiB。请只使用自己或已获准使用的录音。默认 Web 服务只绑定本机，不会创建公开分享链接。

## 命令行

平台安装指南会设置对应的 `$Extra`。请在完成安装和模型检查的同一 PowerShell 会话中运行；新开会话时，先按该指南重新设置 `$Extra`。选择参考录音并创建档案后，可用以下命令生成语音：

```powershell
$env:USE_MODELSCOPE = "false"
$env:HF_HUB_OFFLINE = "1"
uv run --no-sync --no-dev --extra $Extra voice-reader speak --text "Hello there." --profile PROFILE_ID
```

把 `PROFILE_ID` 替换为 `voice-reader profiles list` 返回的档案 ID。命令行的档案管理和文本输入方式见 `voice-reader --help`。

## 本地模型与隐私

模型文件放在 `checkpoints/`，首次初始化使用的辅助模型放在 `checkpoints/hf_cache/`。两处目录都不应提交到 Git。模型下载需要网络；完成预取后，启动器和平台指南会启用离线模式，避免生成期间联网下载。

请遵循 [IndexTTS 上游许可与使用条款](https://github.com/index-tts/index-tts)。
