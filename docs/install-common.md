# 公共准备：Local Voice Reader

本文件只写所有安装方式共用的准备步骤。具体依赖和启动命令在各自平台指南中；先按对应指南设置 `$Extra`，再执行本文件中的模型步骤。

## 前置条件

- Windows PowerShell，Python 3.11（项目支持 3.10–3.11）和 `uv`。
- 建议预留至少 **25 GB** 磁盘空间：运行环境约数 GB，模型主文件约 5.5 GB，辅助模型约 5.3 GB；下载缓存可能需要额外空间。
- 语音模型在本机运行；安装依赖和下载模型需要网络。

## 安装 uv 和 Python 3.11

在项目根目录执行。现有系统 Python 可以是 3.13，用它安装 uv 没问题；项目本身仍须使用 3.11。

```powershell
Set-Location C:\src\Voice2
python -m pip install --user uv --index-url https://mirrors.aliyun.com/pypi/simple
$UvScripts = python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))"
$env:Path = "$UvScripts;$env:Path"
uv --version
uv python install 3.11
uv python find 3.11
```

如果系统完全没有 Python，先安装一个可信来源的 Python 3.11 x64，再用它安装 uv。`uv python install` 的解释器下载没有经过验证的国内镜像，不建议改用不明来源。

## 国内镜像与上游下载源

项目已配置：

- 阿里云 PyPI：普通 Python 依赖。
- SJTU PyTorch wheel 镜像：项目自动选择所需的 wheel。

正常执行 `uv sync` 时会按 `uv.lock` 使用这些地址，不需要额外索引参数。项目默认不安装 pytest 等开发依赖；安装命令也显式使用 `--no-dev`。若修改依赖或镜像地址，请重新运行 `uv lock`，并检查 `uv.lock` 中解析出的来源。

IndexTTS 源码固定在 GitHub 版本归档；模型来自 Hugging Face。当前依赖版本使用 `hf-mirror.com` 曾遇到元数据错误。若镜像下载失败，使用下文的官方 endpoint。上游辅助模型下载器会自行尝试 ModelScope；只有资源缺失时才会访问模型源。

## 下载并预取模型

在对应平台指南设置 `$Extra` 后，于项目根目录执行。主模型约 **5.5 GB**；辅助模型约 **5.3 GB**。这两步必须在生成前完成。

```powershell
$env:HF_ENDPOINT = "https://huggingface.co"
$env:HF_HUB_DISABLE_XET = "1"
uv run --no-sync --no-dev --extra $Extra hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints --max-workers 4
uv run --no-sync --no-dev --extra $Extra python -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available('checkpoints')"
```

`ensure_models_available` 会把 IndexTTS 首次初始化所需的辅助权重准备到 `checkpoints/hf_cache/`。成功时应打印 `All auxiliary models ready.`

### 检查模型文件

预取后运行以下检查。每个路径都必须是非空文件；只有显示“模型文件齐全”才能启动生成：

```powershell
$Required = @(
  "checkpoints/config.yaml",
  "checkpoints/gpt.pth",
  "checkpoints/codec.pth",
  "checkpoints/s2mel.pth",
  "checkpoints/wav2vec2bert_stats.pt",
  "checkpoints/feat1.pt",
  "checkpoints/feat2.pt",
  "checkpoints/multilingual_zh_ja_yue_char_del.tiktoken",
  "checkpoints/qwen0.6bemo4-merge/config.json",
  "checkpoints/qwen0.6bemo4-merge/model.safetensors",
  "checkpoints/qwen0.6bemo4-merge/tokenizer.json",
  "checkpoints/qwen0.6bemo4-merge/tokenizer_config.json",
  "checkpoints/qwen0.6bemo4-merge/vocab.json",
  "checkpoints/qwen0.6bemo4-merge/merges.txt",
  "checkpoints/hf_cache/w2v-bert-2.0/config.json",
  "checkpoints/hf_cache/w2v-bert-2.0/model.safetensors",
  "checkpoints/hf_cache/w2v-bert-2.0/conformer_shaw.pt",
  "checkpoints/hf_cache/w2v-bert-2.0/preprocessor_config.json",
  "checkpoints/hf_cache/semantic_codec_model.safetensors",
  "checkpoints/hf_cache/campplus_cn_common.bin",
  "checkpoints/hf_cache/bigvgan/config.json",
  "checkpoints/hf_cache/bigvgan/bigvgan_generator.pt"
)
$Missing = @(
  foreach ($Path in $Required) {
    if (-not (Test-Path $Path -PathType Leaf)) {
      $Path
    } elseif ((Get-Item $Path).Length -le 0) {
      $Path
    }
  }
)
if ($Missing.Count -gt 0) {
  Write-Host "缺少模型文件：" -ForegroundColor Red
  $Missing
  throw "模型准备未完成；请重新下载缺失的资源。"
}
Write-Host "模型文件齐全。"
```

主模型下载如果出现 `LocalEntryNotFoundError` 或 HEAD/metadata 错误，请保持 `$env:HF_HUB_DISABLE_XET = "1"`，并将 `$env:HF_ENDPOINT` 设为 `https://huggingface.co` 后重试。不要在服务运行时补下模型。

## 运行时离线

完成文件检查后再启动服务。所有平台指南和 `start-auto.bat` 都会设置以下变量：Hugging Face 离线模式，以及禁用辅助模型下载器的自动切源。这样如果漏了模型会立即报缺失，而不是在生成过程中联网下载。

模型目录 `checkpoints/` 已被 Git 忽略，不要提交模型文件。
