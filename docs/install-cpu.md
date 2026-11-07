# CPU 版安装

先完成[公共准备](install-common.md)中的 Python/uv 安装。这个指南只安装 CPU 运行依赖，不安装开发测试依赖。

## 快速安装与预取模型

在项目根目录的 PowerShell 中整段复制执行：

```powershell
Set-Location C:\src\Voice2
$Extra = "cpu"
uv sync --extra $Extra --no-dev

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$env:USE_MODELSCOPE = "true"
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

## CPU 线程数

CPU 模型加载前，应用在每个进程中设置一次 PyTorch intra-op 线程数；不修改 inter-op。
无显式设置时，读取当时的默认值 `D = torch.get_num_threads()` 与逻辑处理器数
`L = os.cpu_count()`，使用启发式 `max(D, min(L, ceil(1.5 × D)))`。
若 `L` 未知，回退到 `D`；即使 `L < D`，也不自动降低当前线程数。
例如默认 4、逻辑处理器 8 时选择 6，默认 4、逻辑处理器 4 时保持 4。

需要固定线程数时，在启动服务前设置环境变量：

```powershell
$env:MKL_NUM_THREADS = "4"
uv run --no-sync --no-dev --extra cpu voice-reader-web --device cpu
```

显式设置优先级为 `MKL_NUM_THREADS` > `OMP_NUM_THREADS` > 自动启发式，
与 PyTorch 的 MKL 优先约定一致；同时存在时只使用前者。
显式值可以低于当前默认值，也不按逻辑处理器数截断。
应用会显式调用 `torch.set_num_threads()`，并读取实际值确认生效，
不依赖环境变量被 PyTorch 自动采纳；所选变量必须是正整数，否则启动模型时报错。
终端记录默认值、逻辑处理器数、目标值、实际值及来源。
修改环境变量后需重启推理进程；在程序内直接调用 PyTorch 设置线程不属于本应用的配置入口。

这个 1.5×规则是确定性启发式，不是跨硬件最优值或固定加速保证。
它不改变模型权重、精度、采样、beam 数或扩散步数，也不影响 GPU 运行策略。
不同线程下的浮点计算或随机采样可能导致音频差异，不能据此承诺“无质量损失”。

本仓库在 i7-11370H（4 核／8 逻辑处理器、PyTorch 2.8.0+cu128 的 CPU 路径）
用同一短句、内置男声、手动平静情绪做过 4／6 线程暖机交错对照，各两轮；
排除模型加载与预热后，中位耗时为 30.11／27.39 秒，RTF 为 23.58／21.45。
但单轮分别波动于 25.24–34.98／21.86–32.92 秒，波动大于中位差，
样本不足以证明稳定加速；仅建议把自动值作为可显式覆盖的候选。
暖机输出均为 22050 Hz、1.277 秒；固定种子的波形高度相关但不逐位相同。
未做主观试听或语音识别质量评测，也未在此处复测 EPYC 或其他 CPU。
