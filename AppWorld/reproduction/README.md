# AppWorld + SGLang 复测入口

本目录把 2026-10-10 的本地适配整理为可迁移入口。使用原版
`life_harness_appworld.runner`、`policies/v009_r4`、AppWorld function-calling
提示词和官方 evaluator；不复制任务数据库、模型、逐步轨迹或原始运行产物。

## 1. 下载这个分支，准备 AppWorld

```bash
git clone --single-branch --branch codex/appworld-sglang-repro-20261011 \
  https://github.com/Wendy-light/Life-Harness.git
cd Life-Harness
python3 AppWorld/reproduction/setup.py --appworld-root ../appworld --python python3.12
```

需要 Linux、Git、Python 3.12 和 venv；安装器优先使用 `uv`，否则使用 pip。
新机器需要能够访问 GitHub 和 AppWorld 数据下载站点。安装器会：

1. 克隆 `stonybrooknlp/appworld`，固定到 `appworld_revision.txt` 中的提交。
2. 应用 `patches/appworld-http-timeout.patch`，只把 HTTP timeout 传给 SDK。
3. 创建 AppWorld 的 `.venv`，安装 AppWorld、实验依赖及本仓库适配器。
4. 执行 `appworld install --repo`，下载 AppWorld 数据。

已有 checkout 必须匹配固定提交；安装器不会擅自切换已有仓库分支。
已经安装好时可加 `--skip-install --skip-download`，只检查版本并补齐补丁。
`constraints-appworld-py312.txt` 记录已测试环境的依赖版本，作为安装约束使用。
SGLang 与 AppWorld 使用**分开的 Python 环境**，避免 OpenAI SDK 版本冲突。

## 2. 如果已经部署好 SGLang，直接连接

`--model` 必须与服务 `/v1/models` 返回的 id 完全一致，可以是模型路径或
`--served-model-name` 指定的别名。`--server-url` 可带或不带 `/v1`。
以下命令均在 Life-Harness 根目录执行：

```bash
export APPWORLD_ROOT="$(realpath ../appworld)"
export SERVER_URL=http://127.0.0.1:30001
export MODEL_ID=Qwen3-4B-Instruct-2507

"$APPWORLD_ROOT/.venv/bin/python" AppWorld/reproduction/run.py \
  --appworld-root "$APPWORLD_ROOT" --server-url "$SERVER_URL" --model "$MODEL_ID" \
  --run-name small-model-preflight --check-only
```

服务需要认证时设置 `NO_API_KEY`（或 `OPENAI_API_KEY`）；密钥不会写入运行清单。
运行入口会为模型请求移除代理设置，确保本机或内网模型服务直连。

## 3. 先跑 5 条，再跑正式 dev57

```bash
# 5 个固定 train 任务，Baseline 和 Life-Harness 各跑一次；不计入 dev 分数。
"$APPWORLD_ROOT/.venv/bin/python" AppWorld/reproduction/run.py \
  --appworld-root "$APPWORLD_ROOT" --server-url "$SERVER_URL" --model "$MODEL_ID" \
  --run-name small-model-pilot --pilot --num-processes 5

# 检查 pilot summary.json、两份 audit 和轨迹后，以独立名称启动全量。
mkdir -p local-logs
nohup "$APPWORLD_ROOT/.venv/bin/python" -u AppWorld/reproduction/run.py \
  --appworld-root "$APPWORLD_ROOT" --server-url "$SERVER_URL" --model "$MODEL_ID" \
  --run-name small-model-dev57 --num-processes 16 \
  > local-logs/small-model-dev57.log 2>&1 < /dev/null &
echo "Queue PID: $!"
```

两个条件默认顺序执行，始终使用同一个模型服务。任务队列**不会停止 SGLang**。
每次使用新的 `--run-name`；入口拒绝覆盖已有输出，防止误复用旧轨迹。
`--conditions baseline` 或 `--conditions harness` 可单独运行一组。
服务器 CPU/内存较小时可以减少 `--num-processes`，并记录这个变化。

结果写到：

```text
$APPWORLD_ROOT/experiments/outputs/reproduction/portable/<run-name>/
  manifest.json           本次参数，不含密钥
  status.json             当前条件 / complete / failed
  summary.json            各条件成功任务数、TGC、SGC、接口错误数
  baseline_audit.json     请求参数、工具调用、接口/执行问题
  harness_audit.json
  *_metrics.json          官方 evaluator 的详细结果（全量模式）
  run_logs/               每个 worker 的日志
  baseline/tasks/         原始轨迹，仅本地保存
  harness/tasks/
```

pilot 必须每个任务都结束、有 API 执行、无接口错误或检测到的格式问题才会成功退出。
兼容性检查通过不等于任务成功，也不能保证所有未见任务都无格式问题。
正式评测保留失败任务及接口错误在 57 条分母中；不会选择性剔除。进程崩溃或
任务缺失会终止计分，避免把不完整结果当成完整 dev 分数。

## 4. 如果需要启动 SGLang

使用适合目标 GPU/CUDA 的独立 SGLang 环境。我们验证过的版本是 SGLang 0.5.12、
PyTorch 2.11.0、Transformers 5.6.0、Triton 3.6.0、FlashInfer 0.6.11.post1；
原服务器是 8 张 RTX PRO 5000 72GB Blackwell，CUDA toolkit 13.0。
另一台机器的 GPU、驱动和 CUDA 扩展需要匹配，不能直接复制原服务器的 `.venv`。
以下 `python` 指 SGLang 环境中的 Python：

```bash
# 小模型单卡示例；把 /models/... 换成目标机器上的完整模型目录。
CUDA_VISIBLE_DEVICES=0 python AppWorld/reproduction/serving/serve.py \
  --model-path /models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen3-4B-Instruct-2507 \
  --tool-call-parser qwen --tp 1 --dp-size 1 --port 30001

# xLAM：保留原启动脚本的自定义 parser 注册方式，取消硬编码环境路径。
CUDA_VISIBLE_DEVICES=0,1,2,3 python AppWorld/reproduction/serving/start_sglang_for_xlam.py \
  --model-path /models/Llama-xLAM-2-70b-fc-r \
  --served-model-name Llama-xLAM-2-70b-fc-r --tp 4 --dp-size 1 --port 30001
```

服务默认绑定 `127.0.0.1`，跨机器连接时按需指定 `--host 0.0.0.0`。
运行在前台；需要后台驻留时用 tmux/systemd 或 nohup 启动这个服务命令。
默认显存比例 0.9、attention backend `triton`、服务 seed 100。

| 模型系列 | parser | 本次测过的 TP / DP | 额外说明 |
|---|---|---|---|
| Qwen2.5-32B-Instruct | qwen | 2 / 1 | 历史本轮使用 context 32768 |
| Qwen3-4B、Qwen3-30B-A3B Instruct-2507 | qwen | 1 / 2 | 原生非思考版本；单卡复测可 DP=1 |
| Qwen3.5-27B、Qwen3.6-27B | qwen3_coder | 1 / 2 | 原生 thinking，没有显式关闭 |
| Qwen3.5-35B-A3B | qwen3_coder | 2 / 1 | 原生 thinking |
| Llama-3.3-70B-Instruct | llama3 | 4 / 1 | 需加 `--llama-history-template` |
| Llama-xLAM-2-70b-fc-r | xlam | 4 / 1 | 自定义 parser，使用原生模板 |

xLAM parser 来源语义是 AppWorld 的
`experiments/code/common/vllm_plugins/xlam_tool_call_parser.py`：解析完整
`[{"name": "...", "arguments": {...}}, ...]`，普通文本及无法解析的内容原样保留。
Llama 3.3 的模板适配把历史消息里的并行调用逐个序列化，保留每个调用及参数；
原模板会拒绝 AppWorld 的并行工具调用示例。不要给 xLAM 或 Qwen 开这个适配。

## 协议与可比性

- 默认官方 dev57、原版 v009_r4、两个条件完全相同的任务集。
- 主模型和 API predictor 均 temperature=1、请求 seed=100；主模型不设显式输出上限，
  predictor 上限 3000，max_steps=50，HTTP timeout=14400 秒。
- 默认 `--thinking native`，不传 `enable_thinking`。`--thinking off` 是单独的 Qwen
  消融设置，不能与此次默认结果混为一组。Instruct-2507 本身就是非思考模型。
- `--max-steps 5` 只适合快速检查程序，不是表格中的正式评测协议。
- SGLang 可能采用模型 generation_config 中其他默认采样项，例如 top_p；
  不同模型/服务版本不保证逐 token 相同。实际可用上下文也受 KV cache 容量影响，
  需要记录启动日志，不能只看模型标称 context length。
- 核心策略保持原样，包括已知的 answer-stripping 行为。此次没有悄悄修策略以提高分数。

关键结果和异常说明见 [RESULTS.md](RESULTS.md)，纯汇总数字见
[key_results.csv](key_results.csv)。没有包含新实验的逐任务轨迹、数据库或模型。
