# 2026-10-10 AppWorld 复测结果

以下均使用同一份官方 dev57，主指标 TGC，另报 SGC。表格为百分数；例如
14.0% 对应比例约 0.140。TGC 是任务整体成功率，不是任务内断言的部分得分。
SGC 要求同一场景下的 3 个任务全部成功。使用官方 evaluator 的舍入结果，
例如 45/57 官方显示为 79.0%。差值从成功任务数计算。

| 模型 | Baseline 成功数 | Harness 成功数 | TGC B / H (%) | SGC B / H (%) | TGC 差值 (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-32B-Instruct | 8/57 | 10/57 | 14.0 / 17.5 | 10.5 / 10.5 | +3.51 |
| Qwen3.5-27B | 41/57 | 40/57 | 71.9 / 70.2 | 52.6 / 47.4 | -1.75 |
| Qwen3.6-27B | 46/57 | 45/57 | 80.7 / 79.0 | 63.2 / 63.2 | -1.75 |
| Qwen3-4B-Instruct-2507 † | 0/57 | 3/57 | 0.0 / 5.3 | 0.0 / 0.0 | +5.26 |
| Qwen3-30B-A3B-Instruct-2507 | 8/57 | 10/57 | 14.0 / 17.5 | 10.5 / 15.8 | +3.51 |
| Qwen3.5-35B-A3B | 34/57 | 38/57 | 59.7 / 66.7 | 31.6 / 36.8 | +7.02 |
| Llama-3.3-70B-Instruct | 8/57 | 9/57 | 14.0 / 15.8 | 10.5 / 15.8 | +1.75 |
| Llama-xLAM-2-70b-fc-r | 2/57 | 5/57 | 3.5 / 8.8 | 0.0 / 5.3 | +5.26 |

这是每组一次 temperature=1 的采样实验，没有多 seed 置信区间。结果不证明
Life-Harness 在所有模型或重复运行上稳定提升。

## 必须保留的异常与协议差异

- **† Qwen3-4B-Instruct-2507 Baseline 有人工限时干预。** `df61dc5_2` 生成数小时后
  按用户要求终止并计失败；最后 3 条分别设置 600 秒任务墙钟时间限制。
  这不是完全不受限的原协议结果，不能当作严格配对的无干预提升。
  本目录的通用入口不默认重现这次临时干预。
- Qwen2.5 的 context=32768；Harness 有 3 个上下文超限 HTTP 400，全部保留在分母。
  Baseline 接口错误为 0。
- Qwen3-30B-A3B-Instruct-2507 Baseline 有 1 个上下文超限，保留在分母。
  该服务实际输入容量约 71874 tokens，而不是模型标称的完整窗口。
- Qwen3.5/3.6-27B 的思考保持开启。运行中把 SDK 默认 600 秒 HTTP timeout 改成
  14400 秒；保留 203 条已完成轨迹，仅重启 25 条未完成尝试。最终两模型四组接口错误均为 0。
  三个服务最大输入 108528，Qwen3.6 Baseline 为 109074；服务 seed 不同，请求 seed 均为 100。
- Llama 3.3 使用并行历史模板适配。原模型目录 6 个分片不完整且只读，补齐到临时目录后
  全部 30 个分片通过来源 SHA256 校验才开始测试。临时目录和模型文件不提交。
  两组最终模型接口错误均为 0。
- xLAM 使用 SGLang 自定义 JSON-array parser，未改原生模板。先完成 train 上
  5 条 × 2 组检查，262 次模型请求、220 次工具调用无接口或格式错误；另验证双工具并行回复。
  正式两组接口错误均为 0；新增成功任务 3 条，无退步任务。

## 与上游报告的区别

上游 `ITERATION_REPORT.md` 的 0.133→0.211 是 Qwen3-4B **train90** 的
12/90→19/90；它的 dev 是 5/57→9/57。Qwen3-8B dev 的 0.158→0.175
对应 9/57→10/57。Qwen2.5-32B-Instruct 的 0.151→0.204 对应
8/53→11/54，历史缺失任务的原因没有记录。

我们统一使用 dev57，且 Qwen3-4B-Instruct-2507 不等于上游的 Qwen3-4B thinking
版本。历史本地 Jsonnet、原始依赖锁与确切 prompt 配置未完整提交，所以当前配置是根据
上游标准 function-calling instructions/demos 和 Life-Harness runner 重建的；服务后端为
SGLang，而原报告使用 vLLM。不能据此承诺逐分复现历史数字。

## 已知策略问题，没有在本分支修分

`src/life_harness_appworld/agent.py` 的 `_is_answer_required` 对部分
“Give me ...” 查询任务判断为不需要答案，`_strip_answer_if_action_only` 会删除
`complete_task` 的 answer。它在两个 27B 模型的轨迹中影响了
`50e1ac9_1/2/3`。这里只记录原因，仍使用原版 v009_r4，不把事后修正当成实测分数。

## 固定源版本与环境

- Life-Harness 实验基线：`22c299b83d888405d34d2c7efc0fcf4d2ccfdcef`。
- AppWorld：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`，加本目录的 timeout-only 补丁。
- AppWorld Python 3.12.3，OpenAI SDK 1.99.8、httpx 0.28.1、jsonnet 0.22.0；
  完整约束见 `constraints-appworld-py312.txt`。
- SGLang 环境：0.5.12 / torch 2.11.0 / transformers 5.6.0，BF16，triton attention。
- 表格从本机官方 `*_metrics.json` 提取并核对；只提交汇总数值和说明，不提交原始指标产物或轨迹。
