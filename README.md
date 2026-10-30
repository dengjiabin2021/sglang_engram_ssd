# SGLang Engram SSD 插件

将研究文档第三章的同步 eager 方案实现为独立插件。基于本仓库 SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`；所有新增文件都在本目录。

本地 SGLang 的官方装饰器名为 `plugin_hook`，插件通过 `plugin_hook as register_hook` 导入，因此代码中的 `@register_hook(...)` 使用的是官方 HookRegistry。安装后由 `sglang.srt.plugins` entry point 自动发现，每个 TP worker 各登记和应用一遍。

## 安装与启动

在已经安装本地 SGLang 及其 CUDA 依赖的 Python 环境中执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./sglang-engram-ssd
```

启动示例，模型路径指向现有的本地 SSD checkpoint：

```bash
SGLANG_PLUGINS=engram_ssd \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared \
SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 \
SGLANG_DSV41_ENGRAM_SSD_CACHE_BYTES=536870912 \
SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS=4096 \
python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --language-model-only \
  --device cuda \
  --tp-size 4 \
  --load-format safetensors \
  --startup-weight-load-mode serial \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000
```

`SGLANG_PLUGINS` 是可选的插件白名单；需要同时启用其他插件时按 SGLang 规则追加名称。TP=4 仅为示例，插件没有固定 TP 数量。首版范围为单节点 NVIDIA CUDA、普通 TP、纯文本、非投机 prefill/decode；不支持 Attention DP/CP、PP、PD 分离和 CUDA graph。保留原 HTTP 接口及 CPU/GPU overlap 调度。

使用包含视觉配置的 V4.1 checkpoint 时，必须添加 `--language-model-only`，跳过视觉模块并使用文本启动预热。否则默认图片预热会触发插件的 `Engram SSD currently supports text-only requests` 检查。

`SGLANG_ENABLE_DSV41_ENGRAM_SSD` 默认为 0，关闭时插件不登记 hook。新增环境变量由插件自己的 `config.py` 读取，无需修改 SGLang `environ.py`。不要启用整 checkpoint prefetch、startup overlap、IPC weight cache、custom loader 或 `weight-loader-disable-mmap`；插件会在模型加载前报告不支持的配置。

插件入口只识别明确关闭值，其余情况先登记 Hook。完整配置校验在模型加载 Hook 的开头执行；非法开关、非整数预算、负预算或非正的 `CHUNK_ROWS` 会在原模型加载函数调用前抛错，阻止模型加载。配置解析错误不再发生于会被 SGLang 捕获并继续的插件注册阶段。

## 代码组织与 Hook 接入

| 文件 | 作用 |
| --- | --- |
| `src/sglang_engram_ssd/__init__.py` | 通用插件入口，启用后导入并登记 hooks。 |
| `config.py` | SSD 开关、每 rank 缓存字节预算、K 查询工作区容量。 |
| `catalog.py` | 原 index/header 解析、集中形状和编码校验、原始文件位置。 |
| `runtime.py` | 每 rank 一份跨层 LRU、原文件行读取、有限 pinned/HBM 工作区。 |
| `weight_loader.py` | 在 `get_tensor()` 前精确跳过 Engram 两张大表。 |
| `hooks.py` | 七个 `@register_hook(..., type=HookType.AROUND)` 接入点。 |

| SGLang Hook 目标 | 接入动作 |
| --- | --- |
| `load_model_utils.load_model_with_memory_saver` | 在原模型构造和启动预取前验证配置、准备 catalog。 |
| `loader.get_model_loader` | 核对实际返回的 loader，防止量化配置选择其他加载路径。 |
| `DefaultModelLoader._get_weights_iterator` | 保留文件发现，改用提前过滤的选择性 iterator。 |
| `DeepseekV4ForCausalLM.load_weights` | 在原 dispatch 外补充 canonical 名称的外置参数过滤。 |
| `EngramEmbedding.__init__` | 创建空参数占位和共享 runtime，跳过完整 DDR 表分配。 |
| `EngramEmbedding.forward` | 按 layer_id 调用 SSD lookup，返回原形状的 BF16 embedding。 |
| `EngramEmbedding.finish_load` | SSD 模式跳过完整 host 表的完成 barrier。 |

HookRegistry 会传播已导入的函数别名，因此 `ModelRunner` 之前导入的 `load_model_with_memory_saver` 也会指向 wrapper。在该入口 Hook 已成功应用的前提下，插件会检查其余必要 Hook，拒绝部分安装的组合；入口 Hook 自身未安装或插件模块导入失败仍需查看 SGLang 插件加载日志。Hasher、history、wkv、gate 和 gather kernel 使用 SGLang 原实现。

## 数据路径与容量

启动解析原 `model.safetensors.index.json` 和相关 shard header，只将配置启用的 `model.layers.<id>.engram.embed.weight/scale` 加入目录。校验在 `build_engram_ssd_catalog()` 完成；catalog 保存 `layer_ids`，runtime 构造时循环 `init_table(layer_id)` 绑定句柄。`open_table(layer_id)` 在层号不存在时抛出带层号的 `KeyError`。

首版要求原生 `F8_E4M3` payload `[N,D]` 和 `F8_E8M0` scale `[N,D/32]`。U8 scale 不自动当作 E8M0 原始指数。weight 与 scale 可以位于不同 shard；每个 shard 的数据区起点分别是 `8 + header_len`，两张表各自保存绝对偏移。不会导出新表、转换权重或生成额外大文件。

一次 lookup 按原查询顺序处理 GPU `indices[T,C]`：同步 D2H → 每块最多 K 次查询 → 各行查 LRU／两段 `pread` → pinned staging → 两次 H2D → 原 gather 反量化 → GPU BF16 `[T,C,D]`。不排序、不去重、不做 Engram lookup TP all-reduce。`device_ids=[0,K)` 仅是当前 GPU 工作区槽号；全局 row_id 只用于原文件读取。

设一行原始数据为 `R=D+D/32` 字节：每 rank 缓存 payload 预算为 B，pinned 工作区为 `K*R`，HBM 工作区为 `K*(R+8)`，每块 q 次查询的 H2D 为 `q*R`。Python 对象、CPU 查询列表和完整 BF16 输出另计。`self.cached_bytes` 记录当前已缓存字节；预算为零时仍保留有限 staging，按需读取但不保留 LRU 行数据。

各 rank 独立维护目录、LRU、fd 和 GPU 工作区。同一节点读取相同文件可能复用内核 page cache；应用层预算不限制全节点 page cache。同进程锁只保护跨层共享工作区的并发线程，不协调各 rank 的 I/O。块末 CUDA event 完成后才复用 staging；当前实现没有后台预取或传输重叠。

## 验证状态

本次仅按本地源码检查 Hook 接口、调用链和 Python 语法，未安装插件、导入运行 SGLang、执行测试、模型推理或性能测试。实际 checkpoint 编码、CUDA 数值一致性和吞吐收益仍需在目标 NVIDIA 环境验证。
