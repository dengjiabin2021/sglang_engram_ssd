# SGLang Engram SSD 插件

基于官方 SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`，通过 HookRegistry 接入，所有修改均位于插件仓库。

## 固定读取路径

每个 TP rank 负责 `[N*r//P,N*(r+1)//P)` 的全局行区间。模型到达 Engram 层后，同步 io_uring 读取所属行，单套 pinned／GPU 工作区串行分块完成 H2D 和原生 `engram_gather`，再将局部 BF16 结果放回完整零张量。模型线程在原 Engram 调用点执行一次 TP all-reduce；空 owner 也参加，TP=1 不通信。

原 checkpoint 保持不变，weight／scale 的定位仅由 reader 解释。插件不分配完整 host table，不依赖原生 HOST_TABLE 布局选择。

## 构建与安装

先在目标 Linux NVIDIA 环境按 [原生读取模块](src/sglang_engram_ssd/query/README.md) 构建，再执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./sglang-engram-ssd
```

原生扩展、liburing 和可初始化的 io_uring 是必需依赖，失败直接报错，不自动回退。

## 配置

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD` | `0` | 总开关；关闭时不登记插件 Hook |
| `SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS` | `4096` | 单套同步中转区的行容量，必须为正 |
| `SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH` | `32` | 在途文件请求数，必须为正；weight、scale 各占一个请求 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP` | `0` | 对整次本 rank 查询去重，读取唯一行后恢复重复位置 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH` | `0` | 调度预取总开关；当前阶段先捕获只读 CPU hash 参数 |

去重在同步内部读取入口完成，保留原顺序、形状和 BF16 数值；关闭时不分配 unique／inverse 张量。

CPU 行号计算直接调用官方 `compute_engram_hash_ids`，参数从模型已构造的 Hasher 复制，不重建 tokenizer，也不提前更新请求 history。其 Hook 仅随调度预取总开关登记。

旧版读取选择、查询模式、按层预取、流水和 reader 缓存配置均已退役；显式设置只在启动时警告一次并忽略。配置只由 `config.py` 解析，非法值在原模型加载前拒绝。

## 启动

```bash
SGLANG_PLUGINS=engram_ssd \
SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 \
python -m sglang.launch_server \
  --model-path /model \
  --trust-remote-code \
  --language-model-only \
  --device cuda \
  --tp-size 8 \
  --load-format safetensors \
  --startup-weight-load-mode serial \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

当前只支持 NVIDIA CUDA、普通 TP、纯文本、非投机 prefill／decode；不支持 attention DP／CP、PP、PD、CUDA Graph。模型必须为本地 safetensors；禁止整 checkpoint 预取、startup overlap、IPC weight cache、自定义 loader 和禁用 mmap 的加载配置。

## 验证状态

本地 macOS 仅做源码静态检查，未编译、未运行 pytest 或模型。测试位于 `tests/`，原生和 CUDA 测试在目标环境执行；实际 TP8 数值、端到端和性能结果仍待验证。
