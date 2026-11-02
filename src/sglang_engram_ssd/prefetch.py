"""hash 生成后统一提交选中层的 SSD lookup，到各层时取得或等待结果。"""

from __future__ import annotations

from concurrent.futures import Future
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .runtime import EngramSsdRuntime


class EngramPrefetchContext:
    """保存一次 forward 的按层查询任务；任务全部在 hash 生成后提交。"""

    def __init__(
        self, runtime: EngramSsdRuntime, *, layer_ids: tuple[int, ...]
    ) -> None:
        # 所有查询继续使用原 runtime 的缓存、读取后端和 A/B 工作区。
        self.runtime = runtime
        # layer_ids=(1,14) 时，hash_ids[:,0] 属于层 1，hash_ids[:,1] 属于层 14。
        # 此字典只负责“模型层号 → hash_ids 第二维的位置”，不保存 hash 数据。
        self.hash_column_by_layer: dict[int, int] = {
            layer_id: column for column, layer_id in enumerate(layer_ids)
        }
        # 层号 → 后台任务；owner 模式的 Future 只保存本 rank 的零填局部结果。
        # TP all-reduce 由前台 Embedding Hook 在实际到达该层时执行。
        self.layer_futures: dict[int, Future[torch.Tensor]] = {}

    def submit(self, hash_ids: torch.Tensor) -> None:
        """由 Hasher Hook 调用一次，将所有选中层立即放入后台查询队列。"""
        # hash_ids 为 [T,L,C]；取 hash_ids[:,column] 得到某一层的 [T,C] 索引。
        # hash kernel 此时可能尚未执行完；后台先等这个 GPU 事件，再读取索引。
        hash_ready = torch.cuda.Event()
        hash_ready.record(torch.cuda.current_stream(device=self.runtime.device))
        for layer_id in sorted(self.runtime.prefetch_layer_ids):
            column = self.hash_column_by_layer[layer_id]
            # 提交不等待结果；任务参数会持有索引视图和事件，直到该任务结束。
            # runtime 只有一个 worker，各层复用同一套 A/B，按队列顺序执行。
            self.layer_futures[layer_id] = self.runtime.prefetch_executor.submit(
                self._lookup,
                layer_id=layer_id,
                indices=hash_ids[:, column],
                hash_ready=hash_ready,
            )

    def _lookup(
        self, layer_id: int, indices: torch.Tensor, hash_ready: torch.cuda.Event
    ) -> torch.Tensor:
        # 后台以 A 的流等待 hash 并准备索引，不向模型主流插入查询工作。
        # lookup 内部仍按块切换到 A/B 各自的 gather 流，返回前等待两者完成。
        stream = self.runtime.buffers[0].gather_stream
        with torch.cuda.device(self.runtime.device), torch.cuda.stream(stream):
            stream.wait_event(hash_ready)
            # Future 完成表示本 rank 查询已完成；owner 结果由前台再做 TP 合并。
            return self.runtime.lookup(layer_id=layer_id, indices=indices)

    def lookup(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """由 Embedding Hook 调用：本层有预取任务就取结果，否则现场查表。"""
        future = self.layer_futures.get(layer_id)
        if future is None:
            # 不预取仍使用插件 lookup：原读取后端、缓存和块间流水均保持生效。
            # 若后台正占用共享 A/B，这次查询会在 runtime.lock 处等待。
            return self.runtime.lookup(layer_id=layer_id, indices=indices)

        # 已完成时立即返回；未完成时等本层任务，不会再次读取本层数据。
        out = future.result()
        del self.layer_futures[layer_id]
        # 输出由后台流分配，模型主流后续异步执行 wkv 时仍需保持其显存有效。
        out.record_stream(torch.cuda.current_stream(device=self.runtime.device))
        return out

    def finish(self, *, raise_error: bool = True) -> None:
        """forward 退出时收齐所有未消费任务，再释放其结果。"""
        first_error = None
        for future in self.layer_futures.values():
            try:
                future.result()
            except BaseException as error:
                # 某层失败后仍收齐其他层，防止后台继续使用即将释放的资源。
                if first_error is None:
                    first_error = error
        self.layer_futures.clear()
        # 模型自身已抛错时保留原异常，正常退出则传播后台查询错误。
        if first_error is not None and raise_error:
            raise first_error


# Model Hook 设置本次上下文，Hasher/Embedding Hook 通过它找到同一份任务字典。
_current_prefetch: ContextVar[EngramPrefetchContext | None] = ContextVar(
    "engram_ssd_prefetch", default=None,
)


def get_current_prefetch() -> EngramPrefetchContext | None:
    return _current_prefetch.get()


def set_current_prefetch(context: EngramPrefetchContext) -> Token:
    return _current_prefetch.set(context)


def reset_current_prefetch(token: Token) -> None:
    _current_prefetch.reset(token)
