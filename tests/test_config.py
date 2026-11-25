"""验证固定读取路径的配置校验与退役参数提示。"""

from __future__ import annotations

from typing import Iterator

import pytest

from sglang_engram_ssd.config import get_config, should_register_hooks

_PREFETCH = "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for name in list(os.environ):
        if name.startswith(("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "SGLANG_DSV41_ENGRAM_SSD")):
            monkeypatch.delenv(name)
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_disabled_registration() -> None:
    """A1：默认不登记插件，配置查询也不创建运行期资源。"""
    assert not should_register_hooks()
    with pytest.raises(RuntimeError, match="requires Engram SSD"):
        get_config()


@pytest.mark.parametrize("value", ["false", "0", "no", "off-invalid"])
def test_registration_defers_validation(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A1：登记只识别明确关闭，非法开关留待模型加载入口拒绝。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", value)
    if value == "off-invalid":
        assert should_register_hooks()
        with pytest.raises(ValueError, match="must be a boolean"):
            get_config()
    else:
        assert not should_register_hooks()


@pytest.mark.parametrize("suffix", ["IO_URING_QUEUE_DEPTH", "CHUNK_ROWS"])
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_positive_reading_configuration(
    monkeypatch: pytest.MonkeyPatch, suffix: str, value: str
) -> None:
    """A0：非法读取容量在任何模型／I/O 初始化之前被拒绝。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv(f"SGLANG_DSV41_ENGRAM_SSD_{suffix}", value)
    with pytest.raises(ValueError):
        get_config()


def test_retired_configuration_is_ignored_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A0：退役配置只提示一次，非法旧值也不会重启旧实现。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    name = "SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND"
    monkeypatch.setenv(name, "invalid-retired-value")
    first = get_config()
    assert get_config() is first
    assert len([record for record in caplog.records if name in record.getMessage()]) == 1


@pytest.mark.parametrize(
    "suffix", ["WINDOW_BATCHES", "POOL_BATCHES", "POOL_BYTES", "LOG_INTERVAL"],
)
def test_disabled_scheduler_options_are_not_parsed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, suffix: str
) -> None:
    """A1：未启用调度预取时，非法从属配置只警告，不分配资源或阻止同步路径。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    name = "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_" + suffix
    monkeypatch.setenv(name, "invalid-value")
    config = get_config()
    assert not config.sched_prefetch_enabled
    assert name in caplog.text


@pytest.mark.parametrize("name,value", [
    ("POOL_BYTES", "0"), ("POOL_BYTES", "-1"), ("POOL_BATCHES", "2"), ("POOL_BATCHES", "-1"),
    ("WINDOW_BATCHES", "0"), ("LOG_INTERVAL", "-1"),
])
def test_invalid_scheduler_options_fail_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """A1：启用功能后，所属非法配置直接报错。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_" + name, value)
    with pytest.raises(ValueError):
        get_config()


@pytest.mark.parametrize("sched_enabled", ["0", "1"])
@pytest.mark.parametrize("suffix", ["SEGMENT_TOKENS", "POLICY", "TIMEOUT_MS", "MAX_DEFER_MS"])
def test_retired_scheduler_options_warn_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, sched_enabled: str, suffix: str,
) -> None:
    """完成标准 4：旧调度配置值直接忽略，无论是否开启预取都只提示一次。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", sched_enabled)
    name = _PREFETCH + suffix
    monkeypatch.setenv(name, "invalid")
    assert get_config() is get_config()
    assert [record.getMessage() for record in caplog.records] == [f"Ignoring retired Engram SSD configuration {name}"]


@pytest.mark.parametrize("window,batches,expected", [(None, None, 3), ("3", None, 4), ("3", "6", 6)])
def test_pool_size_in_batches(
    monkeypatch: pytest.MonkeyPatch, window: str | None, batches: str | None, expected: int,
) -> None:
    """完成标准 4：默认留一批容量，也可显式按批数定容，并且整行取整。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    if window is not None:
        monkeypatch.setenv(_PREFETCH + "WINDOW_BATCHES", window)
    if batches is not None:
        monkeypatch.setenv(_PREFETCH + "POOL_BATCHES", batches)
    config = get_config()
    assert config.window_batches == (2 if window is None else int(window))
    assert config.resolve_pool_bytes(tokens=31, layers=2, columns=4, dim=256, tp_size=8) == expected * 35 * 264


@pytest.mark.parametrize("slots,extra,valid", [(70, 0, False), (70, 263, False), (71, 0, True)])
def test_pool_bytes_must_fit_more_rows_than_window(
    monkeypatch: pytest.MonkeyPatch, slots: int, extra: int, valid: bool,
) -> None:
    """完成标准 4：按实际整行容量校验，等于窗口或小于一行的余量均拒绝。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    size = slots * 264 + extra
    monkeypatch.setenv(_PREFETCH + "POOL_BYTES", str(size))
    config = get_config()
    args = dict(tokens=31, layers=2, columns=4, dim=256, tp_size=8)
    if valid:
        assert config.resolve_pool_bytes(**args) == size
    else:
        with pytest.raises(ValueError, match="POOL_BYTES must hold more rows"):
            config.resolve_pool_bytes(**args)


def test_pool_size_options_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """完成标准 4：同时设置两种行池尺寸报错。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    monkeypatch.setenv(_PREFETCH + "POOL_BATCHES", "4")
    monkeypatch.setenv(_PREFETCH + "POOL_BYTES", "1000000")
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_config()


@pytest.mark.parametrize("suffix", ["DEDUP", "SCHED_PREFETCH"])
def test_invalid_new_boolean(monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    """A1：新增布尔值同样严格校验，不在登记阶段吞掉错误。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_" + suffix, "invalid")
    with pytest.raises(ValueError, match="must be a boolean"):
        get_config()


@pytest.mark.parametrize("raw, expected", [(None, False), ("1", True), ("0", False)])
def test_debug_rows_config(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool) -> None:
    """完成标准 7：随调度预取解析逐位核对开关，默认关闭。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    if raw is not None:
        monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS", raw)
    assert get_config().debug_check_rows is expected
