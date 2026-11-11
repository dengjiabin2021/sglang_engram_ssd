"""验证固定读取路径的配置校验与退役参数提示。"""

from __future__ import annotations

from typing import Iterator

import pytest

from sglang_engram_ssd.config import get_config, should_register_hooks


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


@pytest.mark.parametrize("suffix", ["POOL_BYTES", "SEGMENT_TOKENS", "POLICY", "TIMEOUT_MS", "LOG_INTERVAL"])
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
    ("POOL_BYTES", "0"), ("POOL_BYTES", "-1"), ("SEGMENT_TOKENS", "-1"),
    ("POLICY", "invalid"), ("TIMEOUT_MS", "-1"), ("LOG_INTERVAL", "-1"),
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


@pytest.mark.parametrize("policy", ["BEST_EFFORT", "WAIT_COMPLETE"])
def test_non_timeout_policy_ignores_timeout_option(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, policy: str
) -> None:
    """A1：策略大小写归一，非timeout时不解析显式非法TIMEOUT_MS。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POLICY", policy)
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_TIMEOUT_MS", "invalid")
    config = get_config()
    assert config.policy == policy.lower()
    assert config.timeout_ms == 500
    assert "POLICY is not timeout" in caplog.text


@pytest.mark.parametrize("suffix", ["DEDUP", "SCHED_PREFETCH"])
def test_invalid_new_boolean(monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    """A1：新增布尔值同样严格校验，不在登记阶段吞掉错误。"""
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_" + suffix, "invalid")
    with pytest.raises(ValueError, match="must be a boolean"):
        get_config()
