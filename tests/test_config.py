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
