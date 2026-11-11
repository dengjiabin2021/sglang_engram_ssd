"""在独立 Python 进程核验关闭开关时不登记新增 Hook 或加载新运行期模块。"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("ssd,sched", [("0", "0"), ("1", "0"), ("1", "1")])
def test_feature_gated_registration(ssd: str, sched: str) -> None:
    """A1：独立进程避免已导入模块缓存影响，关闭时没有新增 Hook／运行期资源。"""
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "SGLANG_DSV41_ENGRAM_SSD")):
            del environment[name]
    environment["SGLANG_ENABLE_DSV41_ENGRAM_SSD"] = ssd
    environment["SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH"] = sched
    script = r"""
import sys
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang_engram_ssd import register
register()
if SSD == "0":
    assert not any("sglang_engram_ssd.hooks" in name for name in sys.modules)
else:
    from sglang_engram_ssd.hooks._registry import HOOK_TARGETS, SCHED_PREFETCH_HOOK_TARGETS
    registered = set(HookRegistry._hooks)
    assert set(HOOK_TARGETS) <= registered
    if SCHED == "0":
        assert not (set(SCHED_PREFETCH_HOOK_TARGETS) & registered)
        assert "sglang_engram_ssd.hashing" not in sys.modules
        assert "sglang_engram_ssd.sched_prefetch" not in sys.modules
    else:
        assert set(SCHED_PREFETCH_HOOK_TARGETS) <= registered
assert "sglang_engram_ssd.query._C" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", f"SSD={ssd!r}\nSCHED={sched!r}\n" + script],
        env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
