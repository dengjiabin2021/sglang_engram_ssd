"""通过 SGLang 通用插件入口登记 Engram SSD hooks。"""

from __future__ import annotations


def register() -> None:
    """在官方插件加载上下文中登记当前开关需要的 Hook。

    输入：无显式参数；仅通过 config 的轻量门控判断 SSD 是否明确关闭。
    输出：无返回值；开启时导入 hooks 以登记装饰器，关闭时不导入替换模块或创建 runtime 资源。
    Hook 的实际应用由 SGLang 随后完成，完整配置校验留在模型加载调用链内。
    """
    from .config import should_register_hooks

    # SSD 独立启用；关闭时不登记替换，由 SGLang 原配置选择 HBM 或 DDR。
    # 注册阶段只识别明确关闭值，不在这里解析预算和分块参数。
    # 非法配置留到模型加载 Hook 中报错，避免被 load_plugins 捕获后继续启动。
    if not should_register_hooks():
        return

    # 装饰器在模块首次导入时登记；load_plugins 随后统一 apply_hooks。
    from . import hooks  # noqa: F401
