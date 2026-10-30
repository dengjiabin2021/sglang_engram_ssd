"""通过 SGLang 通用插件入口登记 Engram SSD hooks。"""


def register() -> None:
    from .config import should_register_hooks

    # SSD 独立启用；关闭时不登记替换，由 SGLang 原配置选择 HBM 或 DDR。
    # 注册阶段只识别明确关闭值，不在这里解析预算和分块参数。
    # 非法配置留到模型加载 Hook 中报错，避免被 load_plugins 捕获后继续启动。
    if not should_register_hooks():
        return

    # 装饰器在模块首次导入时登记；load_plugins 随后统一 apply_hooks。
    from . import hooks  # noqa: F401
