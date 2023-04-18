from platform import system

match system():
    case "Windows":
        from ._win import (
            hide_path,
            unhide_path,
            is_hidden_path,
        )
    case _:
        from ._linux import (
            hide_path,
            unhide_path,
            is_hidden_path,
        )


__all__ = [
    "hide_path",
    "unhide_path",
    "is_hidden_path",
]
