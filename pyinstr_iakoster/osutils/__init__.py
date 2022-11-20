from platform import system

match system():
    case "Windows":
        from ._win import (
            hide_path,
            unhide_path,
            is_hidden_path,
        )
    case _ as os_name:
        raise SystemError("unsupported platform: %r" % os_name)


__all__ = [
    "hide_path",
    "unhide_path",
    "is_hidden_path",
]
