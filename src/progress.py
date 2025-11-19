"""通用进度条工具，供各命令行脚本复用。"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class ProgressBar:
    total: int | None = None
    prefix: str = ""
    width: int = 30
    _active: bool = False

    def update(self, current: int, extra: str = "") -> None:
        self._active = True
        total = self.total if self.total and self.total > 0 else None
        width = max(10, self.width)

        if total:
            percent = min(max(current / total, 0.0), 1.0)
            filled = int(width * percent)
            bar = "#" * filled + "-" * (width - filled)
            base = f"{self.prefix} [{bar}] {percent*100:5.1f}% {current}/{total}"
        else:
            bar = "#" * (current % (width + 1))
            base = f"{self.prefix} [{bar:<{width}}] {current}"

        if extra:
            base = f"{base} {extra}"
        sys.stdout.write("\r" + base)
        sys.stdout.flush()

    def finish(self) -> None:
        if self._active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._active = False
