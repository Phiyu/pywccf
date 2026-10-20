"""Minimal logging helpers (replacement of ``pyhipp.core.abc.HasLog``)."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

__all__ = ["Log", "HasLog"]


class Log:
    """Print ``[name] message`` when ``verbose`` is true.

    >>> log = Log('WCCF3D')
    >>> log('N_cells = 10')
    [WCCF3D] N_cells = 10
    >>> with log.timer('pair counting'):
    ...     pass
    """

    def __init__(self, name: str = "", verbose: bool = True, stream=None):
        self.name = str(name)
        self.verbose = bool(verbose)
        self.stream = stream

    def __call__(self, *msgs, **print_kw):
        if not self.verbose:
            return
        prefix = f"[{self.name}] " if self.name else ""
        print(prefix + " ".join(str(m) for m in msgs),
              file=self.stream or sys.stdout, flush=True, **print_kw)

    @contextmanager
    def timer(self, what: str):
        t0 = time.perf_counter()
        yield
        self(f"{what}: {time.perf_counter() - t0:.2f} s")


class HasLog:
    """Mixin giving an object a :class:`Log` as ``self.log``."""

    def __init__(self, verbose: bool = True, name: str | None = None):
        self.log = Log(name or type(self).__name__, verbose)

    @property
    def verbose(self) -> bool:
        return self.log.verbose

    @verbose.setter
    def verbose(self, value: bool):
        self.log.verbose = bool(value)
