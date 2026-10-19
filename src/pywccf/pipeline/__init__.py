"""YAML-driven measurement pipeline.

Command line (installed as ``pywccf-run``)::

    pywccf-run configs/fiducial.yaml --dry-run
    pywccf-run configs/fiducial.yaml

is equivalent to the former ``python code/run.py config.yaml``.
"""
from .runner import (HaloWeb1DCross, build_run_plan, load_config, main,
                     print_config, run)

__all__ = ["HaloWeb1DCross", "build_run_plan", "load_config", "main",
           "print_config", "run"]
