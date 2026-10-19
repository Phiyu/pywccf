"""Data containers, copied from ``pyccb.samples`` (CosmicCoreBox).

``field.py`` and ``halo.py`` are kept identical to the CosmicCoreBox versions
so that results are reproducible; ``catalog.py`` holds the ELUCID-specific
catalogue construction used in Section 2 of the paper.
"""
from . import field, halo, catalog
from .field import TidalField, FieldSample
from .halo import HaloSample
from .catalog import (build_elucid_halo_catalog, join_tree_properties,
                      load_halo_sample, count_mass_bins)

__all__ = ["TidalField", "FieldSample", "HaloSample",
           "build_elucid_halo_catalog", "join_tree_properties",
           "load_halo_sample", "count_mass_bins"]
