"""Helpers: sample selection and synthetic boxes."""
from .selection import tail_masks, mass_match_weights
from .synthetic import (gaussian_random_field, lognormal_delta,
                        poisson_points, synthetic_box)

__all__ = ["tail_masks", "mass_match_weights", "gaussian_random_field",
           "lognormal_delta", "poisson_points", "synthetic_box"]
