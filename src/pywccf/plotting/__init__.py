"""Figures of the paper.

xi     directional slices / contours of the 3D-WCCF and the appendix tests
ratio  amplitude-ratio maps and AR--mass relations
field  tidal-field slice, z_f--M_h plane, web volume-filling fractions

The function names of ``xi`` keep the numbering of the draft
(``plot_1_main_xi`` ...); descriptive aliases are provided below.
"""
from .xi import (plot_1_main_xi, plot_1_main_xi_by_mass,
                 plot_1_main_xi_by_mass_pairs, plot_1_main_xi_by_subproperty,
                 plot_2_main_contour, plot_a1_ss_impact, plot_a2_ss_evolution,
                 plot_b2_lth_impact, PAPER_RC, style_axis, top_legend)
from .ratio import (plot_ratio_3d_contour, plot_anisotropic_A, plot_ratio,
                    plot_ratio_3d_contour2, plot_ratio2, plot_anisotropic_A2)
from .field import (plot_tidal_field_slice, plot_zf_mass,
                    volume_filling_fractions, plot_volume_fractions)

# descriptive aliases (paper figure names)
plot_main_xi = plot_1_main_xi
plot_main_contour = plot_2_main_contour
plot_ss_impact = plot_a1_ss_impact
plot_ss_evolution = plot_a2_ss_evolution
plot_lth_impact = plot_b2_lth_impact
plot_ar_map = plot_ratio_3d_contour
plot_ar_mass = plot_anisotropic_A
