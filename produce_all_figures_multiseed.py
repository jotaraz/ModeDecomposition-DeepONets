"""Seed-averaged versions of the paper's figures.

Same as produce_all_figures.py, but every point/curve is the ARITHMETIC MEAN
over the ten seeds v0..v9 trained by the sweep (see SEED_SWEEP.md), and the
output goes to figures/{pdfs,pngs}/Fig*_multiseed.{pdf,png} so the originals are
never overwritten.

Excluded:
  Figure 6  -- its synthetic datasets (synthv14/15/18/19) do not exist anywhere,
               so those four nets were not part of the sweep.
  Figure 12 -- uses no DeepONets at all, only the raw synthv11 data, so there is
               nothing to average.

Every *_multiseed analysis script has a SHOW_BAND flag at the top (default
False); set it to True to shade the min..max spread across seeds behind each
mean. Missing seeds are skipped rather than fatal, so this runs on a partial
sweep too -- each script prints how many seeds it actually used.
"""

import os

commands = {
    "1_top_right":    "python -m src.analysis.RELEVANT.analyze_whichTs_ga_multiseed",
    "1_bottom_right": "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses_ga_multiseed",
    "2":  "python -m src.analysis.RELEVANT.analyze_whichTs_newlayout2_multiseed 3 40 -1 d5_w100",
    "3":  "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2_multiseed 3 1 1 2",
    "4":  "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate_multiseed 3 4 1 32",
    "5":  "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2_multiseed 3 0 1 2",
    # "6": excluded -- no synthv datasets, see module docstring
    "7":  "python -m src.analysis.RELEVANT.investigate_branch_sb_scale2_multiseed 2 1",
    "8":  "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate2_multiseed 3 0 2 32",
    "9a": "python -m src.analysis.RELEVANT.show_components_mult_multsizes_multiseed 3 -1 4000 1",
    # neptag1=10000 reuses fig 9a's w50 net; see SEED_SWEEP.md
    "9b": "python -m src.analysis.RELEVANT.show_components_2x2_multiseed 3 0 3 10000 10000",
    "10": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate_multiseed 3 5 1 32",
    "11": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate_multiseed 3 11 2 32",
    # "12": excluded -- no DeepONets involved
}


#take_keys = ["2"]
take_keys = commands.keys()

for k, v in commands.items():
    if k in take_keys:
        print(k, ":", v)
        os.system(v)
