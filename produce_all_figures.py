import os 

commands = {
    "1_top_right": "python -m src.analysis.RELEVANT.analyze_whichTs_ga",
    "1_bottom_right": "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses_ga",
    "2": "python -m src.analysis.RELEVANT.analyze_whichTs_newlayout2 3 40 -1 d5_w100",
    "3": "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2 3 1 1 2",
    "4": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 4 1 32",
    "5": "python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2 3 0 1 2",
    "6": "python src/analysis/spectral_bias/plot_res3_sidebyside_mat_gridspec.py 0.2",
    "7": "python -m src.analysis.RELEVANT.investigate_branch_sb_scale2 2 1",
    "8": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate2 3 0 2 32",
    "9a": "python -m src.analysis.RELEVANT.show_components_mult_multsizes 3 -1 4000 1",
    "9b": "python -m src.analysis.RELEVANT.show_components_2x2 3 0 3 10000 10000",  # neptag1 10000, not 4000: reuses fig 9a's w50 net, see SEED_SWEEP.md
    "10": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 5 1 32",
    "11": "python -m src.analysis.RELEVANT.analyze_mode_losses_rotate 3 11 2 32",
    "12": "python src/analysis/RELEVANT/synth_freq_comp_fromFILES.py"
}



#take_keys = ["2"]
take_keys = commands.keys()

for k,v in commands.items():
    if k in take_keys: 
        print(k, ":", v)
        os.system(v)




