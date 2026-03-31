"""
Script to list all net directory names required for reproducing all figures.
Run from repo root: python list_needed_nets.py
"""

all_nets = []

# ===================================================================
# bid=3 (kdvnx401_dt0.0001_nc5_m5000_1999)
# ===================================================================

# From analyze_whichTs_ga.py (bid=3, lrtag=40, LLW=-1, dw="d5_w100", Nep5000, whichT=-1)
for llw in range(2, 100, 8):  # [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98]
    all_nets.append(
        f"whichT-1_doStackedFalse_doSigma0_siscFirst_aT0.0_aB0.0_exp0.0_Nep5000_d5_w100_llw{llw}_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam40_v0"
    )

# From plot_gd_or_adam_modelosses_ga.py (hardcoded single net)
all_nets.append(
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w100_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam40_v0"
)

# From analyze_whichTs_newlayout2.py (bid=3, lrtag=40, LLW=-1, dw="d5_w100")
# whichTids = [-1, 0, 1, 2, 7]
# All use doSigma0_siscFirst regardless of whichT (see line 376 of script)
for wht in [-1, 0, 1, 2, 7]:
    for llw in range(2, 100, 8):
        all_nets.append(
            f"whichT{wht}_doStackedFalse_doSigma0_siscFirst_aT0.0_aB0.0_exp0.0_Nep5000_d5_w100_llw{llw}_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam40_v0"
        )

# From plot_gd_or_adam_modelosses2.py call 1 (bid=3, whichones=1=SGD, sizeid=1, wun=335)
for nep in [10000, 4000]:
    all_nets.append(
        f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
    )

# From plot_gd_or_adam_modelosses2.py call 2 (bid=3, whichones=0=Adam, sizeid=1, wun=335)
for nep in [10000, 4000]:
    all_nets.append(
        f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0"
    )

# From analyze_mode_losses_rotate.py call 1 (bid=3, whichones=4, sizeid=1, lrtag=32): wun=335
for exp in ["-1.0", "-0.5", "0.5", "1.0"]:
    for nep in [4000]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp{exp}_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
        )
for nep in [10000, 4000]:
    all_nets.append(
        f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
    )

# From analyze_mode_losses_rotate.py call 2 (bid=3, whichones=5, sizeid=1, lrtag=32): wun=335
for exp in ["-1.0", "-0.5", "0.5", "1.0"]:
    for nep in [4000]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp{exp}_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0"
        )
for nep in [10000, 4000]:
    all_nets.append(
        f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0"
    )

# From analyze_mode_losses_rotate.py call 3 (bid=3, whichones=11, sizeid=2, lrtag=32): wun=335
for nep in [4000, 5000, 10000]:
    for lr in ["SGD", "Adam"]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lr{lr}32_v0"
        )

# From analyze_mode_losses_rotate2.py (bid=3, whichones=0, sizeid=2, lrtag=32): wst=42, wun=495
all_nets.append(
    "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w42_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0"
)
all_nets.append(
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w495_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0"
)

# From show_components_mult_multsizes.py (bid=3, size_id=-1, neptag="4000")
# wun_s = [50, 100, 220, 335, 495]
for w in [50, 100, 220, 335, 495]:
    for nep in [4000, 10000]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w{w}_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
        )

# From show_components_2x2.py (bid=3, sizeid1=0, sizeid2=3, neptag1="4000", neptag2="10000")
# w1=wun_s[0]=50, w2=wun_s[3]=335
all_nets.append(
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w50_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
)
all_nets.append(
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"
)

# ===================================================================
# investigate_branch_sb_scale2.py (bids=[0,3,4,6], sizeid=2)
# bid=0 (advdiffnx201_dt0.0005_nc20_m1000, endtag=1000): wun=wun_s[2]=332, llw=20
# bid=3 (kdvnx401_dt0.0001_nc5_m5000, endtag=1999): wun=wun_s[2]=335, llw=50
# bid=4 (kdvnx401_dt0.0001_nc5_m5000, endtag=5999): wun=wun_s[2]=335, llw=50
# bid=6 (burgers_dt0.0001_nc10_m3800, endtag=100): wun=wun_s[2]=337, llw=50
# ===================================================================
# bid=0
for nep in [4000, 10000]:
    for lr in ["SGD", "Adam"]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w332_llw20_batadvdiffnx201_dt0.0005_nc20_m1000_1000_numd1000_lr{lr}32_v0"
        )

# bid=3
for nep in [4000, 10000]:
    for lr in ["SGD", "Adam"]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lr{lr}32_v0"
        )

# bid=4
for nep in [4000, 10000]:
    for lr in ["SGD", "Adam"]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_5999_numd1000_lr{lr}32_v0"
        )

# bid=6
for nep in [4000, 10000]:
    for lr in ["SGD", "Adam"]:
        all_nets.append(
            f"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep{nep}_d5_w337_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lr{lr}32_v0"
        )

# ===================================================================
# Spectral bias (synthv data)
# plot_res3_sidebyside_mat_gridspec.py (alpha=0.2, F0=0.2, dw="d5_w50", llw=5)
# sigmascale_to_vid:
#   "ss-0.0001": trunc=12, flip=13, Nep=5000
#   "ss-0.01":   trunc=14, flip=15, Nep=2000
#   "ss-0.1":    trunc=16, flip=17, Nep=2000
#   "ss-0.5":    trunc=18, flip=19, Nep=2000
# (only ss-0.01 and ss-0.5 are looped over in the script for irow in [-0.01, -0.5])
# ===================================================================
sigmascale_to_vid = {
    "ss-0.0001": {"trunc": 12, "flip": 13, "Nep": 5000},
    "ss-0.01":   {"trunc": 14, "flip": 15, "Nep": 2000},
    "ss-0.1":    {"trunc": 16, "flip": 17, "Nep": 2000},
    "ss-0.5":    {"trunc": 18, "flip": 19, "Nep": 2000},
}
alpha = 0.2
F0 = 0.2
dw = "d5_w50"
llw = 5
sharedbase = "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep"

# The script loops over [-0.01, -0.5]
for sigmascale_num in [-0.01, -0.5]:
    sigmascale = "ss" + str(sigmascale_num)
    truncated_id = sigmascale_to_vid[sigmascale]["trunc"]
    flipped_id   = sigmascale_to_vid[sigmascale]["flip"]
    nepochs      = sigmascale_to_vid[sigmascale]["Nep"]
    all_nets.append(
        sharedbase + str(nepochs) + "_" + dw + "_llw" + str(llw) + "_batsynthv" + str(truncated_id) +
        "_n100_N5_m5000_fs" + str(alpha) + sigmascale + "ns" + str(F0) + "_numd5000_lrAdam40_v0"
    )
    all_nets.append(
        sharedbase + str(nepochs) + "_" + dw + "_llw" + str(llw) + "_batsynthv" + str(flipped_id) +
        "_n100_N5_m5000_fs" + str(alpha) + sigmascale + "ns" + str(F0) + "_numd5000_lrAdam40_v0"
    )

# Deduplicate and sort
unique_nets = sorted(set(all_nets))

print(f"Total unique nets needed: {len(unique_nets)}")
print()
for net in unique_nets:
    print(net)
