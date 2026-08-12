source .venv-linux/bin/activate
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
python -m src.analysis.RELEVANT.analyze_mode_losses_rotate2_multiseed 3 0 2 32
MULTISEED_BAND=1 python -m src.analysis.RELEVANT.analyze_mode_losses_rotate2_multiseed 3 0 2 32
