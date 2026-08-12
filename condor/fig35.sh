source .venv-linux/bin/activate
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
MULTISEED_BAND=1 python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2_multiseed 3 1 1 2
MULTISEED_BAND=1 python -m src.analysis.RELEVANT.plot_gd_or_adam_modelosses2_multiseed 3 0 1 2
