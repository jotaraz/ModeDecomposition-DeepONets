source .venv-linux/bin/activate
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
cd /fast/jtaraz/MISC/ModeDecomposition-DeepONets
python condor/plot2d_errdecomp.py
