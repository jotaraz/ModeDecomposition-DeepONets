import matplotlib.pyplot as plt

from generate_grfs import *

U, X, Y = sample_grf_dirichlet(n=50, l=0.15, seed=0)

plt.pcolormesh(X, Y, U, shading="auto")
plt.gca().set_aspect("equal")
plt.colorbar(label="u(x)")
plt.show()