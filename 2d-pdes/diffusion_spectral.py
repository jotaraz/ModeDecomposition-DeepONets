"""Spectral (sine-basis) solver for the 2D heat / diffusion equation.

Solves

    du/dt = D * (u_xx + u_yy)

on the unit square (0, 1)^2 with homogeneous Dirichlet boundary conditions
(u = 0 on the boundary), starting from an initial field u0 sampled on the
interior grid.

Method: the Dirichlet eigenfunctions sin(m*pi*x)sin(n*pi*y) diagonalise the
Laplacian, so in the 2D discrete sine transform (DST-I) each mode a_{n,m}
evolves exactly as a_{n,m}(t) = a_{n,m}(0) * exp(-D * lambda_{n,m} * t) with
lambda_{n,m} = (m^2 + n^2) * pi^2 / side^2. There is no timestep stability
constraint; the only error is spatial truncation / aliasing.

See spec.md for the full derivation.
"""

import numpy as np
from scipy.fft import dstn, idstn

__all__ = ["diffusion_spectral", "interior_grid"]


def interior_grid(nx, ny=None, side=1.0):
    """Return the interior sample points matching the DST-I convention.

    For N interior points on a side of length `side` with the boundary values
    pinned to 0, the sample locations are x_k = k * side / (N + 1), k = 1..N.
    Mode m then evaluates as sin(m*pi*x_k/side) = sin(m*pi*k/(N+1)), exactly the
    DST-I basis.

    Parameters
    ----------
    nx : int
        Number of interior points along x.
    ny : int, optional
        Number of interior points along y (defaults to nx).
    side : float, optional
        Domain side length (default 1.0).

    Returns
    -------
    X, Y : (ny, nx) ndarray
        Interior grid coordinates (row index -> y, column index -> x), matching
        the (Ny, Nx) layout used by ``diffusion_spectral``.
    """
    if ny is None:
        ny = nx
    xs = side * np.arange(1, nx + 1) / (nx + 1)
    ys = side * np.arange(1, ny + 1) / (ny + 1)
    X, Y = np.meshgrid(xs, ys)  # indexing='xy' -> shape (ny, nx)
    return X, Y


def diffusion_spectral(u0, D=1.0, side=1.0, t=0.1):
    """Evolve u0 to time t under the 2D heat equation, spectrally and exactly.

    Parameters
    ----------
    u0 : (Ny, Nx) array_like
        Initial field on the interior grid (boundary values are implicitly 0).
    D : float, optional
        Diffusion coefficient D > 0 (default 1.0).
    side : float, optional
        Domain side length (default 1.0, the unit square).
    t : float, optional
        Evaluation time (default 0.1).

    Returns
    -------
    u_t : (Ny, Nx) ndarray
        The field u(x, y, t) on the same interior grid.
    """
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 2:
        raise ValueError(f"u0 must be 2D (Ny, Nx); got shape {u0.shape}")
    ny, nx = u0.shape

    # Forward DST-I -> mode amplitudes a_{n,m}. Array index (i, j) corresponds to
    # mode numbers n = i + 1 (y) and m = j + 1 (x).
    a = dstn(u0, type=1)

    m = np.arange(1, nx + 1)  # x mode numbers, along axis 1
    n = np.arange(1, ny + 1)  # y mode numbers, along axis 0
    lam = (n[:, None] ** 2 + m[None, :] ** 2) * (np.pi ** 2) / (side ** 2)

    a *= np.exp(-D * lam * t)

    # Inverse DST-I back to grid values. idstn(dstn(., type=1), type=1) is the
    # identity, so the decay factors above are applied exactly per mode.
    return idstn(a, type=1)
