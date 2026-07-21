import numpy as np
from scipy.fft import idstn


def sample_grf_dirichlet(n, l, sigma=1.0, seed=None):
    """Sample a boundary-compatible GRF on the interior of the unit square.

    Synthesises the field directly in the Dirichlet-Laplacian eigenbasis
    sin(m*pi*x)sin(n*pi*y), so it vanishes on the boundary by construction (a
    valid homogeneous-Dirichlet initial condition) instead of being tapered
    after the fact. The mode variances follow the RBF / squared-exponential
    power spectrum, exp(-l**2 * lambda / 2) with lambda = (m^2+n^2)*pi^2, so the
    length scale `l` has the same meaning as in ``sample_grf_2d`` (larger l ->
    smoother). See spec.md, Appendix A, for the derivation and the list of RBF
    properties this construction preserves.

    Parameters
    ----------
    n : int
        Number of interior grid points along each axis (grid is n-by-n). The
        sample points are x_k = k/(n+1), k = 1..n, matching the DST-I basis.
    l : float
        Length scale l > 0. Smaller l -> rougher, faster-varying functions.
    sigma : float, optional
        Target pointwise standard deviation (default 1.0).
    seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    U : (n, n) ndarray
        Sampled function values on the interior grid, vanishing on the boundary.
    X, Y : (n, n) ndarray
        Interior grid coordinates (from np.meshgrid with indexing='ij').
    """
    if l <= 0:
        raise ValueError("length scale l must be positive")

    rng = np.random.default_rng(seed)

    # Dirichlet-Laplacian eigenvalues on the unit square, lambda_{m,n}.
    modes = np.arange(1, n + 1)
    lam = (modes[:, None] ** 2 + modes[None, :] ** 2) * np.pi ** 2  # (n, n)

    # RBF-matched spectral density: variance of each sine mode.
    S = np.exp(-0.5 * l ** 2 * lam)

    # Draw independent mode amplitudes, then synthesise with one inverse DST-I.
    c = rng.standard_normal((n, n)) * np.sqrt(S)
    U = idstn(c, type=1)

    # Normalise to the requested pointwise standard deviation.
    std = U.std()
    if std > 0:
        U = sigma * U / std

    coords = np.arange(1, n + 1) / (n + 1)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    return U, X, Y


def sample_grf_2d(n, l, sigma=1.0, seed=None):
    """
    Sample a function u from a mean-zero Gaussian random field on the unit
    square [0,1]^2, using the RBF (squared-exponential) covariance kernel

        k(x, y) = sigma**2 * exp(-||x - y||^2 / (2 * l**2)),

    and return its values on a regular n-by-n grid.

    Parameters
    ----------
    n : int
        Number of grid coordinates along each axis. The grid has n**2 points.
    l : float
        Length scale l > 0. Smaller l -> rougher, faster-varying functions.
    sigma : float, optional
        Output standard deviation; sets the pointwise variance k(x,x)=sigma**2.
        Default 1.0 (matches the DeepONet paper's unit-variance convention).
    seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    U : (n, n) ndarray
        Sampled function values u(x) on the grid. U[i, j] = u at (X[i,j], Y[i,j]).
    X, Y : (n, n) ndarray
        Grid coordinates (from np.meshgrid with indexing='ij').
    """
    if l <= 0:
        raise ValueError("length scale l must be positive")

    rng = np.random.default_rng(seed)

    # 1-D coordinates on [0,1], then the full n**2 grid of 2-D points.
    coords = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(coords, coords, indexing="ij")   # each (n, n)
    pts = np.column_stack([X.ravel(), Y.ravel()])       # (n**2, 2)

    # Pairwise squared Euclidean distances between all grid points: (n**2, n**2).
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq_norms = np.sum(pts**2, axis=1)
    sqdist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(sqdist, 0.0, out=sqdist)                 # clip tiny negatives

    # RBF covariance matrix K, with a small jitter added to the diagonal so the
    # Cholesky factorization is numerically positive-definite.
    K = sigma**2 * np.exp(-sqdist / (2.0 * l**2))
    K[np.diag_indices_from(K)] += 1e-10 * sigma**2

    # Sample: u = L z, where K = L L^T (Cholesky) and z ~ N(0, I).
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(n * n)
    u = L @ z

    return u.reshape(n, n), X, Y
