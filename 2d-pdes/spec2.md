# spec2 — Synthesizing boundary-compatible GRFs in the Dirichlet–Laplacian eigenbasis

This note expands §"The chosen approach" of Appendix A in [`spec.md`](spec.md). It
states the construction precisely, derives the mode variances, justifies the
approximation being made, and connects the formulas to
[`generate_grfs.py::sample_grf_dirichlet`](generate_grfs.py).

Goal: draw a Gaussian random field $u_0$ on $\Omega=(0,1)^2$ that (i) is a *valid*
homogeneous-Dirichlet initial condition — $u_0=0$ on $\partial\Omega$ exactly —
(ii) has a controllable smoothness matching the RBF length scale $l$, and (iii) is
expressed in the *same* sine basis the solver uses, so no extra transform or dense
covariance factorization is needed.

---

## 1. The Dirichlet–Laplacian eigenbasis

On the unit square, the negative Laplacian with homogeneous Dirichlet boundary
conditions has the exact eigenpairs

$$
\varphi_{m,n}(x,y)=\sin(m\pi x)\sin(n\pi y),\qquad
-\Delta\,\varphi_{m,n}=\lambda_{m,n}\,\varphi_{m,n},\qquad
\lambda_{m,n}=(m^2+n^2)\pi^2,\quad m,n\ge 1 .
$$

Two properties are all we use:

- **Boundary compatibility.** Every $\varphi_{m,n}$ vanishes on $\partial\Omega$ by
  construction, so *any* linear combination does too. This is what makes the field
  an admissible initial condition — no repair step is required.
- **Orthogonality.** $\displaystyle\int_\Omega \varphi_{m,n}\varphi_{m',n'}\,dx\,dy
  = \tfrac14\,\delta_{mm'}\delta_{nn'}$, i.e. the $\varphi_{m,n}$ are orthogonal in
  $L^2(\Omega)$. They therefore form the coordinate system in which a Laplacian-based
  covariance operator is diagonal (§2).

These are the same eigenfunctions and eigenvalues the solver diagonalizes
([`diffusion_spectral.py`](diffusion_spectral.py), which evolves mode $(m,n)$ by
$e^{-D\lambda_{m,n}t}$). The initial-condition generator and the propagator share one
basis.

---

## 2. A GRF as a function of the Laplacian, and its Karhunen–Loève expansion

We define the field through a covariance operator that is a **function of the
Dirichlet Laplacian**,

$$
\mathcal{C}=g(-\Delta_{\text{Dir}}),\qquad g(\lambda)=\exp\!\big(-\tfrac12 l^2\lambda\big)>0 .
$$

Because $\mathcal{C}$ is built from $-\Delta_{\text{Dir}}$, it is diagonalized by the
*same* eigenfunctions: $\mathcal{C}\,\varphi_{m,n}=g(\lambda_{m,n})\,\varphi_{m,n}$.
Its eigenvalues $g(\lambda_{m,n})>0$ are summable (they decay super-exponentially), so
$\mathcal{C}$ is trace-class and defines a genuine Gaussian measure. The
Karhunen–Loève theorem then gives the field as a sum over the *covariance
eigenfunctions* with **independent** Gaussian coefficients whose variances are the
covariance eigenvalues:

$$
\boxed{\;u_0(x,y)=\sum_{m,n\ge 1} c_{m,n}\,\varphi_{m,n}(x,y),\qquad
c_{m,n}\stackrel{\text{indep}}{\sim}\mathcal N\!\big(0,\;S(\lambda_{m,n})\big),\quad
S(\lambda)=g(\lambda)=e^{-l^2\lambda/2}.\;}
$$

This is exact for the operator $\mathcal{C}$: the KL expansion of a
Laplacian-function GRF *is* a sine series with independent, variance-weighted modes.
No conditioning, no tapering, no dense linear algebra — the boundary condition and the
diagonalization come for free from the choice of basis. Truncating to $1\le m,n\le N$
is the only approximation at this stage, and it is the same spatial truncation the
solver already makes.

---

## 3. Why $S(\lambda)=e^{-l^2\lambda/2}$ reproduces the RBF length scale

The target in [`spec.md`](spec.md) family (b) is the squared-exponential (RBF) kernel
$k(\mathbf x,\mathbf x')=\sigma^2\exp(-\lVert\mathbf x-\mathbf x'\rVert^2/2l^2)$. A
*stationary* kernel is characterized by its power spectral density (Bochner's
theorem). On $\mathbb R^2$ the RBF kernel's PSD is again a Gaussian,

$$
\hat k(\mathbf k)\;\propto\;\exp\!\big(-\tfrac12 l^2\lVert\mathbf k\rVert^2\big).
$$

A stationary GRF can be synthesized by giving each Fourier mode $e^{i\mathbf k\cdot\mathbf x}$
an independent Gaussian amplitude with variance $\hat k(\mathbf k)$. Our sine modes
$\varphi_{m,n}$ are the bounded-domain analog of Fourier modes, and the Laplacian
eigenvalue plays the role of the squared wavenumber,

$$
\lambda_{m,n}=(m^2+n^2)\pi^2\;\longleftrightarrow\;\lVert\mathbf k\rVert^2 .
$$

Substituting gives exactly the mode variance used above,

$$
S(\lambda_{m,n})=\hat k\big(\lVert\mathbf k\rVert^2=\lambda_{m,n}\big)
=\exp\!\big(-\tfrac12 l^2\lambda_{m,n}\big)
=\exp\!\big(-\tfrac12 l^2\pi^2(m^2+n^2)\big).
$$

**What is exact vs. approximate.** The map "RBF ⇒ Gaussian PSD in frequency" is exact
on the whole plane. Transplanting that PSD onto the Dirichlet sine spectrum is an
*approximation*: the true Mercer eigenfunctions of the RBF kernel restricted to a
square are not sines, and a bounded domain is not translation-invariant. What the
construction reproduces faithfully is the **spectral shape** — the same Gaussian
frequency decay controlled by the same $l$ — hence the same interior smoothness and
length scale. In the interior, far from $\partial\Omega$, the resulting covariance is
approximately stationary and matches the RBF covariance; near the boundary it must
differ, by design (§5).

---

## 4. Discrete implementation (the code)

The continuous series is realized on the interior grid by one inverse DST-I. This is
lines 42–55 of [`generate_grfs.py`](generate_grfs.py):

| Step | Math | Code |
|------|------|------|
| Eigenvalues | $\lambda_{m,n}=(m^2+n^2)\pi^2$ | `lam = (modes[:,None]**2 + modes[None,:]**2) * np.pi**2` |
| Mode variances | $S(\lambda_{m,n})=e^{-l^2\lambda/2}$ | `S = np.exp(-0.5 * l**2 * lam)` |
| Amplitudes | $c_{m,n}\sim\mathcal N(0,S)$ | `c = rng.standard_normal((n,n)) * np.sqrt(S)` |
| Synthesis | $u_0=\sum c_{m,n}\varphi_{m,n}$ | `U = idstn(c, type=1)` |
| Normalize | rescale to std $\sigma$ | `U = sigma * U / U.std()` |

**Grid / basis match.** For $n$ interior points with the boundary pinned to $0$, the
sample locations are $x_k=k/(n+1)$, $k=1,\dots,n$. Then

$$
\varphi_{m,n}(x_j,y_k)=\sin\!\Big(\frac{m\pi j}{n+1}\Big)\sin\!\Big(\frac{n\pi k}{n+1}\Big),
$$

which is precisely the DST-I kernel, so `idstn(c, type=1)` evaluates
$\sum_{m,n}c_{m,n}\varphi_{m,n}$ at the interior nodes in $O(n^2\log n)$. This is the
same node convention as `interior_grid` in [`diffusion_spectral.py`](diffusion_spectral.py),
so a generated $u_0$ can be fed straight into the solver.

**Normalization.** SciPy's DST-I carries a fixed constant factor, and the exact
continuum-to-discrete scaling of the KL series would need per-mode constants. The code
sidesteps both by rescaling the sample to the requested empirical pointwise standard
deviation $\sigma$ (`U = sigma * U / U.std()`). Only the *relative* mode variances —
the spectral *shape* $S(\lambda)$, which encodes $l$ — matter, and those are preserved;
the overall amplitude is set afterward. (This makes $\sigma$ an empirical spatial std
of the drawn sample rather than an ensemble one, a negligible distinction at
$64\times64$.)

---

## 5. What is preserved, and what changes by design

**Preserved** (see [`spec.md`](spec.md) Appendix A for the same list):

- *Gaussianity* — $u_0$ is a linear combination of independent Gaussians, hence a
  mean-zero Gaussian field exactly.
- *Length-scale control* — the damping $e^{-l^2\lambda/2}$ is the same Gaussian
  frequency decay as the RBF PSD; larger $l$ ⇒ energy in low modes ⇒ smoother, smaller
  $l$ ⇒ high modes admitted ⇒ rougher. Same parameter, same monotone meaning.
- *$C^\infty$ smoothness* — super-exponential spectral decay ⇒ infinitely
  differentiable interior sample paths, as for RBF GRFs.
- *Isotropy* — $S$ depends on $(m,n)$ only through $m^2+n^2=\lVert\mathbf k\rVert^2$,
  so the spectrum is radially symmetric (up to the unavoidable anisotropy of the
  boundary and the discrete lattice).
- *Interior stationarity* — away from $\partial\Omega$ the covariance is approximately
  translation-invariant with the RBF shape.

**Changed by design:**

- *Exact homogeneous Dirichlet BC.* Every sample satisfies $u_0=0$ on $\partial\Omega$
  analytically. The unavoidable price is that **near-boundary variance is suppressed**:
  since each $\varphi_{m,n}\to 0$ at the wall, $\operatorname{Var}\,u_0(x,y)\to 0$ as
  $(x,y)\to\partial\Omega$. This is the 2-D analog of a Brownian bridge — the field
  conditioned, in the spectral sense, to hit zero on the boundary. It is the *minimal*
  principled deformation of the stationary field consistent with the constraint, unlike
  a $\sin\pi x\sin\pi y$ taper, which multiplies by the lowest mode, biases toward low
  frequencies, and corrupts the very length scale $l$ it is supposed to preserve.
- *Native to the solver, and cheap.* The field lives in the exact DST-I basis
  `diffusion_spectral` uses, so its initial amplitudes $c_{m,n}$ are already the solver's
  mode coefficients — one inverse DST ($O(n^2\log n)$) to generate, versus the dense
  $O(N^6)$ Cholesky of `sample_grf_2d` at $N=n^2$ ($\sim 64^2$ unknowns).

---

## 6. Generalization

Only the spectral density $g$ encodes the model; swapping it changes the field family
while keeping the boundary compatibility and the one-DST cost:

- $g(\lambda)=e^{-l^2\lambda/2}$ — squared-exponential / RBF (this note): $C^\infty$,
  Gaussian spectral decay.
- $g(\lambda)=(\kappa^2+\lambda)^{-(\nu+1)}$ — **Whittle–Matérn** fields: algebraic
  spectral decay ⇒ finite, tunable regularity (roughness controlled by $\nu$, range by
  $\kappa$). Use when rougher samples than $C^\infty$ are wanted.

Any positive, summable $g$ yields a valid trace-class covariance operator and hence a
well-defined boundary-vanishing GRF via the identical `idstn` synthesis.
