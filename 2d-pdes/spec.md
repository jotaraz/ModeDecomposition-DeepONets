## Problem

Solve the heat/diffusion equation on the unit square:

$$\frac{\partial u}{\partial t} = D\,\Delta u = D\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

- $u = u(x,y,t)$ : scalar field being diffused
- $D > 0$ : diffusion coefficient
- Domain: $(x,y) \in \Omega = (0,1)^2$
- Boundary condition: homogeneous Dirichlet, $u = 0$ on $\partial\Omega$
- Initial condition: $u(x,y,0) = u_0(x,y)$, arbitrary in theory (must vanish on the boundary) 

In practice, I want to try the following initial conditions
(a) $$u_0(x,y) = \left(\sum_{i=1}^{K} a_i \sin(i \pi x)\right) \left(\sum_{j=1}^{K} b_j \sin(j \pi y)\right)$$ for $a_i$ and $b_j$ uniformly drawn from $[-1,1]$, where $K$ is the number of sine modes per axis, and 
(b) gaussian random fields with varying $l$, you can use generate_grfs.py to generate them.
I want you to save the solution $u(t)$ at some fixed $t$'s. The characteristic
diffusion time on the unit square is $\tau = 1/(D\cdot 2\pi^2) \approx 0.05/D$, so
for the default $D = 1$ the save times $\{0.01, 0.02, 0.05, 0.1\}$ span the transient
decay (the fundamental mode drops from $\approx 0.82$ to $\approx 0.14$ across them).
Larger times such as $\{0.1, 0.2, 0.5, 1.0\}$ collapse the field to $\sim$machine zero
for $D=1$; use them only with a correspondingly smaller $D$ (e.g. $D \approx 0.05$).


## Method

**Space:** Sine-spectral discretization. On the unit square the eigenfunctions $\sin(m\pi x)\sin(n\pi y)$ exactly satisfy the Dirichlet BCs, so the field is represented by its 2D discrete sine transform (DST, type-I) with mode amplitudes $a_{n,m}$ and integer mode numbers $m,n \geq 1$. (For a general square of side length `side`, replace $x,y$ by $x/\text{side},\,y/\text{side}$; the default is `side` $= 1$.)

**Time:** Exact exponential stepping. Each mode is an eigenfunction of $-\Delta$ with eigenvalue

$$\lambda_{n,m} = \frac{(m^2 + n^2)\pi^2}{\text{side}^2},$$

so it evolves exactly as $a_{n,m}(t) = a_{n,m}(0)\,e^{-D\lambda_{n,m} t}$. No timestep stability constraint; error is spatial truncation/aliasing only.

## Algorithm

1. Sample $u_0$ on the interior grid (boundary values are implicitly 0).
2. Forward DST: $u_0 \rightarrow a_{n,m}$.
3. Multiply each mode by $e^{-D\lambda_{n,m} t}$.
4. Inverse DST: $a_{n,m}(t) \rightarrow u(x,y,t)$ on the grid.

## Interface

```
diffusion_spectral(u0, D=1.0, side=1.0, t=0.1) -> u_t
```

- `u0` : `(Ny, Nx)` array of interior grid values
- `D`, `side`, `t` : diffusion coefficient, domain side length, evaluation time
- returns : `(Ny, Nx)` array of $u(x,y,t)$ on the same grid

## Dependencies

- `numpy`
- `scipy.fft` (`dstn`, `idstn`)

## Scope / limitations

- Valid only for constant $D$, rectangular domain, and homogeneous Dirichlet BCs.
- Neumann BCs: swap DST for a discrete cosine transform (`dctn`/`idctn`).
- Periodic BCs: use the full FFT.
- Variable coefficients $D(x,y)$ or source terms break the spectral shortcut — use finite differences with Crank–Nicolson instead.

## Appendix A: Boundary-compatible GRFs via spectral synthesis

### The problem

Initial-condition family (b) draws from a Gaussian random field (GRF). The natural
construction (`generate_grfs.py`) uses a mean-zero GRF with the RBF /
squared-exponential covariance kernel

$$k(\mathbf{x},\mathbf{x}') = \sigma^2 \exp\!\left(-\frac{\lVert \mathbf{x}-\mathbf{x}'\rVert^2}{2 l^2}\right),$$

and samples it on a grid via a dense Cholesky factorization of the $N^2 \times N^2$
covariance. This field is **stationary** and does **not** vanish on $\partial\Omega$,
so it is not a valid initial condition for the homogeneous-Dirichlet problem (the
spec requires $u_0 = 0$ on the boundary). Two fixes are common:

1. **Taper:** multiply the sample by an envelope such as $\sin(\pi x)\sin(\pi y)$.
   This is the crudest option: the envelope is exactly the lowest eigenmode, so it
   suppresses variance *quadratically* through the boundary layer, biases the field
   toward low frequencies, and distorts the length scale $l$ you are trying to
   control. The modification has no principled relationship to the covariance.

2. **Exact Gaussian conditioning:** condition the field on $u = 0$ at the boundary
   nodes, giving conditional covariance $K_{II} - K_{IB}K_{BB}^{-1}K_{BI}$. This is
   faithful (it keeps the exact RBF interior covariance) but still requires the dense
   kernel — an $O(N^6)$ Cholesky at $64\times 64$ — and it is not expressed in the
   basis the solver uses.

### The chosen approach: synthesize in the Dirichlet–Laplacian eigenbasis

Instead of sampling a stationary field and then repairing it, we build the field
**directly** from the eigenfunctions of the Dirichlet Laplacian — the *same* sine
basis the solver transforms into. On the unit square,

$$\varphi_{m,n}(x,y) = \sin(m\pi x)\sin(n\pi y), \qquad
  \lambda_{m,n} = (m^2+n^2)\pi^2, \qquad m,n \ge 1,$$

and every $\varphi_{m,n}$ vanishes on $\partial\Omega$ by construction. We draw
independent mode amplitudes and sum:

$$u_0(x,y) = \sum_{m,n=1}^{N} c_{m,n}\,\varphi_{m,n}(x,y), \qquad
  c_{m,n} \sim \mathcal{N}\!\big(0,\; S(\lambda_{m,n})\big).$$

The spectral density $S(\lambda)$ sets the field's smoothness. To reproduce the RBF
kernel we use its (isotropic, Gaussian) power spectral density: on $\mathbb{R}^2$ the
squared-exponential kernel has $\hat{k}(\mathbf{k}) \propto \exp(-l^2\lVert\mathbf{k}\rVert^2/2)$,
and since the Laplacian eigenvalue plays the role of $\lVert\mathbf{k}\rVert^2$,

$$S(\lambda_{m,n}) = \exp\!\left(-\tfrac{1}{2}\, l^2 \lambda_{m,n}\right)
                   = \exp\!\left(-\tfrac{1}{2}\, l^2 \pi^2 (m^2+n^2)\right).$$

The sum is evaluated with a single inverse DST-I ($O(N^2\log N)$), and the sample is
rescaled to the target pointwise standard deviation $\sigma$.

Formally this is the Karhunen–Loève expansion of a GRF whose covariance operator is
$g(-\Delta_{\text{Dirichlet}})$ with $g(\lambda) = \exp(-l^2\lambda/2)$ — i.e. the
squared-exponential smoothing operator restricted to the domain and made to respect
the boundary. (Swapping $g$ for $(\kappa^2+\lambda)^{-\nu-1}$ yields Whittle–Matérn
fields if rougher, algebraically-decaying spectra are wanted.)

### Which properties of the original GRF carry over

- **Gaussianity** — $u_0$ is a linear combination of independent Gaussians, so it is
  still an exactly mean-zero Gaussian field.
- **Length-scale control via $l$** — the damping factor $\exp(-l^2\lambda/2)$ is the
  *same* Gaussian decay in frequency as the RBF kernel. Larger $l$ concentrates energy
  in the low modes (smoother samples); smaller $l$ admits high modes (rougher). The
  monotone $l \mapsto$ smoothness relationship is preserved, and it is the same
  parameter with the same meaning.
- **$C^\infty$ smoothness** — the super-exponential (Gaussian) spectral decay makes
  interior sample paths infinitely differentiable, exactly as for RBF-kernel GRFs.
- **Isotropy** — $S$ depends on $(m,n)$ only through $m^2+n^2 = \lVert\mathbf{k}\rVert^2$,
  so the spectrum is radially symmetric and the interior field is isotropic, matching
  the RBF kernel (up to the anisotropy the boundary and the discrete lattice
  unavoidably introduce).
- **Interior stationarity** — away from the boundary the covariance is approximately
  translation-invariant and reproduces the RBF covariance shape.

### What changes — by design

- **Exact homogeneous Dirichlet BC.** Every sample vanishes on $\partial\Omega$
  analytically, not approximately. The price is that near-boundary variance is
  suppressed (a 2-D analog of the Brownian bridge). This is the *minimal, principled*
  modification consistent with the constraint — the field conditioned, in the spectral
  sense, to satisfy the BC — rather than the ad-hoc distortion a taper introduces.
- **Native to the solver, and cheap.** The field is expressed in the exact DST-I basis
  `diffusion_spectral` uses, so its initial mode amplitudes $c_{m,n}$ are available
  with no extra transform, and generation costs one inverse DST ($O(N^2\log N)$)
  instead of a dense $O(N^6)$ Cholesky.
