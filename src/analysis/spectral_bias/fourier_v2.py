import numpy as np
import matplotlib.pyplot as plt
import sys

uend = sys.argv[1]

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

# --- User-defined function f : R^d -> R ---
# Example: f(x) = sin(2π x[0]) + 0.5 * cos(4π x[1])
#def f(x):
#    return np.sin(2 * np.pi * x[..., 0]) + 0.5 * np.cos(4 * np.pi * x[..., 1])

# --- Parameters ---
d = 401                     # input dimension
#L = 1.0                   # domain bound: sample x in [-L, L]^d
S = 5000                 # number of Monte Carlo samples for f(x)
N_directions = 30        # directions per wavenumber shell
k_vals = np.logspace(-3, 2, 20)  # wavenumber magnitudes (e.g., from 1 to 100)

# --- Monte Carlo sampling of input domain ---
#x_samples = np.random.uniform(-L, L, size=(S, d))
x_samples = np.loadtxt(base+"/"+"kdvnx401_dt0.0001_nc5_m5000_P.txt")

yy = np.loadtxt(base+"/"+"kdvnx401_dt0.0001_nc5_m5000_"+uend+"_U.txt")
uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T

plt.figure(figsize=(6, 4))

modeids = np.array(1*np.arange(0, 20), dtype=np.int64)

for modeid in modeids:
    print(modeid)
    # --- Function evaluations ---
    #f_vals = f(x_samples)
    f_vals    = vv[:,modeid]

    # --- Estimate power spectrum ---
    power_spectrum = []

    for k in k_vals:
        # Sample random unit directions (on S^{d-1})
        directions = np.random.randn(N_directions, d)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Build wave vectors of norm k
        xis = k * directions  # shape (N_directions, d)
        
        power_vals = []
        for xi in xis:
            # Compute Fourier estimate at this xi
            phases = np.exp(-2j * np.pi * x_samples @ xi)
            ft_estimate = np.mean(f_vals * phases)
            power = np.abs(ft_estimate) ** 2
            power_vals.append(power)
        
        # Average power over directions
        avg_power = np.mean(power_vals)
        power_spectrum.append(avg_power)

    plt.loglog(k_vals, power_spectrum, marker='o')


plt.xlabel("Wavenumber $k = ||\\xi||$")
plt.ylabel("Estimated Power $|\\hat{f}(\\xi)|^2$")
plt.title("Radial Power Spectrum of $f: \\mathbb{R}^d \\to \\mathbb{R}$")
#plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
