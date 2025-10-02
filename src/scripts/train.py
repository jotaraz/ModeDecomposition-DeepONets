"""
Training script for DeepONet and SVDONet models (both stacked and unstacked and re-weighting).
Handles model initialization, training loop execution, and result logging.
"""

from utils import *
import sys

jax.config.update('jax_enable_x64', True)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Parse command line arguments
which_T = int(sys.argv[1])      # Model type: <0 for DeepONet, >=0 for SVDONet
exponent = float(sys.argv[2])   # Loss re-weighting exponent
dostacked = bool(int(sys.argv[3]))  # Use stacked architecture

# Training hyperparameters
CONFIG = {
    'Nepochs': 201,
    'vtag': 0,              # Random seed/version tag
    'depth': 2,
    'width': 10,
    'llw': 10,              # Inner dimension (number of modes)
    'doplot': False,
    'batch_name': "burgers_dt0.0001_nc10_m1000",
    'lrstag': 40,           # Learning rate identifier for directory naming
    'init_lr': 2e-3,
    'decay_rate': 0.95,
    'num_data': 500,
    'dotruesigma': True,    # Use true singular values
    'uendtag': "100",
    'sigmascale': "1.0",
    'doadam': True,         # Use Adam optimizer (False for SGD)
    'alphaT': 0.0,          # Trunk orthogonality regularization weight
    'alphaB': 0.0,          # Branch orthogonality regularization weight
    'lambdarange': 0.0,
    'adaptive_init_lr': False,
    'momentum': True,       # Use momentum in Adam
}

print(f"Optimizer: {'Adam' if CONFIG['doadam'] else 'SGD'}, "
      f"lr={CONFIG['init_lr']}, decay={CONFIG['decay_rate']}")


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

# Initialize random key
key = jax.random.PRNGKey(CONFIG['vtag'])

# Load dataset
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = load_dataset(
    CONFIG['batch_name'], CONFIG['uendtag'], CONFIG['num_data']
)

print(f"Data shapes: nt={nt}, nb={nb}, rtrain={np.shape(rtrain)}, "
      f"ptrain={np.shape(ptrain)}, utrain={np.shape(utrain)}")

# Compute SVD of training data
llw = min(utrain.shape[0], utrain.shape[1], CONFIG['llw'])
uu_train, ss_train, vh_train = jnp.linalg.svd(utrain, full_matrices=False)

n_train, m_train = utrain.shape
VT_train = vh_train[:llw, :]
truesigma = ss_train[:llw]

# Compute test coefficients
VT_test = jnp.matmul(
    jnp.diag(1/truesigma),
    jnp.matmul(uu_train[:, :llw].T, utest)
)

# Setup trunk matrix and singular value scaling
T = None
lambdas = np.zeros(llw)

if CONFIG['dotruesigma']:
    ScaledSigma = ss_train[:llw]
else:
    ScaledSigma = np.ones(llw) * ss_train[0]

# Apply scaling factor
if CONFIG['sigmascale'] == "First":
    scale_factor = 1.0 / ss_train[0]
elif CONFIG['sigmascale'] == "Size":
    scale_factor = 1.0 / np.sqrt(n_train * m_train)
else:
    scale_factor = float(CONFIG['sigmascale'])

print(f"Sigma scaling: {CONFIG['sigmascale']}, factor={scale_factor}")
print(f"Scaled sigma: {ScaledSigma * scale_factor}")
ScaledSigma = jnp.diag(scale_factor * ScaledSigma)


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

if which_T < 0:  # DeepONet (with trunk network)
    if dostacked:
        model = StackedDeepONet(nb, nt, CONFIG['depth'], CONFIG['width'],
                               CONFIG['depth'], CONFIG['width'], llw)
    else:
        model = DeepONet(nb, nt, CONFIG['depth'], CONFIG['width'], llw)
    params = model.init(key, ptrain, rtrain, ScaledSigma)

else:  # SVDONet (fixed trunk from SVD)
    T = uu_train[:, :llw]
    lambdas = 10 ** (CONFIG['lambdarange'] * (2 * np.random.rand(llw) - 1))
    
    if dostacked:
        model = StackedTDeepONet(nb, CONFIG['depth'], CONFIG['width'], llw)
    else:
        model = TDeepONet(nb, CONFIG['depth'], CONFIG['width'], llw)
    params = model.init(key, ptrain, T, ScaledSigma)


# =============================================================================
# OPTIMIZER SETUP
# =============================================================================

# Adjust learning rate based on singular values if needed
init_lr = CONFIG['init_lr']
if CONFIG['adaptive_init_lr']:
    init_lr *= ss_train[0] ** (-2 * exponent)

# Create learning rate schedule
lr_schedule = optax.exponential_decay(
    init_value=init_lr,
    transition_steps=500,
    decay_rate=CONFIG['decay_rate'],
    staircase=True
)

# Select optimizer
if CONFIG['doadam']:
    if CONFIG['momentum']:
        optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        optimizer = optax.adam(learning_rate=lr_schedule, b1=0.0)
else:
    optimizer = optax.sgd(learning_rate=lr_schedule)

# Create training state
state = TrainStateWithUpdates.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)


# =============================================================================
# DIRECTORY NAMING AND CREATION
# =============================================================================

def create_experiment_name(which_T, dostacked, dotruesigma, sigmascale, alphaT, alphaB,
                          exponent, adaptive_init_lr, Nepochs, depth, width, llw,
                          batch_name, uendtag, num_data, doadam, momentum, lrstag, vtag):
    """Generate unique experiment directory name based on configuration."""
    name = f"whichT{which_T}_doStacked{dostacked}_doSigma{dotruesigma}"
    
    # Add scaling and regularization info
    if abs(exponent) > 1e-12 and adaptive_init_lr:
        name += f"_sisc{sigmascale}_aT{alphaT}_aB{alphaB}_expA{exponent}"
    else:
        name += f"_sisc{sigmascale}_aT{alphaT}_aB{alphaB}_exp{exponent}"
    
    # Add architecture info
    name += f"_Nep{Nepochs}_d{depth}_w{width}_llw{llw}"
    
    # Add dataset info
    name += f"_bat{batch_name}_{uendtag}_numd{num_data}"
    
    # Add optimizer info
    if doadam:
        name += f"_lrAdam{lrstag}" if momentum else f"_lrAda{lrstag}"
    else:
        name += f"_lrSGD{lrstag}"
    
    # Add version tag
    name += f"_v{vtag}"
    
    return name


stem = create_experiment_name(
    which_T, dostacked, CONFIG['dotruesigma'], CONFIG['sigmascale'],
    CONFIG['alphaT'], CONFIG['alphaB'], exponent, CONFIG['adaptive_init_lr'],
    CONFIG['Nepochs'], CONFIG['depth'], CONFIG['width'], llw,
    CONFIG['batch_name'], CONFIG['uendtag'], CONFIG['num_data'],
    CONFIG['doadam'], CONFIG['momentum'], CONFIG['lrstag'], CONFIG['vtag']
)

print(f"Experiment name: {stem}")

# Check if experiment already exists
existing_dirs = os.listdir("../nets")
if stem in existing_dirs:
    print("Directory already exists. Skipping training to avoid overwriting results.")
    sys.exit(0)


# =============================================================================
# TRAINING
# =============================================================================

# Create experiment directory
experiment_dir = f"../nets/{stem}"
os.mkdir(experiment_dir)
param_base = f"{experiment_dir}/"

# Save lambdas (for potential future use)
np.savetxt(f"{param_base}lambdas.txt", lambdas)

# Run training loop
print(f"\nStarting training for {CONFIG['Nepochs']} epochs...")
state, opt_params, errors_data_train, errors_orthoT_train, errors_orthoB_train, \
errors_data_test, errors_orthoT_test, errors_orthoB_test = all_mult_updates(
    CONFIG['Nepochs'], state,
    ptrain, rtrain, utrain, ptest, rtest, utest,
    CONFIG['alphaT'], CONFIG['alphaB'],
    param_base, ScaledSigma, T, truesigma,
    VT_train, VT_test, exponent, llw, which_T, uu_train[:, :llw]
)

print("Training completed!")


# =============================================================================
# VISUALIZATION
# =============================================================================

colors = ["black", "red", "blue", "green", "yellow", 
          "orange", "cyan", "pink", "brown", "gray"]

plt.figure(figsize=(10, 6))

# Plot data loss
plt.plot(errors_data_train, label="L_data train", color=colors[0])
plt.plot(errors_data_test, label="L_data test", color=colors[1])

# Plot trunk orthogonality loss
if CONFIG['alphaT'] > 0:
    plt.plot(CONFIG['alphaT'] * errors_orthoT_train,
            label=f"{CONFIG['alphaT']}*L_orthoT train", color=colors[2])
    plt.plot(CONFIG['alphaT'] * errors_orthoT_test,
            label=f"{CONFIG['alphaT']}*L_orthoT test", color=colors[3])
else:
    plt.plot(errors_orthoT_train, label="L_orthoT train", color=colors[2])
    plt.plot(errors_orthoT_test, label="L_orthoT test", color=colors[3])

# Plot branch orthogonality loss
if CONFIG['alphaB'] > 0:
    plt.plot(CONFIG['alphaB'] * errors_orthoB_train,
            label=f"{CONFIG['alphaB']}*L_orthoB train", color=colors[4])
    plt.plot(CONFIG['alphaB'] * errors_orthoB_test,
            label=f"{CONFIG['alphaB']}*L_orthoB test", color=colors[5])
else:
    plt.plot(errors_orthoB_train, label="L_orthoB train", color=colors[4])
    plt.plot(errors_orthoB_test, label="L_orthoB test", color=colors[5])

plt.legend()
plt.yscale("log")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title(f"Training Progress: {stem}", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{param_base}errorcurve.png", dpi=150)

if CONFIG['doplot']:
    plt.show()

print(f"\nResults saved to: {experiment_dir}")