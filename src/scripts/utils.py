"""
Utility functions for SVDONet training and analysis.
Includes data loading, model definitions, training loops, and checkpoint management.
"""

import os
import sys
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from flax.training import train_state
import optax
from typing import Any
import flax.serialization

# Configure matplotlib and JAX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
jax.config.update('jax_enable_x64', True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dic(tag):
    """Map integer tags to dataset configurations (batch_name, endtag, llw)."""
    configs = {
        0: ("advdiffnx201_dt0.0005_nc20_m1000", "1000", 20),
        1: ("advdiffnx201_dt0.0005_nc20_m1000", "1999", 20),
        2: ("kdvnx401_dt0.0001_nc5_m1000", "10", 50),
        3: ("kdvnx401_dt0.0001_nc5_m1000", "1999", 50),
        4: ("kdvnx401_dt0.0001_nc5_m1000", "5999", 50),
        5: ("kdvnx401_dt0.0001_nc5_m1000", "9999", 50),
        6: ("burgers_dt0.0001_nc10_m1000", "100", 50),
        7: ("burgers_dt0.0001_nc10_m1000", "999", 50),
    }
    return configs.get(tag)


def get_dwllw(name):
    """
    Parse directory name to extract network architecture parameters.
    
    Returns:
        depth, width, llw (inner dimension), whichT, batch_name, num_data, endtag
    """
    parts = name.split("_")
    
    whichT = int(parts[0][6:])
    depth = int(parts[8][1:])
    width = int(parts[9][1:])
    llw = int(parts[10][3:])
    
    # Extract batch name and number of data points
    batch_name = parts[11][3:]
    num_data = 0
    
    for j, part in enumerate(parts[12:]):
        if "numd" in part:
            num_data = int(part[4:])
            break
        batch_name += "_" + part
    
    # Clean up batch name (remove last part which is endtag)
    batch_parts = batch_name.split("_")
    batch_name = "_".join(batch_parts[:-1])
    endtag = batch_parts[-1]
    
    return depth, width, llw, whichT, batch_name, num_data, endtag


def contains_all(direc, tags):
    """Check if directory name contains all specified tags."""
    return all(tag in direc for tag in tags)


def contains_all_remove(direc, tags):
    """Check if directory contains all tags and return remaining string."""
    remain = direc
    for tag in tags:
        if tag not in direc:
            return False, ""
        remain = remain.replace(tag, "")
    return True, remain


def get_colors():
    """Load color palette from colors.txt file."""
    with open("colors.txt", "r") as f:
        colors = [line.strip() for line in f.readlines()]
    return colors


def stringify(array):
    """Convert array to space-separated string."""
    return " ".join(str(x) for x in array) + " "


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(batch_name0, uendtag, num_data):
    """
    Load dataset and split into training and test sets.
    
    Args:
        batch_name0: Base name of dataset files
        uendtag: Tag for solution file
        num_data: Total number of data points to load
    
    Returns:
        nt, nb: Dimensions of coordinate and parameter spaces
        rtrain, rtest: Training and test coordinates
        ptrain, ptest: Training and test parameters
        utrain, utest: Training and test solutions
    """
    batch_name = f"../data/{batch_name0}"
    
    # Load data matrices
    R0 = np.loadtxt(f"{batch_name}_R.txt")
    P0 = np.loadtxt(f"{batch_name}_P.txt")
    U0 = np.loadtxt(f"{batch_name}_{uendtag}_U.txt")
    
    print(f"Original shapes: R={np.shape(R0)}, P={np.shape(P0)}")
    
    # Ensure 2D arrays
    if len(np.shape(R0)) == 1:
        R0 = R0.reshape(-1, 1)
    if len(np.shape(P0)) == 1:
        P0 = P0.reshape(-1, 1)
    
    print(f"Reshaped: R={np.shape(R0)}, P={np.shape(P0)}")
    
    # Select subset of data
    P = P0[:num_data, :]
    U = U0[:, :num_data]
    
    # Split into train (90%) and test (10%)
    num_train = int(0.9 * num_data)
    
    rtrain = rtest = R0
    ptrain = P[:num_train, :]
    ptest = P[num_train:, :]
    utrain = U[:, :num_train]
    utest = U[:, num_train:]
    
    nt = rtrain.shape[1]
    nb = ptrain.shape[1]
    
    return nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(params, update, path="checkpoint.msgpack"):
    """Save model parameters and optimizer state to checkpoint file."""
    to_save = {"params": params, "update": update}
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))


def load_checkpoint(init_params, init_update, path="checkpoint.msgpack"):
    """Load model parameters and optimizer state from checkpoint file."""
    with open(path, "rb") as f:
        byte_data = f.read()
    
    template = {"params": init_params, "update": init_update}
    restored = flax.serialization.from_bytes(template, byte_data)
    return restored["params"], restored["update"]


def save_params(params, base_path):
    """Save model parameters to text files (legacy format)."""
    for p_key in params.keys():
        for net_key in params[p_key].keys():
            for layer_key in params[p_key][net_key].keys():
                filename = f"{base_path}{net_key}_{layer_key}.txt"
                
                with open(filename, "w") as f:
                    for param_key in params[p_key][net_key][layer_key].keys():
                        param_array = params[p_key][net_key][layer_key][param_key]
                        
                        # Write parameter array to file
                        for i in range(param_array.shape[0]):
                            if len(param_array.shape) == 2:
                                line = " ".join(str(param_array[i, j]) 
                                              for j in range(param_array.shape[1]))
                            else:
                                line = str(param_array[i])
                            f.write(line + "\n")
                        f.write("\n\n")


# =============================================================================
# TRAIN STATE
# =============================================================================

class TrainState(train_state.TrainState):
    """Standard Flax training state."""
    pass


class TrainStateWithUpdates(train_state.TrainState):
    """Training state that also returns parameter updates."""
    
    def apply_gradients_with_updates(self, *, grads):
        """Apply gradients and return both new state and updates."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state),
            updates
        )


# =============================================================================
# DEEPONET MODEL (with trunk network)
# =============================================================================

class DeepONet(nn.Module):
    """
    Deep Operator Network with learnable trunk and branch networks.
    
    Attributes:
        nb: Branch network input size
        nt: Trunk network input size
        d: Number of hidden layers
        w: Width of hidden layers
        llw: Output dimension (inner dimension)
    """
    nb: int
    nt: int
    d: int
    w: int
    llw: int

    def setup(self):
        """Initialize branch and trunk networks."""
        self.branch_net = self._build_mlp(self.nb)
        self.trunk_net = self._build_mlp(self.nt)

    def _build_mlp(self, input_dim):
        """Build MLP with d hidden layers of width w."""
        layers = [nn.Dense(self.w), nn.gelu]
        for _ in range(self.d - 1):
            layers.append(nn.Dense(self.w))
            layers.append(nn.gelu)
        layers.append(nn.Dense(self.llw))
        return nn.Sequential(layers)

    def __call__(self, p, r, ScaledSigma):
        """
        Forward pass through DeepONet.
        
        Args:
            p: Branch input (parameters/initial conditions)
            r: Trunk input (evaluation coordinates)
            ScaledSigma: Diagonal matrix with singular values
        
        Returns:
            G: Predicted solution
            T: Trunk network output
            B: Branch network output
        """
        B = self.branch_net(p)
        T = self.trunk_net(r)
        G = jnp.matmul(T, jnp.matmul(ScaledSigma, B.T))
        return G, T, B


def get_params(direc, w, llw, lastlayer):
    """Load DeepONet parameters from text files."""
    files = os.listdir(direc)
    branch_dict = {}
    trunk_dict = {}
    
    for filename in files:
        if "test" in filename or "train" in filename:
            continue
        
        # Determine layer width
        layer_width = llw if lastlayer in filename else w
        
        # Load bias and kernel
        bias = jnp.array(np.loadtxt(f"{direc}/{filename}", max_rows=layer_width))
        kern0 = np.loadtxt(f"{direc}/{filename}", skiprows=layer_width + 2)
        
        # Ensure kernel is 2D
        if len(np.shape(kern0)) == 1:
            kern = jnp.zeros((1, len(kern0)))
            kern = kern.at[0, :].set(kern0)
        else:
            kern = jnp.array(kern0)
        
        # Extract layer name
        parts = filename.split("_")
        layer_name = f"{parts[2]}_{parts[3].split('.')[0]}"
        
        # Assign to appropriate network
        if "branch_net" in filename:
            branch_dict[layer_name] = {"kernel": kern, "bias": bias}
        elif "trunk_net" in filename:
            trunk_dict[layer_name] = {"kernel": kern, "bias": bias}
    
    return {"params": {"branch_net": branch_dict, "trunk_net": trunk_dict}}


def get_model(direc, last_layer, d, w, llw, nt, nb):
    """Load DeepONet model and parameters from directory."""
    model = DeepONet(nb, nt, d, w, llw)
    params = get_params(direc, w, llw, last_layer)
    return model, params


# =============================================================================
# DEEPONET LOSS FUNCTIONS AND TRAINING
# =============================================================================

@jax.jit
def all_losses(params, state, p, r, G_true, ScaledSigma, scaleT, scaleB):
    """
    Compute all loss components for DeepONet.
    
    Returns:
        L_data: Data fitting loss
        L_orthoT: Trunk orthogonality regularization
        L_orthoB: Branch orthogonality regularization
    """
    G_pred, T, B = state.apply_fn(params, p, r, ScaledSigma)
    
    L_data = jnp.mean((G_pred - G_true) ** 2)
    
    gram_T = jnp.matmul(T.T, jnp.matmul(T, ScaledSigma))
    L_orthoT = jnp.mean((gram_T - scaleT * ScaledSigma) ** 2)
    
    gram_B = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_B - scaleB * ScaledSigma) ** 2)
    
    return L_data, L_orthoT, L_orthoB


@jax.jit
def train_step(state, p, r, G_true, alphaT, alphaB, ScaledSigma, scaleT, scaleB):
    """Perform one training step for DeepONet."""
    def loss_fn(params):
        L_data, L_orthoT, L_orthoB = all_losses(
            params, state, p, r, G_true, ScaledSigma, scaleT, scaleB
        )
        return L_data + alphaT * L_orthoT + alphaB * L_orthoB
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates


@jax.jit
def all_loss_components(params, state, p, r, VT, ScaledSigma, truesigma, leftsingvec):
    """Compute mode-wise loss components."""
    G_pred, T, B = state.apply_fn(params, p, r, ScaledSigma)
    VTapprox = jnp.matmul(jnp.diag(1.0/truesigma), jnp.matmul(leftsingvec.T, G_pred))
    return jnp.sum((VTapprox.T - VT.T) ** 2, axis=0)


# =============================================================================
# SVDONET MODEL (fixed trunk, learnable branch)
# =============================================================================

class TDeepONet(nn.Module):
    """
    SVDONet: DeepONet with fixed trunk matrix (from SVD).
    Only the branch network is learned.
    
    Attributes:
        nb: Branch network input size
        d: Number of hidden layers
        w: Width of hidden layers
        llw: Output dimension
    """
    nb: int
    d: int
    w: int
    llw: int

    def setup(self):
        """Initialize only the branch network."""
        self.branch_net = self._build_mlp(self.nb)

    def _build_mlp(self, input_dim):
        """Build MLP with d hidden layers of width w."""
        layers = [nn.Dense(self.w), nn.gelu]
        for _ in range(self.d - 1):
            layers.append(nn.Dense(self.w))
            layers.append(nn.gelu)
        layers.append(nn.Dense(self.llw))
        return nn.Sequential(layers)

    def __call__(self, p, T, ScaledSigma):
        """
        Forward pass with fixed trunk matrix T.
        
        Args:
            p: Branch input
            T: Fixed trunk matrix (from SVD)
            ScaledSigma: Diagonal matrix with singular values
        """
        B = self.branch_net(p)
        G = jnp.matmul(T, jnp.matmul(ScaledSigma, B.T))
        return G, T, B


def get_Tparams(direc, w, llw, lastlayer):
    """Load SVDONet (branch-only) parameters from text files."""
    files = os.listdir(direc)
    branch_dict = {}
    
    for filename in files:
        if "test" in filename or "train" in filename:
            continue
        
        layer_width = llw if lastlayer in filename else w
        
        bias = jnp.array(np.loadtxt(f"{direc}/{filename}", max_rows=layer_width))
        kern0 = np.loadtxt(f"{direc}/{filename}", skiprows=layer_width + 2)
        
        if len(np.shape(kern0)) == 1:
            kern = jnp.zeros((1, len(kern0)))
            kern = kern.at[0, :].set(kern0)
        else:
            kern = jnp.array(kern0)
        
        parts = filename.split("_")
        layer_name = f"{parts[2]}_{parts[3].split('.')[0]}"
        branch_dict[layer_name] = {"kernel": kern, "bias": bias}
    
    return {"params": {"branch_net": branch_dict}}


def get_Tmodel(direc, last_layer, d, w, llw, nb):
    """Load SVDONet model and parameters."""
    model = TDeepONet(nb, d, w, llw)
    params = get_Tparams(direc, w, llw, last_layer)
    return model, params


# =============================================================================
# SVDONET LOSS FUNCTIONS AND TRAINING
# =============================================================================

@jax.jit
def Tall_losses(params, state, p, T, G_true, ScaledSigma, scaleB):
    """Compute loss components for SVDONet."""
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)
    
    L_data = jnp.mean((G_pred - G_true) ** 2)
    
    gram_B = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_B - scaleB * ScaledSigma) ** 2)
    
    return L_data, L_orthoB


@jax.jit
def Ttrain_step(state, p, T, G_true, alphaB, ScaledSigma, scaleB):
    """Perform one training step for SVDONet."""
    def loss_fn(params):
        L_data, L_orthoB = Tall_losses(params, state, p, T, G_true, ScaledSigma, scaleB)
        return L_data + alphaB * L_orthoB
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates


@jax.jit
def Tall_loss_components(params, state, p, T, VT, ScaledSigma):
    """Compute mode-wise loss components for SVDONet."""
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)
    return jnp.sum((B - VT.T) ** 2, axis=0)


@jax.jit
def TEall_losses(params, state, p, T, truesigma, VT, exponent, ScaledSigma, scaleB):
    """Compute losses for SVDONet with modified loss weighting (exponent e)."""
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)
    
    # Weighted loss using exponent
    target = jnp.matmul(jnp.diag(truesigma ** (1 + exponent)), VT)
    pred = jnp.matmul(jnp.diag(truesigma ** exponent), jnp.matmul(ScaledSigma, B.T))
    L_data = jnp.mean((target - pred) ** 2)
    
    gram_B = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_B - scaleB * ScaledSigma) ** 2)
    
    return L_data, L_orthoB


@jax.jit
def TEtrain_step(state, p, T, truesigma, VT, exponent, alphaB, ScaledSigma, scaleB):
    """Training step for SVDONet with loss re-weighting."""
    def loss_fn(params):
        L_data, L_orthoB = TEall_losses(
            params, state, p, T, truesigma, VT, exponent, ScaledSigma, scaleB
        )
        return L_data + alphaB * L_orthoB
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates


# =============================================================================
# STACKED ARCHITECTURES
# =============================================================================

class FeedforwardNet(nn.Module):
    """Simple feedforward network for stacked architectures."""
    input_dim: int
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x


class TrunkNet(nn.Module):
    """Trunk network for stacked DeepONet."""
    input_dim: int
    hidden_dim: int
    depth: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class StackedDeepONet(nn.Module):
    """
    Stacked DeepONet: separate branch network for each mode.
    
    Attributes:
        branch_input_dim: Branch input dimension
        trunk_input_dim: Trunk input dimension
        depth_tr, hidden_dim_tr: Trunk network architecture
        depth_br, hidden_dim_br: Branch network architecture
        p: Number of modes (number of branch sub-networks)
    """
    branch_input_dim: int
    trunk_input_dim: int
    depth_tr: int
    hidden_dim_tr: int
    depth_br: int
    hidden_dim_br: int
    p: int

    def setup(self):
        """Initialize p separate branch networks and one trunk network."""
        self.branch_nets = [
            FeedforwardNet(self.branch_input_dim, self.hidden_dim_br, self.depth_br)
            for _ in range(self.p)
        ]
        self.trunk_net = TrunkNet(
            self.trunk_input_dim, self.hidden_dim_tr, self.depth_tr, self.p
        )

    def __call__(self, branch_input, trunk_input, ScaledSigma):
        """Forward pass through stacked architecture."""
        # Evaluate each branch network separately
        branch_outputs = [net(branch_input) for net in self.branch_nets]
        B = jnp.concatenate(branch_outputs, axis=1)
        
        T = self.trunk_net(trunk_input)
        
        return jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)), T, B


class StackedTDeepONet(nn.Module):
    """Stacked SVDONet: separate branch network for each mode, fixed trunk."""
    branch_input_dim: int
    depth_br: int
    hidden_dim_br: int
    p: int

    def setup(self):
        """Initialize p separate branch networks."""
        self.branch_nets = [
            FeedforwardNet(self.branch_input_dim, self.hidden_dim_br, self.depth_br)
            for _ in range(self.p)
        ]

    def __call__(self, branch_input, T, ScaledSigma):
        """Forward pass with fixed trunk matrix."""
        branch_outputs = [net(branch_input) for net in self.branch_nets]
        B = jnp.concatenate(branch_outputs, axis=1)
        return jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)), T, B


def get_STparams(direc, depth, w, llw):
    """Load stacked DeepONet parameters (placeholder)."""
    return {}


def get_STmodel(direc, last_layer, d, w, llw, nt, nb):
    """Load stacked DeepONet model (placeholder)."""
    return 0, 0


def get_TSTparams(direc, depth, w, llw):
    """Load stacked SVDONet parameters from text files."""
    files = os.listdir(direc)
    lastlayer = f"Dense_{depth}"
    branch_nets = {}
    
    for j in range(llw):
        branch_dict = {}
        
        for filename in files:
            if f"branch_nets_{j}_Dense" not in filename:
                continue
            
            # Determine layer width
            layer_width = 1 if lastlayer in filename else w
            
            # Load bias and kernel
            bias0 = jnp.array(np.loadtxt(f"{direc}/{filename}", max_rows=layer_width))
            kern0 = np.loadtxt(f"{direc}/{filename}", skiprows=layer_width + 2)
            
            # Handle 1D arrays
            if len(np.shape(kern0)) == 1 and lastlayer in filename:
                kern = jnp.zeros((len(kern0), 1))
                kern = kern.at[:, 0].set(kern0)
            elif len(np.shape(kern0)) == 1:
                kern = jnp.zeros((1, len(kern0)))
                kern = kern.at[0, :].set(kern0)
            else:
                kern = jnp.array(kern0)
            
            if np.ndim(bias0) == 0:
                bias = jnp.array([bias0])
            else:
                bias = bias0
            
            # Extract layer name
            parts = filename.split("_")
            layer_name = f"Dense_{parts[4].split('.')[0]}"
            branch_dict[layer_name] = {"kernel": kern, "bias": bias}
        
        branch_nets[f"branch_nets_{j}"] = branch_dict
    
    return {"params": branch_nets}


def get_TSTmodel(direc, last_layer, d, w, llw, nb):
    """Load stacked SVDONet model and parameters."""
    model = StackedTDeepONet(nb, d, w, llw)
    params = get_TSTparams(direc, d, w, llw)
    return model, params


# =============================================================================
# TRAINING LOOP
# =============================================================================

def all_mult_updates(Nepochs, state, ptrain, rtrain, utrain, ptest, rtest, utest,
                     alphaT, alphaB, param_base, ScaledSigma, T,
                     truesigma, VT_train, VT_test, exponent, llw, whichT, leftsingvec):
    """
    Main training loop for DeepONet or SVDONet.
    
    Args:
        Nepochs: Number of training epochs
        state: Training state
        whichT: If < 0, train DeepONet; otherwise train SVDONet
        exponent: Loss re-weighting exponent
        llw: Number of modes
    
    Saves checkpoints and logs to param_base directory.
    """
    print(f"Training: {param_base}")
    
    # Setup
    n_train, m_train = utrain.shape
    n_test, m_test = utest.shape
    
    scaleT_train, scaleT_test = n_train ** 2, n_test ** 2
    scaleB_train, scaleB_test = m_train ** 2, m_test ** 2
    
    # Compute normalization factors
    utrain_norm = jnp.mean(utrain ** 2)
    utest_norm = jnp.mean(utest ** 2)
    utrainE_norm = jnp.mean(jnp.matmul(jnp.diag(truesigma ** (1+exponent)), VT_train) ** 2)
    utestE_norm = jnp.mean(jnp.matmul(jnp.diag(truesigma ** (1+exponent)), VT_test) ** 2)
    
    # Initialize optimal parameters
    if whichT < 0:
        opt_state, _, update = train_step(
            state, ptrain, rtrain, utrain, alphaT, alphaB, 
            ScaledSigma, scaleT_train, scaleB_train
        )
    else:
        opt_state, _, update = TEtrain_step(
            state, ptrain, T, truesigma, VT_train, exponent, 
            alphaB, ScaledSigma, scaleB_train
        )
    
    opt_params = opt_state.params
    opt_update = update
    last_params = opt_state.params
    last_update = update
    after_opt_params = opt_state.params
    after_opt_update = update
    
    # Initialize tracking arrays
    min_mode_losses_train = 1e8 * np.ones(llw)
    min_mode_losses_test = 1e8 * np.ones(llw)
    minloss = 1e8
    last_opt_it = 0
    
    errors_data_train = jnp.zeros(Nepochs)
    errorsE_data_train = jnp.zeros(Nepochs)
    errors_orthoT_train = jnp.zeros(Nepochs)
    errors_orthoB_train = jnp.zeros(Nepochs)
    errors_data_test = jnp.zeros(Nepochs)
    errorsE_data_test = jnp.zeros(Nepochs)
    errors_orthoT_test = jnp.zeros(Nepochs)
    errors_orthoB_test = jnp.zeros(Nepochs)
    
    # Training loop
    for i in range(Nepochs):
        # Perform gradient step
        if whichT < 0:  # DeepONet with trainable trunk
            state, _, update = train_step(
                state, ptrain, rtrain, utrain, alphaT, alphaB,
                ScaledSigma, scaleT_train, scaleB_train
            )
            
            # Compute loss components
            L_data_train, L_orthoT_train, L_orthoB_train = all_losses(
                state.params, state, ptrain, rtrain, utrain,
                ScaledSigma, scaleT_train, scaleB_train
            )
            L_data_test, L_orthoT_test, L_orthoB_test = all_losses(
                state.params, state, ptest, rtest, utest,
                ScaledSigma, scaleT_test, scaleB_test
            )
            
            mode_losses_train = all_loss_components(
                state.params, state, ptrain, rtrain, VT_train,
                ScaledSigma, truesigma, leftsingvec
            )
            mode_losses_test = all_loss_components(
                state.params, state, ptest, rtest, VT_test,
                ScaledSigma, truesigma, leftsingvec
            )
            
            LE_data_train = LE_data_test = 0
            
        else:  # SVDONet (fixed trunk)
            state, _, update = TEtrain_step(
                state, ptrain, T, truesigma, VT_train, exponent,
                alphaB, ScaledSigma, scaleB_train
            )
            
            # Compute weighted loss
            LE_data_train, _ = TEall_losses(
                state.params, state, ptrain, T, truesigma, VT_train, exponent,
                ScaledSigma, scaleB_train
            )
            LE_data_test, _ = TEall_losses(
                state.params, state, ptest, T, truesigma, VT_test, exponent,
                ScaledSigma, scaleB_test
            )
            
            # Compute standard loss
            L_data_train, L_orthoB_train = Tall_losses(
                state.params, state, ptrain, T, utrain,
                ScaledSigma, scaleB_train
            )
            L_data_test, L_orthoB_test = Tall_losses(
                state.params, state, ptest, T, utest,
                ScaledSigma, scaleB_test
            )
            
            mode_losses_train = Tall_loss_components(
                state.params, state, ptrain, T, VT_train, ScaledSigma
            )
            mode_losses_test = Tall_loss_components(
                state.params, state, ptest, T, VT_test, ScaledSigma
            )
            
            L_orthoT_train = L_orthoT_test = 0
        
        # Update minimum mode losses
        min_mode_losses_train = np.minimum(mode_losses_train, min_mode_losses_train)
        min_mode_losses_test = np.minimum(mode_losses_test, min_mode_losses_test)
        
        # Store errors
        errors_data_train = errors_data_train.at[i].set(L_data_train / utrain_norm)
        errorsE_data_train = errorsE_data_train.at[i].set(LE_data_train / utrainE_norm)
        errors_orthoT_train = errors_orthoT_train.at[i].set(L_orthoT_train)
        errors_orthoB_train = errors_orthoB_train.at[i].set(L_orthoB_train)
        errors_data_test = errors_data_test.at[i].set(L_data_test / utest_norm)
        errorsE_data_test = errorsE_data_test.at[i].set(LE_data_test / utestE_norm)
        errors_orthoT_test = errors_orthoT_test.at[i].set(L_orthoT_test)
        errors_orthoB_test = errors_orthoB_test.at[i].set(L_orthoB_test)
        
        # Track optimal parameters
        if L_data_test < minloss:
            opt_params = state.params.copy()
            opt_update = update
            last_opt_it = i
            minloss = L_data_test
        
        if last_opt_it + 1 == i:
            after_opt_params = state.params.copy()
            after_opt_update = update
        
        # Logging
        if i % 50 == 0:
            # Write to log file
            with open(f"{param_base}log.txt", "a") as f:
                log_line = f"{i} "
                for error in [errors_data_train[i], errors_orthoT_train[i], errors_orthoB_train[i],
                            errors_data_test[i], errors_orthoT_test[i], errors_orthoB_test[i],
                            minloss/utest_norm, errorsE_data_train[i], errorsE_data_test[i]]:
                    log_line += f"{np.log10(error)} "
                log_line += f"{last_opt_it} \n"
                f.write(log_line)
            
            # Print to console
            print(log_line.strip())
            
            # Write mode losses to file
            with open(f"{param_base}log_modes.txt", "a") as f:
                mode_line = (f"{i} " +
                           stringify(mode_losses_train) +
                           stringify(min_mode_losses_train) +
                           stringify(mode_losses_test) +
                           stringify(min_mode_losses_test) + "\n")
                f.write(mode_line)
        
        # Save checkpoints
        if i % 100 == 0:
            save_checkpoint(state.params, update, 
                          path=f"{param_base}{i+1}cur_chp")
            save_checkpoint(opt_params, opt_update,
                          path=f"{param_base}{i+1}opt_chp")
            save_checkpoint(after_opt_params, after_opt_update,
                          path=f"{param_base}{i+1}aop_chp")
            save_checkpoint(last_params, last_update,
                          path=f"{param_base}{i+1}las_chp")
        
        # Update last parameters
        last_update = update.copy()
        last_params = state.params.copy()
    
    return (state, opt_params, errors_data_train, errors_orthoT_train, errors_orthoB_train,
            errors_data_test, errors_orthoT_test, errors_orthoB_test)

