"""
Analysis script for comparing SVDONet training results across different configurations.
Specifically focuses on comparing different loss re-weighting exponents (e values).
Generates plots showing loss curves and mode-wise losses over training epochs.
"""

from utils import *

# Configuration
CONFIG = {
    'num_columns': 3,
    'num_epochs_to_plot': 80,
    'epoch_step': 4,
    'num_data_points': 1000,
}

DIRECTORIES = [
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0",
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0",
    "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd4000_lrSGD32_v0",
]


def load_experiment_data(direc):
    """Load loss and mode loss data from experiment directory."""
    modeloss_data = np.loadtxt(f"../nets/{direc}/log_modes.txt")
    loss_data = np.loadtxt(f"../nets/{direc}/log.txt")
    
    return {
        'epochs': loss_data[:, 0],
        'train_loss': loss_data[:, 1] / 2,
        'test_loss': loss_data[:, 4] / 2,
        'modeloss_data': modeloss_data,
    }


def extract_mode_losses(modeloss_data, llw):
    """Extract training and test mode losses from modeloss data."""
    train_modes = modeloss_data[:, 1:1+llw]
    test_modes = modeloss_data[:, 1+2*llw:1+3*llw]
    return train_modes, test_modes


def parse_exponent_from_directory(direc):
    """Parse directory name to extract exponent value for plot title."""
    parts = direc.split("_doSigma1_sisc1.0_aT0.0_aB0.0_")
    exp_part = parts[1].split("bat")[0].split("_Nep")[0]
    exponent = exp_part[3:]  # Remove 'exp' prefix
    return rf"$e={exponent}$"


def calculate_base_losses(utrain, utest, llw):
    """
    Calculate base losses and SVD components.
    
    Returns:
        ss_train: Singular values from training data
        base_loss_test: Base loss values for test data
        T: Truncated left singular vectors (trunk matrix)
    """
    uu_train, ss_train, _ = np.linalg.svd(utrain, full_matrices=False)
    
    T = uu_train[:, :llw]
    VT_test = T.T @ utest
    base_loss_test = np.array([np.linalg.norm(VT_test[i, :]) for i in range(llw)])
    
    return ss_train, base_loss_test, T


def setup_figure(num_columns):
    """
    Create figure with custom gridspec layout.
    
    Layout: Top row has 1 wide plot (loss curves),
            Bottom has 2 rows × num_columns (mode losses: train and test)
    """
    fig = plt.figure(figsize=(8, 6))
    
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.25)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    gs_bottom = gridspec.GridSpecFromSubplotSpec(2, num_columns, subplot_spec=outer[1], hspace=0)
    
    # Create axes: 1 top plot + 2*num_columns bottom plots
    axs = [fig.add_subplot(gs_top[0])]
    for i in range(num_columns):
        axs.append(fig.add_subplot(gs_bottom[0, i]))  # Training mode losses
        axs.append(fig.add_subplot(gs_bottom[1, i]))  # Test mode losses
    
    return fig, axs


def plot_loss_curves(ax, epochs, train_loss, test_loss, color, label, num_epochs):
    """Plot training and test loss curves on log scale."""
    ax.plot(epochs[:num_epochs], 10**train_loss[:num_epochs], '--', color=color, label='_nolegend_')
    ax.plot(epochs[:num_epochs], 10**test_loss[:num_epochs], '.-', color=color, label=label)


def plot_mode_losses(ax_train, ax_test, train_modes, test_modes, ss_train, base_loss_test,
                     mtrain, mtest, llw, num_epochs, epoch_step, color, title, xmin, xmax):
    """
    Plot mode-wise losses for training and test data.
    
    Shows base losses (singular values) and mode losses at different epochs
    with color gradient indicating training progress.
    """
    # Plot base losses (from singular values)
    ax_train.plot(ss_train[:llw]**2 / mtrain, '.-', color="k", label='_nolegend_')
    ax_test.plot(base_loss_test**2 / mtest, '.-', color="k", label='_nolegend_')
    
    # Initialize y-limits with base loss values
    y_limits = {
        'train_min': ss_train[llw-1]**2 / mtrain,
        'train_max': ss_train[0]**2 / mtrain,
        'test_min': np.min(base_loss_test)**2 / mtest,
        'test_max': np.max(base_loss_test)**2 / mtest
    }
    
    # Plot mode losses at different epochs with color gradient
    last_epoch = 0
    for j in range(0, num_epochs, epoch_step):
        progress = j / num_epochs
        color_intensity = 0.2 + 0.7 * progress
        
        ytmp_train = ss_train[:llw]**2 * train_modes[j, :] / mtrain
        ytmp_test = ss_train[:llw]**2 * test_modes[j, :] / mtest
        
        # Update y-limits
        y_limits['train_min'] = min(y_limits['train_min'], np.min(ytmp_train))
        y_limits['train_max'] = max(y_limits['train_max'], np.max(ytmp_train))
        y_limits['test_min'] = min(y_limits['test_min'], np.min(ytmp_test))
        y_limits['test_max'] = max(y_limits['test_max'], np.max(ytmp_test))
        
        ax_train.plot(ytmp_train, '--', color=(color_intensity, 0, 0), alpha=0.5, label='_nolegend_')
        ax_test.plot(ytmp_test, '--', color=(0, 0, color_intensity), alpha=0.5, label='_nolegend_')
        last_epoch = j
    
    # Add title boxes with background color
    for ax, ymax_key in [(ax_train, 'train_max'), (ax_test, 'test_max')]:
        txt = ax.text(1.05*llw, y_limits[ymax_key], title, color="k", fontsize=14,
                     horizontalalignment='right', verticalalignment='top')
        txt.set_bbox(dict(facecolor=color, alpha=0.4, edgecolor=color))
    
    # Configure axes
    for ax in [ax_train, ax_test]:
        ax.set_xlim(xmin, xmax)
        ax.set_yscale("log")
    
    return y_limits, last_epoch


def get_reference_values(direc, train_modes, test_modes, ss_train, mtrain, mtest, 
                        llw, last_epoch):
    """
    Extract reference values for standard configuration (e=0.0).
    These are used to draw horizontal reference lines.
    """
    # Only calculate for standard configuration
    if "exp0.0" not in direc or "doStackedFalse" not in direc or "_w50" in direc:
        return None
    
    # Use epoch slightly before last to avoid noise
    ref_epoch = max(0, last_epoch - 4)
    
    train_value = np.max(ss_train[:llw]**2 * train_modes[ref_epoch, :] / mtrain)
    test_value = np.max(ss_train[:llw]**2 * test_modes[ref_epoch, :] / mtest)
    
    return {'train': train_value, 'test': test_value}


def finalize_plot(fig, axs, num_columns, llw, xmin, xmax, y_limits_all, reference_values):
    """Add labels, legends, and reference lines to complete the plot."""
    # Main x-label
    fig.text(0.45, 0.04, r"Mode index $i$", fontsize=12)
    
    # Configure top loss curve plot
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Epochs", fontsize=12)
    axs[0].set_ylabel(r"Relative Error $\delta$", fontsize=12)
    
    # Add legend entries for train/test distinction
    axs[0].plot([], [], '.-', color="gray", label="Test")
    axs[0].plot([], [], '--', color="gray", label="Train")
    axs[0].legend(ncol=4, fontsize=12, loc='upper right')
    
    # Y-labels for mode loss plots
    axs[1].set_ylabel("Weighted Training\nMode Losses", fontsize=12, multialignment='center')
    axs[2].set_ylabel("Weighted Test\nMode Losses", fontsize=12, multialignment='center')
    
    # Configure each column of mode loss plots
    for k in range(num_columns):
        ref_vals = reference_values[k]
        
        # Add reference lines if available
        if ref_vals:
            for kk in range(num_columns):
                axs[1+2*kk].plot([xmin, xmax], [ref_vals['train']]*2, '--',
                                color="fuchsia", alpha=0.8, linewidth=1, label='_nolegend_')
                axs[2+2*kk].plot([xmin, xmax], [ref_vals['test']]*2, '--',
                                color="fuchsia", alpha=0.8, linewidth=1, label='_nolegend_')
        
        # Remove y-ticks for all columns except the first
        if k != 0:
            axs[1+2*k].set_yticks([])
            axs[2+2*k].set_yticks([])
        
        # Set y-limits with some padding
        axs[1+2*k].set_ylim(y_limits_all['train_min']/2, 2*y_limits_all['train_max'])
        axs[2+2*k].set_ylim(y_limits_all['test_min']/2, 2*y_limits_all['test_max'])
    
    plt.subplots_adjust(wspace=0.0, hspace=0.2)


def find_valid_directories(directories, nets_path="../nets"):
    """Find directories that exist and contain required log files."""
    available_dirs = os.listdir(nets_path)
    valid_dirs = [
        d for d in directories 
        if d in available_dirs and "log_modes.txt" in os.listdir(f"{nets_path}/{d}")
    ]
    return valid_dirs

def get_singular_values(batch_name0, endtag, num_data):
    batch_name = f"../data/{batch_name0}"
    name = f"{batch_name}_{endtag}_numd{num_data}_singvals.txt"
    return np.loadtxt(name)

def main():
    """Main analysis routine."""
    num_columns = CONFIG['num_columns']
    
    # Load dataset information from first directory
    _, _, llw, _, batch_name, num_data, endtag = get_dwllw(DIRECTORIES[0])
    print(f"Dataset: {batch_name}, Tag: {endtag}")
    
    mtrain = 900
    mtest  = 100

    # Load the singular values of the training data matrix
    ss_train = get_singular_values(batch_name, endtag, CONFIG['num_data_points'])
    
    # This only approximates the test base loss
    base_loss_test = ss_train[:llw] * np.sqrt(mtest / mtrain)
    
    # Setup plot
    xmin, xmax = -0.1 * llw, 1.1 * llw
    fig, axs = setup_figure(num_columns)
    
    # Find valid directories
    valid_dirs = find_valid_directories(DIRECTORIES)
    print(f"Found {len(valid_dirs)}/{len(DIRECTORIES)} valid directories")
    
    if len(valid_dirs) == 0:
        print("Error: No valid directories found!")
        return
    
    # Initialize tracking variables
    colors = get_colors()
    reference_values = []
    
    # Track global y-limits across all plots
    y_limits_all = {
        'train_min': ss_train[llw-1]**2 / mtrain,
        'train_max': ss_train[0]**2 / mtrain,
        'test_min': np.min(base_loss_test)**2 / mtest,
        'test_max': np.max(base_loss_test)**2 / mtest
    }
    
    # Process each directory
    for k, direc in enumerate(valid_dirs[:num_columns]):
        print(f"\nProcessing [{k+1}/{min(len(valid_dirs), num_columns)}]: {direc}")
        
        # Load experiment data
        data = load_experiment_data(direc)
        train_modes, test_modes = extract_mode_losses(data['modeloss_data'], llw)
        
        # Extract title from directory name
        title = parse_exponent_from_directory(direc)
        
        # Plot loss curves in top panel
        plot_loss_curves(
            axs[0], data['epochs'], data['train_loss'], data['test_loss'],
            colors[k], title, CONFIG['num_epochs_to_plot']
        )
        
        # Plot mode losses in bottom panels
        y_lim, last_epoch = plot_mode_losses(
            axs[1+2*k], axs[2+2*k], train_modes, test_modes, ss_train, base_loss_test,
            mtrain, mtest, llw, CONFIG['num_epochs_to_plot'], CONFIG['epoch_step'],
            colors[k], title, xmin, xmax
        )
        
        # Update global y-limits
        for key in y_lim:
            if 'min' in key:
                y_limits_all[key] = min(y_limits_all[key], y_lim[key])
            else:
                y_limits_all[key] = max(y_limits_all[key], y_lim[key])
        
        # Get reference values for standard configuration
        ref_vals = get_reference_values(
            direc, train_modes, test_modes, ss_train, mtrain, mtest, llw, last_epoch
        )
        reference_values.append(ref_vals)
    
    # Finalize and show plot
    finalize_plot(fig, axs, num_columns, llw, xmin, xmax, y_limits_all, reference_values)
    plt.show()


if __name__ == "__main__":
    main()
