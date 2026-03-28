import matplotlib.pyplot as plt

MOE_COLOR = '#1f77b4'
DENSE_COLOR = '#d62728'

TITLES = [
    ["Training Loss", "Validation Loss"], 
    ["Training Top-1 Accuracy", "Validation Top-1 Accuracy"]
]
YLABELS = [
    ["Loss", "Loss"], 
    ["Accuracy (%)", "Accuracy (%)"]
]

def apply_plot_style():
    """Set global matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",        
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,      
        "axes.spines.right": False,   
    })

def fetch_and_plot(api, run_path, metric_mapping, color, linestyle, base_label):
    """Fetch wandb data and plot available metrics."""
    run = api.run(run_path)
    df = run.history(samples=None)
    df['epoch'] = df['epoch'].ffill().bfill()
    
    label = base_label
    # Clean MoE specific name if present
    if "Test-Tiny-test" in run.name:
        clean_name = run.name.replace("Test-Tiny-test ", "")
        label = f"{base_label} ({clean_name})"

    for metric, ax in metric_mapping.items():
        if metric in df.columns:
            df_clean = df.dropna(subset=['epoch', metric])
            if not df_clean.empty:
                ax.plot(
                    df_clean['epoch'], df_clean[metric], 
                    color=color, linestyle=linestyle, 
                    linewidth=2.0, alpha=0.9, label=label
                )

def log_loss_accuracy_moe_vs_dense(BEST_MOE_PTH, DENSE_MODEL_PTH, TRAIN_PTH, VAL_PTH, api):
    apply_plot_style()
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    moe_mapping = {
        f'{TRAIN_PTH}/train_total_loss': axs[0, 0],
        f'{VAL_PTH}/val_class_loss': axs[0, 1],
        f'{TRAIN_PTH}/train_top1': axs[1, 0],
        f'{VAL_PTH}/val_top1': axs[1, 1]
    }
    
    dense_mapping = {
        f'{TRAIN_PTH}/train_loss': axs[0, 0],
        f'{VAL_PTH}/val_loss': axs[0, 1],
        f'{TRAIN_PTH}/train_top1': axs[1, 0],
        f'{VAL_PTH}/val_top1': axs[1, 1]
    }

    # Plot lines
    fetch_and_plot(api, BEST_MOE_PTH, moe_mapping, MOE_COLOR, '-', "Best MoE")
    fetch_and_plot(api, DENSE_MODEL_PTH, dense_mapping, DENSE_COLOR, '--', "Dense Baseline (ResNet18)")

    # Format subplots
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]
            ax.set_xlabel("Epoch")
            ax.set_ylabel(YLABELS[row][col])
            ax.set_title(TITLES[row][col])
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Smart legend to avoid duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, edgecolor='black')

    plt.suptitle("Best PCE Model vs Dense Baseline", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()