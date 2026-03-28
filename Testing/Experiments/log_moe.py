import matplotlib.pyplot as plt

COLOR_PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

def apply_plot_style():
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

def clean_run_name(name):
    return name.replace("Test-Tiny-test ", "")

def fetch_and_prep_history(run, x_axis, samples=None):
    df = run.history(samples=samples)
    if df.empty:
        return df, None

    x_col = x_axis if x_axis in df.columns else 'epoch'
    if x_col not in df.columns:
        x_col = '_step'
    
    df[x_col] = df[x_col].ffill().bfill()
    return df, x_col

def apply_smart_legend(ax, fontsize=11):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, edgecolor='black', fontsize=fontsize)

def format_subplot(ax, x_label, y_label, title):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)

def log_loss_accuracy_moe(PROJECT_PATH, X_AXIS, TRAIN_PTH, VAL_PTH, api):
    apply_plot_style()
    run_colors = {}
    color_idx = 0

    loss_metrics = [f'{TRAIN_PTH}/train_class_loss', f'{VAL_PTH}/val_class_loss']
    acc_metrics = [f'{TRAIN_PTH}/train_top1', f'{VAL_PTH}/val_top1']
    all_metrics = loss_metrics + acc_metrics
    
    runs = api.runs(PROJECT_PATH)
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
        
    for run in runs: 
        if run.state not in ['finished', 'running']: 
            continue
        if any(skip in run.name for skip in ['ResNet18', 'Kernel', 'post_block']):
            continue

        if run.name not in run_colors:
            run_colors[run.name] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            color_idx += 1
            
        color = run_colors[run.name]
        clean_name = clean_run_name(run.name)
        df, x_col = fetch_and_prep_history(run, X_AXIS)

        if df.empty:
            continue

        for metric in all_metrics:
            if metric not in df.columns:
                continue

            df_clean = df.dropna(subset=[x_col, metric])
            if df_clean.empty:
                continue

            # Determine subplot location based on metric
            if metric == loss_metrics[0]: ax = axs[0, 0]
            elif metric == loss_metrics[1]: ax = axs[0, 1]
            elif metric == acc_metrics[0]: ax = axs[1, 0]
            elif metric == acc_metrics[1]: ax = axs[1, 1]

            ax.plot(df_clean[x_col], df_clean[metric], color=color, linewidth=1.8, alpha=0.9, label=clean_name)

    titles = [["Training Loss", "Validation Loss"], ["Training Top-1 Accuracy", "Validation Top-1 Accuracy"]]
    ylabels = [["Loss", "Loss"], ["Accuracy (%)", "Accuracy (%)"]]
    x_label = X_AXIS.capitalize() if X_AXIS else "Epoch"
    
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]
            format_subplot(ax, x_label, ylabels[row][col], titles[row][col])
            apply_smart_legend(ax)

    plt.suptitle("Positional Convolution Experts (PCE) Performance", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

def log_loss_router_moe(PROJECT_PATH, X_AXIS, ROUTER_PTH, api):
    apply_plot_style()
    run_colors = {}
    color_idx = 0

    metrics_info = {
        f'{ROUTER_PTH}/entropy_norm_mean': {'title': 'Normalized Entropy', 'ylabel': 'Entropy (0-1)'},
        f'{ROUTER_PTH}/imbalance_mean': {'title': 'Imbalance Mean (Max/Min)', 'ylabel': 'Ratio'},
        f'{ROUTER_PTH}/drop_rate': {'title': 'Drop Rate', 'ylabel': 'Drop Rate (0-1)'},
        f'{ROUTER_PTH}/mean_capacity_ratio_mean': {'title': 'Capacity Efficiency', 'ylabel': 'Efficiency Ratio'},
        f'{ROUTER_PTH}/spec_entropy_mean': {'title': 'Router Spec Entropy', 'ylabel': 'Entropy'}
    }
    
    print(f"Searching runs in: {PROJECT_PATH}...")
    runs = api.runs(PROJECT_PATH)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axes = axs.flatten()
    fig.delaxes(axes[5]) 
        
    for run in runs: 
        if run.state not in ['finished', 'running']: 
            continue
        if 'ResNet18' in run.name:
            continue

        if run.name not in run_colors:
            run_colors[run.name] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            color_idx += 1
            
        color = run_colors[run.name]
        clean_name = clean_run_name(run.name)

        print(f"\n---> Analyzing run: {clean_name}")
        df, x_col = fetch_and_prep_history(run, X_AXIS, samples=2000)
        
        if df.empty:
            print(f"  ⚠️ No history data for {clean_name}")
            continue

        for idx, metric in enumerate(metrics_info.keys()):
            ax = axes[idx]
            if metric not in df.columns:
                print(f"  ❌ Missing metric: {metric}")
                continue
                
            df_plot = df[[x_col, metric]].dropna()
            if df_plot.empty:
                print(f"  ⚠️ All NaN data for: {metric}")
                continue
                
            ax.plot(df_plot[x_col], df_plot[metric], color=color, linewidth=1.8, alpha=0.9, label=clean_name)
            print(f"  ✅ Plotted: {metric}")

    x_label = X_AXIS.capitalize() if X_AXIS else "Epoch"
    for idx, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[idx]
        format_subplot(ax, x_label, info['ylabel'], info['title'])
        apply_smart_legend(ax)

    plt.suptitle("Router MoE Metrics Comparison", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

def refining_models_log(MOE_16_PTH, MOE_K1_PTH, MOE_PB1_PTH, X_AXIS, TRAIN_PTH, VAL_PTH, api):
    apply_plot_style()
    fig, axs = plt.subplots(3, 1, figsize=(5, 7.5))
    
    models_to_plot = [
        {'path': MOE_16_PTH,  'color': '#1f77b4'},
        {'path': MOE_K1_PTH,  'color': '#ff7f0e'},
        {'path': MOE_PB1_PTH, 'color': '#2ca02c'}
    ]

    metrics_config = {
        f'{TRAIN_PTH}/train_class_loss': {'ax': axs[0], 'title': 'Training Class Loss', 'ylabel': 'Loss'},
        f'{VAL_PTH}/val_class_loss':     {'ax': axs[1], 'title': 'Validation Class Loss', 'ylabel': 'Loss'},
        f'{VAL_PTH}/val_top1':           {'ax': axs[2], 'title': 'Validation Top-1 Accuracy', 'ylabel': 'Accuracy (%)'}
    }

    for model_info in models_to_plot:
        try:
            run = api.run(model_info['path'])
            clean_name = clean_run_name(run.name)
            color = model_info['color']
            
            df, x_col = fetch_and_prep_history(run, X_AXIS)
            if df.empty:
                continue

            for metric, config in metrics_config.items():
                if metric in df.columns:
                    df_clean = df.dropna(subset=[x_col, metric])
                    if not df_clean.empty:
                        ax = config['ax']
                        style = '--' if 'val_' in metric else '-'
                        ax.plot(df_clean[x_col], df_clean[metric], color=color, linestyle=style, linewidth=2.0, alpha=0.9, label=clean_name)
                        
        except Exception as e:
            print(f"⚠️ Failed to process run {model_info['path']}: {e}")

    x_label = X_AXIS.capitalize() if X_AXIS else "Epoch"
    for idx, (metric, config) in enumerate(metrics_config.items()):
        ax = config['ax']
        format_subplot(ax, x_label, config['ylabel'], config['title'])
        
        # Legend only on the first plot to save space
        if idx == 0:
            apply_smart_legend(ax, fontsize=9)

    plt.suptitle("Refining Models Comparison", fontsize=16, y=1.02)
    plt.tight_layout(pad=0.5, h_pad=0.5)
    # plt.savefig("img/MoE_16_refining_and_not.png", bbox_inches='tight', dpi=300)
    plt.show()