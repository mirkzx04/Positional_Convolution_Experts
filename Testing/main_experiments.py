import os
import sys
import warnings
from pathlib import Path

os.environ['WANDB_SILENT'] = 'true'
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import wandb 
from Testing.Experiments.log_moe import log_loss_accuracy_moe, log_loss_router_moe, refining_models_log
from Testing.Experiments.log_dense_vs_moe import log_loss_accuracy_moe_vs_dense
# Aggiunti gli import per l'inferenza
from Testing.Experiments.eval_model import compute_params, inference_moe, inference_dense

print("PYTHON EXECUTABLE:", sys.executable)
print("WANDB VERSION:", wandb.__version__)

# Global configuration paths
PROJECT_PATH = 'mirkzx-sapienza-universit-di-roma/PCE'
DENSE_MODEL_PTH = f'{PROJECT_PATH}/oevtrijo'
BEST_MOE_PTH  = f'{PROJECT_PATH}/85yn0fv8'
MOE_16_PTH = f'{PROJECT_PATH}/w2tbu4t7'
MOE_K1_PTH = f'{PROJECT_PATH}/0nbk6a5l'
MOE_PB1_PTH = f'{PROJECT_PATH}/z1eavwvr'

TRAIN_PTH = 'training'
VAL_PTH = 'validation'
ROUTER_PTH = 'router-train'
X_AXIS = 'epoch'

# ANSI Color codes
C = '\033[96m'  
M = '\033[95m'  
Y = '\033[93m'  
G = '\033[92m'  
R = '\033[91m'  
W = '\033[97m'  
B = '\033[1m'   
RESET = '\033[0m'

# Initialize WandB API globally
wandb.login(key='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', relogin=False)
api = wandb.Api()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_menu():
    """Print the interactive CLI menu."""
    clear_screen()
    print(f"\n{C}{B}╔═════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{C}{B}║{RESET} {M}{B}            🚀 DEBUGGING SECTION FOR PCE PROJECT          {C}{B}║{RESET}")
    print(f"{C}{B}╚═════════════════════════════════════════════════════════════╝{RESET}\n")

    print(f" {Y}{B}★ MAIN MENU ★{RESET}\n")
    print(f"  {G}[1]{RESET} {B}MoE vs MoE{RESET}\n      {W}Comparative analysis: Loss & Accuracy between MoEs{RESET}\n")
    print(f"  {G}[2]{RESET} {B}Best MoE vs Dense{RESET}\n      {W}Direct comparison: Best MoE vs Dense baseline{RESET}\n")
    print(f"  {G}[3]{RESET} {B}Final Checkpoint Eval{RESET}\n      {W}Testing: Evaluate parameters and inference performance{RESET}\n")
    print(f"  {G}[4]{RESET} {B}Router MoE metrics{RESET}\n      {W}Deep dive: Detailed router metrics per MoE{RESET}\n")
    print(f"  {G}[5]{RESET} {B}Best MoE vs Refine models{RESET}\n      {W}Architecture tweaking: Best MoE vs Refined versions{RESET}\n")
    print(f"  {R}[0]{RESET} {B}Exit CLI{RESET}\n")
    print(f"{C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

def print_params_table(models_params):
    """Format and print the model parameters table."""
    print(f"\n{Y}{B}📊 MODEL PARAMETERS REPORT{RESET}")
    print(f"{C}╔════════════╦═══════════════════════╦═══════════════════════╗{RESET}")
    print(f"{C}║{RESET}{B}  EXPERTS   {C}║{RESET}{B}   TOTAL PARAMS        {C}║{RESET}{B}  TRAINABLE PARAMS     {C}║{RESET}")
    print(f"{C}╠════════════╬═══════════════════════╬═══════════════════════╣{RESET}")

    for exp, data in models_params.items():
        total = f"{data['total_params']:,}".replace(',', '.')
        trainable = f"{data['trainable_params']:,}".replace(',', '.')
        print(f"{C}║{RESET} {W}{exp:<10} {C}║{RESET} {G}{total:>21} {C}║{RESET} {M}{trainable:>21} {C}║{RESET}")

    print(f"{C}╚════════════╩═══════════════════════╩═══════════════════════╝{RESET}\n")

def print_inference_table(moe_results, dense_results):
    """Format and print the inference results table."""
    print(f"\n{Y}{B}⚡ INFERENCE PERFORMANCE REPORT{RESET}")
    print(f"{M}{B}ℹ️  INFO: The subset used for inference is a subset of the validation set.{RESET}\n")
    
    print(f"{C}╔════════════╦══════════════════╦══════════════════╦══════════════════╗{RESET}")
    print(f"{C}║{RESET}{B}  MODEL     {C}║{RESET}{B}   ACCURACY (%)   {C}║{RESET}{B} AVG LATENCY (ms) {C}║{RESET}{B} STD LATENCY (ms) {C}║{RESET}")
    print(f"{C}╠════════════╬══════════════════╬══════════════════╬══════════════════╣{RESET}")

    # Stampa i risultati Dense
    dense_data = dense_results.get('dense', {})
    acc = f"{dense_data.get('Accuracy (%)', 0):.2f}"
    avg_lat = f"{dense_data.get('Avg Latency (ms)', 0):.2f}"
    std_lat = f"{dense_data.get('Std Latency (ms)', 0):.2f}"
    print(f"{C}║{RESET} {R}Dense (18){RESET} {C}║{RESET} {G}{acc:>16} {C}║{RESET} {M}{avg_lat:>16} {C}║{RESET} {Y}{std_lat:>16} {C}║{RESET}")

    # Linea separatrice tra Dense e MoE
    print(f"{C}╠════════════╬══════════════════╬══════════════════╬══════════════════╣{RESET}")

    # Stampa i risultati MoE
    for exp, data in moe_results.items():
        name = f"MoE ({exp})"
        acc = f"{data.get('Accuracy (%)', 0):.2f}"
        avg_lat = f"{data.get('Avg Latency (ms)', 0):.2f}"
        std_lat = f"{data.get('Std Latency (ms)', 0):.2f}"
        print(f"{C}║{RESET} {W}{name:<10} {C}║{RESET} {G}{acc:>16} {C}║{RESET} {M}{avg_lat:>16} {C}║{RESET} {Y}{std_lat:>16} {C}║{RESET}")

    print(f"{C}╚════════════╩══════════════════╩══════════════════╩══════════════════╝{RESET}\n")


def main():
    menu = True

    while menu:
        draw_menu()

        try:
            choice = int(input(f"\n  {Y}➤ Enter section code:{RESET}  "))
        except ValueError:
            print(f"{R}Please enter a valid number!{RESET}")
            input(f"\n{Y}➤ Press ENTER to continue...{RESET}")
            continue

        if choice == 1:
            print(f"\n{C}{B}  [⚙️] Processing: Plotting MoE vs MoE...{RESET}")
            log_loss_accuracy_moe(PROJECT_PATH, X_AXIS, TRAIN_PTH, VAL_PTH, api)
            print(f"{G}{B}  [✔] Done!{RESET}\n")
        
        elif choice == 2: 
            print(f"\n{M}{B}  [⚔️] Processing: Plotting Best MoE vs Dense...{RESET}")
            log_loss_accuracy_moe_vs_dense(BEST_MOE_PTH, DENSE_MODEL_PTH, TRAIN_PTH, VAL_PTH, api)
            print(f"{G}{B}  [✔] Done!{RESET}\n")
        
        elif choice == 3:
            print(f"\n{C}{B}  [🧮] Calculating model parameters...{RESET}")
            models_params = compute_params()
            print_params_table(models_params)
            
            print(f"{M}{B}  [🚀] Running inference on models (this might take a moment)...{RESET}")
            moe_results = inference_moe()
            dense_results = inference_dense()
            print_inference_table(moe_results, dense_results)

        elif choice == 4:
            print(f"\n{M}{B}  [🔀] Processing: Plotting Router metrics...{RESET}")
            log_loss_router_moe(PROJECT_PATH, X_AXIS, ROUTER_PTH, api)
            print(f"{G}{B}  [✔] Done!{RESET}\n")
        
        elif choice == 5: 
            print(f"\n{C}{B}  [🔬] Processing: Extracting refined models logs...{RESET}")
            refining_models_log(MOE_16_PTH, MOE_K1_PTH, MOE_PB1_PTH, X_AXIS, TRAIN_PTH, VAL_PTH, api)
            print(f"{G}{B}  [✔] Done!{RESET}\n")

        elif choice == 0:
            print(f"\n{R}{B}  [👋] Exiting program. See you soon!{RESET}\n")
            menu = False
            
        if menu:
            input(f"\n{Y}➤ Press ENTER to return to the main menu...{RESET}")

main()