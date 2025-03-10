import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import re
import glob

# Set consistent plot style
plt.style.use('seaborn-v0_8-whitegrid')
BLUE_PALETTE = ['#2978B5', '#64A0D0', '#8FB3D9', '#A3C4DC']
FIG_SIZE = (10, 6)
SAVE_PLOTS = True
OUTPUT_DIR = './flad/plot/figures/'

def setup_environment():
    """Setup the environment for plotting"""
    # Create output directory if it doesn't exist
    if SAVE_PLOTS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Set consistent font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def process_csv_files():
    """Process CSV files and extract execution times"""
    path = os.getcwd()
    files = glob.glob(path + "./flad/plot/pipelines*.csv")
    results = {}
    
    print("Processing CSV files...")
    for file in files:
        try:
            df = pd.read_csv(file)
            cluster_size = len(df)
            avg_execution_time = df["Execution_Time"].mean()
            results[cluster_size] = avg_execution_time
            
            print(f"File: {os.path.basename(file)}")
            print(f"  Cluster Size: {cluster_size}")
            print(f"  Average Execution Time: {avg_execution_time:.4f}s")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    print(f"Processed {len(files)} files")
    return results

def plot_execution_times():
    """Plot execution times comparing Base vs Swift"""
    # Data for execution time comparison
    base_results = {3: 1081.6197, 5: 2494.7078, 7: 910.5060, 9: np.nan}  # Use NaN for incomplete
    swift_results = {3: 1064.1329, 5: 2339.8869, 7: 899.6769, 9: 901.1903}
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Extract data
    x_values = list(swift_results.keys())
    x = np.array(x_values)
    base_values = [base_results.get(k, np.nan) for k in x_values]
    swift_values = list(swift_results.values())
    
    # Plot Swift data
    swift_bars = ax.bar(x + 0.2, swift_values, width=0.4, align='center', 
                        label='Swift', color=BLUE_PALETTE[0], edgecolor='black', linewidth=1)
    
    # Plot completed Base data
    completed = ~np.isnan(base_values)
    if any(completed):
        base_bars = ax.bar(x[completed] - 0.2, np.array(base_values)[completed], width=0.4, 
                           align='center', label='Base', color=BLUE_PALETTE[2], 
                           edgecolor='black', linewidth=1)
    
    # Plot incomplete Base data
    incomplete = np.isnan(base_values)
    if any(incomplete):
        inc_bars = ax.bar(x[incomplete] - 0.2, [100] * sum(incomplete), width=0.4, align='center',
                          hatch='////', color='lightgray', edgecolor='black', linewidth=1,
                          label='Base (Did not complete)')
        
        
    
    # Customize plot
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Average Execution Time (s)')
    ax.set_title('Execution Time Comparison by Cluster Size', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}execution_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimization_times():
    """Plot optimization times comparing Phase1 vs Phase2"""
    base_optimization = {3: 0.01, 5: 0.01, 7: 0.01, 9: np.nan}  # Use NaN for incomplete
    swift_optimization = {3: 0.04, 5: 0.29, 7: 0.4, 9: 0.25}
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Extract data
    x_values = list(swift_optimization.keys())
    x = np.array(x_values)
    base_values = [base_optimization.get(k, np.nan) for k in x_values]
    
    # Plot Swift/Phase2 data
    swift_bars = ax.bar(x + 0.2, swift_optimization.values(), width=0.4, align='center', 
                        label='Phase2', color=BLUE_PALETTE[0], edgecolor='black', linewidth=1)
    
    # Plot completed Base/Phase1 data
    completed = ~np.isnan(base_values)
    if any(completed):
        base_bars = ax.bar(x[completed] - 0.2, np.array(base_values)[completed], width=0.4, 
                         align='center', label='Phase1', color=BLUE_PALETTE[2], 
                         edgecolor='black', linewidth=1)
    
    # Plot special bar for incomplete Phase1 data
    incomplete = np.isnan(base_values)
    if any(incomplete):
        ax.bar(x[incomplete] - 0.2, [0.01] * sum(incomplete), width=0.4, align='center',
              hatch='////', color='lightgray', edgecolor='black', linewidth=1,
              label='Phase1 (Did not complete)')
        

    
    # Customize plot
    ax.set_xlabel('Problem Scale')
    ax.set_ylabel('Average Optimization Time (s)')
    ax.set_title('Optimization Time Comparison by Problem Scale', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}optimization_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_size_comparison():
    """Plot execution times by model size"""
    base_model_size = {"5.55 GB": 1390.7427, "11.10 GB": 2998.2761, "14.01 GB": np.nan}
    swift_model_size = {"5.55 GB": 1252.8087, "11.10 GB": 2911, "14.01 GB": 2944.5439}
    
    # Get keys and positions
    model_sizes = list(swift_model_size.keys())
    bar_position = np.arange(len(model_sizes))
    base_values = [base_model_size.get(k, np.nan) for k in model_sizes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot Swift data
    swift_bars = ax.bar(bar_position + 0.2, swift_model_size.values(), width=0.4, align='center', 
                      label='Swift', color=BLUE_PALETTE[0], edgecolor='black', linewidth=1)
    
    # Plot completed Base data
    completed = ~np.isnan(base_values)
    completed_positions = bar_position[completed]
    completed_values = np.array(base_values)[completed]
    if len(completed_values) > 0:
        base_bars = ax.bar(completed_positions - 0.2, completed_values, width=0.4, align='center', 
                         label='Base', color=BLUE_PALETTE[2], edgecolor='black', linewidth=1)
    
    # Plot special bar for incomplete data
    incomplete = np.isnan(base_values)
    if any(incomplete):
        incomplete_positions = bar_position[incomplete]
        ax.bar(incomplete_positions - 0.2, [100] * len(incomplete_positions), width=0.4, align='center',
              hatch='////', color='lightgray', edgecolor='black', linewidth=1, 
              label='Base (Did not complete)')
        
    
    
    
    # Customize plot
    plt.xticks(bar_position, model_sizes)
    plt.xlabel('Model Size')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Execution Time Comparison by Model Size', fontsize=16, pad=20)
    plt.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}model_size_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_recovery_time():
    """Plot recovery time comparison"""
    Recovery_time = {'Relaunch': 50, 'Elastic': 30, 'FHDP': 6}
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot with improved aesthetics
    bars = ax.bar(
        list(Recovery_time.keys()), 
        list(Recovery_time.values()),
        width=0.6,
        color=BLUE_PALETTE[0:3],
        edgecolor='black',
        linewidth=1
    )
    
    # Highlight SWIFT with hatching
    bars[2].set_hatch('\\\\')

    # Customize plot
    ax.set_title('Recovery Time Comparison Between Methods', fontsize=16, pad=20)
    ax.set_xlabel('Recovery Method', fontsize=14, labelpad=10)
    ax.set_ylabel('Recovery Time (s)', fontsize=14, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(Recovery_time.values()) * 1.15)  # Add 15% headroom
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}recovery_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_VE_throughout():
    """
    Plot throughput comparison (samples/min) between different pipeline configurations:
    - Standalone: The entire model on a single device
    - FHDP: Model split at encoder beginning
    - Random: Model split at decoder beginning
    
    All configurations use batch size 2*2
    """
    # Throughput data in samples/minute
    VE_Throughput = {'Standalone': 21, 'FHDP': 14.75, 'Random': 9.627}
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot with improved aesthetics
    bars = ax.bar(
        list(VE_Throughput.keys()),
        list(VE_Throughput.values()),
        width=0.6,
        color=[BLUE_PALETTE[0], BLUE_PALETTE[1], BLUE_PALETTE[2]],
        edgecolor='black',
        linewidth=1
    )
    
    # Customize plot
    ax.set_title('Vision Encoder Throughput Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Schemes', fontsize=14, labelpad=10)
    ax.set_ylabel('Throughput (samples/minute)', fontsize=14, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
     
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note
    
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}vision_encoder_throughput.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_VE_mem():
    """
    Plot memory usage comparison (GB) between different pipeline configurations:
    - Standalone: The entire model on a single device
    - Pipeline 1: Model split at encoder beginning
    - Pipeline 2: Model split at decoder beginning
    
    Values represent average memory consumption per device.
    """
    # Memory usage data in GB (per device)
    memory_usage = {
        'Standalone': 6.5,  # 3.1+3.4
        'FHDP': 3.1,  # (3.1+3.1)/2
        'Random': 3.3   # (3.1+3.5)/2
    }
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot with improved aesthetics
    bars = ax.bar(
        list(memory_usage.keys()),
        list(memory_usage.values()),
        width=0.6,
        color=[BLUE_PALETTE[0], BLUE_PALETTE[1], BLUE_PALETTE[2]],
        edgecolor='black',
        linewidth=1
    )
    
    
    # Customize plot
    ax.set_title('Memory Usage Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Schemes', fontsize=14, labelpad=10)
    ax.set_ylabel('Memory Usage per Device (GB)', fontsize=14, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note
    
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}vision_encoder_memory.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    

if __name__ == "__main__":
    setup_environment()
    # Process CSV files if needed
    # results = process_csv_files()
    
    # Generate all plots
    #plot_execution_times()
    #plot_optimization_times()
    #plot_model_size_comparison()
    #plot_recovery_time()
    plot_VE_throughout()
    plot_VE_mem()
    
    print(f"{'Plots saved to '+OUTPUT_DIR if SAVE_PLOTS else 'Plots displayed but not saved'}")
