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
FIG_SIZE = (4, 5)  # Updated to specified dimensions
SAVE_PLOTS = True
OUTPUT_DIR = './flad/plot/figures/'

def setup_environment():
    """Setup the environment for plotting"""
    # Create output directory if it doesn't exist
    if SAVE_PLOTS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Set consistent font sizes with updated values
    plt.rcParams['font.size'] = 16  # Base font size
    plt.rcParams['axes.labelsize'] = 25  # Increased for x and y labels
    plt.rcParams['axes.titlesize'] = 25  # Increased for title
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16  # Increased for legend

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
    base_results = {3: 1608.9600, 5: 2494.7078, 7: 3436.8156, 9: np.nan}  # Use NaN for incomplete
    swift_results = {3: 1367.0400, 5: 2339.8869, 7: 2768.0314, 9: 2950.6813}
    
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
        inc_bars = ax.bar(x[incomplete] - 0.2, [1100] * sum(incomplete), width=0.4, align='center',
                          hatch='////', color='lightgray', edgecolor='black', linewidth=1,
                          label='Failed')
        
        
    
    # Customize plot
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Avg Execution Time (s)')
    ax.set_ylim(1000, 3500)  # Set y-axis limit to 4000s
    #ax.set_title('Execution Time Comparison by Cluster Size', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}execution_time_comparison.png", dpi=400, bbox_inches='tight')
    plt.show()

def plot_optimization_times():
    """Plot optimization times comparing Phase1 vs Phase2"""
    base_optimization = {'3': 0.01, '4': 0.01,'5': 0.01, '7': 0.01}  # Use NaN for incomplete
    swift_optimization = {'3': 0.04,'4': 0.04,'5': 0.29, '7': 0.4}
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Extract data
    x_values = list(swift_optimization.keys())
    
    # Convert string keys to numeric positions for plotting
    x_positions = np.arange(len(x_values))
    base_values = [base_optimization.get(k, np.nan) for k in x_values]
    swift_values = list(swift_optimization.values())
    
    # Plot Swift/Phase2 data - use numeric positions instead of string keys
    swift_bars = ax.bar(x_positions + 0.2, swift_values, width=0.4, align='center', 
                        label='Phase2', color=BLUE_PALETTE[0], edgecolor='black', linewidth=1)
    
    # Plot completed Base/Phase1 data
    completed = ~np.isnan(base_values)
    if any(completed):
        completed_positions = x_positions[completed]
        completed_values = np.array(base_values)[completed]
        base_bars = ax.bar(completed_positions - 0.2, completed_values, width=0.4, 
                         align='center', label='Phase1', color=BLUE_PALETTE[2], 
                         edgecolor='black', linewidth=1)
    
    # Plot special bar for incomplete Phase1 data
    incomplete = np.isnan(base_values)
    if any(incomplete):
        incomplete_positions = x_positions[incomplete]
        ax.bar(incomplete_positions - 0.2, [0.01] * sum(incomplete), width=0.4, align='center',
              hatch='////', color='lightgray', edgecolor='black', linewidth=1,
              label='Failed')
    
    # Customize plot
    ax.set_xlabel('Problem Scale')
    ax.set_ylabel('Avg Optimization Time (s)')
    
    # Set x-ticks at numeric positions but with string labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values)
    
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}optimization_time_comparison.png", dpi=400, bbox_inches='tight')
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
              label='Failed')
        
    
    
    
    # Customize plot
    plt.xticks(bar_position, model_sizes)
    plt.xlabel('Model Size')
    plt.ylabel('Avg Execution Time (s)')
    #plt.title('Execution Time Comparison by Model Size', fontsize=16, pad=20)
    plt.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}model_size_comparison.png", dpi=400, bbox_inches='tight')
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
    #ax.set_title('Recovery Time Comparison Between Methods', fontsize=16, pad=20)
    ax.set_xlabel('Recovery Method', fontsize=24, labelpad=10)
    ax.set_ylabel('Recovery Time (s)', fontsize=24, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(Recovery_time.values()) * 1.15)  # Add 15% headroom
    
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}recovery_time_comparison.png", dpi=400, bbox_inches='tight')
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
    #ax.set_title('Vision Encoder Throughput Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Schemes', fontsize=24, labelpad=10)
    ax.set_ylabel('Throughput (samples/min)', fontsize=24, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
     
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note
    
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}vision_encoder_throughput.png", dpi=400, bbox_inches='tight')
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
    #ax.set_title('Memory Usage Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Schemes', fontsize=24, labelpad=10)
    ax.set_ylabel('Avg Memory Usage (GB)', fontsize=24, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note
    
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}vision_encoder_memory.png", dpi=400, bbox_inches='tight')
    plt.show()
    
def plot_model_architecture():
    """
    Plot a simplified visualization of the Vision-Encoder model architecture.
    This diagram shows the high-level structure rather than all individual layers.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from matplotlib.path import Path
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors
    COLORS = {
        'rgb': '#64A0D0',      # Light blue
        'lidar': '#8FB3D9',    # Mid blue
        'encoder': '#2978B5',  # Dark blue
        'decoder': '#2978B5',  # Dark blue
        'heads': '#A3C4DC',    # Very light blue
        'arrow': '#333333',    # Dark gray
        'text': '#000000',     # Black
        'border': '#000000',   # Black
    }
    
    # Helper function to draw blocks
    def draw_block(x, y, width, height, label, color, sublabels=None):
        rect = patches.FancyBboxPatch(
            (x, y), width, height, 
            boxstyle=patches.BoxStyle("Round", pad=0.6),
            facecolor=color, edgecolor=COLORS['border'], linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        
        # Add main label
        ax.text(x + width/2, y + height/2, label, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add sublabels if provided
        if sublabels:
            y_offset = height / (len(sublabels) + 1)
            for i, sublabel in enumerate(sublabels):
                ax.text(x + width/2, y + (i+1)*y_offset, 
                        sublabel, ha='center', va='center', fontsize=9)
    
    # Helper function to draw arrows
    def draw_arrow(start, end, label=None):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(facecolor=COLORS['arrow'], shrink=0.05, 
                                  width=1.5, headwidth=8, alpha=0.9))
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color=COLORS['text'])
    
    # Draw title
    ax.text(50, 95, 'Vision-Encoder Model Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Draw input nodes
    draw_block(10, 80, 20, 10, "RGB Input", "#e0e0e0", ["3×H×W"])
    draw_block(70, 80, 20, 10, "LiDAR Input", "#e0e0e0", ["N×9 points"])
    draw_block(40, 80, 20, 10, "Velocity", "#e0e0e0", ["1D"])
    
    # Draw backbone blocks
    draw_block(10, 65, 20, 10, "RGB Backbone", COLORS['rgb'], ["ResNet"])
    draw_block(70, 65, 20, 10, "LiDAR Backbone", COLORS['lidar'], ["PointPillar"])
    
    # Draw feature processing
    draw_block(10, 50, 20, 10, "RGB Features", COLORS['rgb'], ["2048→256"])
    draw_block(70, 50, 20, 10, "LiDAR Features", COLORS['lidar'], ["192→256"])
    draw_block(40, 50, 20, 10, "Velocity Embed", COLORS['rgb'], ["1→256"])
    
    # Draw encoder block
    draw_block(40, 35, 20, 10, "Transformer\nEncoder", COLORS['encoder'], ["1 layer"])
    
    # Draw decoder block
    draw_block(40, 20, 20, 10, "Transformer\nDecoder", COLORS['decoder'], ["3 layers"])
    
    # Draw prediction heads
    draw_block(10, 5, 15, 10, "Traffic\nPrediction", COLORS['heads'])
    draw_block(30, 5, 15, 10, "Waypoints", COLORS['heads'], ["GRU"])
    draw_block(50, 5, 15, 10, "Traffic Light", COLORS['heads'])
    draw_block(70, 5, 15, 10, "Stop Sign", COLORS['heads'])
    
    # Draw arrows
    # Input to backbone
    draw_arrow((20, 80), (20, 75))
    draw_arrow((80, 80), (80, 75))
    draw_arrow((50, 80), (50, 50))
    
    # Backbone to features
    draw_arrow((20, 65), (20, 60))
    draw_arrow((80, 65), (80, 60))
    
    # Features to encoder
    draw_arrow((20, 50), (40, 40))
    draw_arrow((50, 50), (45, 40))
    draw_arrow((80, 50), (50, 40))
    
    # Encoder to decoder
    draw_arrow((50, 35), (50, 30))
    
    # Decoder to heads
    draw_arrow((40, 20), (17.5, 15))
    draw_arrow((43, 20), (37.5, 15))
    draw_arrow((47, 20), (57.5, 15))
    draw_arrow((50, 20), (77.5, 15))
    
    # Add a note about split points
    ax.text(5, 95, "Pipeline Split Points:", ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Draw split point indicators
    # Encoder beginning
    ax.axhline(y=42, xmin=0.05, xmax=0.95, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(95, 42, "Encoder\nBeginning", ha='right', va='center', color='red', fontsize=10)
    
    # Decoder beginning
    ax.axhline(y=27, xmin=0.05, xmax=0.95, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(95, 27, "Decoder\nBeginning", ha='right', va='center', color='blue', fontsize=10)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}model_architecture.png", dpi=400, bbox_inches='tight')
    plt.show()

def plot_traffic_light_training_progress():
    """
    Plot training progress using data from 100roundVEmodel.csv
    Shows just traffic lights accuracy metrics over training rounds
    Samples data every 5 rounds to reduce visual clutter
    """
    # Read the CSV file
    csv_path = os.path.join(os.path.dirname(os.getcwd()), 'D:/EXP/pippyoldversion/flad/used_with_stats.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found. Looking for file in current directory...")
        try:
            df = pd.read_csv('100roundVEmodel.csv')
        except FileNotFoundError:
            print("Error: 100roundVEmodel.csv not found in current directory either.")
            return
    
    # Sample the data every 5 points
    sampled_df = df[(df['round'] % 2 == 0) & (df['round'] < 70)]
    
    # Also include the first round if not already included
    if not sampled_df.empty and sampled_df.iloc[0]['round'] != 1:
        first_row = df.iloc[0:1]
        sampled_df = pd.concat([first_row, sampled_df])
    
    # Create figure
    plt.figure(figsize=FIG_SIZE)
    
    # Plot traffic lights accuracy
    plt.plot(sampled_df['round'], sampled_df['traffic_lights_acc'], 'o-', color=BLUE_PALETTE[0], 
            label='Traffic Light Accuracy', markersize=5)
    
    # Add error bands using standard deviation
    plt.fill_between(sampled_df['round'], 
                    sampled_df['traffic_lights_acc'] - sampled_df['traffic_lights_std'], 
                    sampled_df['traffic_lights_acc'] + sampled_df['traffic_lights_std'], 
                    alpha=0.2, color=BLUE_PALETTE[0])
    
    # Set labels and title with updated font sizes
    plt.xlabel('Round', fontdict={'size': 24})
    plt.ylabel('Accuracy', fontdict={'size': 24})
    #plt.title('Traffic Light Detection Accuracy Over Training', fontdict={'size': 25}, pad=20)
    
    # Set y-axis limits
    plt.ylim(0.4, 1.05)
    
    # Add gridlines and legend with updated styles
    plt.grid(color='silver', linestyle='--', linewidth=1, alpha=0.3)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}traffic_light_accuracy_sampled.png", dpi=400, bbox_inches='tight')
    
    plt.show()

def plot_stop_sign_training_progress():
    """
    Plot training progress using data from 100roundVEmodel.csv
    Shows just traffic lights accuracy metrics over training rounds
    Samples data every 5 rounds to reduce visual clutter
    """
    # Read the CSV file
    csv_path = os.path.join(os.path.dirname(os.getcwd()), '100roundVEmodel.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found. Looking for file in current directory...")
        try:
            df = pd.read_csv('100roundVEmodel.csv')
        except FileNotFoundError:
            print("Error: 100roundVEmodel.csv not found in current directory either.")
            return
    
    # Sample the data every 5 points
    # Fix: Use proper pandas boolean operations with & (and) operator instead of Python's 'and'
    sampled_df = df[(df['round'] % 2 == 0) & (df['round'] < 1001)]  # Get rows where round is divisible by 5 and < 65
    
    # Also include the first round if not already included
    if not sampled_df.empty and sampled_df.iloc[0]['round'] != 1:
        first_row = df.iloc[0:1]
        sampled_df = pd.concat([first_row, sampled_df])
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot traffic lights accuracy
    ax.plot(sampled_df['round'], sampled_df['stop_sign_acc'], 'o-', color=BLUE_PALETTE[0], 
            label='Traffic Light Accuracy', markersize=5)
    
    # Add error bands using standard deviation
    ax.fill_between(sampled_df['round'], 
                    sampled_df['stop_sign_acc'] - sampled_df['stop_sign_std'], 
                    sampled_df['stop_sign_acc'] + sampled_df['stop_sign_std'], 
                    alpha=0.2, color=BLUE_PALETTE[0])
    
    # Set labels and title
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Accuracy')
    #plt.title('Stop Sign Detection Accuracy Over Training', pad=20)
    
    # Set y-axis limits
    ax.set_ylim(0.4, 1.05)
    
    # Add gridlines and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}stop_sign_accuracy_sampled.png", dpi=400, bbox_inches='tight')
    
    plt.show()

def plot_route_completion_score_bar():
    """
    Plot a bar chart comparing route completion scores for different model configurations:
    - Untrained LLM + Our VE
    - Untrained LLM + Raw VE
    - Trained LLM + Raw VE
    - Trained LLM + Our VE
    
    Values are shown with error bars representing the min/max range.
    """
    # Data with mean values and min/max bounds
    data = {
        "Untrained LLM\n+ Raw VE": {"mean": 0.535, "min": 0.27, "max": 0.8},
        "Untrained LLM\n+ FLAD VE": {"mean": 0.585, "min": 0.35, "max": 0.82},
        "Trained LLM\n+ Raw VE": {"mean": 14.5, "min": 0.0, "max": 29.0},
        "Trained LLM\n+ FLAD VE": {"mean": 30.8, "min": 24.7, "max": 37.43}
    }
    
    # Extract values
    labels = list(data.keys())
    means = [data[label]["mean"] for label in labels]
    
    # Calculate errors for error bars (distance from mean to min/max)
    lower_errors = [means[i] - data[labels[i]]["min"] for i in range(len(labels))]
    upper_errors = [data[labels[i]]["max"] - means[i] for i in range(len(labels))]
    
    # Create asymmetric error bars
    yerr = [lower_errors, upper_errors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Create bar colors with the first two using one color (untrained) and the last two using another (trained)
    colors = [BLUE_PALETTE[2], BLUE_PALETTE[2], BLUE_PALETTE[0], BLUE_PALETTE[0]]
    
    # Plot bars with error bars
    bars = ax.bar(
        range(len(labels)),
        means,
        width=0.7,
        color=colors,
        edgecolor='black',
        linewidth=1,
        capsize=8,
        yerr=yerr,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5}
    )
    
    # Highlight "Our VE" bars with patterns
    bars[1].set_hatch('///')
    bars[3].set_hatch('///')
    
    # Set axis labels and title
    ax.set_xlabel('AD Model Configuration', fontsize=24, labelpad=10)
    ax.set_ylabel('Route Completion Score', fontsize=24, labelpad=10)
    #ax.set_title('Route Completion Performance by Model Configuration', fontsize=16, pad=20)
    
    # Set x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontdict={'fontsize': 13})
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to start from 0 with some padding at the top
    y_max = max([data[label]["max"] for label in labels])
    ax.set_ylim(0, y_max * 1.1)
    
    # Add a legend for the patterns
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', hatch='///', edgecolor='black', label='FLAD VE'),
        Patch(facecolor='gray', edgecolor='black', label='Raw VE')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5)
    """
    # Add text annotations for the values
    for i, bar in enumerate(bars):
        height = means[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + yerr[1][i] + 0.8,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    """
    plt.tight_layout()
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}route_completion_scores.png", dpi=400, bbox_inches='tight')
    
    plt.show()

def plot_infraction_score_bar():
    """
    Plot a bar chart comparing infraction scores for different model configurations:
    - Untrained LLM + Raw VE (did not complete routes)
    - Untrained LLM + FLAD VE (did not complete routes)
    - Trained LLM + Raw VE
    - Trained LLM + FLAD VE
    
    Lower scores are better - indicating fewer infractions.
    """
    # Data with mean values and min/max bounds
    data = {
        "Untrained LLM\n+ Raw VE": {"mean": 0, "status": "failure"},
        "Untrained LLM\n+ FLAD VE": {"mean": 0, "status": "failure"},
        "Trained LLM\n+ Raw VE": {"mean": 2.49, "min": 0.0, "max": 4.99},
        "Trained LLM\n+ FLAD VE": {"mean": 0.38, "min": 0, "max": 0.76}
    }
    
    # Extract values
    labels = list(data.keys())
    
    # Create a list for means, handling 'failure' cases
    means = []
    failed_indices = []
    for i, label in enumerate(labels):
        if "status" in data[label] and data[label]["status"] == "failure":
            means.append(0)  # Placeholder for failed cases
            failed_indices.append(i)
        else:
            means.append(data[label]["mean"])
    
    # Calculate errors for error bars (distance from mean to min/max)
    lower_errors = []
    upper_errors = []
    
    for i, label in enumerate(labels):
        if i in failed_indices:
            lower_errors.append(0)
            upper_errors.append(0)
        else:
            lower_errors.append(means[i] - data[label].get("min", means[i]))
            upper_errors.append(data[label].get("max", means[i]) - means[i])
    
    # Create asymmetric error bars
    yerr = [lower_errors, upper_errors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Create bar colors - lower is better for infractions, so use green for low values
    colors = [BLUE_PALETTE[3], BLUE_PALETTE[3], BLUE_PALETTE[1], BLUE_PALETTE[0]]
    
    # Plot bars with error bars for completed runs
    valid_indices = [i for i in range(len(labels)) if i not in failed_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_means = [means[i] for i in valid_indices]
    valid_yerr = [[lower_errors[i] for i in valid_indices], [upper_errors[i] for i in valid_indices]]
    valid_colors = [colors[i] for i in valid_indices]
    
    bars = ax.bar(
        valid_indices,
        valid_means,
        width=0.7,
        color=valid_colors,
        edgecolor='black',
        linewidth=1,
        capsize=8,
        yerr=valid_yerr,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5}
    )
    
    # For failed runs, create a special bar with hatching
    if failed_indices:
        failed_bars = ax.bar(
            failed_indices,
            [0.1] * len(failed_indices),  # Small height for visibility
            width=0.7,
            color='lightgray',
            edgecolor='black',
            linewidth=1,
            hatch='xxx',
            label='Failed'
        )
    
    # Highlight "FLAD VE" bars with patterns
    if 1 in valid_indices:
        bars[valid_indices.index(1)].set_hatch('///')
    if 3 in valid_indices:
        bars[valid_indices.index(3)].set_hatch('///')
    
    # Set axis labels and title
    ax.set_xlabel('AD Model Configuration', fontsize=24, labelpad=10)
    ax.set_ylabel('Infraction Score', fontsize=24, labelpad=10)
    #ax.set_title('Infraction Score by Model Configuration', fontsize=16, pad=20)
    
    # Set x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontdict={'fontsize': 13})
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to start from 0 with some padding at the top
    max_value = max([data[label].get("max", 0) for label in labels if "max" in data[label]])
    ax.set_ylim(0, max_value * 1.2)
    
    # Add a legend for the patterns
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', hatch='///', edgecolor='black', label='FLAD VE'),
        Patch(facecolor='gray', edgecolor='black', label='Raw VE')
    ]
    if failed_indices:
        legend_elements.append(Patch(facecolor='lightgray', hatch='xxx', edgecolor='black', label='Failed'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5)
    """
    # Add text annotations for the values
    for i, value in enumerate(valid_means):
        position = valid_indices[i]
        ax.text(
            position,
            value + valid_yerr[1][i] + 0.1,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    """
    # Add "Failed" text for the failed runs
    for idx in failed_indices:
        ax.text(
            idx,
            0.3,
            'Failed',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='darkred'
        )
    
    plt.tight_layout()
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}infraction_scores.png", dpi=400, bbox_inches='tight')
    
    plt.show()



def plot_combined_driving_score_bar():
    """
    Plot a bar chart showing the combined driving score, calculated as:
    Combined Driving Score = Route Completion Score - Infraction Score
    
    This combined metric provides a single measure of overall driving performance,
    balancing route progress against safety violations.
    """
    # Data with mean values and min/max bounds
    route_completion_data = {
        "Untrained LLM\n+ Raw VE": {"mean": 0.535, "min": 0.27, "max": 0.8},
        "Untrained LLM\n+ FLAD VE": {"mean": 0.585, "min": 0.35, "max": 0.82},
        "Trained LLM\n+ Raw VE": {"mean": 14.5, "min": 0.0, "max": 29.0},
        "Trained LLM\n+ FLAD VE": {"mean": 30.8, "min": 24.7, "max": 37.43}
    }

    infraction_data = {
        "Untrained LLM\n+ Raw VE": {"mean": 0, "status": "failure"},
        "Untrained LLM\n+ FLAD VE": {"mean": 0, "status": "failure"},
        "Trained LLM\n+ Raw VE": {"mean": 2.49, "min": 0.0, "max": 4.99},
        "Trained LLM\n+ FLAD VE": {"mean": 0.38, "min": 0, "max": 0.76}
    }
    
    # Calculate combined driving scores
    labels = list(route_completion_data.keys())
    combined_data = {}
    
    for label in labels:
        # For failed models, set combined score equal to route completion score
        if "status" in infraction_data[label] and infraction_data[label]["status"] == "failure":
            combined_data[label] = {
                "mean": route_completion_data[label]["mean"],
                "min": route_completion_data[label]["min"],
                "max": route_completion_data[label]["max"],
                "status": "minimal_progress"
            }
        else:
            # Calculate combined score as route completion minus infractions
            mean_score = route_completion_data[label]["mean"] - infraction_data[label]["mean"]
            
            # Calculate min and max for error bars
            # Min combined = Min route - Max infraction
            min_score = route_completion_data[label]["min"] - infraction_data[label].get("max", 0)
            # Max combined = Max route - Min infraction
            max_score = route_completion_data[label]["max"] - infraction_data[label].get("min", 0)
            
            combined_data[label] = {
                "mean": mean_score,
                "min": min_score,
                "max": max_score
            }
    
    # Extract values for plotting
    means = [combined_data[label]["mean"] for label in labels]
    
    # Calculate errors for error bars (distance from mean to min/max)
    lower_errors = [means[i] - combined_data[labels[i]].get("min", means[i]) for i in range(len(labels))]
    upper_errors = [combined_data[labels[i]].get("max", means[i]) - means[i] for i in range(len(labels))]
    
    # Create asymmetric error bars
    yerr = [lower_errors, upper_errors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Create bar colors - use a gradient from light to dark blue based on performance
    colors = [BLUE_PALETTE[3], BLUE_PALETTE[3], BLUE_PALETTE[1], BLUE_PALETTE[0]]
    
    # Plot bars with error bars
    bars = ax.bar(
        range(len(labels)),
        means,
        width=0.7,
        color=colors,
        edgecolor='black',
        linewidth=1,
        capsize=8,
        yerr=yerr,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5}
    )
    
    # Highlight FLAD VE bars with patterns
    bars[1].set_hatch('///')
    bars[3].set_hatch('///')
    
    # Set axis labels and title with updated font sizes
    ax.set_xlabel('AD Model Configuration', fontdict={'size': 25})
    ax.set_ylabel('Driving Score', fontdict={'size': 25})
    #ax.set_title('Driving Score by AD Model Configuration', fontdict={'size': 25}, pad=20)
    
    # Set x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontdict={'fontsize': 13})
    
    # Add a grid for better readability with updated style
    ax.grid(color='silver', linestyle='--', linewidth=1, alpha=0.3)
    ax.grid(True)
    
    # Determine y-axis limits, ensuring 0 is included
    y_min = min(0, min([combined_data[label].get("min", 0) for label in labels]))
    y_max = max([combined_data[label].get("max", 0) for label in labels])
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add a legend for the patterns with updated font size
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', hatch='///', edgecolor='black', label='FLAD VE'),
        Patch(facecolor='gray', edgecolor='black', label='Raw VE')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}combined_driving_score.png", dpi=400, bbox_inches='tight')
    
    plt.show()

def execute_csv_std(file_path):
    """
    Compute average accuracy and corresponding standard deviation for each row in a CSV file.
    Adds 'avg acc' and 'std' columns to the dataframe.
    
    Args:
        file_path (str): Path to the CSV file containing accuracy data.
        
    Returns:
        pd.DataFrame: DataFrame with added 'avg acc' and 'std' columns.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        
        # Print columns to help debug
        print(f"Columns in the CSV: {df.columns.tolist()}")
        
        # Identify accuracy columns (containing 'acc' in their name)
        accuracy_columns = [col for col in df.columns if 'acc' in col.lower()]
        
        if not accuracy_columns:
            print("No accuracy metrics found in the CSV file.")
            return df
        
        print(f"Found accuracy columns: {accuracy_columns}")
        
        # Calculate row-wise mean (average) for accuracy columns
        df['avg acc'] = df[accuracy_columns].mean(axis=1)
        
        # Calculate row-wise standard deviation for accuracy columns
        df['std'] = df[accuracy_columns].std(axis=1)
        
        # Print summary statistics
        print("\nSummary statistics:")
        print(f"Overall average accuracy: {df['avg acc'].mean():.4f}")
        print(f"Overall standard deviation: {df['std'].mean():.4f}")
        
        # Optionally save the modified DataFrame
        output_path = file_path.replace('.csv', '_with_stats.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
        
        return df
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        
        return None

def plot_diff_LLM__scores_bar():
    """
    Plot bar charts comparing different LLM models with FLAD VE:
    - Route Completion Score (higher is better)
    - Infraction Score (lower is better)
    - Overall Driving Score
    
    Each metric is saved as a separate figure.
    """
    # Data with mean values and min/max bounds
    RC_data = {
        "Llamma\n+ FLAD VE": {"mean": 30.8},
        "Llava\n+ FLAD VE": {"mean": 4},
        "Vicuna\n+ FLAD VE": {"mean": 11.4}
    }
    IS_data = {
        "Llamma\n+ FLAD VE": {"mean": 0.38},
        "Llava\n+ FLAD VE": {"mean": 20},
        "Vicuna\n+ FLAD VE": {"mean": 13}
    }
    DS_data = {
        "Llamma\n+ FLAD VE": {"mean": 30},
        "Llava\n+ FLAD VE": {"mean": -16},
        "Vicuna\n+ FLAD VE": {"mean": -1.6}
    }
    
    # Plot Route Completion Scores
    plot_llm_comparison(RC_data, "Route Completion Score", "route_completion_llm_comparison.png", 
                        higher_is_better=True)
    
    # Plot Infraction Scores
    plot_llm_comparison(IS_data, "Infraction Score", "infraction_score_llm_comparison.png", 
                        higher_is_better=False)
    
    # Plot Driving Scores
    plot_llm_comparison(DS_data, "Driving Score", "driving_score_llm_comparison.png", 
                        higher_is_better=True, has_error_bars=False)

def plot_llm_comparison(data, y_label, filename, higher_is_better=True, has_error_bars=True):
    """
    Helper function to plot LLM comparison bar charts with consistent styling
    
    Args:
        data: Dictionary containing the data to plot
        y_label: Label for the y-axis
        filename: Filename to save the plot
        higher_is_better: Whether higher values are better (affects color scheme)
        has_error_bars: Whether to include error bars (some data doesn't have min/max values)
    """
    # Extract values
    labels = list(data.keys())
    means = [data[label]["mean"] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Choose colors based on performance metric (higher is better or lower is better)
    if higher_is_better:
        # For metrics where higher is better (e.g., route completion)
        # Use darker blue for better performance
        performance_order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
        colors = [BLUE_PALETTE[min(i, len(BLUE_PALETTE)-1)] for i in performance_order]
    else:
        # For metrics where lower is better (e.g., infractions)
        # Use darker blue for better performance (lower values)
        performance_order = sorted(range(len(means)), key=lambda i: means[i])
        colors = [BLUE_PALETTE[min(i, len(BLUE_PALETTE)-1)] for i in performance_order]
    
    # Sort colors back to original data order
    colors_in_order = [colors[performance_order.index(i)] for i in range(len(means))]
    
    # Add error bars if applicable
    if has_error_bars:
        # Calculate errors for error bars (distance from mean to min/max)
        lower_errors = [means[i] - data[labels[i]].get("min", means[i]) for i in range(len(labels))]
        upper_errors = [data[labels[i]].get("max", means[i]) - means[i] for i in range(len(labels))]
        yerr = [lower_errors, upper_errors]
        
        # Plot bars with error bars
        bars = ax.bar(
            range(len(labels)),
            means,
            width=0.6,
            color=colors_in_order,
            edgecolor='black',
            linewidth=1,
            capsize=8,
            yerr=yerr,
            error_kw={'elinewidth': 1.5, 'capthick': 1.5}
        )
    else:
        # Plot bars without error bars
        bars = ax.bar(
            range(len(labels)),
            means,
            width=0.6,
            color=colors_in_order,
            edgecolor='black',
            linewidth=1
        )
    
    # Set axis labels and title
    ax.set_xlabel('LLM Model', fontsize=24, labelpad=10)
    ax.set_ylabel(y_label, fontsize=24, labelpad=10)
    
    # Set x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontdict={'fontsize': 16})
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis
    if higher_is_better:
        # For metrics where higher is better
        y_min = min(0, min(means) * 1.1)  # Include 0 or lower if negative values exist
        y_max = max(means) * 1.2  # Add 20% headroom
    else:
        # For metrics where lower is better
        y_min = 0  # Start at 0
        y_max = max(means) * 1.2  # Add 20% headroom
    
    ax.set_ylim(y_min, y_max)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = means[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (y_max - y_min) * 0.02,  # Position slightly above bar
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=400, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    setup_environment()
    # Process CSV files if needed
    # results = process_csv_files()
    
    # Generate all plots
    plot_execution_times()
    #plot_optimization_times()
    #plot_model_size_comparison()
    #plot_recovery_time()
    #plot_VE_throughout()
    #plot_VE_mem()
    #plot_model_architecture()
    #plot_traffic_light_training_progress()
    #plot_stop_sign_training_progress()
    #plot_route_completion_score_bar()
    #plot_infraction_score_bar()
    #plot_combined_driving_score_bar()
    file_path = "flad/used.csv"
    #execute_csv_std(file_path)
    #plot_diff_LLM__scores_bar()
    #plot_diff_LLM__scores_bar()
    print(f"{'Plots saved to '+OUTPUT_DIR if SAVE_PLOTS else 'Plots displayed but not saved'}")


