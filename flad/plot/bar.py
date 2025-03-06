import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import re
import glob

#read csv files in current directory
path = os.getcwd()
files = glob.glob(path + "./flad/plot/pipelines*.csv")
#for each file compute the average of columns "Execution_Time" and the lines of this file
results = {}
for file in files:
    df = pd.read_csv(file)
    print("File: ", file)
    print("Average Execution Time: ", df["Execution_Time"].mean())
    print("Number of lines: ", len(df))
    results[len(df)] = df["Execution_Time"].mean()
    print(results)
    
#draw results as a bar plot the x-axis is the number of lines in the file and the y-axis is the average of the column "Execution_Time"


base_results = {3: 1081.6197, 5: 2494.7078, 7: 910.5060, 9: 0}
swift_results = {3: 1064.1329, 5: 2339.8869, 7: 899.6769, 9: 901.1903}
#draw results as a bar plot the x-axis is the number of lines in the file and the y-axis is the average of the column "Execution_Time"
#don't overlap the bars
plt.bar(np.array(list(base_results.keys())) - 0.2, base_results.values(), width=0.4, align='center', label='Base')
plt.bar(np.array(list(swift_results.keys())) + 0.2, swift_results.values(), width=0.4, align='center', label='Swift')
plt.xlabel('Cluster Size')
plt.ylabel('Average Execution Time (s)')
plt.legend()
plt.show()

base_optimization = {3: 0.01, 5: 0.01, 7: 0.01, 9: np.nan}  # Use NaN for incomplete
swift_optimization = {3: 0.04, 5: 0.29, 7: 0.4, 9: 0.25}

x = np.array(list(base_optimization.keys()))
base_values = list(base_optimization.values())

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot swift data normally
ax.bar(x + 0.2, swift_optimization.values(), width=0.4, align='center', label='Phase2')

# Plot completed base data
completed = ~np.isnan(base_values)
ax.bar(x[completed] - 0.2, np.array(base_values)[completed], 
       width=0.4, align='center', label='Phase1')

# Plot special bar for incomplete
incomplete = np.isnan(base_values)
if any(incomplete):
    # Add a special bar for "did not complete"
    ax.bar(x[incomplete] - 0.2, [0.01], width=0.4, align='center',
           hatch='////', color='lightgray', label='Phase1 (Did not complete)')
    
    # Add text annotation
    for i in x[incomplete]:
        ax.annotate('Did not\ncomplete', xy=(i-0.2, 0.03), ha='center', va='bottom')

plt.xlabel('Problem Scale')
plt.ylabel('Average Optimization Time (s)')
plt.legend()
plt.tight_layout()
plt.show()

# Modify base_model_size to use NaN for incomplete tasks
base_model_size = {"5.55 GB": 1390.7427, "11.10 GB": 2998.2761, "14.01 GB": np.nan}
swift_model_size = {"5.55 GB": 1252.8087, "11.10 GB": 2911, "14.01 GB": 2944.5439}

# Get keys and positions
model_sizes = list(base_model_size.keys())
bar_position = np.arange(len(model_sizes))
base_values = list(base_model_size.values())

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot swift data normally
ax.bar(bar_position + 0.2, swift_model_size.values(), width=0.4, align='center', label='Swift')

# Plot completed base data
completed = ~np.isnan(base_values)
completed_positions = bar_position[completed]
completed_values = np.array(base_values)[completed]
ax.bar(completed_positions - 0.2, completed_values, width=0.4, align='center', label='Base')

# Plot special bar for incomplete
incomplete = np.isnan(base_values)
if any(incomplete):
    incomplete_positions = bar_position[incomplete]
    # Add a special bar for "did not complete"
    ax.bar(incomplete_positions - 0.2, [0.01], width=0.4, align='center',
           hatch='////', color='lightgray', label='Base (Did not complete)')
    
    # Add text annotation
    for i in incomplete_positions:
        ax.annotate('Did not\ncomplete', xy=(i-0.2, 150), ha='center', va='bottom')

plt.xticks(bar_position, model_sizes)
plt.xlabel('Model Size')
plt.ylabel('Average Execution Time (s)')
plt.legend()
plt.tight_layout()
plt.show()