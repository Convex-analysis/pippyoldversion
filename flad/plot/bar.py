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

base_optimization = {3: 0.01, 5: 0.01, 7: 0.01, 9: 0}
swift_optimization = {3: 0.04, 5: 0.29, 7: 0.4, 9: 0.25}
plt.bar(np.array(list(base_optimization.keys())) - 0.2, base_optimization.values(), width=0.4, align='center', label='Phase1')
plt.bar(np.array(list(swift_optimization.keys())) + 0.2, swift_optimization.values(), width=0.4, align='center', label='Phase2')
plt.xlabel('Problem Scale')
plt.ylabel('Average Optimization Time (s)')
plt.legend()
plt.show()

base_model_size = {"5.55 GB": 1390.7427, "11.10 GB": 2998.2761, "14.01 GB": 0}
swift_model_size = {"5.55 GB": 1252.8087, "11.10 GB": 2911, "14.01 GB": 2944.5439}
bar_potision = np.arange(len(base_model_size))
plt.bar(bar_potision - 0.2, base_model_size.values(), width=0.4, align='center', label='Base')
plt.bar(bar_potision + 0.2, swift_model_size.values(), width=0.4, align='center', label='Swift')
plt.xticks(bar_potision, base_model_size.keys())
plt.xlabel('Model Size')
plt.ylabel('Average Execution Time (s)')
plt.legend()
plt.show()
