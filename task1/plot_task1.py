import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

# Create the images directory to save the generated plots
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Compile the C++ files using the provided Makefile
print("Building the project...")
subprocess.run(["make"], check=True)

# The stages defined in the harness
stages = ["reorder", "unroll", "tile", "simd", "optimized"]
labels = ["Loop reordering", "Loop unrolling", "Tiling", "SIMD", "All Combined"]

def run_workload(h, w, k):
    """Runs the conv executable for a specific workload and extracts speedups."""
    cmd = ["./bin/conv", "all", str(h), str(w), str(k)]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Regex to match the output table format: stage, correct, time, GFLOP/s, speedup
    # Example: "tile            yes          16.613        4.54      1.02x"
    pattern = re.compile(r"([a-z]+(?:\s+\(ref\))?)\s+(yes|NO)\s+[\d.]+\s+[\d.]+\s+([\d.]+)x")
    
    speedups = {}
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if match:
            stage_name = match.group(1).strip()
            speedup = float(match.group(3))
            speedups[stage_name] = speedup
            
    return speedups

def plot_grouped_bar(categories, data_dict, title, xlabel, ylabel, filename):
    """Generates and saves a grouped bar chart."""
    x = np.arange(len(categories))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
    
    for attribute, measurement in data_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir}/{filename}")

# --- Experiment 1: Speedup vs Matrix Size (Fixed K=3) ---
# Note: W must be a multiple of 8, and K must be odd.
matrix_sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
k_fixed = 3

size_results = {label: [] for label in labels}
size_labels = []

for h, w in matrix_sizes:
    size_labels.append(f"{h}x{w}")
    speedups = run_workload(h, w, k_fixed)
    
    for stage, label in zip(stages, labels):
        size_results[label].append(speedups.get(stage, 0.0))

plot_grouped_bar(
    categories=size_labels,
    data_dict=size_results,
    title=f"Normalized Speedup vs Matrix Size (Kernel = {k_fixed}x{k_fixed})",
    xlabel="Matrix Size (H x W)",
    ylabel="Normalized Speedup (vs Naive)",
    filename="task1_speedup_vs_size.png"
)

# --- Experiment 2: Speedup vs Kernel Size (Fixed H=2048, W=2048) ---
h_fixed, w_fixed = 2048, 2048
kernel_sizes = [3, 5, 7, 9]

kernel_results = {label: [] for label in labels}
kernel_labels = []

for k in kernel_sizes:
    kernel_labels.append(f"K={k}")
    speedups = run_workload(h_fixed, w_fixed, k)
    
    for stage, label in zip(stages, labels):
        kernel_results[label].append(speedups.get(stage, 0.0))

plot_grouped_bar(
    categories=kernel_labels,
    data_dict=kernel_results,
    title=f"Normalized Speedup vs Kernel Size (Matrix = {h_fixed}x{w_fixed})",
    xlabel="Kernel Size (K)",
    ylabel="Normalized Speedup (vs Naive)",
    filename="task1_speedup_vs_kernel.png"
)