import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Configuration & Physics Parameters ---
Nv = 32          # Velocity grid points
vmax, vmin = 4, -4
POINTS_PER_PLOT = 100  # Resolution for our x-axis sweeps

# Parameter Ranges [min, max]
ranges = {
    'omega_n':  [0.0, 20.0],
    'omega_Ti': [10.0, 60.0],
    'ky':       [0.1, 1.0],
    'kz':       [1.0, 4.0]
}

# Automatically calculate midpoints for the parameters
midpoints = {key: (val[0] + val[1]) / 2.0 for key, val in ranges.items()}

# --- 2. Static Grid & Matrix Pre-computation ---
v = np.linspace(vmin, vmax, Nv)
dv = (vmax - vmin) / (Nv - 1)
v_matrix = np.diag(v)
F0 = (1.0 / np.pi**0.5) * np.exp(-v**2)
F0_matrix = np.diag(F0)
W = np.ones((Nv, Nv)) * dv
Integral_Operator = F0_matrix @ W 

# --- 3. Core Growth Rate Function ---
def get_max_growth_rate(omega_n, omega_Ti, ky, kz):
    """
    Constructs the Vlasov operator A and returns the maximum growth rate 
    (the maximum real part of the resulting eigenvalues).
    """
    bracket_term = omega_n + (v**2 - 0.5) * omega_Ti
    bracket_matrix = np.diag(bracket_term)
    
    A1 = -1j * kz * v_matrix
    A2 = -1j * kz * (v_matrix @ Integral_Operator)
    A3 = -1j * ky * (bracket_matrix @ Integral_Operator)
    A = A1 + A2 + A3
    
    # We only need eigenvalues, so right=False saves computation time
    evals = eig(A, right=False)  
    
    # The growth rate is the maximum real part of lambda
    return np.max(evals.real)

# --- 4. Perform Parameter Scans ---
print("Calculating growth rates across parameter ranges...")
results = {}

for param_name, param_range in ranges.items():
    # Create the x-axis values for this specific parameter
    x_vals = np.linspace(param_range[0], param_range[1], POINTS_PER_PLOT)
    y_vals = np.zeros(POINTS_PER_PLOT)
    
    # Sweep through the parameter, keeping others fixed at midpoints
    for i, val in enumerate(tqdm(x_vals, desc=f"Scanning {param_name}")):
        current_params = midpoints.copy()
        current_params[param_name] = val
        
        y_vals[i] = get_max_growth_rate(
            current_params['omega_n'],
            current_params['omega_Ti'],
            current_params['ky'],
            current_params['kz']
        )
        
    results[param_name] = (x_vals, y_vals)

# --- 5. Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Maximum Linear Growth Rate vs. Physics Parameters', fontsize=16)

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each parameter's results
for i, (param_name, (x_vals, y_vals)) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(x_vals, y_vals, color='firebrick', linewidth=2)
    ax.set_title(f'Varying {param_name}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Max Growth Rate $Re(\lambda)$')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text box indicating the values of the fixed parameters
    fixed_text = "\n".join([f"{k} = {v:.2f}" for k, v in midpoints.items() if k != param_name])
    ax.text(0.05, 0.95, f"Fixed at:\n{fixed_text}", transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

import itertools

# --- 6. 2D Heatmap Configurations ---
RESOLUTION_2D = 100  # Keep this moderate so the 6 scans don't take forever

# Generate all unique pairs of parameters (4 parameters = 6 combinations)
param_pairs = list(itertools.combinations(ranges.keys(), 2))

print(f"\nCalculating 2D heatmaps for {len(param_pairs)} parameter pairs...")
heatmap_results = {}

# --- 7. Perform 2D Parameter Scans ---
for p1, p2 in param_pairs:
    print(f"Scanning {p1} vs {p2}...")
    
    # Create the grids for the two active parameters
    x_vals = np.linspace(ranges[p1][0], ranges[p1][1], RESOLUTION_2D)
    y_vals = np.linspace(ranges[p2][0], ranges[p2][1], RESOLUTION_2D)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    # Iterate over the grid
    for i in tqdm(range(RESOLUTION_2D), desc="Rows", leave=False):
        for j in range(RESOLUTION_2D):
            # Start with all parameters at their midpoints
            current_params = midpoints.copy()
            
            # Override with the scanning parameters
            current_params[p1] = X[i, j]
            current_params[p2] = Y[i, j]
            
            # Calculate and store the maximum growth rate
            Z[i, j] = get_max_growth_rate(
                current_params['omega_n'],
                current_params['omega_Ti'],
                current_params['ky'],
                current_params['kz']
            )
            
    heatmap_results[(p1, p2)] = (X, Y, Z)

# # --- 8. Plotting the Heatmaps ---
# fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
# fig2.suptitle('2D Parameter Scans: Maximum Linear Growth Rate', fontsize=18, y=0.98)
# axes2 = axes2.flatten()

# for idx, ((p1, p2), (X, Y, Z)) in enumerate(heatmap_results.items()):
#     ax = axes2[idx]
    
#     # Use contourf for a smooth, filled heatmap
#     cp = ax.contourf(X, Y, Z, levels=30, cmap='inferno')
#     fig2.colorbar(cp, ax=ax, label='Max Growth Rate $Re(\lambda)$')
    
#     ax.set_title(f'{p1} vs {p2}', fontsize=14)
#     ax.set_xlabel(p1, fontsize=12)
#     ax.set_ylabel(p2, fontsize=12)
    
#     # Add a text box indicating the values of the fixed parameters
#     fixed_params = [f"{k} = {v:.2f}" for k, v in midpoints.items() if k not in (p1, p2)]
#     fixed_text = "Fixed at:\n" + "\n".join(fixed_params)
    
#     # Place text box in the corner, styling it to be visible against the heatmap
#     ax.text(0.05, 0.95, fixed_text, transform=ax.transAxes, 
#             fontsize=10, color='white', verticalalignment='top', 
#             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none'))

# plt.tight_layout(pad=2.0)
# plt.show()

# # --- 8. Plotting the Heatmaps (Normalized Colormap) ---

# # 1. Find the global minimum and maximum Z values across all heatmaps
# global_min = min([np.min(Z) for _, _, Z in heatmap_results.values()])
# global_max = max([np.max(Z) for _, _, Z in heatmap_results.values()])

# # 2. Define consistent contour levels based on global min/max
# shared_levels = np.linspace(global_min, global_max, 30)

# fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
# fig2.suptitle('2D Parameter Scans: Maximum Linear Growth Rate', fontsize=18, y=0.98)
# axes2 = axes2.flatten()

# for idx, ((p1, p2), (X, Y, Z)) in enumerate(heatmap_results.items()):
#     ax = axes2[idx]
    
#     # Apply the shared levels and vmin/vmax to normalize the colormap
#     cp = ax.contourf(X, Y, Z, levels=shared_levels, cmap='inferno', 
#                      vmin=global_min, vmax=global_max)
    
#     # Add colorbar to each plot (they will all show the same range now)
#     fig2.colorbar(cp, ax=ax, label='Max Growth Rate Re(λ)')
    
#     ax.set_title(f'{p1} vs {p2}', fontsize=14)
#     ax.set_xlabel(p1, fontsize=12)
#     ax.set_ylabel(p2, fontsize=12)
    
#     # Add a text box indicating the values of the fixed parameters
#     fixed_params = [f"{k} = {v:.2f}" for k, v in midpoints.items() if k not in (p1, p2)]
#     fixed_text = "Fixed at:\n" + "\n".join(fixed_params)
    
#     ax.text(0.05, 0.95, fixed_text, transform=ax.transAxes, 
#             fontsize=10, color='white', verticalalignment='top', 
#             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none'))

# plt.tight_layout(pad=2.0)
# plt.show()

# --- 8. Plotting the Heatmaps (Normalized Colormap & LaTeX) ---

# Define LaTeX labels for clean plot rendering
latex_labels = {
    'omega_n':  r'$\omega_n$',
    'omega_Ti': r'$\omega_{Ti}$',
    'ky':       r'$k_y$',
    'kz':       r'$k_z$'
}

# 1. Find the global minimum and maximum Z values across all heatmaps
global_min = min([np.min(Z) for _, _, Z in heatmap_results.values()])
global_max = max([np.max(Z) for _, _, Z in heatmap_results.values()])

# 2. Define consistent contour levels based on global min/max
shared_levels = np.linspace(global_min, global_max, 30)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('2D Parameter Scans: Maximum Growth Rate', fontsize=18, y=0.98)
axes2 = axes2.flatten()

for idx, ((p1, p2), (X, Y, Z)) in enumerate(heatmap_results.items()):
    ax = axes2[idx]
    
    # Apply the shared levels and vmin/vmax to normalize the colormap
    cp = ax.contourf(X, Y, Z, levels=shared_levels, cmap='inferno', 
                     vmin=global_min, vmax=global_max)
    
    # Add colorbar (using LaTeX for lambda)
    fig2.colorbar(cp, ax=ax, label=r'Max Growth Rate $Re(\lambda)$')
    
    # Apply LaTeX formatting to titles and axes
    ax.set_title(f'{latex_labels[p1]} vs {latex_labels[p2]}', fontsize=14)
    ax.set_xlabel(latex_labels[p1], fontsize=14)
    ax.set_ylabel(latex_labels[p2], fontsize=14)
    
    # Apply LaTeX formatting to the fixed parameters text box
    fixed_params = [f"{latex_labels[k]} = {v:.2f}" for k, v in midpoints.items() if k not in (p1, p2)]
    fixed_text = "Fixed at:\n" + "\n".join(fixed_params)
    
    ax.text(0.05, 0.95, fixed_text, transform=ax.transAxes, 
            fontsize=12, color='white', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none'))

plt.tight_layout(pad=2.0)
plt.show()