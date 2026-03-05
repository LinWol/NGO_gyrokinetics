import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from Functions_and_classes import PDEDataset
from model_setup import VlasovModel
from basis_functions_classes import make_basis_matrix

Nt, Nv = 16, 32
N_basis_t, N_basis_v = 12, 12
basis_matrix = make_basis_matrix(Nt=Nt, Nv=Nv, N_basis_t=N_basis_t, N_basis_v=N_basis_v)
data_path = "C:/Thesis/Scripts/NGO_for_simple_gyrokinetics/NGO_all/vlasov_ngo_data_t_10_16_points.npz"
full_dataset = PDEDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_set = random_split(full_dataset, [train_size, val_size])

# Load model
model = VlasovModel(input_dim=4, Nv=32, Nt=16, basis_matrix=basis_matrix)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "vlasov_model.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ============================================================
# CUSTOM INPUT: Set your own parameters and initial condition
# ============================================================
# Replace these values with your own. `custom_params` should have
# the same number of entries as the params in your dataset (here: 4).
custom_params = [5, 50, 0.7,2.0]  # <-- Edit these values

# `custom_initial_cond` should be a flat list/array of length Nv=32
# representing the initial distribution f(v) at t=0.
v = np.linspace(-4,4,Nv)
custom_initial_cond = 0.001 * np.exp(-v**2 / 2) / np.sqrt(2 * np.pi)    # <-- Edit this (e.g. a Gaussian)

# Set to True to run the custom prediction, False to skip it
RUN_CUSTOM = True
# ============================================================

# --- Compare several validation samples ---
num_samples = 2
fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

for i in range(num_samples):
    params, initial_cond, solution = val_set[i]

    with torch.no_grad():
        y_pred = model(params.unsqueeze(0), initial_cond.unsqueeze(0)).squeeze(0)

    true_grid = solution.view(16, 32).abs().numpy()
    pred_grid = y_pred.view(16, 32).abs().numpy()

    t_min, t_max = 0.0, 10.0
    v_min, v_max = -4, 4
    plot_extent = [v_min, v_max, t_min, t_max]

    im1 = axes[i, 0].imshow(true_grid, aspect='auto', extent=plot_extent, origin='lower')
    axes[i, 0].set_title(f"True (inputs: {params.numpy()})")
    plt.colorbar(im1, ax=axes[i, 0])
    axes[i, 0].set_xlabel("Velocity space (a.u.)")
    axes[i, 0].set_ylabel("Time (a.u.)")

    im2 = axes[i, 1].imshow(pred_grid, aspect='auto', extent=plot_extent, origin='lower')
    axes[i, 1].set_title(f"Predicted")
    plt.colorbar(im2, ax=axes[i, 1])
    axes[i, 1].set_xlabel("Velocity space (a.u.)")
    axes[i, 1].set_ylabel("Time (a.u.)")

plt.tight_layout()
plt.show()

# --- Custom input prediction ---
if RUN_CUSTOM:
    custom_params_tensor = torch.tensor(custom_params, dtype=torch.float32).unsqueeze(0)
    custom_ic_tensor = torch.tensor(custom_initial_cond, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        custom_pred = model(custom_params_tensor, custom_ic_tensor).squeeze(0)

    custom_grid = custom_pred.view(16, 32).abs().numpy()

    t_min, t_max = 0.0, 10.0
    v_min, v_max = -4, 4
    plot_extent = [v_min, v_max, t_min, t_max]

    fig2, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(custom_grid, aspect='auto', extent=plot_extent, origin='lower')
    ax.set_title(f"Custom Prediction\nParams: {custom_params}")
    ax.set_xlabel("Velocity space (a.u.)")
    ax.set_ylabel("Time (a.u.)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()