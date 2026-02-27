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

data_path = "C:/Thesis/Scripts/NGO_for_simple_gyrokinetics/DeepONet_all/vlasov_ngo_data_t_10_16_points.npz"

full_dataset = PDEDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_set = random_split(full_dataset, [train_size, val_size])

# Load model
model = VlasovModel(input_dim=4, Nv=32, Nt=16, basis_matrix=basis_matrix)
# model.load_state_dict(torch.load("vlasov_model.pth", map_location="cpu"))
# model.eval()
import os

# This gets the directory where the script file is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "vlasov_model.pth")

model.load_state_dict(torch.load(model_path, map_location="cpu"))

# --- Compare several validation samples ---
model.eval()

num_samples = 2
fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))

for i in range(num_samples):
    params, initial_cond, solution = val_set[i]

    # Predict
    with torch.no_grad():
        y_pred = model(params.unsqueeze(0), initial_cond.unsqueeze(0)).squeeze(0)

    # Plot
    true_grid = solution.view(16, 32).abs().numpy()
    pred_grid = y_pred.view(16, 32).abs().numpy()

    t_min, t_max = 0.0, 10.0
    v_min, v_max = -4, 4 
    plot_extent = [v_min, v_max, t_min, t_max]

    im1 = axes[i, 0].imshow(true_grid, aspect='auto',extent = plot_extent, origin = 'lower')
    axes[i, 0].set_title(f"True (inputs: {params.numpy()})")
    plt.colorbar(im1, ax=axes[i, 0])
    plt.xlabel("Velocity space (a.u.)")
    plt.ylabel("Time (a.u.)")
    
    im2 = axes[i, 1].imshow(pred_grid, aspect='auto',extent = plot_extent, origin = 'lower')
    axes[i, 1].set_title(f"Predicted")
    plt.colorbar(im2, ax=axes[i, 1])
    plt.xlabel("Velocity space (a.u.)")
    plt.ylabel("Time (a.u.)")

plt.tight_layout()
plt.show()