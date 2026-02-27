# --- 3. Execution Block ---
from model_setup import VlasovModel
from Functions_and_classes import PDEDataset, DataLoader
data_path = "C:/Thesis/Scripts/NGO_for_simple_gyrokinetics/DeepONet_all/vlasov_ngo_data_t_10_16_points.npz"

import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum

from basis_functions_classes import make_basis_matrix


# Parameters
Nt, Nv = 16, 32
N_basis_t = 12 
N_basis_v = 12 


basis_matrix = make_basis_matrix(Nt=Nt,Nv=Nv,N_basis_t=N_basis_t,N_basis_v=N_basis_v)

if __name__ == "__main__":
    # 1. Load and Split Data
    full_dataset = PDEDataset(data_path)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    
    # basis_matrix_calc = compute_basis_matrix(Nt, Nv, 100, 20)
    # 2. Initialize Model
    model = VlasovModel(input_dim=4, Nv=Nv, Nt=Nt, basis_matrix=basis_matrix)  # Changed from 1 to 96
    
    # 3. Train
    trainer = L.Trainer(max_epochs=5000, accelerator="auto")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the trained model
    torch.save(model.state_dict(), "vlasov_model.pth")
    print("Model saved to vlasov_model.pth")

