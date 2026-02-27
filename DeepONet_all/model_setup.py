import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from torch import nn, optim, utils
from torch.utils.data import Dataset, DataLoader, random_split
import opt_einsum

from Functions_and_classes import integrate, relative_l2_loss

class VlasovModel(L.LightningModule):
    def __init__(self, input_dim=4, Nv=32, Nt=32, basis_matrix=None):
        super().__init__()
        self.output_dim = Nt * Nv  # 256
        self.Nt = Nt
        self.Nv = Nv
        
        # Basis setup
        if basis_matrix is None:
            raise ValueError("Must provide precomputed basis_matrix")
        self.K = basis_matrix.shape[0]  # Number of basis functions
        
        # Register basis as a fixed buffer (not trained)
        # For complex solutions, we use the SAME real basis for both real/imag parts
        self.register_buffer('basis_real', basis_matrix)  # (K, 256)
        self.register_buffer('basis_imag', torch.zeros_like(basis_matrix))  # No imaginary basis
        
        # Branch network: outputs K complex coefficients (2K reals)
        initial_size = Nv * 2  # Initial condition flattened (real + imag)
        total_input = input_dim + initial_size
        
        self.branch = nn.Sequential(
            nn.Linear(total_input, 512),
            nn.ReLU(),
            nn.Linear(512, 1028),
            nn.ReLU(),
            nn.Linear(1028,1028),
            nn.ReLU(),
            nn.Linear(1028, self.K * 2)  # K complex = 2K reals
        )
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, params, initial_conditions):
        batch_size = params.shape[0]
        
        # Flatten initial conditions and split real/imag
        init_flat = initial_conditions.view(batch_size, -1)
        init_real_imag = torch.cat([init_flat.real, init_flat.imag], dim=1)
        
        # Concatenate params and initial conditions
        x = torch.cat([params, init_real_imag], dim=1)
        
        # Branch network outputs coefficients
        out = self.branch(x)  # (Batch, 2*K)
        c_real = out[..., :self.K]   # (Batch, K)
        c_imag = out[..., self.K:]   # (Batch, K)
        
        # Reconstruct solution: u = Σ c_k * B_k
        # Since basis is real: u = (c_r + i*c_i) * B_r
        pred_real = c_real @ self.basis_real  # (Batch, 256)
        pred_imag = c_imag @ self.basis_real  # (Batch, 256)
        
        return torch.complex(pred_real, pred_imag)
    
    def training_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        preds = self(params, initial_conditions)

        y_hat = preds.view(-1, 32, 32)
        y_true = solution.view(-1, 32, 32)

        W = torch.ones(32, device=self.device)
        diff_sq = (y_hat.real - y_true.real)**2 + (y_hat.imag - y_true.imag)**2
        true_sq = y_true.real**2 + y_true.imag**2

        l2_diff = integrate(W, diff_sq)**(0.5)
        l2_norm = integrate(W, true_sq)**(0.5)
        
        loss_back_up = torch.mean(l2_diff / torch.clamp(l2_norm, min=1e-7))

        loss = relative_l2_loss(y_hat,y_true)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        params, initial_conditions, solution = batch
        preds = self(params, initial_conditions)
        y_hat_val = preds.view(-1, 32, 32)
        y_true_val = solution.view(-1, 32, 32)
        val_loss_org = self.loss_fn(preds.real, solution.real) + self.loss_fn(preds.imag, solution.imag)
        val_loss = relative_l2_loss(y_hat_val,y_true_val)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)