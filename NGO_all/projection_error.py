import numpy as np
import torch

from basis_functions_classes import make_basis_matrix

from Functions_and_classes import PDEDataset
data_path = "C:/Thesis/Scripts/NGO_for_simple_gyrokinetics/NGO_all/vlasov_ngo_data_t_10_16_points.npz"

# Parameters
Nt, Nv = 16, 32
N_basis_t = 12 
N_basis_v = 12 


basis_matrix = make_basis_matrix(Nt=Nt,Nv=Nv,N_basis_t=N_basis_t,N_basis_v=N_basis_v)

full_dataset = PDEDataset(data_path)
solutions = full_dataset.y
sample = solutions[700]

import torch

print(sample.shape,basis_matrix.shape)

def find_projection_error(solution, basis_matrix):
    """
    solution: complex tensor (Nt, Nv)
    basis_matrix: real tensor (K, Nt*Nv)
    """

    # Flatten solution
    b = solution.reshape(-1)  # (Nt*Nv,)

    # Transpose basis to get columns
    A = basis_matrix.T       # (Nt*Nv, K)

    # IMPORTANT: Match dtype
    A = A.to(b.dtype)

    # Solve least squares
    result = torch.linalg.lstsq(A, b)
    coeffs = result.solution

    # Reconstruct
    u_proj = A @ coeffs

    # Compute errors
    residual = b - u_proj
    abs_error = torch.linalg.norm(residual)
    rel_error = abs_error / torch.linalg.norm(b)

    return abs_error.item(), rel_error.item(), u_proj.reshape(solution.shape)

# Example of usage with your existing function:
# basis_mat = make_basis_matrix(Nt, Nv, N_basis_t, N_basis_v)
err, rel_err, proj = find_projection_error(sample, basis_matrix)

print(rel_err)