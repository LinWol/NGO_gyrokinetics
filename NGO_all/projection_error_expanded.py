import torch
import numpy as np

from basis_functions_classes import make_basis_matrix, BSplineBasis

from Functions_and_classes import PDEDataset
data_path = "C:/Thesis/Scripts/NGO_for_simple_gyrokinetics/NGO_all/vlasov_ngo_data_t_10_16_points.npz"

# Parameters
Nt, Nv = 16, 32
N_basis_t = 30
N_basis_v = 30


basis_matrix = make_basis_matrix(Nt=Nt,Nv=Nv,N_basis_t=N_basis_t,N_basis_v=N_basis_v)
basis_t = BSplineBasis(N=N_basis_t, p=3, C=2, dtype=torch.float32, device='cpu')
basis_v = BSplineBasis(N=N_basis_v, p=3, C=2, dtype=torch.float32, device='cpu')


full_dataset = PDEDataset(data_path)
solutions = full_dataset.y

def compute_projection_statistics(solutions, basis_matrix, n_samples=1000):
    """
    solutions: (num_samples, Nt, Nv)
    basis_matrix: (K, Nt*Nv)
    """

    # ---- 1. Random subset ----
    total_samples = solutions.shape[0]
    indices = torch.randperm(total_samples)[:n_samples]
    subset = solutions[indices]

    # ---- 2. Prepare projection operator (QR once) ----
    A = basis_matrix.T.to(torch.complex128)
    Q, R = torch.linalg.qr(A, mode='reduced')

    errors = []

    print(( subset.to(torch.complex128)).shape, 'eeeeeeeee')

    for sol in subset:
        b = sol.reshape(-1).to(torch.complex128)

        # Orthogonal projection
        u_proj = Q @ (Q.conj().T @ b)

        rel_error = torch.linalg.norm(b - u_proj) / torch.linalg.norm(b)
        errors.append(rel_error.item())

    errors = np.array(errors)

    # ---- 3. Statistics ----
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)

    ci95 = 1.96 * std_error / np.sqrt(n_samples)

    return mean_error, std_error, ci95


def compute_projection_error_v4(solutions, n_samples, Nt, Nv, 
                                basis_t,basis_v):

    total_samples = solutions.shape[0]
    indices = torch.randperm(total_samples)[:n_samples]
    subset = solutions[indices]

    # grids
    t = torch.linspace(0.0, 10.0, Nt)
    v = torch.linspace(-4.0, 4.0, Nv)

    dt = t[1] - t[0]
    dv = v[1] - v[0]

    # normalize to [0,1] for B-splines
    t_norm = (t - t.min()) / (t.max() - t.min())
    v_norm = (v - v.min()) / (v.max() - v.min())

    # evaluate all 1D basis functions
    B_t = basis_t.forward(t_norm)  # (Nt, N_basis_t)
    B_v = basis_v.forward(v_norm)  # (Nv, N_basis_v)

    B_t_T = B_t.squeeze().T
    B_v_T = B_v.squeeze().T

    # mass matrices

    M_t = B_t_T.T @ B_t_T * dt
    M_v = B_v_T.T @ B_v_T * dv

    M_t_inv = torch.linalg.inv(M_t)
    M_v_inv = torch.linalg.inv(M_v)

    errors = []

    for sol in subset:

        U = sol.reshape(Nt, Nv)

        # RHS tensor
        RHS = B_t_T.T @ U @ B_v_T * dt * dv

        # coefficient matrix
        C = M_t_inv @ RHS @ M_v_inv

        # projected solution
        U_proj = B_t @ C @ B_v.T

        # L2 norms
        error_sq = torch.sum((U - U_proj)**2) * dt * dv
        u_norm_sq = torch.sum(U**2) * dt * dv

        relative_error = torch.sqrt(error_sq / u_norm_sq)

        errors.append(relative_error)

    return torch.stack(errors)


t = torch.linspace(0.0, 10.0, Nt)
v = torch.linspace(-4.0, 4.0, Nv)

dt = t[1] - t[0]
dv = v[1] - v[0]

# normalize to [0,1] for B-splines
t_norm = (t - t.min()) / (t.max() - t.min())
v_norm = (v - v.min()) / (v.max() - v.min())

# evaluate all 1D basis functions
B_t = basis_t.forward(t_norm)  # (Nt, N_basis_t)
B_v = basis_v.forward(v_norm)  # (Nv, N_basis_v)

B_t_T = B_t.squeeze().T
B_v_T = B_v.squeeze().T


print(B_t_T.shape)
print(B_v_T.shape)


import torch

def compute_projection_error_v6(solutions, n_samples, Nt, Nv, basis_t, basis_v):
    total_samples = solutions.shape[0]
    indices = torch.randperm(total_samples)[:n_samples]
    subset = solutions[indices]

    single_sample = subset[1]

    # Grids and spacing
    t = torch.linspace(0.0, 10.0, Nt)
    v = torch.linspace(-4.0, 4.0, Nv)
    dt = t[1] - t[0]
    dv = v[1] - v[0]

    # Normalize for B-splines
    t_norm = (t - t.min()) / (t.max() - t.min())
    v_norm = (v - v.min()) / (v.max() - v.min())

    w_t = torch.ones(Nt, dtype=t.dtype, device=t.device) * dt
    w_t[0] *= 0.5
    w_t[-1] *= 0.5

    w_v = torch.ones(Nv, dtype=v.dtype, device=v.device) * dv
    w_v[0] *= 0.5
    w_v[-1] *= 0.5
    
    # Ensure bases are real (as specified)
    B_t = basis_t.forward(t_norm).squeeze().to(single_sample.dtype).T
    B_v = basis_v.forward(v_norm).squeeze().to(single_sample.dtype).T

    # Mass matrices remain real
    # M_t = (B_t.T @ B_t) * dt
    # M_v = (B_v.T @ B_v) * dv

    B_t_w = B_t * w_t.unsqueeze(1)  # (Nt, K_t), rows scaled by w_t
    B_v_w = B_v * w_v.unsqueeze(1)  # (Nv, K_v), rows scaled by w_v

    M_t = B_t_w.T @ B_t  # (K_t, K_t)
    M_v = B_v_w.T @ B_v

    print(f"M_t condition number: {torch.linalg.cond(M_t*M_v).item():.2e}")
    print(f"M_v condition number: {torch.linalg.cond(M_v).item():.2e}")

    eps = 1e-10  # tune this if needed
    M_t_reg = M_t + eps * torch.eye(M_t.shape[0], dtype=M_t.dtype, device=M_t.device)
    M_v_reg = M_v + eps * torch.eye(M_v.shape[0], dtype=M_v.dtype, device=M_v.device)



    errors = []

    for sol in subset:
        # U is complex (Nt x Nv)
        U = sol.reshape(Nt, Nv)

        # 1. RHS is now complex-valued
        # (K_t, Nt) @ (Nt, Nv) @ (Nv, K_v) -> (K_t, K_v) complex
        RHS = B_t_w.T @ U @ B_v_w

        # 2. Solve for complex coefficients C
        # X = torch.linalg.solve(M_t, RHS)
        # C = torch.linalg.solve(M_v, X.T).T

        C = torch.linalg.solve(M_t, torch.linalg.solve(M_v, RHS.T).T)

        # 3. Reconstruct (Nt x Nv) complex
        U_proj = B_t @ C @ B_v.T

        # 4. Correct L2 norm for complex numbers: sum(|U - U_proj|^2)
        # We use .abs().pow(2) or (diff.conj() * diff).real
        diff = U - U_proj
        error_sq = torch.sum(diff.abs()**2 * w_t.unsqueeze(1) * w_v.unsqueeze(0))
        u_norm_sq = torch.sum(U.abs()**2 * w_t.unsqueeze(1) * w_v.unsqueeze(0))
        
        relative_error = torch.sqrt(error_sq / u_norm_sq)
        errors.append(relative_error)

    return torch.stack(errors)

    
error = torch.mean(compute_projection_error_v6(solutions, 1000, Nt, Nv,basis_t, basis_v))

print(error)