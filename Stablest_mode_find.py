import numpy as np
from scipy.linalg import eig
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Configuration & Physics Parameters ---
N_SAMPLES = 5000
Nv = 32
Nt = 20
t_max = 8
vmax, vmin = 4, -4
TOLERANCE = 1e-7  # Threshold to exclude "zero" growth rates

ranges = {
    'omega_n':  [0.0, 20.0],
    'omega_Ti': [10.0, 60.0],
    'ky':       [0.1, 1.0],
    'kz':       [1.0, 4.0],
    'c_i':      [0.01, 1.0]
}

# --- 2. Static Grid & Matrix Pre-computation ---
v = np.linspace(vmin, vmax, Nv)
t = np.linspace(0, t_max, Nt)
dv = (vmax - vmin) / (Nv - 1)
v_matrix = np.diag(v)
F0 = (1.0 / np.pi**0.5) * np.exp(-v**2)
F0_matrix = np.diag(F0)
W = np.ones((Nv, Nv)) * dv
Integral_Operator = F0_matrix @ W 

# Tracking variables
min_growth_rate = float('inf')
best_params = None
best_solution = None

print(f"Searching for the lowest non-zero growth rate...")

for i in tqdm(range(N_SAMPLES)):
    # Random Parameters
    omega_n  = np.random.uniform(*ranges['omega_n'])
    omega_Ti = np.random.uniform(*ranges['omega_Ti'])
    ky       = np.random.uniform(*ranges['ky'])
    kz       = np.random.uniform(*ranges['kz'])
    
    # Construct Vlasov Operator A
    bracket_term = omega_n + (v**2 - 0.5) * omega_Ti
    bracket_matrix = np.diag(bracket_term)
    
    A = (-1j * kz * v_matrix) + \
        (-1j * kz * (v_matrix @ Integral_Operator)) + \
        (-1j * ky * (bracket_matrix @ Integral_Operator))
    
    # Solve for Eigenvalues (No eigenvectors yet for speed)
    evals = eig(A, left=False, right=False)
    
    # Identify the growth rate: max(Re(eigenvalues))
    current_growth_rate = np.max(evals.real)
    
    # --- EXCLUSION LOGIC ---
    # We skip if the growth rate is effectively zero
    if np.abs(current_growth_rate) < TOLERANCE:
        continue
        
    # Check if this is the lowest (most stable/negative) growth rate found
    if current_growth_rate < min_growth_rate:
        min_growth_rate = current_growth_rate
        best_params = {
            'omega_n': omega_n,
            'omega_Ti': omega_Ti,
            'ky': ky,
            'kz': kz,
            'growth_rate': current_growth_rate
        }
        
        # Re-calculate with eigenvectors only for the new record holder
        evals_full, evecs_full = eig(A)
        coeffs = np.random.uniform(*ranges['c_i'], size=Nv) + \
                 1j * np.random.uniform(*ranges['c_i'], size=Nv)
        time_evol = np.exp(np.outer(t, evals_full))
        best_solution = np.einsum('tj, j, vj -> tv', time_evol, coeffs, evecs_full)

# --- 3. Output Results ---
if best_params:
    print("\n" + "="*40)
    print(f"LOWEST NON-ZERO GROWTH RATE: {min_growth_rate:.6e}")
    print("="*40)
    for key, val in best_params.items():
        print(f"{key:15}: {val:.4f}")

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(v, t, best_solution.real, shading='auto', cmap='RdBu_r')
    plt.colorbar(label='Re(f)')
    plt.title(f"Most Stable Mode (Growth Rate: {min_growth_rate:.4f})")
    plt.xlabel("Velocity (v)")
    plt.ylabel("Time (t)")
    plt.show()
else:
    print("No non-zero growth rates found within the sample size.")