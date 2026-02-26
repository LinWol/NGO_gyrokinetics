import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import eval_hermite

# All initial constants and such
N_SAMPLES = 1000
Nv = 32           
Nt = 32       
t_max = 5.0     
vmax, vmin = 4, -4
N_HERMITE = 16  # Number of polynomials to use

ranges = {
    'omega_n':  [0.0, 20.0],
    'omega_Ti': [10.0, 60.0],
    'ky':       [0.1, 1.0],
    'kz':       [1.0, 4.0]
}

v = np.linspace(vmin, vmax, Nv)
t = np.linspace(0, t_max, Nt)
dv = (vmax - vmin) / (Nv - 1)

# parts of matrix that dont depend on inputs
v_matrix = np.diag(v)
F0 = (1.0 / np.pi**0.5) * np.exp(-v**2)
F0_matrix = np.diag(F0)
W = np.ones((Nv, Nv)) * dv
Integral_Operator = F0_matrix @ W 

# Some empty arrays to store
X_params = np.zeros((N_SAMPLES, 4)) 
Y_solutions = np.zeros((N_SAMPLES, Nt, Nv), dtype=np.complex128)
S_forcing = np.zeros((N_SAMPLES, Nt, Nv), dtype=np.complex128)

# Pre-compute Hermite polynomials matrix for efficiency? 
# Or just compute on the fly. On the fly is fine for 1000 samples.

for i in tqdm(range(N_SAMPLES)):
    # 1. Random System Parameters
    omega_n  = np.random.uniform(*ranges['omega_n'])
    omega_Ti = np.random.uniform(*ranges['omega_Ti'])
    ky       = np.random.uniform(*ranges['ky'])
    kz       = np.random.uniform(*ranges['kz'])
    X_params[i] = [omega_n, omega_Ti, ky, kz]
    
    # 2. Setup A Matrix
    bracket_matrix = np.diag(omega_n + (v**2 - 0.5) * omega_Ti)
    A = -1j*kz*v_matrix - 1j*kz*(v_matrix @ Integral_Operator) - 1j*ky*(bracket_matrix @ Integral_Operator)
    
    # 3. Construct MMS Target Solution (Hermite Expansion)
    w_real_range = [-15,15]
    w_imag_range = [0.2,3.0]
    w_real = np.random.uniform(*w_real_range)
    w_imag = np.random.uniform(*w_imag_range)
    w_mms = w_real - 1j * w_imag
    amp = 0.1
    
    # Generate random coefficients for the 16 polynomials
    # We scale them slightly down as order increases to encourage some smoothness, 
    # though strict convergence isn't required for MMS.
    coeffs = np.random.uniform(-1, 1, N_HERMITE)
    
    # Construct velocity profile: sum(c_n * H_n(v)) * exp(-v^2)
    hermite_sum = np.zeros_like(v)
    for n in range(N_HERMITE):
        hermite_sum += coeffs[n] * eval_hermite(n, v)
        
    # Multiply by Gaussian envelope (standard physics practice to ensure integrability)
    velocity_profile = hermite_sum * np.exp(-v**2)
    
    # Normalize profile to prevent huge numbers (H_15 is very large at v=4)
    # This keeps the amplitude controlled regardless of the random coeffs
    velocity_profile = velocity_profile / np.max(np.abs(velocity_profile))
    
    # Time profile
    time_profile = np.exp(1j * w_mms * t)
    # df/dt = i * w * exp(i * w * t)
    time_deriv   = 1j * w_mms * time_profile
    
    # Combine: f(v, t) = A * T(t) * V(v)
    f_target = amp * np.outer(time_profile, velocity_profile)
    
    # 4. Analytical Derivative df/dt
    df_dt = amp * np.outer(time_deriv, velocity_profile)
    
    # 5. Calculate Forcing S = df/dt - Af
    # We compute (A @ f) for all time steps efficiently
    # f_target has shape (Nt, Nv), A has (Nv, Nv). 
    # We want (A @ f.T).T
    Af = (A @ f_target.T).T
    
    S_forcing[i] = df_dt - Af
    Y_solutions[i] = f_target

# Save all solutions
np.savez('vlasov_ms_new.npz', inputs=X_params, solutions=Y_solutions, forcing=S_forcing)

# --- Plotting to Verify ---
idx = np.random.randint(N_SAMPLES)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

# Plot Real part of f
im1 = ax[0].pcolormesh(v, t, Y_solutions[idx].real, shading='auto', cmap='RdBu_r')
ax[0].set_title(f"Target Solution $f$ (Real)\nSample {idx}")
plt.colorbar(im1, ax=ax[0])

# Plot Real part of S
im2 = ax[1].pcolormesh(v, t, S_forcing[idx].real, shading='auto', cmap='magma')
ax[1].set_title("Required Forcing $S$ (Real)")
plt.colorbar(im2, ax=ax[1])

# Plot the velocity profile at t=0 (or max amplitude time) to see the polynomial structure
f_slice = Y_solutions[idx][Nt//4, :].real # Approx peak time
ax[2].plot(v, f_slice, label='Velocity Slice')
ax[2].set_title("Velocity Structure (Hermite Sum)")
ax[2].set_xlabel("v")
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

hermite_sum = np.zeros_like(v)
for n in range(N_HERMITE):
    hermite_sum = eval_hermite(n, v)

plt.figure()
plt.plot(v, hermite_sum)
plt.show()