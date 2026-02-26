import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# --- 1. Setup Parameters (No Nz needed!) ---
Nv = 32
vmax = 4
vmin = -4
v = np.linspace(vmin, vmax, Nv)
dv = (vmax - vmin) / (Nv - 1)

# Physics Parameters
omega_n  = 6.88 #11.8877
omega_Ti = 50.1 #45.2720
ky = 0.89 #0.2026
kz = 2.155 #2.6729  # The single Fourier mode you are solving for

# --- 2. Construct Matrices (Nv x Nv only) ---

# Velocity vector as a diagonal matrix
v_matrix = np.diag(v)

# Equilibrium F0 (Maxwellian)
F0 = (1.0 / np.pi**0.5) * np.exp(-v**2)
F0_matrix = np.diag(F0)

# The integration weight matrix (row vector of ones * dv)
# This performs the integral over v for every row
W = np.ones((Nv, Nv)) * dv

# The bracket term: [omega_n + (v^2 - 0.5) * omega_Ti]
# This varies with v, so it is a diagonal matrix
bracket_term = omega_n + (v**2 - 0.5) * omega_Ti
bracket_matrix = np.diag(bracket_term)

# --- 3. Assemble the Vlasov Operator A ---
# Equation: df/dt = -ikz*v*f - ikz*v*F0*Phi - [bracket]*F0*iky*Phi

# Term 1: -v_parallel * d/dz f  ->  -i * kz * v * f
A1 = -1j * kz * v_matrix

# Term 2: -v_parallel * F0 * d/dz Phi -> -i * kz * v * F0 * (Integral f dv)
# Note: F0_matrix @ W creates a matrix where every column is F0 * dv.
# When this multiplies vector f, it computes F0 * sum(f*dv).
A2 = -1j * kz * (v_matrix @ F0_matrix @ W)

# Term 3: - [bracket] * F0 * i * ky * Phi
A3 = -1j * ky * (bracket_matrix @ F0_matrix @ W)

# Total Matrix
A = A1 + A2 + A3

# --- 4. Solve Eigenvalues ---
eigen_values, eigen_vectors = eig(A)

print(f"Max growth rate: {np.max(eigen_values.real):.4f}")
print(np.linalg.matrix_rank(eigen_vectors))

plt.scatter(eigen_values.real, eigen_values.imag,marker='x')


plt.xlabel('Growth rate ($\\gamma$)')
plt.ylabel('Frequency')
plt.title('Plot of Eigenvalues')

# --- Scientific-style formatting ---
plt.tick_params(
    direction='in',   # ticks inside
    which='both',     # apply to major and minor ticks
    top=True,          # show ticks on top
    right=True         # show ticks on right
)

# Optional: add minor ticks
plt.minorticks_on()

# Optional: slightly thicker axis lines
ax = plt.gca()
ax.spines['top'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

plt.show()