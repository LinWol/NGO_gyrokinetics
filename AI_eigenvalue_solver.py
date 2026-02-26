import numpy as np
from numpy.linalg import eig
# your original small test grid
Nz = 32
Nv = 32
zmin = -np.pi
zmax = np.pi
z = np.linspace(zmin, zmax, Nz)
dz = z[1] - z[0]

vmax = 4.0
vmin = -4.0
v = np.linspace(vmin, vmax, Nv)
dv = (vmax - vmin) / (Nv - 1)

# Example F0 and derived vectors (kept from you)
F0 = (1.0/np.sqrt(np.pi))*np.exp(-v**2)
omega_n  = 5.0
omega_Ti = 10.0
ky = 0.1

# helper derivative (kept)
def first_derivative_matrix(N, h):
    D = np.zeros((N, N))
    for i in range(N):
        D[i, (i-1) % N] = -1/(2*h)
        D[i, (i+1) % N] =  1/(2*h)
    return D

Dz = first_derivative_matrix(Nz, dz)

# weights (kept)
w = np.ones(Nv) * dv
w[0] *= 0.5
w[-1] *= 0.5

# derived vectors (kept)
u = - v * F0        # coefficient for Term A (sign according to your eqn)
c = -1j * ky * (omega_n + (v**2 - 0.5)*omega_Ti) * F0

# -----------------------
# ---- CHANGES START ----
# -----------------------

# Total size
N = Nz * Nv

# 1) Correct streaming term (term1): build block-diagonal -v_l * Dz using kron
# previously you had: A_1 = np.diag(-v) @ Dz  <-- WRONG (32x32 only)
A1 = np.kron(np.diag(-v), Dz)   # shape (Nv*Nz, Nv*Nz)

print(np.shape(A1))
# 2) Build S: operator that maps flattened f -> Phi(z) by integrating over v with weights w
#    S has shape (Nz, Nz*Nv). For flattened ordering p = l*Nz + i (v slow, z fast),
#    S = [ w0*I_z | w1*I_z | ... | w_{Nv-1}*I_z ]
I_z = np.eye(Nz)
S_blocks = [w[m] * I_z for m in range(Nv)]
S = np.hstack(S_blocks)        # shape (Nz, Nz*Nv)

# Dz @ S maps flattened f -> dPhi/dz (shape Nz x Nz*Nv)
DzS = Dz @ S

# 3) Build A2 and A3 which couple across v via S and DzS.
#    For each v-block (rows l*Nz:(l+1)*Nz) multiply DzS (or S) by the prefactors:
A2 = np.zeros((N, N), dtype=complex)   # for term 2: - v_l * F0_l * dPhi/dz
A3 = np.zeros((N, N), dtype=complex)   # for term 3: c_l * Phi(z)

for l in range(Nv):
    row_slice = slice(l*Nz, (l+1)*Nz)
    coeff2 = - v[l] * F0[l]        # matches Fortran: -vpar(l) * F0(l) * dPhi/dz
    coeff3 = -1j * ky * (omega_n + (v[l]**2 - 0.5) * omega_Ti) * F0[l]  # same as c[l]

    # assign the block rows for this v
    # A2: each v-block's rows get coeff2 * (Dz @ S)
    A2[row_slice, :] = coeff2 * DzS

    # A3: each v-block's rows get coeff3 * S   (Phi -> multiplied by prefactor, identity in z)
    A3[row_slice, :] = coeff3 * S

# Combine operators (full (Nz*Nv)x(Nz*Nv) matrix)
A = A1 + A2 + A3

# -----------------------
# ---- CHANGES END  -----
# -----------------------

# eigenvalues (dense eig for clarity; for larger problems use sparse eigs)
eigen_values, eigen_vectors = eig(A)

# interpret growth rate: Fortran used i*ky ... so growth is imag part
growth_rates = eigen_values.imag
max_growth_idx = np.argmax(growth_rates)
max_growth_rate = growth_rates[max_growth_idx]

print("max growth rate (imag part):", max_growth_rate)
