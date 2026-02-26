epsilon_D = [0,0.5,0.9]

for epsilon in epsilon_D:
    import numpy as np
    from scipy.sparse import kron, eye, diags, csr_matrix
    from scipy.sparse.linalg import eigs
    from scipy.linalg import block_diag
    import matplotlib.pyplot as plt

    Nz = 24
    Nv = 32
    zmin = -np.pi
    zmax = np.pi
    z = np.linspace(zmin, zmax, Nz)
    dz = (zmax - zmin) / Nz

    vmax = 4
    vmin = -4
    v = np.linspace(vmin, vmax, Nv)
    dv = (vmax - vmin) / (Nv - 1)
    omega_n  = 3.0
    omega_Ti = 30.0
    ky = 0.3
    epsilon_D = epsilon



    def first_derivative_matrix(N, h,k):

        D = np.zeros((N, N))
        
        for i in range(N):
            D[i, (i-1) % N] = -1/(2*h)
            D[i, (i+1) % N] =  1/(2*h)
        
        D_full = np.kron(np.eye(k, dtype=int), D)


        return D_full

    def second_derivative_matrix(N, h, k):
        
        D_2 = np.zeros((N, N))
        
        for i in range(N):
            D_2[i, i]           = -2.0 / (h**2)
            D_2[i, (i-1) % N]   =  1.0 / (h**2)
            D_2[i, (i+1) % N]   =  1.0 / (h**2)
        
        D_full_second = np.kron(np.eye(k, dtype=int), D_2)

        return D_full_second



    Dz_full = first_derivative_matrix(Nz,dz,Nv)
    Dz_2_full = second_derivative_matrix(Nz,dz,Nv)


    w = np.ones(Nv) * dv
    w[0] *= 0.5
    w[-1] *= 0.5


    I_small = np.identity(Nz)*dv
    # I_small[0,0] *= 0.5
    # I_small[-1,-1] *=0.5

    I_full = np.tile(I_small,(Nv,Nv))



    blocks = []
    for vi in v:
        # choose one of the two:
        block = np.identity(Nz)*vi     
        # block = vi * np.eye(Nz)  # OR each block = v[i] * I_Nz
        blocks.append(block)

    # assemble block-diagonal matrix (dense)
    v_big = block_diag(*blocks)      # shape (Nv*Nz, Nv*Nz)

    blocks_2 = []
    for vi in v:
        block2 = np.identity(Nz) * vi * (1.0/np.sqrt(np.pi))*np.exp(-vi**2)
        blocks_2.append(block2)

    v_F0_big = block_diag(*blocks_2)

    blocks_3 = []
    for vi in v:
        block3 = np.identity(Nz) * 1j * ky * (omega_n + (vi**2 - 0.5)*omega_Ti) * (1.0/np.sqrt(np.pi))*np.exp(-vi**2)
        blocks_3.append(block3)

    c_big = block_diag(*blocks_3)

    blocks_4 = []
    for vi in v:
        block4 = np.identity(Nz) * epsilon_D
        blocks_4.append(block4)

    epsilon_matrix = block_diag(*blocks_4)

    # dPhi_dz_op = R @ (Dz @ S) 
    # Phi_op = R @ S  

    dphi_dz = Dz_full @ I_full
    diffusion_term = (dz/2)**2 * epsilon_D

    A_1 = -v_big @ Dz_full
    A_2 = -v_F0_big @ dphi_dz
    A_3 = -c_big @ I_full
    A_4= diffusion_term * Dz_2_full


    A = A_1 + A_2 + A_3 + A_4

    eigen_values, eigen_vectors = np.linalg.eig(A)

    growth_rates = eigen_values.real
    max_idx = np.argmax(growth_rates)

    max_growth_rate = growth_rates[max_idx]
    max_frequency = eigen_values[max_idx].imag
    eigen_vector_max = eigen_vectors[:, max_idx]

    growing_modes_list = []
    idx_list = []
    for idx, item in enumerate(growth_rates):
        if item > 0.1:
            growing_modes_list.append(item)
            idx_list.append(idx)

    pairs = sorted(zip(growing_modes_list, idx_list))
    growing_modes_list, idx_list = map(list, zip(*pairs))

    # print(growing_modes_list)

    # print(len(idx_list)*2)
    # print(max_growth_rate)
    # #print(eigen_vector_max)

    # eigen_vector = eigen_vectors[:, idx_list[7]]
    # eigen_vector_2 = eigen_vectors[:,idx_list[4]]

    # mode = eigen_vector_max.reshape(Nv, Nz)
    # mode_1 = eigen_vector.reshape(Nv,Nz)
    # mode_2 = eigen_vector_2.reshape(Nv,Nz)

    k_list = []
    growth_list = []

    # Precompute parallel wavenumber grid for Nz points
    f_grid = np.fft.fftfreq(Nz, d=dz)   # cycles per unit length along z
    k_grid = 2 * np.pi * f_grid         # parallel wavenumber (rad/unit)

    for item in idx_list:
        eigen_vector_loop = eigen_vectors[:, item]
        mode_loop = eigen_vector_loop.reshape(Nv, Nz)
        #fz_loop = mode_loop[Nv//2, :]        # slice along z (parallel structure)
        fz_loop = np.sum(mode_loop,axis=0)
        
        F_loop = np.fft.fft(fz_loop)         # FFT along z
        mag = np.abs(F_loop)
        
        peak_idx = np.argmax(mag)
        k_peak = k_grid[peak_idx]            # parallel wavenumber
        k_abs = np.abs(k_peak)               # optional: positive axis only

        k_list.append(k_abs)
        growth_list.append(growth_rates[item])  # keep pairing correct

    k_list_sorted, growth_list_sorted = zip(*sorted(zip(k_list, growth_list)))

    # plot growth rate vs parallel wavenumber
    plt.plot(k_list_sorted, growth_list_sorted,label=epsilon)
    plt.xlabel('$k_\\parallel$')
    plt.ylabel('Growth rate')
    plt.ylim(0,1)

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


    # fz = mode_2[Nv//2,:]   


    # F = np.fft.fft(fz)                 # complex spectrum (indices 0..Nz-1)
    # k_par = np.argmax(np.abs(F))         # index of largest amplitude

    # if k_par < Nz/2: 
    #     f_par = k_par / (Nz * dz)
    # if k_par > Nz/2:
    #     f_par = (k_par - Nz) / (Nz * dz)

    # k_parr = 2 * np.pi * f_par

    # k_abs = abs(k_parr)

plt.legend()
plt.show()