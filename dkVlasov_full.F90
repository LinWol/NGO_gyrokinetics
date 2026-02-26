PROGRAM dkVlasov
  IMPLICIT NONE
  INTEGER :: k, l, t, Ntsteps, Nz, Nv, vmin, vmax, k_par, write_output
  REAL :: dt, ky, omega_n, omega_Ti, inv_Nz, pi, dv, inv_2dz, inv_dz2, zmin, zmax, Di, epsilon_D
  COMPLEX :: i
  REAL, ALLOCATABLE :: z(:), vpar(:), Phiabs(:), Phiavg_out(:)
  COMPLEX, ALLOCATABLE :: F(:,:), F0(:), Fft(:), Ftemp(:,:), Phi(:), k1(:,:), k23(:,:)
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! start input parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Ntsteps = 2000 ! number of time steps
  dt = 0.01 ! time step, should obey Courant limit: dt < dz/vmax

  k_par = 1 ! parallel wavenumber = number of periods initialized
  Nz = 16 ! resolution in z; note: z will contain additional boundary points
  Nv = 32 ! resolution in v_par
  epsilon_D = 0.9 ! diffusion term coefficient

  ky = 0.65 ! y-wavenumber
  omega_n = 5.3 ! omega_n = L_z/L_n
  omega_Ti = 54.8 ! omega_Ti = L_z/L_Ti
  
  vmin = - 4 ! box size in v_par in units of thermal velocity
  vmax = 4

  write_output = 0 ! write output files: 0 = none; 1 = Phi(t); 2 = Phi(z); 3 = both
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! end input parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! box size in z (inboard-outboard-inboard); periodic BC's -> zmax at (Nz + 1)
  pi = 2.0 * ASIN(1.0)
  zmin = - pi
  zmax = pi

  ! auxiliary quantities
  i = (0.0,1.0) ! imaginary number
  dv = (vmax - vmin) / (Nv - 1.0)
  inv_2dz = 0.5 * Nz / (zmax - zmin)
  inv_dz2 = (Nz / (zmax - zmin))**2
  inv_Nz = 1.0 / Nz
  Di = epsilon_D * 0.25 / inv_dz2

  ! allocation of arrays
  ALLOCATE(z(1:Nz),vpar(1:Nv),Phiabs(1:Nz),Phiavg_out(1:Ntsteps),k1(1:Nz,1:Nv),k23(1:Nz,1:Nv))
  ALLOCATE(F0(1:Nv),F(0:(Nz+1),1:Nv),Ftemp(0:(Nz+1),1:Nv),Fft(1:Nv),Phi(0:(Nz+1)))
  
  ! create z, v grids, set F0 as Maxwellian
  DO l = 1, Nv
    vpar(l) = vmin + (vmax - vmin) * (l - 1) / (Nv - 1.0)
    F0(l) = EXP(-vpar(l)*vpar(l)) / SQRT(pi)
  END DO
  DO k = 1, Nz
    z(k) = zmin + (zmax - zmin) * (k - 1) * inv_Nz
  END DO

  ! set initial perturbed distribution function F (complex exponential in z, Maxwellian in vpar)
  DO k = 1, Nz
    F(k,:) = 0.0001 * F0(:) * EXP(i*z(k)*k_par*2.0*pi/(zmax-zmin))
  END DO

  ! calculate Phi from F
  CALL Phicalc(Phi,F)
  ! periodic BC's
  CALL perBC(Phi,F)

  ! time stepping (3rd-order Runge-Kutta, or Heun, scheme): get F(t) from F(t-1) in three steps
  DO t = 1, Ntsteps
    ! Heun, step 1
    CALL kieval(k1,F) !! get Vlasov terms
    Ftemp(1:Nz,1:Nv) = F(1:Nz,1:Nv) + dt * k1(1:Nz,1:Nv) * 0.3333333333

    CALL Phicalc(Phi,Ftemp)
    CALL perBC(Phi,Ftemp)
    
    ! Heun, step 2
    CALL kieval(k23,Ftemp)
    Ftemp(1:Nz,1:Nv) = F(1:Nz,1:Nv) + dt * k23(1:Nz,1:Nv) * 0.66666666667

    CALL Phicalc(Phi,Ftemp)
    CALL perBC(Phi,Ftemp)

    ! Heun, step 3
    CALL kieval(k23,Ftemp)
    F(1:Nz,1:Nv) = F(1:Nz,1:Nv) + 0.25 * dt * (k1(1:Nz,1:Nv) + 3.0 * k23(1:Nz,1:Nv))
    
    ! suppress unwanted k_par modes via Fourier transform
    ! forward transform
    Fft(:) = (0.0,0.0)
    DO k = 1, Nz
      Fft(:) = Fft(:) + F(k,:) * EXP(-i*2.0*pi*k_par*k*inv_Nz)
    END DO
    Fft(:) = Fft(:) * inv_Nz
    ! back transform of the singled-out k_par
    DO k = 1, Nz
      F(k,:) = Fft(:) * EXP(i*2.0*pi*k_par*k*inv_Nz)
    END DO

    CALL Phicalc(Phi,F)
    CALL perBC(Phi,F)

    ! save output for this timestep
    Phiavg_out(t) = SUM(ABS(Phi(1:Nz))) * inv_Nz

    ! print instantaneous growth rate
    IF (MODULO(t,100) .EQ. 0) THEN
      PRINT*, "time =", t * dt, ": gamma =", LOG(Phiavg_out(t)/Phiavg_out(t-1)) / dt
    END IF
  END DO ! end time loop

  ! output of z-averaged ABS(Phi)
  IF (MODULO(write_output,2) .EQ. 1) THEN
    OPEN(11,FILE="Phi_of_t.dat")
    DO t = 1, Ntsteps
      WRITE(11,FMT="(G11.4)",ADVANCE="NO") t * dt
      WRITE(11,FMT="(G15.8)") Phiavg_out(t)
    END DO
    CLOSE(11)
  END IF

  ! output of Re(Phi) at last time step
  IF (write_output .GE. 2) THEN
    OPEN(12,FILE="Phi_of_z.dat")
    DO k = 1, Nz
      WRITE(12,FMT="(G11.4)",ADVANCE="NO") z(k)
      WRITE(12,FMT="(G14.7)") REAL(Phi(k))
    END DO
    CLOSE(12)
  END IF

CONTAINS

  ! field equation: calculate Phi(t) from F(t)
  SUBROUTINE Phicalc(Phi,F)
    INTEGER :: k, l
    COMPLEX, DIMENSION (0:(Nz+1),1:Nv) :: F(0:(Nz+1),1:Nv)
    COMPLEX, DIMENSION (0:(Nz+1)) :: Phi(0:(Nz+1))

    Phi(:) = (0.0,0.0)
    DO k = 1, Nz
      Phi(k) = Phi(k) + F(k,1) * dv * 0.5
      DO l = 2, Nv - 1
        Phi(k) = Phi(k) + F(k,l) * dv
      END DO
      Phi(k) = Phi(k) + F(k,Nv) * dv * 0.5
    END DO
  END SUBROUTINE Phicalc

  ! ki evaluator: k1, k2, k3 in Heun
  SUBROUTINE kieval(ki,F)
    INTEGER :: k, l
    COMPLEX, DIMENSION (1:Nz,1:Nv) :: ki(1:Nz,1:Nv)
    COMPLEX, DIMENSION (0:(Nz+1),1:Nv) :: F(0:(Nz+1),1:Nv)

    DO k = 1, Nz
      DO l = 1, Nv
        ki(k,l) = - F0(l) * ((omega_n + omega_Ti * (vpar(l) * vpar(l) - 0.5)) * i * ky * Phi(k) + &
          vpar(l) * (Phi(k+1) - Phi(k-1)) * inv_2dz) - vpar(l) * (F(k+1,l) - F(k-1,l)) * inv_2dz + &
          Di * (F(k+1,l) - 2 * F(k,l) + F(k-1,l)) * inv_dz2
      END DO
    END DO
  END SUBROUTINE
  
  ! apply periodic boundary conditions to Phi, F
  SUBROUTINE perBC(Phi,F)
    COMPLEX, DIMENSION (0:(Nz+1)) :: Phi(0:(Nz+1))
    COMPLEX, DIMENSION (0:(Nz+1),1:Nv) :: F(0:(Nz+1),1:Nv)

    Phi(0) = Phi(Nz)
    Phi(Nz+1) = Phi(1)
    F(0,:) = F(Nz,:)
    F(Nz+1,:) = F(1,:)
  END SUBROUTINE
END PROGRAM dkVlasov