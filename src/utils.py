import numpy as np
from mpi4py import MPI
from parallel_fft import parallel_fftn, parallel_ifftn

def compute_velocity_from_vorticity(w, kx, ky, kz, N):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if w.shape[2] == 0:
        return np.zeros((N, N, 0)), np.zeros((N, N, 0)), np.zeros((N, N, 0))
    
    if rank == 0:
        print(f"compute_velocity: w.shape={w.shape}, kx.shape={kx.shape}, ky.shape={ky.shape}, kz.shape={kz.shape}")
    
    # FFT de las componentes de vorticidad
    w_hat = np.zeros((3, N, N, w.shape[2]), dtype=np.complex128)
    for i in range(3):
        w_hat[i] = parallel_fftn(w[i], comm)
    
    # Calcular k^2
    k2 = kx**2 + ky**2 + kz**2
    if w.shape[2] > 0:
        k2[0, 0, 0] = 1.0
    
    # Resolver ecuación de Poisson para el vector potencial A: ∇²A = -ω
    A_hat = np.zeros_like(w_hat)
    for i in range(3):
        A_hat[i] = -w_hat[i] / k2
    
    # Velocidad: u = ∇ × A
    u_hat = 1j * (ky * A_hat[2] - kz * A_hat[1])
    v_hat = 1j * (kz * A_hat[0] - kx * A_hat[2])
    w_hat_vel = 1j * (kx * A_hat[1] - ky * A_hat[0])
    
    # Transformar de vuelta
    u = parallel_ifftn(u_hat, comm).real
    v = parallel_ifftn(v_hat, comm).real
    w = parallel_ifftn(w_hat_vel, comm).real
    
    return u, v, w