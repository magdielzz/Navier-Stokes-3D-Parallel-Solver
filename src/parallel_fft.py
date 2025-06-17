import numpy as np
from mpi4py import MPI
from scipy.fft import fftn, ifftn

def parallel_fftn(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = data.shape[0]
    N_local = data.shape[2] if len(data.shape) > 2 else 0
    
    if rank == 0:
        print(f"parallel_fftn: N={N}, N_local={N_local}, size={size}, data.shape={data.shape}")
    
    if N_local == 0:
        return np.zeros((N, N, 0), dtype=np.complex128)
    
    # FFT en x e y (localmente)
    data_hat = fftn(data, axes=(0, 1), norm='ortho')
    
    # Reunir datos
    local_sendcount = data_hat.size
    sendcounts = np.zeros(size, dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])
    
    displacements = np.zeros(size, dtype=int)
    if rank == 0:
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)
    
    if rank == 0:
        print(f"parallel_fftn: sendcounts={sendcounts}, displacements={displacements}")
    
    # Juntar datos en un arreglo global
    data_hat_global = np.zeros((N, N, N), dtype=np.complex128)
    try:
        comm.Allgatherv(data_hat, [data_hat_global, sendcounts, displacements, MPI.DOUBLE_COMPLEX])
    except Exception as e:
        if rank == 0:
            print(f"Error en Allgatherv: {e}")
        comm.Abort(1)
    
    # FFT en z
    data_hat_global = fftn(data_hat_global, axes=(2,), norm='ortho')
    
    # Distribuir de vuelta a local
    data_hat_local = np.zeros((N, N, N_local), dtype=np.complex128)
    for i in range(size):
        start = displacements[i] // (N * N)
        end = start + (sendcounts[i] // (N * N))
        if rank == i:
            data_hat_local[:, :, :] = data_hat_global[:, :, start:end]
    
    return data_hat_local

def parallel_ifftn(data_hat, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = data_hat.shape[0]
    N_local = data_hat.shape[2] if len(data_hat.shape) > 2 else 0
    
    if rank == 0:
        print(f"parallel_ifftn: N={N}, N_local={N_local}, size={size}, data_hat.shape={data_hat.shape}")
    
    if N_local == 0:
        return np.zeros((N, N, 0), dtype=np.complex128)
    
    # Juntar datos
    local_sendcount = data_hat.size
    sendcounts = np.zeros(size, dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])
    
    displacements = np.zeros(size, dtype=int)
    if rank == 0:
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)
    
    data_global = np.zeros((N, N, N), dtype=np.complex128)
    try:
        comm.Allgatherv(data_hat, [data_global, sendcounts, displacements, MPI.DOUBLE_COMPLEX])
    except Exception as e:
        if rank == 0:
            print(f"Error en Allgatherv: {e}")
        comm.Abort(1)
    
    # IFFT en z
    data_global = ifftn(data_global, axes=(2,), norm='ortho')
    
    # Distribuir de vuelta
    data_local = np.zeros((N, N, N_local), dtype=np.complex128)
    for i in range(size):
        start = displacements[i] // (N * N)
        end = start + (sendcounts[i] // (N * N))
        if rank == i:
            data_local[:, :, :] = data_global[:, :, start:end]
    
    # IFFT en x e y
    data_local = ifftn(data_local, axes=(0, 1), norm='ortho')
    
    return data_local
