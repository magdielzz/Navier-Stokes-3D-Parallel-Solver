import numpy as np
from scipy.fft import fftn, ifftn
from mpi4py import MPI
from parallel_fft import parallel_fftn, parallel_ifftn
from utils import compute_velocity_from_vorticity

class NavierStokesSolver:
    def __init__(self, grid_size, viscosity, dt, forcing_amplitude):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Parámetros
        self.N = grid_size
        self.nu = viscosity
        self.dt = dt
        self.A = forcing_amplitude
        
        if self.size > self.N:
            if self.rank == 0:
                print(f"Error: Número de procesos ({self.size}) es mayor que el tamaño de la malla ({self.N}).")
            self.comm.Abort(1)
        
        # Distribución de la malla en z
        self.N_local = self.N // self.size
        if self.rank < self.N % self.size:
            self.N_local += 1
        
        # Malla espacial
        self.x = np.linspace(0, 1, self.N, endpoint=False)
        self.y = self.x
        self.z = self.x
        self.dx = 1.0 / self.N
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        
        # Mallas de números de onda locales (en z)
        if self.N_local > 0:
            self.kx, self.ky, self.kz_local = np.meshgrid(
                self.k, self.k, self.k[self.rank * self.N_local:(self.rank + 1) * self.N_local], indexing='ij'
            )
        else:
            self.kx, self.ky, self.kz_local = np.zeros((self.N, self.N, 0)), np.zeros((self.N, self.N, 0)), np.zeros((self.N, self.N, 0))
        
        # Inicializamos vorticidad (3 componentes) con un anillo vortical gaussiano
        self.w = np.zeros((3, self.N, self.N, self.N_local), dtype=np.complex128)
        if self.rank == 0:
            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            self.w[0] = np.exp(-50 * (r - 0.2)**2) * (Y - 0.5)  # Componente x
            self.w[1] = -np.exp(-50 * (r - 0.2)**2) * (X - 0.5)  # Componente y
            self.w[2] = np.zeros_like(X)  # Componente z (opcional)
        self.w = self.comm.bcast(self.w, root=0)
        
        # Precalculamos el Laplaciano
        self.k2 = self.kx**2 + self.ky**2 + self.kz_local**2
        if self.N_local > 0:
            self.k2[0, 0, 0] = 1.0  # Evitar división por cero
        
        if self.rank == 0:
            print(f"Rank {self.rank}: w.shape={self.w.shape}, kx.shape={self.kx.shape}, k2.shape={self.k2.shape}")
    
    def step(self):
        if self.N_local == 0:
            return np.zeros((3, self.N, self.N, 0))
        
        # Calcular campo de velocidades
        u, v, w = compute_velocity_from_vorticity(self.w, self.kx, self.ky, self.kz_local, self.N)
        
        # Evolucionar cada componente de la vorticidad
        w_new = np.zeros_like(self.w)
        for i in range(3):
            w_hat = parallel_fftn(self.w[i], self.comm)
            u_grad_w = self.compute_nonlinear_term(u, v, w, self.w[i], w_hat)
            vortex_stretch = self.compute_vortex_stretching(u, v, w, w_hat)
            laplacian_w = -self.k2 * w_hat
            
            # Forzamiento en x
            if self.rank == 0:
                forcing = self.A * np.sin(2 * np.pi * self.x)[:, None, None] * np.ones((1, self.N, self.N_local))
            else:
                forcing = np.zeros((self.N, self.N, self.N_local))
            forcing_hat = parallel_fftn(forcing, self.comm)
            
            dw_dt = -u_grad_w + vortex_stretch + self.nu * laplacian_w + forcing_hat
            w_hat += self.dt * dw_dt
            w_new[i] = parallel_ifftn(w_hat, self.comm).real
        
        self.w = w_new
        return self.w.real
    
    def compute_nonlinear_term(self, u, v, w, w_i, w_hat_i):
        if self.N_local == 0:
            return np.zeros((self.N, self.N, 0), dtype=np.complex128)
        
        w_i_real = parallel_ifftn(w_hat_i, self.comm).real
        dw_dx = parallel_ifftn(1j * self.kx * w_hat_i, self.comm).real
        dw_dy = parallel_ifftn(1j * self.ky * w_hat_i, self.comm).real
        dw_dz = parallel_ifftn(1j * self.kz_local * w_hat_i, self.comm).real
        nonlinear = u * dw_dx + v * dw_dy + w * dw_dz
        return parallel_fftn(nonlinear, self.comm)
    
    def compute_vortex_stretching(self, u, v, w, w_hat_i):
        if self.N_local == 0:
            return np.zeros((self.N, self.N, 0), dtype=np.complex128)
        
        # Derivadas de u, v, w
        u_hat = parallel_fftn(u, self.comm)
        v_hat = parallel_fftn(v, self.comm)
        w_hat = parallel_fftn(w, self.comm)
        du_dx = parallel_ifftn(1j * self.kx * u_hat, self.comm).real
        du_dy = parallel_ifftn(1j * self.ky * u_hat, self.comm).real
        du_dz = parallel_ifftn(1j * self.kz_local * u_hat, self.comm).real
        dv_dx = parallel_ifftn(1j * self.kx * v_hat, self.comm).real
        dv_dy = parallel_ifftn(1j * self.ky * v_hat, self.comm).real
        dv_dz = parallel_ifftn(1j * self.kz_local * v_hat, self.comm).real
        dw_dx = parallel_ifftn(1j * self.kx * w_hat, self.comm).real
        dw_dy = parallel_ifftn(1j * self.ky * w_hat, self.comm).real
        dw_dz = parallel_ifftn(1j * self.kz_local * w_hat, self.comm).real
        
        # Término de estiramiento: (ω · ∇)u
        w_i_real = parallel_ifftn(w_hat_i, self.comm).real
        stretch = (w_i_real * du_dx + w_i_real * dv_dy + w_i_real * dw_dz)
        return parallel_fftn(stretch, self.comm)