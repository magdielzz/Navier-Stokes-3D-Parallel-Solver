import sys
sys.path.append('src')
import numpy as np
from mpi4py import MPI
from navier_stokes_solver import NavierStokesSolver
from mayavi import mlab  # Import mayavi for 3D visualization

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"Iniciando la simulación 3D con {comm.Get_size()} procesos")
    
    # Parámetros de la simulación
    grid_size = 64            # Tamaño de la malla (N x N x N)
    viscosity = 1e-4          # Viscosidad del fluido
    dt = 0.001                # Paso de tiempo
    T = 0.5                   # Tiempo total de simulación (aumentado para evolución)
    forcing_amplitude = 0.1   # Amplitud del forzamiento
    n_steps = int(T / dt)     # Número de pasos de tiempo
    
    # Inicializamos el solucionador
    try:
        solver = NavierStokesSolver(grid_size, viscosity, dt, forcing_amplitude)
        if rank == 0:
            print(f"Solver inicializado: grid_size={grid_size}, N_local={solver.N_local}, w.shape={solver.w.shape}")
    except Exception as e:
        if rank == 0:
            print(f"Error al inicializar el solver: {e}")
        comm.Abort(1)
    
    # Ejecutamos la simulación
    for step in range(n_steps):
        try:
            w = solver.step()
            if rank == 0 and step % 100 == 0:
                print(f"Paso {step}/{n_steps}, Tiempo: {step * dt:.3f}, w.shape={w.shape}")
        except Exception as e:
            if rank == 0:
                print(f"Error en el paso {step}: {e}")
            comm.Abort(1)
    
    # Recolectamos el campo de vorticidad (componente z) para visualización
    if rank == 0:
        print("Recolectando resultados para visualizar")
    w_global = None
    if rank == 0:
        w_global = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    
    local_sendcount = solver.w[2].size  # Componente z de la vorticidad
    sendcounts = np.zeros(comm.Get_size(), dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])
    
    displacements = np.zeros(comm.Get_size(), dtype=int)
    if rank == 0:
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)
    
    try:
        if rank == 0:
            print(f"Gatherv: sendcounts={sendcounts}, displacements={displacements}, w_global.shape={w_global.shape}")
        comm.Gatherv(solver.w[2], [w_global, sendcounts, displacements, MPI.DOUBLE], root=0)
    except Exception as e:
        if rank == 0:
            print(f"Error en Gatherv: {e}")
        comm.Abort(1)
    
    # Visualización 3D usando mayavi (solo rank 0)
    if rank == 0:
        print("Graficando resultados en 3D")
        x, y, z = np.meshgrid(solver.x, solver.y, solver.z, indexing='ij')
        mlab.contour3d(x, y, z, w_global, contours=10, transparent=True, colormap='viridis')
        mlab.colorbar(title='Vorticidad (componente z)', orientation='vertical')
        mlab.title(f'Vorticidad de Navier-Stokes 3D en t={T:.2f}, nu={viscosity}')
        mlab.xlabel('x')
        mlab.ylabel('y')
        mlab.zlabel('z')
        mlab.show()  # Muestra la ventana interactiva
        print("Simulación terminada, visualización 3D generada")

if __name__ == "__main__":
    main()