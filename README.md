# Solvedor Paralelo de Navier-Stokes 3D

## Resumen del Proyecto

Este proyecto implementa un solvedor paralelo para las ecuaciones de Navier-Stokes en 3D en forma de vorticidad usando un método pseudo-espectral. La simulación evoluciona un campo de vorticidad tridimensional en un dominio periódico, inicializado con un anillo vortical Gaussiano, e incluye una pequeña viscosidad y un término de forzado sinusoidal. El solvedor paraleliza los cálculos entre múltiples procesos usando MPI, con FFTs distribuidas mediante la librería `mpi4py`. El campo final de vorticidad (componente \( $\omega_z \$)) se visualiza como una representación volumétrica o de contornos en 3D.

### Cómo Funciona

* **Física**: Se resuelven las ecuaciones de Navier-Stokes en 3D en forma de vorticidad:
  $\(\frac{\partial \boldsymbol{\omega}}{\partial t} + (\boldsymbol{\omega} \cdot \nabla) \mathbf{u} - (\mathbf{u} \cdot \nabla) \boldsymbol{\omega} = \nu \nabla^2 \boldsymbol{\omega} + \nabla \times \mathbf{f}\)$, donde $\(\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)\)$ es la vorticidad, $\(\mathbf{u} = (u, v, w)\)$ es el campo de velocidad, $\(\nu\)$ es la viscosidad, y $\(\mathbf{f}\)$ es un término de forzado.
* **Método Numérico**: Usa un método pseudo-espectral:

  * Las derivadas espaciales se calculan en el espacio de Fourier usando FFTs paralelas.
  * El paso de tiempo se realiza con Euler hacia adelante.
* **Paralelización**:

  * La rejilla 3D se particiona entre procesos, con cada proceso manejando un subconjunto de planos a lo largo del eje \(z\).
  * Las FFTs se paralelizan usando un enfoque en dos etapas: FFTs locales a lo largo de \(x\) e \(y\), seguidas por FFTs globales a lo largo de \(z\) después de redistribuir los datos vía MPI.
* **Salida**: El campo de vorticidad al tiempo final se reúne en el proceso raíz y se visualiza en 3D como `vorticity_3d.png` usando Mayavi.

### Archivos

* `main.py`: Punto de entrada; configura parámetros, ejecuta la simulación y genera la visualización 3D.
* `src/navier_stokes_solver.py`: Define la clase `NavierStokesSolver` para los pasos de tiempo y cálculos principales en 3D.
* `src/parallel_fft.py`: Implementa funciones de FFT e inversa de FFT paralelas para el dominio 3D.
* `src/utils.py`: Función utilitaria para calcular el campo de velocidad a partir de la vorticidad en 3D.
* `run`: Script bash para ejecutar el programa con MPI.
* `requirements.txt`: Lista de paquetes necesarios en Python.

## Requisitos

* **Sistema**: Ubuntu (probado en WSL sobre Windows) o cualquier sistema con soporte MPI.
* **Software**:

  * Python 3.6 o superior
  * Implementación de MPI (OpenMPI o MPICH recomendado)
* **Paquetes de Python** (listados en `requirements.txt`):

  * `numpy>=1.21.0`
  * `mpi4py>=3.1.0`
  * `mayavi>=4.7.0`
  * `PyQt5>=5.15.0`

## Instrucciones de Configuración

### 1. Clonar o Crear el Directorio del Proyecto

```bash
mkdir ~/navier_stokes_3d_parallel_solver
cd ~/navier_stokes_3d_parallel_solver
mkdir src
### 2. Guardar los Archivos del Proyecto

* Coloca los siguientes archivos en `~/navier_stokes_3d_parallel_solver`:

  * `main.py`
  * `run`
  * `requirements.txt`
  * Este `README.md`
* Coloca los siguientes archivos en `~/navier_stokes_3d_parallel_solver/src`:

  * `navier_stokes_solver.py`
  * `parallel_fft.py`
  * `utils.py`

**Nota**: Asegúrate de que el contenido de estos archivos coincida con las versiones proporcionadas en la documentación del proyecto. Están configurados para funcionar juntos sin errores.

### 3. Instalar Dependencias

Instala los paquetes del sistema y dependencias de Python necesarias.

#### Paquetes del Sistema (Ubuntu/WSL)

```bash
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```

#### Paquetes de Python

```bash
pip3 install --user -r requirements.txt
```

### 4. Hacer Ejecutable `run`

```bash
chmod +x run
```

### 5. Ejecutar la Simulación

Ejecuta la simulación con 4 procesos:

```bash
./run
```

### 6. Ver la Salida

La simulación generará `vorticity_3d.png` en el directorio del proyecto. Si estás corriendo en WSL, copia el archivo a Windows para verlo:

```bash
cp vorticity_3d.png /mnt/c/Users/TuNombreDeUsuarioWindows/Desktop/
```

Abre `vorticity_3d.png` para ver el campo de vorticidad 3D al tiempo final.

## Salida Esperada

* **Salida en Consola**: Mensajes de progreso, incluyendo:

  ```
  Iniciando la simulación 3D con 4 procesos
  Solver inicializado: grid_size=32, N_local=8, w.shape=(3, 32, 32, 8)
  Paso 0/2000, Tiempo: 0.000, w.shape=(3, 32, 32, 8)
  Paso 2000/2000, Tiempo: 1.000, w.shape=(3, 32, 32, 8)
  Simulación terminada, imagen vorticity_3d.png generada
  ```

* **Gráfico**: `vorticity_3d.png` muestra un anillo vortical Gaussiano centrado en (0.5, 0.5, 0.5) con vorticidad pico de \~0.08, difundido por la viscosidad.

## Solución de Problemas

* **Errores de MPI**:

  * Si encuentras errores relacionados con MPI, intenta reinstalar `mpi4py`:

    ```bash
    pip3 uninstall -y mpi4py
    pip3 install --user --force-reinstall mpi4py
    ```

  * Alternativamente, usa MPICH en vez de OpenMPI:

    ```bash
    sudo apt remove --purge openmpi-bin openmpi-common libopenmpi-dev
    sudo apt install -y mpich libmpich-dev
    pip3 install --user --force-reinstall mpi4py
    ```

* **Problemas en WSL**:

  * Aumenta los recursos de WSL editando `/mnt/c/Users/TuNombreDeUsuarioWindows/.wslconfig`:

    ```
    [wsl2]
    memory=4GB
    processors=4
    ```

    Reinicia WSL:

    ```bash
    wsl --shutdown
    ```

* **No se Genera el Gráfico**:

  * Ejecuta en modo de un solo proceso para depurar:

    ```bash
    python3 main.py
    ```

  * Revisa si hay errores de `mayavi` y asegúrate de que esté instalado correctamente junto con `PyQt5`.

## Personalización

* **Parámetros de Simulación** (en `main.py`):

  * `grid_size`: Aumenta a 64 para mayor resolución.
  * `T`: Aumenta a 5.0 para correr por más tiempo.
  * `forcing_amplitude`: Aumenta a 0.5 para un forzado más fuerte.

* **Condición Inicial** (en `navier_stokes_solver.py`):

  * Agrega múltiples anillos vorticales:

    ```python
    self.w[0] = (np.exp(-50 * ((X - 0.3)**2 + (Y - 0.3)**2 + (Z - 0.3)**2 - 0.2)**2) * (Y - 0.3) +
                  np.exp(-50 * ((X - 0.7)**2 + (Y - 0.7)**2 + (Z - 0.7)**2 - 0.2)**2) * (Y - 0.7))
    self.w[1] = (-np.exp(-50 * ((X - 0.3)**2 + (Y - 0.3)**2 + (Z - 0.3)**2 - 0.2)**2) * (X - 0.3) +
                  -np.exp(-50 * ((X - 0.7)**2 + (Y - 0.7)**2 + (Z - 0.7)**2 - 0.2)**2) * (X - 0.7))
    ```


