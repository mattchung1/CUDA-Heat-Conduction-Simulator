# CUDA 2D Heat Conduction Simulator

A high-performance parallel computing application that simulates steady-state heat distribution across a 2D plate. This project implements the **Finite Difference Method** (Jacobi Iteration) using **NVIDIA CUDA** to accelerate calculations on the GPU, achieving significant speedups over serial CPU implementations.

Heat Simulation Demo
https://youtu.be/QkSclliFg-I

## Features
* **GPU Acceleration:** Utilizes custom CUDA kernels to perform parallel Jacobi iterations on the GPU.
* **Optimized Memory Management:** Efficient data transfer between Host (CPU) and Device (GPU) memory.
* **Data Visualization:** Includes a Python script to animate the heat propagation and convergence using `matplotlib`.
* **Configurable Simulation:** Supports command-line arguments to adjust grid size and iteration count.

## Prerequisites

To run this simulation, you need the following hardware and software:

* **Hardware:** NVIDIA GPU with CUDA Compute Capability.
* **Software:**
    * [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Tested on v12.6).
    * Microsoft Visual Studio 2022 (for the MSVC compiler `cl.exe`).
    * Python 3.x.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mattchung1/CUDA-Heat-Conduction-Simulator.git
    cd CUDA-Heat-Conduction-Simulator/code
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install numpy matplotlib pandas
    ```

## Compilation (Windows)

This project requires the NVIDIA CUDA Compiler (`nvcc`) and the MSVC compiler (`cl.exe`).

1.  Open the **x64 Native Tools Command Prompt for VS 2022** (Search for this in the Windows Start menu).
2.  Navigate to the `code` directory.
3.  Compile the source code:
    ```cmd
    nvcc Lab4.cu -o Lab4 -arch=native
    ```
    *Note: The `-arch=native` flag optimizes the build for your specific GPU architecture.*

## Usage

### 1. Run the Simulation
Run the executable to perform the calculations. The program will generate a data file (e.g., `finalTemperatures.csv` or binary output).

```cmd
.\Lab4.exe
