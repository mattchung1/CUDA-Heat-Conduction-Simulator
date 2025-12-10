# CUDA 2D Heat Conduction Simulator

A high-performance parallel computing application that simulates steady-state heat distribution across a 2D plate. This project implements the **Finite Difference Method** (Jacobi Iteration) using **NVIDIA CUDA** to accelerate calculations on the GPU, achieving significant speedups over serial CPU implementations.

### 🎥 Simulation Demo
[![Heat Simulation Demo](https://img.youtube.com/vi/QkSclliFg-I/0.jpg)](https://youtu.be/QkSclliFg-I)

> *Click the image above to watch the demonstration video.*

## Features
* **GPU Acceleration:** Utilizes custom CUDA kernels to perform parallel Jacobi iterations on the GPU.
* **Optimized Memory Management:** Efficient data transfer between Host (CPU) and Device (GPU) memory.
* **Data Visualization:** Includes a Python script to animate the heat propagation and convergence using `matplotlib`.
* **Configurable Simulation:** Supports command-line arguments to adjust grid size and iteration count.

## Mathematical Model & Algorithm

The simulation solves for the steady-state temperature distribution on a 2D plate, which is governed by **Laplace's Equation**.

### 1. Governing Equation
For a 2D region with no internal heat generation, the steady-state temperature $T(x,y)$ satisfies:

$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

### 2. Discretization (Finite Difference Method)
The continuous 2D plate is discretized into a grid of nodes $(i, j)$ with uniform spacing $\Delta x$ and $\Delta y$.



The second-order partial derivatives are approximated using the central difference formula. Assuming $\Delta x = \Delta y = h$:

$$\frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{h^2}$$

$$\frac{\partial^2 T}{\partial y^2} \approx \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{h^2}$$

Substituting these back into Laplace's equation yields the discrete form:

$$T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j} = 0$$

### 3. Jacobi Iteration Formula
To solve this system numerically, we rearrange the equation to solve for $T_{i,j}$. This leads to an iterative update rule where the new temperature at a node is the average of its four neighbors from the previous iteration step $(k)$.

$$T_{i,j}^{(k+1)} = \frac{1}{4} \left( T_{i+1,j}^{(k)} + T_{i-1,j}^{(k)} + T_{i,j+1}^{(k)} + T_{i,j-1}^{(k)} \right)$$

### 4. Parallel Execution Logic
A key property of the Jacobi method is that the calculation for a node $T_{i,j}^{(k+1)}$ depends **only** on values from the previous step $(k)$. This independence allows us to map each grid node to a separate CUDA thread, calculating the updates for millions of nodes simultaneously on the GPU.


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
