/*
Author: Myung Chung
Class: ECE6122 Section A
Last Date Modified: 10/30/2025

Description: 
This program performs the calculation of the steady state heat of a 2D plate in response to a constant heat source.
CUDA is used to run the calculations in the GPU in parallel.

*/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <math.h>
#include <fstream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cmath>

#define N 100000000
#define MAX_ERR 1e-6
int numPoints = 256; // Default number of points along one side of the square
int numIterations = 10000; // Default number of iterations



/*
*********************************
Calculate Heat Conduction
*********************************
*/
__global__ void heat_calc(double *g_new, double *h_old, int gridSize) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (i > 0 && i < (gridSize - 1) && j > 0 && j < (gridSize - 1)) 
    {
        int index = j * gridSize + i;

        g_new[index] = 0.25 * (h_old[index - 1] + h_old[index + 1] + h_old[index - gridSize] + h_old[index + gridSize]);
    }
}



int main(int argc, char* argv[])
{
    /*
    *********************************
    ARGUMENT HANDLING
    *********************************
    */
    for (int count = 1; count < argc; count++)
    {
        std::string arg = argv[count];

        // Set number of threads
        if (arg == "-N")
        {
            if (count + 1 < argc) 
            { 
                // Set the number of points along square to user submitted value
                numPoints = std::stoi(argv[count + 1]);
                count++; // Increment count to skip next argument since it has been processed
            } 
            else
            {
                std::cerr << "Error: -N flag requires a number after it.\n";
                return 1; // Exit the program with an error code
            }
        }
        else if(arg == "-I")
        {
            if (count + 1 < argc) 
            { 
                // Set the number of iterations to user submitted value
                numIterations = std::stoi(argv[count + 1]);
                count++;
            } 
            else
            {
                std::cerr << "Error: -I flag requires a number after it.\n";
                return 1; // Exit the program with an error code
            }
        }
        else if(arg == "-q")
        {
            return 0; // Quit program
        }
        else
        {
            std::cerr << "Error: Unknown argument " << arg << "\n";
            return 1; // Exit the program with an error code
        }
    }

    /*
    *********************************
    INITIALIZATION
    *********************************
    */
    // Create first and second temperature grid g and h
    int gridSize = numPoints + 2;
    int arraySize = gridSize * gridSize;
    double * g = (double*)malloc(arraySize * sizeof(double));
    double * h = (double*)malloc(arraySize * sizeof(double));

    // Initialize heat source boundaries ( truncation )
    int heatSourceStart = (int)((numPoints + 1) * (3.0f/10.0f));
    int heatSourceEnd = (int)((numPoints + 1) * (7.0f/10.0f));

    // Use this if you want to round instead and include cmath
    // int heatSourceStart = (int) round(((numPoints + 1) * (3.0f/10.0f)));
    // int heatSourceEnd = (int) round(((numPoints + 1) * (7.0f/10.0f)));


    // Initialize first grid h with boundary conditions
    for(int i = 0; i <= (numPoints + 1); i++) // i is row index
    {
        for(int j = 0; j <= (numPoints + 1); j++) // j is column index
        {
            int index = j * gridSize + i;

            if (j == 0 || i == gridSize - 1 || j == gridSize - 1)
            {
                h[index] = 20.0; // Left, Right, and Bottom boundaries
            }
            else if(i == 0)
            {
                if(j >= heatSourceStart && j <= heatSourceEnd)
                {
                    h[index] = 100.0; // Top boundary with heat source
                }
                else
                {
                    h[index] = 20.0; // Top boundary without heat source
                }
            }
            else
            {
                h[index] = 0.0; // Initial interior points
            }

        }
    }

    std::copy(h, h + arraySize, g); // Copy initial grid h to g


    // Initialize device memory
    double *d_g, *d_h;

    cudaMalloc((void**)&d_g, sizeof(double) * arraySize);
    cudaMalloc((void**)&d_h, sizeof(double) * arraySize);

    cudaMemcpy(d_h, h, sizeof(double) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g, sizeof(double) * arraySize, cudaMemcpyHostToDevice);


    /*
    *********************************
    CUDA EVENT TIMING SETUP
    *********************************
    */
    cudaEvent_t start_event, stop_event;

    // Create events
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);


    /*
    *********************************
    ITERATION LOOP (KERNEL LAUNCH)
    *********************************
    */
    // Define grid and block dimensions
    // Number of blocks is rounded up to cover all points
    dim3 THREADS_PER_BLOCK(16, 16);
    dim3 NUM_BLOCKS((gridSize + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
                    (gridSize + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);

    // Record the start time
    cudaEventRecord(start_event, 0);

    // Perform iterations
    for(int iter = 0; iter < numIterations; iter++)
    {
        heat_calc<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_g, d_h, gridSize);

        // Swap pointers for subsequent iteration
        std::swap(d_g, d_h);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    /*
    *********************************
    DATA TRANSFER: DEVICE -> HOST
    *********************************
    */
    // Receive data from device
    cudaMemcpy(g, d_h, sizeof(double) * arraySize, cudaMemcpyDeviceToHost);


    // Calculate time
    float ms = 0;
    cudaEventElapsedTime(&ms, start_event, stop_event);


    /*
    *********************************
    OUTPUT
    *********************************
    */
    // Print final grid into a text file called "finalTemperatures.csv" and each temperature value is on a separate line
    // Also prints number of milliseconds taken to reach steady state

    // Print execution time
    std::cout << "Time elapsed: " << ms << " ms \n";

    std::ofstream file("finalTemperatures.csv");
    if(file.is_open())
    {
        for(int i = 0; i <= (numPoints + 1); i++) // i is row index
        {
            for(int j = 0; j <= (numPoints + 1); j++) // j is column index
            {
                int index = j * gridSize + i;

                file << g[index];

                if(j < gridSize - 1)
                {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Error: Could not open output file.\n";
        return 1; // Exit the program with an error code
    }


    /*
    *********************************
    CLEANUP
    *********************************
    */
    // Free device memory
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    // Deallocate device memory
    cudaFree(d_g);
    cudaFree(d_h);

    // Free host memory
    free(g);
    free(h);


    return 0;
}