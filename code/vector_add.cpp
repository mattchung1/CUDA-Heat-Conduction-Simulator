//https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono> 
#include <iostream>

#define MAX_ERR 1e-6
#define N 100'000'000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    // Record the starting time
    auto start_time = std::chrono::steady_clock::now();
    // Main function
    vector_add(out, a, b, N);
        // Record the starting time
    auto end_time = std::chrono::steady_clock::now();

     // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print the duration
    std::cout << "Execution time: " << duration.count() << " milliseconds." << std::endl;
  
       // Verification
    for(int i = 0; i < N; i++)
    {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");
    
    free(a); 
    free(b); 
    free(out);
}