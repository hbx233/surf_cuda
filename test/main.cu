#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include "cuda_runtime.h"
__global__ void add_two_vectors(int* v1, int* v2, int* result){
  int idx = threadIdx.x;
  result[idx] = v1[idx] + v2[idx];
  //printf("%i, ",result[idx]);
}
int main(int argc, char **argv) {
    int* v1_host = (int*)malloc(64*sizeof(int));
    int* v2_host = (int*)malloc(64*sizeof(int));
    int* result_host = (int*)malloc(64*sizeof(int));
    //memset(v1_host,64,1);
    //memset(v2_host,64,2);
    for(int i=0;i<64;i++){
      v1_host[i] = 1;
      v2_host[i] = 2;
    }
    int* v1_dev;
    int* v2_dev;
    int* result_dev;
    cudaMalloc(&v1_dev, 64*sizeof(int));
    cudaMalloc(&v2_dev, 64*sizeof(int));
    cudaMalloc(&result_dev, 64*sizeof(int));
    //copy memory 
    cudaMemcpy(v1_dev,v1_host,64*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev,v2_host,64*sizeof(int),cudaMemcpyHostToDevice);
    //launch kernel 
    dim3 grid(1,1,1);
    dim3 block(64,1,1);
    add_two_vectors<<<grid, block>>>(v1_dev, v2_dev, result_dev);
    //sync
    cudaDeviceSynchronize();
    //copy memory from device to host 
    cudaMemcpy(result_host, result_dev, 64*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<64;i++){
        printf("%i, ",result_host[i]);
    }
    free(v1_host);
    free(v2_host);
    free(result_host);
    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(result_dev);
    std::shared_ptr<int> iptr = std::make_shared<int>(4);
    printf("%i",*iptr);
}
