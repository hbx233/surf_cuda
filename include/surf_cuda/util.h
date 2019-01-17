#include "surf_cuda/common.h"
//compare memory in data1 and data2
//data1: pointer to data1 consecutive memory
//data2: pointer to data2 consecutive memory
//size: size of element 
//msg: provide message to print for comparison 
template <typename T> 
bool compare(T* data1, T* data2, size_t size, char msg[]){
    printf("%s\n",msg);
    for(int i=0; i<size; i++){
        if((data1[i]-data2[i])>=0.000001){
            printf("[COMPARE] Different\n");
            return false;
        }
        //printf("%f, %f\n",data1[i], data2[i]);
    }
    printf("[COMPARE] Same\n");
    return true;
}

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}