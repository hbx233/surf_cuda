#ifndef CUDA_MAT_H_
#define CUDA_MAT_H_
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"
namespace surf_cuda{
class CudaMat{
public:
  CudaMat(const int& rows, const int& cols);
  CudaMat(const Mat& mat);//const ref of header
  ~CudaMat();
  //only allocated 2D new memory at the beginning, reallocated only have to change the size 
  void allocate();
  //write data on host to Device 
  void writeDevice(float* hostmem, size_t hostpitch, int width, int height);
  //read Device data to allocated memory on host
  void readDevice(float* hostmem, size_t hostpitch, int width, int height);
  //write Mat data on host to Device 
  void writeDeviceFromMat_32F(const Mat& mat);
  //read Device data to Mat on host 
  void readDeviceToMat_32F(const Mat& mat);
public:
  float* data;
  __host__ __device__ const int rows() const;
  __host__ __device__ const int cols() const;
  __host__ __device__ const size_t pitch() const;
private:
  //number of rows and columns 
  int cols_;//column number
  int rows_;//row number 
  size_t pitch_;
  //set true if allocated using allocate(), and will deallocate when destruct
  bool internalAllocated_;
};

 



}
#endif /* CUDA_MAT_H_*/