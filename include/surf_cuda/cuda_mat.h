#ifndef CUDA_MAT_H_
#define CUDA_MAT_H_
#include "surf_cuda/common.h"

namespace surf_cuda{
class CudaMat{
public:
  CudaMat(const size_t& width, const size_t& height);
  CudaMat(const Mat& mat);//const ref of header
  ~CudaMat();
  //only allocated 2D new memory at the beginning, reallocated only have to change the size 
  void allocate();
  //write data on host to Device 
  void writeDevice(float* hostmem, size_t hostpitch, size_t width, size_t height);
  //read Device data to allocated memory on host
  void readDevice(float* hostmem, size_t hostpitch, size_t width, size_t height);
  //write Mat data on host to Device 
  void writeDeviceFromMat_32F(const Mat& mat);
  //read Device data to Mat on host 
  void readDeviceToMat_32F(const Mat& mat);
public:
  float* data;
  __host__ __device__ const size_t rows() const;
  __host__ __device__ const size_t cols() const;
  __host__ __device__ const size_t pitch() const;
private:
  //number of rows and columns 
  size_t width_;//column number
  size_t height_;//row number 
  size_t pitch_;
  //set true if allocated using allocate(), and will deallocate when destruct
  bool internalAllocated_;
};

 



}
#endif /* CUDA_MAT_H_*/