#include "surf_cuda/cuda_mat.h"

namespace surf_cuda{
CudaMat::CudaMat(const int& rows, const int& cols):rows_(rows),cols_(cols),internalAllocated_(false),data(NULL)
{}


CudaMat::CudaMat(const Mat& mat):cols_(mat.cols),rows_(mat.rows),internalAllocated_(false),data(NULL)
{}


CudaMat::~CudaMat()
{
  if(internalAllocated_){
    cudaFree(data);
  }
}


void CudaMat::allocate(){
  //allocate 2D memory 
  cudaError_t err = CudaSafeCall(cudaMallocPitch(&data,&pitch_,cols_*sizeof(float),rows_));
  if(err==cudaSuccess){
    internalAllocated_ = true;
  }
}


void CudaMat::writeDevice(float* hostmem, size_t hostpitch, int width, int height)
{
  if(data==NULL){
    printf("[CUDA] [Write Device], data not allocated\n");
    return;
  }
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D(data, pitch_, hostmem, hostpitch, width*sizeof(float), height, cudaMemcpyHostToDevice));
    if(err==cudaSuccess){
      printf("[CUDA] Wrote %i bytes data to Device\n", width*height*sizeof(float));
    } 
  } else{
    fprintf(stderr,"[CUDA] [Write Device] Dimension of Host Source Memory and Device Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
  }
}

void CudaMat::writeDeviceFromMat_32F(const Mat& mat){
  //check the data type of Mat
  if(mat.type()!=CV_32F){
    printf("[CUDA] [Write Device] Input cv::Mat must of type CV_32F");
    return;
  } else{
    if(mat.isContinuous()==true){
      writeDevice((float*)mat.data,mat.cols*sizeof(float),mat.cols,mat.rows);
    } else{
      writeDevice((float*)mat.data,mat.step,mat.cols,mat.rows);
    }
  }
}


void CudaMat::readDevice(float* hostmem, size_t hostpitch, int width, int height)
{
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D(hostmem,hostpitch, data, pitch_, width*sizeof(float),height, cudaMemcpyDeviceToHost));
    if(err == cudaSuccess){
      printf("[CUDA] Read %i bytes data from Device\n", width*height*sizeof(float));
    }
  } else{
    fprintf(stderr,"[CUDA] [Read Device] Dimension of Device Source Memory and Source Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
    exit(-1);
  }
}

void CudaMat::readDeviceToMat_32F(const Mat& mat){
  if(mat.type()!=CV_32F){
    printf("[CUDA] [Write Device] Input cv::Mat must of type CV_32F");
    return;
  } else{
    if(mat.isContinuous()==true){
      readDevice((float*)mat.data,mat.cols*sizeof(float),mat.cols,mat.rows);
    } else{
      readDevice((float*)mat.data,mat.step,mat.cols,mat.rows);
    }
  }
  
}

__host__ __device__ const int CudaMat::rows() const{
  return rows_;
}
__host__ __device__ const int CudaMat::cols() const{
  return cols_;
}
__host__ __device__ const size_t CudaMat::pitch() const{
  return pitch_;
}
}