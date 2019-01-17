#include "surf_cuda/cuda_mat.h"

namespace surf_cuda{
CudaMat::CudaMat(const size_t& width, const size_t& height):width_(width),height_(height),internalAllocated_(false),data(NULL)
{}


CudaMat::CudaMat(const Mat& mat):width_(mat.cols),height_(mat.rows),internalAllocated_(false),data(NULL)
{}


CudaMat::~CudaMat()
{
  if(internalAllocated_){
    cudaFree(data);
  }
}


void CudaMat::allocate(){
  //allocate 2D memory 
  cudaError_t err = cudaMallocPitch(&data,&pitch_,width_*sizeof(float),height_);
  //TODO: Error Handle
  if(err==cudaSuccess){
    internalAllocated_ = true;
  } else{
    printf("[CUDA] Bad Allocation: <%s>\n",cudaGetErrorString(err));
  }
}


void CudaMat::writeDevice(float* hostmem, size_t hostpitch, size_t width, size_t height)
{
  if(data==NULL){
    printf("[CUDA] [Write Device], data not allocated\n");
    return;
  }
  if(width==width_ && height==height_){
    cudaError_t err = cudaMemcpy2D(data, pitch_, hostmem, hostpitch, width*sizeof(float), height, cudaMemcpyHostToDevice);
    if(err==cudaSuccess){
      printf("[CUDA] Wrote %i bytes data to Device\n", width*height*sizeof(float));
    } else{
      printf("[CUDA] [Write Device] Data transaction Error, <%s>\n", cudaGetErrorString(err));
    }
  } else{
    printf("[CUDA] [Write Device] Dimension of Host Source Memory and Device Memory do not Match\n");
    printf("[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    printf("[CUDA] Device Memory width (in element): %i, height(in element): %i \n",width_, height_);
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


void CudaMat::readDevice(float* hostmem, size_t hostpitch, size_t width, size_t height)
{
  if(width==width_ && height==height_){
    cudaError_t err = cudaMemcpy2D(hostmem,hostpitch, data, pitch_, width*sizeof(float),height, cudaMemcpyDeviceToHost);
    if(err == cudaSuccess){
      printf("[CUDA] Read %i bytes data from Device\n", width*height*sizeof(float));
    } else{
      printf("[CUDA] [Read Device] Data transaction Error, <%s>\n",cudaGetErrorString(err));
    }
  } else{
    printf("[CUDA] [Read Device] Dimension of Device Source Memory and Source Memory do not Match\n");
    printf("[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    printf("[CUDA] Device Memory width (in element): %i, height(in element): %i \n",width_, height_);
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

__host__ __device__ const size_t CudaMat::rows() const{
  return height_;
}
__host__ __device__ const size_t CudaMat::cols() const{
  return width_;
}
__host__ __device__ const size_t CudaMat::pitch() const{
  return pitch_;
}
}