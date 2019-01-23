#include "surf_cuda/cuda_mat.h"

namespace surf_cuda{

void CudaMat::allocate()
{
  cudaError_t err;
  bool leagal_type=true;
  switch(type_){
    //8bit unsigned char 
    case CV_8U:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(unsigned char),rows_));
      depth_ = sizeof(unsigned char);
      elemSize_ = depth_;
      break;
    case CV_32S:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(int),rows_));
      depth_ = sizeof(int);
      elemSize_ = depth_;
      break;
    case CV_32F:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(float),rows_));
      depth_ = sizeof(float);
      elemSize_ = depth_;
      break;
    case CV_64F:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(double),rows_));
      depth_ = sizeof(double);
      elemSize_ = depth_;
      break;
    default:
      //TODO: Error handling 
      fprintf(stderr,"Unsupported depth");
      leagal_type=false;
      exit(-1);
      break;
  }
  if(err==cudaSuccess && leagal_type==true){
    internalAllocated_ = true;
  }
}


void CudaMat::writeDevice(void* hostmem, size_t hostpitch_bytes, int width, int height)
{
  if(data==NULL){
    fprintf(stderr,"[CUDA] [Write Device], data not allocated\n");
    return;
  }
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D((void*)data, pitch_bytes_, (void*)hostmem, hostpitch_bytes, width*elemSize_, height, cudaMemcpyHostToDevice));
    if(err==cudaSuccess){
      printf("[CUDA] Wrote %i bytes data to Device\n", width*height*elemSize_);
    } 
  } else{
    fprintf(stderr,"[CUDA] [Write Device] Dimension of Host Source Memory and Device Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
  }
}

void CudaMat::readDevice(void* hostmem, size_t hostpitch_bytes, int width, int height){
  if(data==NULL){
    fprintf(stderr,"[CUDA] [Read Device], data not allocated\n");
    return;
  }
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D((void*)hostmem,hostpitch_bytes, (void*)data, pitch_bytes_, width*elemSize_,height, cudaMemcpyDeviceToHost));
    if(err == cudaSuccess){
      printf("[CUDA] Read %i bytes data from Device\n", width*height*elemSize_);
    }
  } else{
    fprintf(stderr,"[CUDA] [Read Device] Dimension of Device Source Memory and Source Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
    exit(-1);
  }
}

void CudaMat::copyFromMat(const Mat& mat)
{
  if(mat.type()==type_){
    writeDevice((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Write Device] Mat type not compatible");
    exit(-1);
  }
}
void CudaMat::copyToMat(Mat& mat)
{
  if(mat.type()==type_){
    mat = Mat(rows_, cols_, type_);
    readDevice((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Read device] Mat type not compatible");
  }
}



__host__ __device__ const int CudaMat::rows() const{
  return rows_;
}
__host__ __device__ const int CudaMat::cols() const{
  return cols_;
}
__host__ __device__ const size_t CudaMat::pitch_bytes() const{
  return pitch_bytes_;
}
__host__ __device__ const int CudaMat::depth() const{
  return depth_;
}
__host__ __device__ const int CudaMat::type() const{
  return type_;
}
__host__ __device__ const int CudaMat::elemSize() const{
  return elemSize_;
}
}