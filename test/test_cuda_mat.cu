#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/cuda_util.cuh"

using namespace surf_cuda;
template<typename T>
__global__ void kernel_memset_1(unsigned char* devmem, size_t pitch_bytes, int rows, int cols){
  int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(row_idx<rows){
    T* ptr = (T*)((char*)devmem + row_idx * pitch_bytes);
    for(int c=0; c<cols; c++){
      ptr[c]=1;
    }
  }
}


int main(){
  int rows=100;
  int cols=100;
  CudaMat m_uchar_gpu(rows, cols, CV_8U);
  CudaMat m_int_gpu(rows, cols, CV_32S);
  CudaMat m_float_gpu(rows, cols, CV_32F);
  CudaMat m_double_gpu(rows, cols, CV_64F);
  Mat m_uchar_cpu= Mat::zeros(rows, cols, CV_8U);
  Mat m_int_cpu=Mat::zeros(rows, cols, CV_32S);
  Mat m_float_cpu=Mat::zeros(rows, cols, CV_32F);
  Mat m_double_cpu=Mat::zeros(rows, cols, CV_64F);
  
  m_uchar_gpu.allocate();
  m_int_gpu.allocate();
  m_float_gpu.allocate();
  m_double_gpu.allocate();
  
  m_uchar_gpu.copyFromMat(m_uchar_cpu);
  m_int_gpu.copyFromMat(m_int_cpu);
  m_float_gpu.copyFromMat(m_float_cpu);
  m_double_gpu.copyFromMat(m_double_cpu);
  
  int block_dim = 128;
  dim3 block(128,1,1);
  dim3 grid(rows/block_dim+1,1,1);
  
  kernel_memset_1<unsigned char> <<<grid,block>>>(m_uchar_gpu.data, m_uchar_gpu.pitch_bytes(), m_uchar_gpu.rows(), m_uchar_gpu.cols());
  kernel_memset_1<int> <<<grid,block>>>(m_int_gpu.data, m_int_gpu.pitch_bytes(), m_int_gpu.rows(), m_int_gpu.cols());
  kernel_memset_1<float> <<<grid,block>>>(m_float_gpu.data, m_float_gpu.pitch_bytes(), m_float_gpu.rows(), m_float_gpu.cols());
  kernel_memset_1<double> <<<grid,block>>>(m_double_gpu.data, m_double_gpu.pitch_bytes(), m_double_gpu.rows(), m_double_gpu.cols());
  
  m_uchar_gpu.copyToMat(m_uchar_cpu);
  m_int_gpu.copyToMat(m_int_cpu);
  m_float_gpu.copyToMat(m_float_cpu);
  m_double_gpu.copyToMat(m_double_cpu);
  cout<<m_uchar_cpu<<endl;
  cout<<m_int_cpu<<endl;
  cout<<m_float_cpu<<endl;
  cout<<m_double_cpu<<endl;
  
}