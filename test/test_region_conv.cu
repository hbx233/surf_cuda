#include "surf_cuda/common.h"
#include "surf_cuda/DoH_filter.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/surf.h"
using surf_cuda::CudaMat;
using surf_cuda::SURF;
using surf_cuda::WeightedRegionIntegral;

__global__ void test_kernel_weighted_region(WeightedRegionIntegral<int> region,unsigned char* integral_mat, unsigned char* weight_conv_mat, size_t pitch, int rows, int cols){
  int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_idx<rows){
    //float* row_addr_in = (float*)((char*)integral_mat + row_idx * pitch);
    int* row_addr_out = (int*)(weight_conv_mat + row_idx * pitch);
    for(int i=0; i<cols; i++){
      row_addr_out[i] = region(integral_mat,pitch, row_idx, i, rows, cols);
    }
  }
}

__global__ void test_kernel_weighted_region_tex(WeightedRegionIntegral<int> region, cudaTextureObject_t integral_mat_tex, unsigned char* weight_conv_mat, size_t pitch, int rows, int cols){
  int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_idx<rows){
    //float* row_addr_in = (float*)((char*)integral_mat + row_idx * pitch);
    int* row_addr_out = (int*)(weight_conv_mat + row_idx * pitch);
    for(int i=0; i<cols; i++){
      row_addr_out[i] = region(integral_mat_tex,pitch, row_idx, i, rows, cols);
    }
  }
}

int main(){
  int rows = 20;
  int cols = 10;
  Mat mat_in = Mat::ones(rows,cols,CV_32F);
  Mat mat_integral = Mat::zeros(rows,cols,CV_32F);
  Mat mat_conv = Mat::zeros(rows,cols,CV_32F);
  CudaMat cuda_mat_in(mat_in);
  CudaMat cuda_mat_integral(mat_in);
  CudaMat cuda_mat_conv(mat_in);
  //allocate two CudaMat
  cuda_mat_in.allocate();
  cuda_mat_integral.allocate();
  cuda_mat_conv.allocate();
  cout<<cuda_mat_in.pitch()<<endl;
  cout<<cuda_mat_integral.pitch()<<endl;
  cout<<cuda_mat_conv.pitch()<<endl;
  
  //write data to in 
  cuda_mat_in.writeDeviceFromMat_32F(mat_in);
  SURF surf;
  //compute integral image 
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral,32,32);
  //read integral image to host
  cuda_mat_integral.readDeviceToMat_32F(mat_integral);
  cout<<mat_integral<<endl;
  //compute weighted region sum 
  //WeightedRegionIntegral wri(-4,-4,-2,-2,1);
  WeightedRegionIntegral wri(2,2,4,4,1);
  dim3 block(32,1,1);
  dim3 grid(1,1,1);
  printf("rows: %i\n",cuda_mat_conv.rows());
  test_kernel_weighted_region<<<block,grid>>>(wri, cuda_mat_integral.data, cuda_mat_conv.data, cuda_mat_integral.pitch(), cuda_mat_integral.rows(), cuda_mat_integral.cols());
  cudaDeviceSynchronize();
  //copy back 
  cuda_mat_conv.readDeviceToMat_32F(mat_conv);
  cout<<mat_conv<<endl;
}