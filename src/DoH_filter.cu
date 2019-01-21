#include "surf_cuda/DoH_filter.h"

namespace surf_cuda {
//kernel function that take inpute integral image, stride, BoxFilters to compute output response map 
__global__ void kernel_DoH_Filter(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride, BoxFilter_xx bf_xx, BoxFilter_xy bf_xy, BoxFilter_yy bf_yy, float weight){
  //first check if the output response map with current stride parameter can fit in the memory of input response map
  //will not use the fast few columns for sub sampling 
  if(response_rows==integral_rows/stride || response_cols==integral_cols/stride){
    //Just one kernel per row's computation 
    //TODO: upgrade the parallelism 
    int row_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      float* row_response_addr = (float*)((char*)response_mat + row_response_idx * response_pitch);
      int row_integral_idx = row_response_idx * stride;
      //loop through columns 
      //TODO: Add unrolling 
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = bf_xx(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols) * 
				bf_yy(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols) -
				  weight * powf(bf_xy(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols),2);
      }
    }
  }
}
void DoHFilter::operator()(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride)
{
  //launch kernel 
  dim3 block(32,1,1);
  dim3 grid(response_rows/32+1,1,1);
  kernel_DoH_Filter<<<grid, block>>>(integral_mat,integral_pitch,integral_rows, integral_cols, response_mat,response_pitch,response_rows,response_cols,stride,box_filter_xx,box_filter_xy,box_filter_yy,weight);
  CudaCheckError();
  //sync
  cudaDeviceSynchronize();
}

  
}