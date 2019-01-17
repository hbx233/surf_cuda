#include "surf_cuda/surf.h"

namespace surf_cuda{
__global__ void compRowIntegral(float* mat_in, float* mat_out, size_t rows, size_t cols, size_t pitch){
  size_t row_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_idx<rows){
        float* row_addr_in = (float*)((char*)mat_in + row_idx*pitch);
        float* row_addr_out = (float*)((char*)mat_out + row_idx*pitch);
	//compute integral along the row 
        float integral_cache;
        for(size_t c=0;c<cols; c++){
            if(c==0){
                //initial value
                integral_cache=row_addr_in[c];
                row_addr_out[c]=integral_cache;
            } else{
                integral_cache = row_addr_in[c] + integral_cache;
                row_addr_out[c] = integral_cache;
            }
        }
    }
}
__global__ void compColIntegral(float* mat_in, float* mat_out, size_t rows, size_t cols, size_t pitch){
  size_t col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(col_idx<cols){
    float integral_cache;
    for(int r=0; r<rows; r++){
      //compute row address
      float* row_addr_in = (float*)((char*)mat_in + r*pitch);
      float* row_addr_out = (float*)((char*)mat_out + r*pitch);
      if(r==0){
	integral_cache = row_addr_in[col_idx];
	row_addr_out[col_idx] = integral_cache;
      } else{
	integral_cache = row_addr_in[col_idx]+integral_cache;
	row_addr_out[col_idx] = integral_cache;
      }
    }
  }
}

void SURF::compIntegralImage(const CudaMat& img_in, const CudaMat& img_out, const size_t& block_size_row, const size_t& block_size_col){
  printf("[CUDA] Computing Integral Image\n");
  //first compute integral along rows
  dim3 block_row(block_size_row,1,1);
  dim3 grid_row(img_in.cols()/block_size_row + 1,1,1);
  compRowIntegral<<<block_row,grid_row>>>(img_in.data, img_out.data,img_in.rows(), img_in.cols(), img_in.pitch());
  //sync
  cudaDeviceSynchronize();
  //then compute integral along cols
  dim3 block_col(block_size_col,1,1);
  dim3 grid_col(img_in.rows()/block_size_col + 1,1,1);
  compColIntegral<<<block_row, grid_row>>>(img_out.data, img_out.data,img_out.rows(), img_out.cols(), img_out.pitch());
  cudaDeviceSynchronize();
}

}