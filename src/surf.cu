#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"
namespace surf_cuda{
  
template <typename T>
__global__ void compRowIntegral(unsigned char* mat_in, unsigned char* mat_out, size_t rows, size_t cols, size_t pitch_bytes){
  int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_idx<rows){
        T* row_addr_in = (T*)(mat_in + row_idx*pitch_bytes);
        T* row_addr_out = (T*)(mat_out + row_idx*pitch_bytes);
	//compute integral along the row 
        T integral_cache;
//#pragma unroll 4
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
template <typename T>
__global__ void compColIntegral(unsigned char* mat_in, unsigned char* mat_out, size_t rows, size_t cols, size_t pitch_bytes){
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(col_idx<cols){
    T integral_cache;
//#pragma unroll 4
    for(int r=0; r<rows; r++){
      //compute row address
      T* row_addr_in = (T*)(mat_in + r*pitch_bytes);
      T* row_addr_out = (T*)(mat_out + r*pitch_bytes);
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

void SURF::compIntegralImage(const CudaMat& img_in, const CudaMat& img_integral){
  printf("[CUDA] Computing Integral Image\n");
  //first compute integral along rows
  size_t block_dim_x_row = 128;
  size_t block_dim_x_col = 128;
  dim3 block_row(block_dim_x_row,1,1);
  dim3 grid_row(img_in.cols()/block_dim_x_row + 1,1,1);
  compRowIntegral<int> <<<block_row,grid_row>>>(img_in.data, img_integral.data,img_in.rows(), img_in.cols(), img_in.pitch_bytes());
  //sync
  cudaDeviceSynchronize();
  //then compute integral along cols
  dim3 block_col(block_dim_x_col,1,1);
  dim3 grid_col(img_in.rows()/block_dim_x_col + 1,1,1);
  compColIntegral<int> <<<block_row, grid_row>>>(img_integral.data, img_integral.data,img_integral.rows(), img_integral.cols(), img_integral.pitch_bytes());
  cudaDeviceSynchronize();
}

__global__ void kernel_DoH_Filter(unsigned char* integral_mat, size_t integral_pitch_bytes, int integral_rows, int integral_cols, unsigned char* response_mat, size_t response_pitch_bytes, int response_rows, int response_cols, int stride, DoHFilter doh_filter){
  //first check if the output response map with current stride parameter can fit in the memory of input response map
  //will not use the fast few columns for sub sampling 
  if(response_rows==integral_rows/stride || response_cols==integral_cols/stride){
    //Just one kernel per row's computation 
    int row_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      float* row_response_addr = (float*)(response_mat + row_response_idx * response_pitch_bytes);
      int row_integral_idx = row_response_idx * stride;
      //loop through columns 
//#pragma unroll 4
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = doh_filter(integral_mat, integral_pitch_bytes, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
  }
}

void SURF::compDoHBlobResponseMap(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter ,const int& stride){
  //check CudaMat type 
  if(img_integral.type()==CV_32S && img_doh_response.type() == CV_32F){
    size_t block_dim_x = 128;
    dim3 block(block_dim_x,1,1);
    dim3 grid(img_doh_response.rows()/block_dim_x + 1,1,1);
    kernel_DoH_Filter<<<grid,block>>> (img_integral.data,img_integral.pitch_bytes(),img_integral.rows(),img_integral.cols(),img_doh_response.data,img_doh_response.pitch_bytes(),img_doh_response.rows(),img_doh_response.cols(),stride, doh_filter);
    CudaCheckError();
    cudaDeviceSynchronize();
  } else{
    fprintf(stderr,"[CUDA] [DoH Response Map] The CudaMat type should be CV_32S for inpute, CV_32F for output");
  }
}




}