#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"
namespace surf_cuda{
#if 0
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
#endif
template <typename T>
__global__ void compRowIntegral(cudaTextureObject_t mat_in_tex, unsigned char* mat_out, size_t rows, size_t cols, size_t pitch_bytes){
  int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_idx<rows){
        //T* row_addr_in = (T*)(mat_in + row_idx*pitch_bytes);
        T* row_addr_out = (T*)(mat_out + row_idx*pitch_bytes);
	//compute integral along the row 
        T integral_cache;
//#pragma unroll 4
	for(size_t c=0;c<cols; c++){
            if(c==0){
                //initial value
                integral_cache=tex2D<T>(mat_in_tex,c,row_idx);
                row_addr_out[c]=integral_cache;
            } else{
                integral_cache = tex2D<T>(mat_in_tex,c,row_idx) + integral_cache;
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
  compRowIntegral<int> <<<block_row,grid_row>>>(img_in.texture_object(), img_integral.data(),img_in.rows(), img_in.cols(), img_in.pitch_bytes());
  //sync
  cudaDeviceSynchronize();
  //then compute integral along cols
  dim3 block_col(block_dim_x_col,1,1);
  dim3 grid_col(img_in.rows()/block_dim_x_col + 1,1,1);
  compColIntegral<int> <<<block_row, grid_row>>>(img_integral.data(), img_integral.data(),img_integral.rows(), img_integral.cols(), img_integral.pitch_bytes());
  cudaDeviceSynchronize();
}

void SURF::allocateInputAndIntegralImage(){
  //allocate input image memory on device 
  cuda_img_in_.allocate();
  cuda_img_in_.allocateArray();
  //allocate integral image memory on device 
  cuda_img_integral_.allocate();
  cuda_img_integral_.allocateArray();
  //set integral image's Texture object 
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeBorder;
  texDesc.addressMode[1]   = cudaAddressModeBorder;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  
  //set texture object 
  cuda_img_integral_.setTextureObjectInterface(texDesc);
}

void SURF::allocateOctaves(const vector<vector<int>>& filter_sizes, const vector<int>& strides){
  //initialize octaves with specified filter size and stride 
  for(int o=0; o<filter_sizes.size(); o++){
    octaves_.push_back(Octave(rows_/strides[o],cols_/strides[o],strides[o],filter_sizes[o]));
  }
  //allocate memories for every octave 
  for(int o=0; o<filter_sizes.size(); o++){
    octaves_[o].allocateMemAndArray();//internally set the texture object 
  }
}

void SURF::extractKeyPoints(Mat img_input){
  //copy image to Device Memory 
  cuda_img_in_.copyFromMat(img_input);
  //compute integral image 
  compIntegralImage(cuda_img_in_,cuda_img_integral_);
  //copy integral image to texture memory 
  cuda_img_integral_.copyToArray();
  //fill all octaves with DoH response maps 
  for(int o=0; o<octaves_.size();o++){
    octaves_[o].fill(cuda_img_integral_);
  }
  for(int o=0; o<octaves_.size();o++){
    octaves_[o].copyResponseMapsToArray();
  }
  for(int o=0; o<octaves_.size();o++){
    octaves_[o].thresholdNonMaxSupAndFindKeyPoints(threshold_);
  }
  //copy keypoints 
  for(int o=0; o<octaves_.size();o++){
    for(int i=0; i<octaves_[o].keypoints_num; i++){
      keypoints_.push_back(cv::Point2f(octaves_[o].keypoints_x[i],octaves_[o].keypoints_y[i]));
    }
  }
}



}