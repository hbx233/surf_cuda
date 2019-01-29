#include "surf_cuda/octave.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/common.h"
namespace surf_cuda {
Octave::Octave(const int& rows, const int& cols, const int& stride, const std::vector<int>& filters_size)
: rows_(rows), cols_(cols), stride_(stride),level_num_(filters_size.size())
{
  response_maps.clear();
  filters.clear();
  for(int i=0; i<filters_size.size(); i++){
    //create response maps, cannot use vector initialization 
    response_maps.emplace_back(rows_,cols_,CV_32F);
    //create filters with given size 
    filters.push_back(DoHFilter(filters_size[i]));
  }
}

  
void Octave::allocateMemAndArray()
{
  //allocate memory for every level of DoH response map in the Octave
  for(int i=0; i<response_maps.size(); i++){
    response_maps[i].allocate();
    response_maps[i].allocateArray();
  }
}

void Octave::fill(const CudaMat& integral_mat){
  //check the compatibility of stride and size 
  bool valid = false;
  if(stride_==1){
    //if stride is one, size should be identical 
    if(integral_mat.rows()==rows_ && integral_mat.cols()==cols_){
      valid = true;
    }
  } else{
    if(integral_mat.rows()/stride_ == rows_ && integral_mat.cols()/stride_ == cols_){
      valid = true;
    }
  }
  if(valid){
    //compute response maps for every filter 
    for(int i=0;i<level_num_;i++){
      compDoHResponseMap_texture(integral_mat,response_maps[i],filters[i],stride_);
    }
  } else{
    fprintf(stderr, "[CUDA] The input image size is not sompatible with current size and stride");
  }
}

void Octave::copyResponseMapsToArray()
{
  //copy response maps to their allocated and set the texture object 
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;//set the address mode to be Clamp for min max computation in sliding window
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  for(int i=0; i<response_maps.size(); i++){
    response_maps[i].copyToArray();
    response_maps[i].setTextureObjectInterface(texDesc);
  }
}

void Octave::readDoHResponseMap(vector< Mat >& images_cpu)
{
  //allocate memory, assume the provided header is not always compatible with current image size  
  images_cpu.clear();
  images_cpu = vector<Mat>(level_num_);
  for(int i=0; i<level_num_; i++){
    images_cpu[i] = Mat::ones(rows_,cols_,CV_32F);
  }
  //read device memory 
  for(int i=0; i<level_num_; i++){
    response_maps[i].copyToMat(images_cpu[i]);
    cout<<"Computed Map result"<<endl;
  }
}

void Octave::readDoHResponseMapAfterSupression(vector< Mat >& images_cpu)
{
  //allocate memory, assume the provided header is not always compatible with current image size  
  images_cpu.clear();
  images_cpu = vector<Mat>(level_num_-2);
  for(int i=0; i<level_num_-2; i++){
    images_cpu[i] = Mat::ones(rows_,cols_,CV_32F);
  }
  //read device memory 
  for(int i=1; i<level_num_-1; i++){
    response_maps[i].copyToMat(images_cpu[i-1]);
    cout<<"Computed Map result"<<endl;
  }
}

__device__ float find_max_in_3x3_window(cudaTextureObject_t& map, int& row, int& col){
  float max_row_1 = fmaxf(fmaxf(tex2D<float>(map,col-1,row-1), tex2D<float>(map,col,row-1)), tex2D<float>(map,col+1,row-1));
  float max_row_2 = fmaxf(fmaxf(tex2D<float>(map,col-1,row  ), tex2D<float>(map,col,row  )), tex2D<float>(map,col+1,row  ));
  float max_row_3 = fmaxf(fmaxf(tex2D<float>(map,col-1,row+1), tex2D<float>(map,col,row+1)), tex2D<float>(map,col+1,row+1));
  return fmaxf(fmaxf(max_row_1,max_row_2),max_row_3);
}
__device__ float find_min_in_3x3_window(cudaTextureObject_t& map, int& row, int& col){
  float max_row_1 = fminf(fminf(tex2D<float>(map,col-1,row-1), tex2D<float>(map,col,row-1)), tex2D<float>(map,col+1,row-1));
  float max_row_2 = fminf(fminf(tex2D<float>(map,col-1,row  ), tex2D<float>(map,col,row  )), tex2D<float>(map,col+1,row  ));
  float max_row_3 = fminf(fminf(tex2D<float>(map,col-1,row+1), tex2D<float>(map,col,row+1)), tex2D<float>(map,col+1,row+1));
  return fminf(fminf(max_row_1,max_row_2),max_row_3);
}

__global__ void kernel_threshold_nonMaxSuppression(cudaTextureObject_t map_low, cudaTextureObject_t map_mid, cudaTextureObject_t map_up, unsigned char* map_out, int rows, int cols, size_t out_pitch_bytes, float threshold){
  //assume texture object in clamp mode 
  //get the row index 
  int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
  //check validity 
  if(row_idx<rows){
    //get output row coordinate 
    //TODO: implement shared memory here 
    int col_idx = threadIdx.y + blockDim.y * blockIdx.y;
    if(col_idx<cols){
    float* row_ptr_out = (float*)(map_out + out_pitch_bytes * row_idx);
    //for(int c=0; c<cols; c++){
      //first apply thresholding, only do non max suppression for the point in middle level 
      //whose response value exceed the threshold 
      float response_value = tex2D<float>(map_mid,col_idx,row_idx);
      //have warp divergence here, I think it cannot be avoided, since some thread have to execute the following 
      //code to get min and max while others don't have to 
      int larger_than_threshold = 0;
      int is_max = 0;
      int is_min = 0;
      int max_not_eq_min = 0;
      if(response_value>=threshold || response_value<=-threshold){
	//typically, first several comparison will exclude the point to be extrema, 
	//but here do full comparison any way for less divergence problem 
	larger_than_threshold = 1;
        float max_low = find_max_in_3x3_window(map_low,row_idx,col_idx);
        float max_mid = find_max_in_3x3_window(map_mid,row_idx,col_idx);
        float max_up = find_max_in_3x3_window(map_up,row_idx,col_idx);
        float min_low = find_min_in_3x3_window(map_low,row_idx,col_idx);
        float min_mid = find_min_in_3x3_window(map_mid,row_idx,col_idx);
        float min_up = find_min_in_3x3_window(map_up,row_idx,col_idx);
	float maximum = fmaxf(fmaxf(max_low,max_mid),max_up);
	float minimum = fminf(fminf(min_low,min_mid),min_up);
	is_max = (int)(maximum == response_value);
	is_min = (int)(minimum == response_value);
	//if maximum equals to minimum, 3x3x3 response values are all the same, not extrema 
	max_not_eq_min = (int)(maximum != minimum);
	//avoid warp divergence, response value cannot be maximum and minimum at the same time 
      }
      row_ptr_out[col_idx] = larger_than_threshold * max_not_eq_min * (is_max + is_min) * response_value;
    }
  }
}


void Octave::thresholdAndNonMaxSuppression()
{
  //loop through all the middle level response maps in octave
  int block_dim_x = 32; //block size along row axis 
  int block_dim_y = 32;
  dim3 block(block_dim_x,block_dim_y,1);
  dim3 grid(rows_/block_dim_x + 1,cols_/block_dim_y + 1,1);
  for(int l = 1; l<response_maps.size()-1; l++){
    kernel_threshold_nonMaxSuppression<<<grid,block>>>(response_maps[l-1].texture_object(),response_maps[l].texture_object(),response_maps[l+1].texture_object(),response_maps[l].data(),rows_,cols_,response_maps[l].pitch_bytes(),threshold_);
    CudaCheckError();
  }
}
}  
