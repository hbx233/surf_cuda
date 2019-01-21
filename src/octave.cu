#include "surf_cuda/octave.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/common.h"
namespace surf_cuda {
Octave::Octave(const int& level_num, const int& rows, const int& cols, const std::vector< int >& filters_size)
:level_num_(level_num),rows_(rows), cols_(cols)
{
  if(level_num != filters_size.size()){
    fprintf(stderr, "[CUDA] Number of Filters is not same as number of levels in Octave");
  } else{
    //create CudaMat header
    images = vector<CudaMat>(level_num_, CudaMat(rows_,cols_));
    //create filters 
    //filters = vector<DoHFilter>(level_num_);
    for(int i=0;i<filters_size.size();i++){
      //create filters with specific size for each level of image 
      filters.push_back(DoHFilter(filters_size[i]));
    }
  }
}

  
void Octave::allocateGpu()
{
  //allocate memory for every level 
  for(int i=0; i<images.size(); i++){
    images[i].allocate();
  }
}


//kernel functions that calculate one blob response map from integral image 
__global__ void filter_kernel
		(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, 
		 float* response_mat, size_t response_pitch, int response_rows, int response_cols, 
		 DoHFilter filter, 
		 int stride
		)
{
  if(response_rows==integral_rows/stride || response_cols==integral_cols/stride){
    //One Kernel just compute one row's blob response  
    //TODO: upgrade the parallelism 
    int row_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      float* row_response_addr = (float*)((char*)response_mat + row_response_idx * response_pitch);
      int row_integral_idx = row_response_idx * stride;
      //loop through columns 
      //TODO: May add loop unrolling here
      #pragma unroll 5
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = filter(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
  }
}

void Octave::fill(const CudaMat& integral_mat, int stride)
{
  //first check if the current Octave image size compatible with the provided stride and integral_mat 
  if(integral_mat.rows()/stride == rows_ && integral_mat.cols()/stride == cols_ ){
    //launch kernels that calculate DoH response 
    int blockDim_x = 128;
    dim3 block(blockDim_x,1,1);
    dim3 grid(rows_/blockDim_x+1,1,1);
    //compute each level's blob response map 
    for(int i=0; i<level_num_; i++){
      cout<<"Filter size:  "<<filters[i].size<<endl;
      filter_kernel<<<grid,block>>>(integral_mat.data,integral_mat.pitch(),integral_mat.rows(), integral_mat.cols(),
			  	    images[i].data,   images[i].pitch(),   images[i].rows(),    images[i].cols(),
			  	    filters[i],
			  	    stride);
      CudaCheckError();
    }
  } else{
    fprintf(stderr, "[CUDA] [Wrong Dimension] Provided Stride not compatible");
  }
}

void Octave::readDoHResponseMap(vector< Mat >& images_cpu)
{
  //allocate memory, assume the provided header is not always compatible with current image size  
  images_cpu.clear();
  images_cpu = vector<Mat>(level_num_);
  for(int i=0; i<level_num_; i++){
    images_cpu[i].create(rows_,cols_,CV_32F);
  }
  //read device memory 
  for(int i=0; i<level_num_; i++){
    images[i].readDeviceToMat_32F(images_cpu[i]);
    cout<<"Computed Map result"<<endl;
  }
}


}
