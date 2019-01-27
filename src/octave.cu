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

  
void Octave::allocate()
{
  //allocate memory for every level of DoH response map in the Octave
  for(int i=0; i<response_maps.size(); i++){
    response_maps[i].allocate();
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
    if(stride_ == integral_mat.rows()/rows_ && stride_ == integral_mat.cols()/cols_){
      valid = true;
    }
  }
  if(valid){
    //compute response maps for every filter 
    for(int i=0;i<level_num_;i++){
      compDoHResponseMap(integral_mat,response_maps[i],filters[i],stride_);
    }
  } else{
    fprintf(stderr, "[CUDA] The input image size is not sompatible with current size and stride");
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
    response_maps[i].copyToMat(images_cpu[i]);
    cout<<"Computed Map result"<<endl;
  }
}

__global__ void kernel_threshold_nonMaxSuppression(unsigned char* map_low, unsigned char* map_mid, unsigned char* map_up, int rows, int cols, float threshold){
  
}
}