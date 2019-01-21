#include "surf_cuda/surf.h"
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/cuda_util.cuh"
using surf_cuda::SURF;
using surf_cuda::CudaMat;

int main(){
  Mat mat = Mat::ones(100,200,CV_32F);
  Mat mat_out = Mat::zeros(100,200,CV_32F);
  CudaMat cuda_mat_in(mat);
  CudaMat cuda_mat_out(mat);
  //allocate two CudaMat
  cuda_mat_in.allocate();
  cuda_mat_out.allocate();
  //write data to in 
  cuda_mat_in.writeDeviceFromMat_32F(mat);
  SURF surf;
  //compute integral image 
  surf.compIntegralImage(cuda_mat_in, cuda_mat_out,128,128);
  //read integral image to host
  cuda_mat_out.readDeviceToMat_32F(mat_out);
  cout<<mat_out<<endl;
  
  
}
