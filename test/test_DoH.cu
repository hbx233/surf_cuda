#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"

using namespace surf_cuda;


__global__ void kernel_BoxFilter_xx(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride, BoxFilter_xx bf_xx){
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
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = bf_xx(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
  }
}
__global__ void kernel_BoxFilter_yy(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride, BoxFilter_yy bf_yy){
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
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = bf_yy(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols);
      }
    }
  }
}
__global__ void kernel_BoxFilter_xy(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride, BoxFilter_xy bf_xy){
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
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = bf_xy(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols);
      }
    }
  }
}
__global__ void kernel_DoHFilter(float* integral_mat, size_t integral_pitch, int integral_rows, int integral_cols, float* response_mat, size_t response_pitch, int response_rows, int response_cols, int stride, DoHFilter doh_filter){
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
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = doh_filter(integral_mat, integral_pitch, row_integral_idx, c*stride, integral_rows, integral_cols);
      }
    }
  }
}
int main(){
  int rows = 10;
  int cols = 10;
  Mat mat_in = Mat::ones(rows,cols,CV_32F);
  Mat mat_integral = Mat::zeros(rows,cols,CV_32F);
  Mat mat_box_filter_xx_stride1=Mat::zeros(rows,cols,CV_32F);
  Mat mat_box_filter_yy_stride1=Mat::zeros(rows,cols,CV_32F);
  Mat mat_box_filter_xy_stride1=Mat::zeros(rows,cols,CV_32F);
  Mat mat_DoH_filter_stride1 = Mat::zeros(rows,cols,CV_32F);
  Mat mat_box_filter_xx_stride2=Mat::zeros(rows/2,cols/2,CV_32F);
  Mat mat_box_filter_yy_stride2=Mat::zeros(rows/2,cols/2,CV_32F);
  Mat mat_box_filter_xy_stride2=Mat::zeros(rows/2,cols/2,CV_32F);
  Mat mat_DoH_filter_stride2 = Mat::zeros(rows/2,cols/2,CV_32F);
  CudaMat cuda_mat_in(mat_in);
  CudaMat cuda_mat_integral(mat_in);
  CudaMat cuda_mat_box_filter_xx_stride1(mat_in);
  CudaMat cuda_mat_box_filter_yy_stride1(mat_in);
  CudaMat cuda_mat_box_filter_xy_stride1(mat_in);
  CudaMat cuda_mat_DoH_filter_stride1(mat_in);
  CudaMat cuda_mat_box_filter_xx_stride2(mat_box_filter_xx_stride2);
  CudaMat cuda_mat_box_filter_yy_stride2(mat_box_filter_xx_stride2);
  CudaMat cuda_mat_box_filter_xy_stride2(mat_box_filter_xx_stride2);
  CudaMat cuda_mat_DoH_filter_stride2(mat_box_filter_xx_stride2);
  //allocate two CudaMat
  cuda_mat_in.allocate();
  cuda_mat_integral.allocate();
  cuda_mat_box_filter_xx_stride1.allocate();
  cuda_mat_box_filter_xy_stride1.allocate();
  cuda_mat_box_filter_yy_stride1.allocate();
  cuda_mat_DoH_filter_stride1.allocate();
  cuda_mat_box_filter_xx_stride2.allocate();
  cuda_mat_box_filter_xy_stride2.allocate();
  cuda_mat_box_filter_yy_stride2.allocate();
  cuda_mat_DoH_filter_stride2.allocate();
  //write data to in 
  cuda_mat_in.writeDeviceFromMat_32F(mat_in);
  SURF surf;
  //compute integral image 
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral,32,32);
  //read integral image to host
  cuda_mat_integral.readDeviceToMat_32F(mat_integral);
  cout<<mat_integral<<endl;
  //initiate BoxFilters 
  BoxFilter_xx box_filter_xx(9);
  BoxFilter_xy box_filter_xy(9);
  BoxFilter_yy box_filter_yy(9);
  DoHFilter doh_filter(9);
  dim3 block(32,1,1);
  dim3 grid(rows/32+1,1,1);
  kernel_BoxFilter_xx<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_xx_stride1.data, cuda_mat_box_filter_xx_stride1.pitch(), cuda_mat_box_filter_xx_stride1.rows(),cuda_mat_box_filter_xx_stride1.cols(),
				      1, box_filter_xx);
  kernel_BoxFilter_xy<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_xy_stride1.data, cuda_mat_box_filter_xy_stride1.pitch(), cuda_mat_box_filter_xy_stride1.rows(),cuda_mat_box_filter_xy_stride1.cols(),
				      1, box_filter_xy);
  kernel_BoxFilter_yy<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_yy_stride1.data, cuda_mat_box_filter_yy_stride1.pitch(), cuda_mat_box_filter_yy_stride1.rows(),cuda_mat_box_filter_yy_stride1.cols(),
				      1, box_filter_yy);
  kernel_DoHFilter<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_DoH_filter_stride1.data, cuda_mat_DoH_filter_stride1.pitch(), cuda_mat_DoH_filter_stride1.rows(),cuda_mat_DoH_filter_stride1.cols(),
				      1, doh_filter);
  kernel_BoxFilter_xx<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_xx_stride2.data, cuda_mat_box_filter_xx_stride2.pitch(), cuda_mat_box_filter_xx_stride2.rows(),cuda_mat_box_filter_xx_stride2.cols(),
				      2, box_filter_xx);
  kernel_BoxFilter_xy<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_xy_stride2.data, cuda_mat_box_filter_xy_stride2.pitch(), cuda_mat_box_filter_xy_stride2.rows(),cuda_mat_box_filter_xy_stride2.cols(),
				      2, box_filter_xy);
  kernel_BoxFilter_yy<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_box_filter_yy_stride2.data, cuda_mat_box_filter_yy_stride2.pitch(), cuda_mat_box_filter_yy_stride2.rows(),cuda_mat_box_filter_yy_stride2.cols(),
				      2, box_filter_yy);
  kernel_DoHFilter<<<grid,block>>>(cuda_mat_integral.data, cuda_mat_integral.pitch(),cuda_mat_integral.rows(), cuda_mat_integral.cols(),
                                      cuda_mat_DoH_filter_stride2.data, cuda_mat_DoH_filter_stride2.pitch(), cuda_mat_DoH_filter_stride2.rows(),cuda_mat_DoH_filter_stride2.cols(),
				      2, doh_filter);
  cudaDeviceSynchronize();
  //copy back
  cuda_mat_box_filter_xx_stride1.readDeviceToMat_32F(mat_box_filter_xx_stride1);
  cuda_mat_box_filter_xy_stride1.readDeviceToMat_32F(mat_box_filter_xy_stride1);
  cuda_mat_box_filter_yy_stride1.readDeviceToMat_32F(mat_box_filter_yy_stride1);
  cuda_mat_box_filter_xx_stride2.readDeviceToMat_32F(mat_box_filter_xx_stride2);
  cuda_mat_box_filter_xy_stride2.readDeviceToMat_32F(mat_box_filter_xy_stride2);
  cuda_mat_box_filter_yy_stride2.readDeviceToMat_32F(mat_box_filter_yy_stride2);
  cuda_mat_DoH_filter_stride1.readDeviceToMat_32F(mat_DoH_filter_stride1);
  cuda_mat_DoH_filter_stride2.readDeviceToMat_32F(mat_DoH_filter_stride2);
  cout<<mat_box_filter_xx_stride1<<endl;
  cout<<mat_box_filter_xy_stride1<<endl;
  cout<<mat_box_filter_yy_stride1<<endl;
  cout<<mat_box_filter_xx_stride2<<endl;
  cout<<mat_box_filter_xy_stride2<<endl;
  cout<<mat_box_filter_yy_stride2<<endl;
  cout<<mat_DoH_filter_stride1<<endl;
  cout<<mat_DoH_filter_stride2<<endl;
}