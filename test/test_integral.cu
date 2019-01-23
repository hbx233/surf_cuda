#include "surf_cuda/surf.h"
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/cuda_util.cuh"
using surf_cuda::SURF;
using surf_cuda::CudaMat;

void computeIntegralImage_cpu(Mat mat_in, Mat& mat_out){
  //compute integral of row
  Mat integral_row = Mat::zeros(mat_in.rows, mat_in.cols, CV_32S);
  for(int r=0; r<mat_in.rows;r++){
    int integral_cache=0;
    int* ptr_integral_row = integral_row.ptr<int>(r);
    int* ptr_mat_in = mat_in.ptr<int>(r);
    for(int c=0; c<mat_in.cols; c++){
      if(c==0){
	integral_cache = ptr_mat_in[0];
	ptr_integral_row[0] = integral_cache;
      } else{
	integral_cache += ptr_mat_in[c];
	ptr_integral_row[c] = integral_cache;
      }
    }
  }
  mat_out = Mat::zeros(mat_in.rows, mat_in.cols, CV_32S);
  for(int c=0; c<mat_in.cols;c++){
    int integral_cache=0;
    for(int r=0; r<mat_in.rows;r++){
      if(r==0){
	integral_cache = *integral_row.ptr<int>(r,c);
	*mat_out.ptr<int>(r,c) = integral_cache;
      } else{
	integral_cache += *integral_row.ptr<int>(r,c);
	*mat_out.ptr<int>(r,c) = integral_cache;
      }
    }
  }
}

#define TEST_IMG 0
int main(){
  Mat mat = Mat::ones(2000,2000,CV_32S)*200;
#if TEST_IMG
  Mat img = cv::imread("./data/img1.png");
  Mat gray_img;
  cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
  gray_img.convertTo(mat, CV_32S);
#endif
  
  int rows = mat.rows;
  int cols = mat.cols;
  
  Mat mat_out_gpu = Mat::zeros(rows,cols,CV_32S);
  CudaMat cuda_mat_in(rows,cols,CV_32S);
  CudaMat cuda_mat_out(rows,cols,CV_32S);
  //allocate two CudaMat
  cuda_mat_in.allocate();
  cuda_mat_out.allocate();
  //write data to in 
  cuda_mat_in.copyFromMat(mat);
  SURF surf;
  //compute integral image 
  surf.compIntegralImage(cuda_mat_in, cuda_mat_out);
  //read integral image to host
  cuda_mat_out.copyToMat(mat_out_gpu);
  Mat mat_out_cpu;
  computeIntegralImage_cpu(mat, mat_out_cpu);
  cout<<mat_out_cpu.ptr<int>(rows-1)[cols-1]<<endl;
  cout<<mat_out_gpu.ptr<int>(rows-1)[cols-1]<<endl;
  cout<<mat_out_gpu(cv::Rect(rows-10,cols-10,10,10))<<endl;
  compare(mat_out_cpu,mat_out_gpu);
}
