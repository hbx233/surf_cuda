#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/cuda_util.cuh"

int main(){
  //100 * 100 matrix
  size_t width = 1000;//columns
  size_t height = 1000;//rows
  float* host_mat = (float*)malloc(width*height*sizeof(float));
  float* host_mat_gpu = (float*)malloc(width*height*sizeof(float));
  memset(host_mat, 4, width*height*sizeof(float));
  //get cuda_mat
  surf_cuda::CudaMat cuda_mat(width, height);
  cuda_mat.allocate();
  //copy to and back 
  cuda_mat.writeDevice(host_mat, width*sizeof(float),width,height);
  cuda_mat.readDevice(host_mat_gpu, width*sizeof(float), width, height);
  compare(host_mat_gpu, host_mat, width*height,"Compare copy data to and from device");
  //test image transfer 
  Mat img = cv::imread("./data/img1.png");
  Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  Mat gray_f;
  gray.convertTo(gray_f,CV_32F);
  //gray_f*= 1./255;
  cout<<(int)gray.at<uchar>(1,2)<<endl;
  cout<<gray_f.at<float>(1,2)<<endl;
  surf_cuda::CudaMat cuda_img(gray_f.cols,gray_f.rows);
  cuda_img.allocate();
  cuda_img.writeDevice((float*)gray_f.data,gray_f.cols*sizeof(float),gray_f.cols,gray_f.rows);
  float* host_img_gpu = (float*)malloc(gray_f.cols * gray_f.rows * sizeof(float));
  cuda_img.readDevice(host_img_gpu, gray_f.cols*sizeof(float), gray_f.cols, gray_f.rows);
  compare<float>(host_img_gpu,(float*)gray_f.data,gray_f.cols*gray_f.rows,"Compare copy image data");
  
  Mat img_gpu_f(gray_f.rows, gray_f.cols, CV_32F,(void*)host_img_gpu);
  img_gpu_f*=1./255;
  cv::namedWindow("Display");
  cv::imshow("Display",img_gpu_f);
  cv::waitKey(0);
}