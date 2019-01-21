#include "surf_cuda/octave.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"
using namespace surf_cuda;


struct DoHFilter_cpu{
  float weight{0.9};
  int size;
  Mat kernel_xx;
  Mat kernel_xy;
  Mat kernel_yy;
  DoHFilter_cpu(int s):size(s){
    //create kernels of approximated second derivative of Gaussian
    kernel_xx = Mat::zeros(s,s,CV_32F);
    kernel_xx(cv::Rect((size/3+1)/2,0,2*size/3 -1, size/3)).setTo(1);
    kernel_xx(cv::Rect((size/3+1)/2,size/3,2*size/3 -1, size/3)).setTo(-2);
    kernel_xx(cv::Rect((size/3+1)/2,2*size/3,2*size/3-1, size/3)).setTo(1);
    //kernel_yy is the rotation of kernel_xx
    cv::rotate(kernel_xx,kernel_yy,cv::ROTATE_90_CLOCKWISE);
    //kernel_xy
    kernel_xy = Mat::zeros(s,s,CV_32F);
    kernel_xy(cv::Rect(size/6,size/6,size/3,size/3)).setTo(1);
    kernel_xy(cv::Rect(size/2+1,size/6,size/3,size/3)).setTo(-1);
    kernel_xy(cv::Rect(size/6,size/2+1,size/3,size/3)).setTo(-1);
    kernel_xy(cv::Rect(size/2+1,size/2+1,size/3,size/3)).setTo(1);
  }
  void operator()(Mat src, Mat& dst){
    Mat filter_map_xx;
    Mat filter_map_xy;
    Mat filter_map_yy;
    //filter 
    cv::filter2D(src,filter_map_xx,CV_32F,kernel_xx,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
    cv::filter2D(src,filter_map_xy,CV_32F,kernel_xy,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
    cv::filter2D(src,filter_map_yy,CV_32F,kernel_yy,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
    Mat temp1;
    cv::multiply(filter_map_xx,filter_map_yy,temp1,1);
    Mat temp2;
    cv::pow(filter_map_xy*weight,2,temp2);
    dst = (temp1 - temp2)/(float)(size*size);
  }
};

#define TEST_IMG 0

int main(){
#if TEST_IMG
  //compute integral image 
  cv::Mat img;
  img = cv::imread("./data/img3.png");
  //convert image to gray scale image
  Mat gray_img;
  cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
  //convert image type from CV_8U to CV_32F
#endif
  cv::Mat mat_in = Mat::ones(1080,1920,CV_32F);
#if TEST_IMG
  gray_img.convertTo(mat_in, CV_32F);
#endif
  //get image size 
  const int rows = mat_in.rows;
  const int cols = mat_in.cols;
  Mat mat_integral = Mat::zeros(rows,cols,CV_32F);
  CudaMat cuda_mat_in(mat_in);
  CudaMat cuda_mat_integral(mat_in);
  cuda_mat_in.allocate();
  cuda_mat_integral.allocate();
  cuda_mat_in.writeDeviceFromMat_32F(mat_in);
  SURF surf;
  //compute integral image 
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral,32,32);
  //read integral image to host
  cuda_mat_integral.readDeviceToMat_32F(mat_integral);
  //cout<<mat_integral<<endl;
  //create octave 
  Octave octave_stride1(4,rows,cols,{9,15,21,27});
  //Octave octave_stride1(2,rows,cols,{9,15});
  cout<<"octave filters size"<<endl;
  for(int i=0;i<octave_stride1.filters.size();i++){
    cout<<octave_stride1.filters[i].size<<endl;
  }
  cout<<"Allocate Octave"<<endl;
  octave_stride1.allocateGpu();
  cout<<"fill octave"<<endl;
  octave_stride1.fill(cuda_mat_integral,1);
  vector<Mat> doh_map_gpu;
  octave_stride1.readDoHResponseMap(doh_map_gpu);
  
  vector<Mat> doh_map_cpu(4);
  for(int i=0;i<4;i++){
    doh_map_cpu[i].create(rows,cols,CV_32F);
  }
  vector<DoHFilter_cpu> filters_cpu { 
    DoHFilter_cpu(9),
    DoHFilter_cpu(15),
    DoHFilter_cpu(21),
    DoHFilter_cpu(27)
  };
  for(int i=0;i<4;i++){
    filters_cpu[i](mat_in,doh_map_cpu[i]); 
  }
  cout<<"Diff between CPU and GPU result"<<endl;
  for(int i=0;i<4;i++){
    //cout<<doh_map_cpu[i]-doh_map_gpu[i]<<endl;
    compare(doh_map_cpu[i],doh_map_gpu[i]);
  }
}