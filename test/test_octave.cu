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

#define TEST_IMG 1

int main(){
#if TEST_IMG
  //compute integral image 
  cv::Mat img;
  img = cv::imread("./data/img1.png");
  //convert image to gray scale image
  Mat gray_img;
  cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
#endif
  cv::Mat mat_in;
#if TEST_IMG
  //convert image type from CV_8U to CV_32F
  gray_img.convertTo(mat_in, CV_32S);
#else
  mat_in = Mat::ones(1080,1920,CV_32S)*255;
#endif
  //get image size 
  const int rows = mat_in.rows;
  const int cols = mat_in.cols;
  CudaMat cuda_mat_in(mat_in);
  CudaMat cuda_mat_integral(mat_in);
  cuda_mat_in.allocate();
  cuda_mat_integral.allocate();
  //compute integral image 
  SURF surf;
  //create octave
  Octave octave_1(rows,cols,1,{9,15,21,27});
  Octave octave_2(rows/2, cols/2, 2, {15,27,39,51});
  Octave octave_3(rows/4, cols/4, 4, {27,51,75,99});
  //allocate octave 
  octave_1.allocate();
  octave_2.allocate();
  octave_3.allocate();
  //compute integral image
  surf.compIntegralImage(cuda_mat_in,cuda_mat_integral);
  //fill octaves 
  octave_1.fill(cuda_mat_integral);
  octave_2.fill(cuda_mat_integral);
  octave_3.fill(cuda_mat_integral);
  //read octave results
  vector<Mat> octave_response_1;
  vector<Mat> octave_response_2;
  vector<Mat> octave_response_3;
  octave_1.readDoHResponseMap(octave_response_1);
  octave_2.readDoHResponseMap(octave_response_2);
  octave_3.readDoHResponseMap(octave_response_3);
  //show image 
  cv::namedWindow("1");
  cv::namedWindow("2");
  cv::namedWindow("3");
  cv::imshow("1",octave_response_1[1]);
  cv::imshow("2",octave_response_2[1]);
  cv::imshow("3",octave_response_3[1]);
  
  
}