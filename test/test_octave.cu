#include "surf_cuda/octave.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"
using namespace surf_cuda;

Mat normalize(Mat map){
  Mat norm = map.clone();
  double max_val;
  double min_val;
  cv::minMaxLoc(norm,&min_val,&max_val);
  cout<<min_val<<' '<<max_val<<endl;
  norm = (norm-min_val)/(float)(max_val - min_val);
  return norm;
}
#define TEST_IMG 1

void drawKeyPoints(const Octave& octave, Mat image){
  cout<<"Keypoints size: "<<octave.keypoints_num<<endl;
  for(int i=0;i<octave.keypoints_num;i++){
    cv::circle(image,cv::Point(octave.keypoints_x[i],octave.keypoints_y[i]),2,cv::Scalar(0,0,200),2);
  }
}

int main(){
#if TEST_IMG
  //compute integral image 
  cv::Mat img;
  img = cv::imread("./data/img2.png");
  //convert image to gray scale image
  Mat gray_img;
  cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
#endif
  cv::Mat mat_in;
#if TEST_IMG
  //convert image type from CV_8U to CV_32F
  gray_img.convertTo(mat_in, CV_32S);
#else
  mat_in = Mat::ones(10,20,CV_32S);
#endif
  //get image size 
  const int rows = mat_in.rows;
  const int cols = mat_in.cols;
  CudaMat cuda_mat_in(rows,cols,CV_32S);
  CudaMat cuda_mat_integral(rows,cols,CV_32S);
  cuda_mat_in.allocate();
  cuda_mat_in.allocateArray();
  cuda_mat_integral.allocate();
  cuda_mat_integral.allocateArray();
  //compute integral image 
  SURF surf;
  //create octaves
  Octave octave_1(rows,cols,1,{9,15,21,27});
  Octave octave_2(rows/2, cols/2, 2, {15,27,39,51});
  Octave octave_3(rows/4, cols/4, 4, {27,51,75,99});
  //allocate octave in Global memory and Texture Memory 
  octave_1.allocateMemAndArray();
  octave_2.allocateMemAndArray();
  octave_3.allocateMemAndArray();
  
  //Gpu Timer
  GpuTimer gpu_timer;
  //compute integral image
  //copy image from host to device 
  
  cuda_mat_in.copyFromMatToArray(mat_in);
  cudaTextureDesc texDesc_integral;
  memset(&texDesc_integral, 0, sizeof(texDesc_integral));
  texDesc_integral.addressMode[0]   = cudaAddressModeClamp;
  texDesc_integral.addressMode[1]   = cudaAddressModeClamp;
  texDesc_integral.filterMode       = cudaFilterModePoint;
  texDesc_integral.readMode         = cudaReadModeElementType;
  texDesc_integral.normalizedCoords = 0;
  
  //set texture object 
  cuda_mat_in.setTextureObjectInterface(texDesc_integral);
  //compute integral image 
  gpu_timer.elapsedTimeStart();
  surf.compIntegralImage(cuda_mat_in,cuda_mat_integral);
  gpu_timer.elapsedTimeStop();
  //Mat integral(rows,cols,CV_32S);
  //cuda_mat_integral.copyToMat(integral);
  //cout<<integral<<endl;
  //copy integral image to texture memory 
  cuda_mat_integral.copyToArray();
  // Specify texture object parameters
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeBorder;
  texDesc.addressMode[1]   = cudaAddressModeBorder;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  
  //set texture object 
  cuda_mat_integral.setTextureObjectInterface(texDesc);
  //set texture 
  //fill octaves with DoH resopnse maps 
  gpu_timer.elapsedTimeStart();
  octave_1.fill(cuda_mat_integral);
  octave_2.fill(cuda_mat_integral);
  octave_3.fill(cuda_mat_integral);
  gpu_timer.elapsedTimeStop();
#if 1
  gpu_timer.elapsedTimeStart();
  octave_1.copyResponseMapsToArray();
  octave_2.copyResponseMapsToArray();
  octave_3.copyResponseMapsToArray();
  gpu_timer.elapsedTimeStop();
  
  gpu_timer.elapsedTimeStart();
  float threshold = 500;
  octave_1.thresholdNonMaxSupAndFindKeyPoints(threshold);
  octave_2.thresholdNonMaxSupAndFindKeyPoints(threshold);
  octave_3.thresholdNonMaxSupAndFindKeyPoints(threshold);
  gpu_timer.elapsedTimeStop();
#endif
  //read octave results
  vector<Mat> octave_response_1;
  vector<Mat> octave_response_2;
  vector<Mat> octave_response_3;
  octave_1.readDoHResponseMap(octave_response_1);
  octave_2.readDoHResponseMap(octave_response_2);
  octave_3.readDoHResponseMap(octave_response_3);
  //octave_2.readDoHResponseMapAfterSupression(octave_response_2);
  //octave_3.readDoHResponseMapAfterSupression(octave_response_3);
  //octave_1.readDoHResponseMapAfterSupression(octave_response_1);
  //show image
  //cout<<octave_response_1.size();
  //draw keypoints on image 
  drawKeyPoints(octave_1,img);
  drawKeyPoints(octave_2,img);
  drawKeyPoints(octave_3,img);
  
  cv::namedWindow("1");
  cv::imshow("1",img);
  cv::imwrite("extracted_key_points2.png",img);
  cv::waitKey(0);
#if 0
  cv::imwrite("./image/octave_1_supression_1.png",normalize(octave_response_1[0])*255);
  cv::imwrite("./image/octave_1_supression_2.png",normalize(octave_response_1[1])*255);
  cv::imwrite("./image/octave_2_supression_1.png",normalize(octave_response_2[0])*255);
  cv::imwrite("./image/octave_2_supression_2.png",normalize(octave_response_2[1])*255);
  cv::imwrite("./image/octave_3_supression_1.png",normalize(octave_response_3[0])*255);
  cv::imwrite("./image/octave_3_supression_2.png",normalize(octave_response_3[1])*255);
#endif
#if 0
  cv::imwrite("./image/octave_1_level_1.png",normalize(octave_response_1[0])*255);
  cv::imwrite("./image/octave_1_level_2.png",normalize(octave_response_1[1])*255);
  cv::imwrite("./image/octave_1_level_3.png",normalize(octave_response_1[2])*255);
  cv::imwrite("./image/octave_1_level_4.png",normalize(octave_response_1[3])*255);
  cv::imwrite("./image/octave_2_level_1.png",normalize(octave_response_2[0])*255);
  cv::imwrite("./image/octave_2_level_2.png",normalize(octave_response_2[1])*255);
  cv::imwrite("./image/octave_2_level_3.png",normalize(octave_response_2[2])*255);
  cv::imwrite("./image/octave_2_level_4.png",normalize(octave_response_2[3])*255);
  cv::imwrite("./image/octave_3_level_1.png",normalize(octave_response_3[0])*255);
  cv::imwrite("./image/octave_3_level_2.png",normalize(octave_response_3[1])*255);
  cv::imwrite("./image/octave_3_level_3.png",normalize(octave_response_3[2])*255);
  cv::imwrite("./image/octave_3_level_4.png",normalize(octave_response_3[3])*255);
#endif
}