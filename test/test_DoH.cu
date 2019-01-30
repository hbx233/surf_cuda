#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.cuh"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/surf.h"

using namespace surf_cuda;
//cpu DoH function that use opencv provided filter2D functionality 
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
    dst = (temp1 - temp2)/(float)(size*size*size*size);
  }
  void box_filter_xx(Mat src, Mat& dst){
    cv::filter2D(src,dst,CV_32F,kernel_xx,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
  }
  void box_filter_xy(Mat src, Mat& dst){
    cv::filter2D(src,dst,CV_32F,kernel_xy,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
  }
  void box_filter_yy(Mat src, Mat& dst){
    cv::filter2D(src,dst,CV_32F,kernel_yy,cv::Point(-1,-1),0,cv::BORDER_CONSTANT);
  }
};

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
#define PRINT 0
int main(){
  Mat mat_in_cpu = Mat::ones(10,10,CV_32S);
#if TEST_IMG 
  Mat img = cv::imread("./data/img1.png");
  Mat gray_img;
  cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
  gray_img.convertTo(mat_in_cpu,CV_32S);
#endif
  
  
  int rows = mat_in_cpu.rows;
  int cols = mat_in_cpu.cols;
  //inpute image 
  CudaMat cuda_mat_in(rows,cols,CV_32S);
  cuda_mat_in.allocate();
  cuda_mat_in.allocateArray(); 
  //integral_image 
  Mat mat_integral_cpu = Mat::zeros(rows,cols,CV_32S);
  Mat mat_integral_gpu = Mat::zeros(rows,cols,CV_32S);
  CudaMat cuda_mat_integral(rows,cols,CV_32S);
  cuda_mat_integral.allocate();
  cuda_mat_integral.allocateArray();
  
  //COPY IMAGE TO DEVICE
  cuda_mat_in.copyFromMatToArray(mat_in_cpu);
  cudaTextureDesc texDesc_integral;
  memset(&texDesc_integral, 0, sizeof(texDesc_integral));
  texDesc_integral.addressMode[0]   = cudaAddressModeClamp;
  texDesc_integral.addressMode[1]   = cudaAddressModeClamp;
  texDesc_integral.filterMode       = cudaFilterModePoint;
  texDesc_integral.readMode         = cudaReadModeElementType;
  texDesc_integral.normalizedCoords = 0;
  
  //set texture object 
  cuda_mat_in.setTextureObjectInterface(texDesc_integral);
   
  SURF surf;
  
  //Create DoH filter 
  DoHFilter doh_filter_gpu(9);
  DoHFilter_cpu doh_filter_cpu(9);
  
  //Blob response map 
  //===============================
  //Stride one test
  //CPU result
  Mat response_map_stride1_cpu = Mat::zeros(rows,cols,CV_32F);
  //GPU result using global memory on Host
  Mat response_map_stride1_gpu = Mat::zeros(rows,cols,CV_32F);
  //GPU result using texture memory on Host 
  Mat response_map_stride1_gpu_tex = Mat::zeros(rows,cols,CV_32F);
  //GPU result using global memory on Device 
  CudaMat cuda_response_map_stride1(rows,cols,CV_32F);
  cuda_response_map_stride1.allocate();
  //GPU result using texture memory on Device 
  CudaMat cuda_response_map_stride1_tex(rows, cols, CV_32F);
  cuda_response_map_stride1_tex.allocate();
  //===============================
  //Stride two test
  Mat response_map_stride2_cpu = Mat::zeros(rows/2,cols/2,CV_32F);
  Mat response_map_stride2_gpu = Mat::zeros(rows/2,cols/2,CV_32F);
  Mat response_map_stride2_gpu_tex = Mat::zeros(rows/2,cols/2,CV_32F);
  CudaMat cuda_response_map_stride2(rows/2,cols/2,CV_32F);
  cuda_response_map_stride2.allocate();
  CudaMat cuda_response_map_stride2_tex(rows/2, cols/2, CV_32F);
  cuda_response_map_stride2_tex.allocate();
  //===============================
  //Stride four test
  Mat response_map_stride4_cpu = Mat::zeros(rows/4,cols/4,CV_32F);
  Mat response_map_stride4_gpu = Mat::zeros(rows/4,cols/4,CV_32F);
  Mat response_map_stride4_gpu_tex = Mat::zeros(rows/4,cols/4,CV_32F);
  CudaMat cuda_response_map_stride4(rows/4,cols/4,CV_32F);
  cuda_response_map_stride4.allocate();
  CudaMat cuda_response_map_stride4_tex(rows/4, cols/4, CV_32F);
  cuda_response_map_stride4_tex.allocate();
  //===============================
  //Stride eight test
  Mat response_map_stride8_cpu = Mat::zeros(rows/8,cols/8,CV_32F);
  Mat response_map_stride8_gpu = Mat::zeros(rows/8,cols/8,CV_32F);
  Mat response_map_stride8_gpu_tex = Mat::zeros(rows/8,cols/8,CV_32F);
  CudaMat cuda_response_map_stride8(rows/8,cols/8,CV_32F);
  cuda_response_map_stride8.allocate();
  CudaMat cuda_response_map_stride8_tex(rows/8, cols/8, CV_32F);
  cuda_response_map_stride8_tex.allocate();
  
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
  
  //create timer
  GpuTimer gpu_timer;
  CpuTimer cpu_timer;
  //=====================================
  //compute integral image
  gpu_timer.elapsedTimeStart();
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral);
  gpu_timer.elapsedTimeStop();
  //=====================================
  //copy integral image from global memory to texture memory 
  gpu_timer.elapsedTimeStart();
  //cudaMemcpy2DToArray(cuda_array_integral, 0, 0, (void*)cuda_mat_integral.data, cuda_mat_integral.pitch_bytes(), cuda_mat_integral.cols() * sizeof(int), cuda_mat_integral.rows(), cudaMemcpyDeviceToDevice);
  cuda_mat_integral.copyToArray();
  gpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using Global Memory 
  gpu_timer.elapsedTimeStart();
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride1,doh_filter_gpu,1);
  gpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using Texture Memory 
  gpu_timer.elapsedTimeStart();
  compDoHResponseMap_texture(cuda_mat_integral,cuda_response_map_stride1_tex,doh_filter_gpu,1);
  gpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using GPU
  cpu_timer.elapsedTimeStart();
  doh_filter_cpu(mat_in_cpu,response_map_stride1_cpu);
  cpu_timer.elapsedTimeStop();
  
  cout<<"Compute Stride 2"<<endl;
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride2,doh_filter_gpu,2);
  compDoHResponseMap_texture(cuda_mat_integral,cuda_response_map_stride2_tex,doh_filter_gpu,2);
  cout<<"Compute Stride 4"<<endl;
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride4,doh_filter_gpu,4);
  compDoHResponseMap_texture(cuda_mat_integral,cuda_response_map_stride4_tex,doh_filter_gpu,4);
  cout<<"Compute Stride 8"<<endl;
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride8,doh_filter_gpu,8);
  compDoHResponseMap_texture(cuda_mat_integral,cuda_response_map_stride8_tex,doh_filter_gpu,8);
  //copy response map to host 
  cuda_response_map_stride1.copyToMat(response_map_stride1_gpu);
  cuda_response_map_stride1_tex.copyToMat(response_map_stride1_gpu_tex);
  cuda_response_map_stride2.copyToMat(response_map_stride2_gpu);
  cuda_response_map_stride2_tex.copyToMat(response_map_stride2_gpu_tex);
  cuda_response_map_stride4.copyToMat(response_map_stride4_gpu);
  cuda_response_map_stride4_tex.copyToMat(response_map_stride4_gpu_tex);
  cuda_response_map_stride8.copyToMat(response_map_stride8_gpu);
  cuda_response_map_stride8_tex.copyToMat(response_map_stride8_gpu_tex);
  //compare 
  compare(response_map_stride1_cpu,response_map_stride1_gpu);
  compare(response_map_stride1_cpu,response_map_stride1_gpu_tex);
  compare(response_map_stride2_gpu,response_map_stride2_gpu_tex);
  compare(response_map_stride4_gpu,response_map_stride4_gpu_tex);
  compare(response_map_stride8_gpu,response_map_stride8_gpu_tex);
#if 1
  cv::namedWindow("CPU DoH");
  cv::namedWindow("GPU DoH");
  cv::namedWindow("GPU DoH stride2");
  cv::imshow("CPU DoH",normalize(response_map_stride1_cpu));
  cv::imshow("GPU DoH",normalize(response_map_stride1_gpu));
  cv::imshow("GPU DoH stride2",normalize(response_map_stride2_gpu));
  cv::imshow("GPU DoH stride2",normalize(response_map_stride2_gpu_tex));
  cv::imshow("GPU DoH stride4",normalize(response_map_stride4_gpu));
  cv::imshow("GPU DoH stride4",normalize(response_map_stride4_gpu_tex));
  cv::imshow("GPU DoH stride8",normalize(response_map_stride8_gpu));
  cv::imshow("GPU DoH stride8",normalize(response_map_stride8_gpu_tex));
  cv::imwrite("./image/doh_map.png",response_map_stride1_gpu);
  cv::waitKey(0);
#endif
#if PRINT
  cout<<response_map_stride1_cpu<<endl;
  cout<<response_map_stride1_gpu<<endl;
  cout<<response_map_stride1_gpu_tex<<endl;
  cout<<response_map_stride1_gpu-response_map_stride1_gpu_tex<<endl;
  //cout<<response_map_stride1_cpu - response_map_stride1_gpu<<endl;
  cout<<response_map_stride2_cpu<<endl;
  cout<<response_map_stride2_gpu<<endl;
  cout<<response_map_stride2_gpu_tex<<endl;
#endif
}