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
    dst = (temp1 - temp2)/(float)(size*size);
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

#define TEST_IMG 1

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
  
  //integral_image 
  Mat mat_integral_cpu = Mat::zeros(rows,cols,CV_32S);
  Mat mat_integral_gpu = Mat::zeros(rows,cols,CV_32S);
  CudaMat cuda_mat_integral(rows,cols,CV_32S);
  cuda_mat_integral.allocate();
  cuda_mat_integral.allocateArray();
  
  //COPY IMAGE TO DEVICE
  cuda_mat_in.copyFromMat(mat_in_cpu);
   
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
  Mat response_map_stride2_gpu_tex = Mat::zeros(rows,cols,CV_32F);
  CudaMat cuda_response_map_stride2(rows/2,cols/2,CV_32F);
  cuda_response_map_stride2.allocate();
  CudaMat cuda_response_map_stride2_tex(rows, cols, CV_32F);
  cuda_response_map_stride2_tex.allocate();
  
  //Channel Descriptor for cuda Array 
  //cudaChannelFormatDesc channelDesc =
  //cudaCreateChannelDesc(32, 0, 0, 0,
  //			cudaChannelFormatKindSigned);
  //Allocate texture memory 
  //cudaArray* cuda_array_integral;
  //cudaMallocArray(&cuda_array_integral, &channelDesc, cols, rows);
  
  //Create texture object 
  // Specify texture
  //cudaResourceDesc resDesc;
  //memset(&resDesc, 0, sizeof(resDesc));
  //resDesc.resType = cudaResourceTypeArray;
  //resDesc.res.array.array = cuda_array_integral;

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
  CpuTimer cpu_timer;
  //=====================================
  //compute integral image
  cpu_timer.elapsedTimeStart();
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral);
  cpu_timer.elapsedTimeStop();
  //=====================================
  //copy integral image from global memory to texture memory 
  cpu_timer.elapsedTimeStart();
  //cudaMemcpy2DToArray(cuda_array_integral, 0, 0, (void*)cuda_mat_integral.data, cuda_mat_integral.pitch_bytes(), cuda_mat_integral.cols() * sizeof(int), cuda_mat_integral.rows(), cudaMemcpyDeviceToDevice);
  cuda_mat_integral.copyToArray();
  cpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using Global Memory 
  cpu_timer.elapsedTimeStart();
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride1,doh_filter_gpu,1);
  cpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using Texture Memory 
  cpu_timer.elapsedTimeStart();
  compDoHResponseMap_texture(cuda_mat_integral.texture_object(),rows, cols, cuda_response_map_stride1_tex,doh_filter_gpu,1);
  cpu_timer.elapsedTimeStop();
  //=====================================
  //Compute Blob response Map Using GPU
  cpu_timer.elapsedTimeStart();
  doh_filter_cpu(mat_in_cpu,response_map_stride1_cpu);
  cpu_timer.elapsedTimeStop();
  
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride2,doh_filter_gpu,2);
  
  //copy response map to host 
  cuda_response_map_stride1.copyToMat(response_map_stride1_gpu);
  cuda_response_map_stride1_tex.copyToMat(response_map_stride1_gpu_tex);
  cuda_response_map_stride2.copyToMat(response_map_stride2_gpu);
  
  //compare 
  compare(response_map_stride1_cpu,response_map_stride1_gpu);
  compare(response_map_stride1_cpu,response_map_stride1_gpu_tex);
#if 0
  cv::namedWindow("CPU DoH");
  cv::namedWindow("GPU DoH");
  cv::namedWindow("GPU DoH stride2");
  cv::imshow("CPU DoH",response_map_stride1_cpu/255);
  cv::imshow("GPU DoH",response_map_stride1_gpu/255);
  cv::imshow("GPU DoH stride2",response_map_stride2_gpu/255);
  cv::waitKey(0);
#endif
#if 0
  cout<<response_map_stride1_cpu<<endl;
  cout<<response_map_stride1_gpu<<endl;
  cout<<response_map_stride1_gpu_tex<<endl;
  cout<<response_map_stride1_gpu-response_map_stride1_gpu_tex<<endl;
  //cout<<response_map_stride1_cpu - response_map_stride1_gpu<<endl;
#endif
}