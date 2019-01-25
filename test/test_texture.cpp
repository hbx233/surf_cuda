
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

__global__ void kernel_DoH_filter_texture(cudaTextureObject_t integral_tex, int integral_rows, int integral_cols, unsigned char* response_map, size_t response_pitch_bytes, int response_rows, int response_cols, int stride){
  
}


#define TEST_IMG 1

int main(){
  Mat mat_in_cpu = Mat::ones(2000,2000,CV_32S);
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
  //Allocate Texture memory
  cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(32, 0, 0, 0,
                                   cudaChannelFormatKindSigned);
  cudaArray* cuda_mat_integral_tex;
  cudaMallocArray(&cuda_mat_integral_tex, &channelDesc, cols, rows);

  
  //COPY IMAGE TO DEVICE
  cuda_mat_in.copyFromMat(mat_in_cpu);
   
  SURF surf;
  
  //Create DoH filter 
  DoHFilter doh_filter_gpu(51);
  DoHFilter_cpu doh_filter_cpu(51);
  
  //Blob response map 
  Mat response_map_stride1_cpu = Mat::zeros(rows,cols,CV_32F);
  Mat response_map_stride1_gpu = Mat::zeros(rows,cols,CV_32F);
  CudaMat cuda_response_map_stride1(rows,cols,CV_32F);
  cuda_response_map_stride1.allocate();
  Mat response_map_stride2_cpu = Mat::zeros(rows/2,cols/2,CV_32F);
  Mat response_map_stride2_gpu = Mat::zeros(rows/2,cols/2,CV_32F);
  CudaMat cuda_response_map_stride2(rows/2,cols/2,CV_32F);
  cuda_response_map_stride2.allocate();
  
  
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_mat_integral_tex;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBoarder;
    texDesc.addressMode[1]   = cudaAddressModeBoarder;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t integral_tex = 0;
    cudaCreateTextureObject(&integral_tex, &resDesc, &texDesc, NULL);
  
  //create timer
  GpuTimer gpu_timer;
  CpuTimer cpu_timer;
  //compute integral image
  //float start_gpu = gpu_timer.elapsedTime();
  float start_gpu = cpu_timer.elapsedTime();
  surf.compIntegralImage(cuda_mat_in, cuda_mat_integral);
  //float elapsed_gpu = gpu_timer.elapsedTime() - start_gpu;
  float elapsed_gpu = cpu_timer.elapsedTime() - start_gpu;
  cout<<"[GPU] Computation Time: "<<elapsed_gpu<<"ms"<<endl;
  //copy integral_image from global device memory to texture memory 
  cudaMemcpy2DToArray(cuda_mat_integral_tex, 0, 0, (void*)cuda_mat_integral.data, cuda_mat_integral.pitch_bytes(), cuda_mat_integral.cols * sizeof(int),cuda_mat_integral.rows,
                    cudaMemcpyDeviceToDevice);
  //Compute Blob response Map 
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride1,doh_filter_gpu,1);
  
  elapsed_gpu = cpu_timer.elapsedTime() - start_gpu;
  cout<<"[GPU] Computation Time: "<<elapsed_gpu<<"ms"<<endl;
  
  
  float start_cpu = cpu_timer.elapsedTime();
  doh_filter_cpu(mat_in_cpu,response_map_stride1_cpu);
  float elapsed_cpu = cpu_timer.elapsedTime() - start_cpu;
  cout<<"[CPU] Computation Time: "<<elapsed_cpu<<"ms"<<endl;
  
  
  compDoHResponseMap(cuda_mat_integral,cuda_response_map_stride2,doh_filter_gpu,2);
  
  
  //copy response map to host 
  cuda_response_map_stride1.copyToMat(response_map_stride1_gpu);
  cuda_response_map_stride2.copyToMat(response_map_stride2_gpu);
  //compare 
  compare(response_map_stride1_cpu,response_map_stride1_gpu);
#if 0
  cv::namedWindow("CPU DoH");
  cv::namedWindow("GPU DoH");
  cv::namedWindow("GPU DoH stride2");
  cv::imshow("CPU DoH",response_map_stride1_cpu/255);
  cv::imshow("GPU DoH",response_map_stride1_gpu/255);
  cv::imshow("GPU DoH stride2",response_map_stride2_gpu/255);
  cv::waitKey(0);
#endif
  //cout<<response_map_stride1_cpu<<endl;
  //cout<<response_map_stride1_gpu<<endl;
  //cout<<response_map_stride1_cpu - response_map_stride1_gpu<<endl;
  
}