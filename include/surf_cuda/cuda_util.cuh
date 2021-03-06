//#ifndef CUDA_UTIL_
//#define CUDA_UTIL_
#pragma once 

#include "surf_cuda/common.h"

#define CUDA_ERROR_CHECK

/*!
 * @brief:compare memory in data1 and data2
 * @param: data1: pointer to data1 consecutive memory
 * @param: data2: pointer to data2 consecutive memory
 * @param: size: size of element 
 * @param: msg: provide message to print for comparison 
 */
template <typename T> 
inline bool compare(T* data1, T* data2, size_t size, char msg[]){
    printf("%s\n",msg);
    for(int i=0; i<size; i++){
        if(std::abs(data1[i]-data2[i])>=0.00001){
            printf("[COMPARE] Different\n");
            return false;
        }
    }
    printf("[COMPARE] Same\n");
    return true;
}

inline bool compare(Mat src1, Mat src2, cv::Point* max_loc=NULL) {
  Mat diff;
  cv::absdiff(src1,src2,diff);
  double max_val;
  double min_val;
  cv::Point min_loc;
  cv::minMaxLoc(diff, &min_val, &max_val,  &min_loc, max_loc);
  printf("[COMPARE] Max Difference: %f \n",max_val);
  if (max_val >= 0.00001) {
    printf("[COMPARE] Different\n");
    if(max_loc!=NULL){
      printf("[COMPARE] At: (%i,%i)",max_loc->y,max_loc->x);
      printf("[COMPARE]: %f\n",src1.at<float>(max_loc->y,max_loc->x));
      printf("[COMPARE]: %f\n",src2.at<float>(max_loc->y,max_loc->x));
    }
    return false;
  } else{
    printf("[COMPARE] Same\n");
    return true;
  }
}

//cuda Error check and safe call 
#define CudaSafeCall( err ) __cudaSafeCall(err, __FILE__ , __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline cudaError_t __cudaSafeCall(cudaError_t err, const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if(cudaSuccess != err){
    fprintf(stderr, "[CUDA] cudaSafeCall() failed at %s:%i: %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return err;
}

inline cudaError_t __cudaCheckError(const char* file, const int line){
#ifdef CUDA_ERROR_CHECK
  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    fprintf(stderr, "[CUDA] cudaCheckError() failed at %s:%i: %s\n", file, line, cudaGetErrorString(err));
  
    exit(-1);
  }
#endif
  return err;
}

struct GpuTimer{
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  bool started_{false};
  GpuTimer(cudaStream_t stream = 0):stream_(stream){
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer(){
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void elapsedTimeStart(){
    cout<<"[GPU Timer] Timer Start"<<endl;
    cudaEventRecord(start_,stream_);
    started_ = true;
  }
  float elapsedTimeStop(){
    if(started_==true){
      cudaEventRecord(stop_,stream_);
      cudaEventSynchronize(stop_);
      float elapsed_t;
      cudaEventElapsedTime(&elapsed_t,start_,stop_);
      cout<<"[GPU Timer] Elapsed Time from Latest elapsedTimeStart Function: "<<elapsed_t<<"ms"<<endl;
      started_ = false;
      return elapsed_t;
    } else{
      cerr<<"[GPU Timer] Have to Call elapsedTimeStart before calling elapsedTimeStop"<<endl;
      return -1;
    }
  }
  
};

struct CpuTimer{
  high_resolution_clock::time_point start_;
  high_resolution_clock::time_point stop_;
  bool started_{false};
  CpuTimer(){
  }
  void elapsedTimeStart(){
    cout<<"[CPU Timer] Timer Start"<<endl;
    start_ = high_resolution_clock::now();
    started_ = true;
  }
  float elapsedTimeStop(){
    if(started_==true){
      stop_ = high_resolution_clock::now();
      auto elapsed_ = stop_-start_;
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_).count();
      cout<<"[CPU Timer] Elapsed Time from Latest elapsedTimeStart Function: "<<ms<<"ms"<<endl;
      started_ = false;
      return static_cast<float>(ms);
    } else{
      cerr<<"[CPU Timer] Have to Call elapsedTimeStart before calling elapsedTimeStop"<<endl;
      return -1;
    }
  }
};


//#endif


