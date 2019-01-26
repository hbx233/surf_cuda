#ifndef CUDA_MAT_H_
#define CUDA_MAT_H_
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"
namespace surf_cuda{
class CudaMat{
public:
  CudaMat():rows_(0),cols_(0),pitch_bytes_(0),type_(0), depth_(0), elemSize_(0), data(NULL),internalAllocated_(false){}
  CudaMat(const int& rows, const int& cols, int type):rows_(rows),cols_(cols),type_(type){}
  CudaMat(const CudaMat& ) = delete;
  ~CudaMat(){
    cout<<"*"<<endl;
    if(internalAllocated_){
      //CudaSafeCall(cudaFree(data));
      cudaFree(data);
    }
  }
  /*!
   * @brief Allocate Global Memory on GPU according to the rows columns and type 
   * only allocated 2D new memory once, reallocated only when have to change the size 
   */
  void allocate();
  /*!
   * @brief Allocate Read-only texture memory on GPU, will be used ONLY in coordinate with Allocated Global Memory
   * @note Need to first transfer the data in Global Memory to Texture Memory, then Read Texture memory  
   */
  void allocateArray();
  
  /*!
   * @brief Set the Texture Object member in the class as interface for fetching data in texture memory \
   * @param tex_desc texture descriptor that is used to create texture object 
   */
  void setTextureObjectInterface(cudaTextureDesc tex_desc);
  
  /*!
   * @brief Transfer the internal data in global memory to texture memory for fast memory accessing  
   */
  void copyToArray();
  
  /*!
   * @brief Write Device memory from host  
   * @param hostmem pointer of host memory 
   * @param hostpitch pitch of host memory, in byte 
   * @param width width of host memory, in element 
   * @param height height of host memory, in element 
   */ 
  void writeDevice(void* hostmem, size_t hostpitch_bytes, int width, int height);
  
  
  /*!
   * @brief Read Device data to host memory  
   * @param hostmem pointer of host memory 
   * @param hostpitch pitch of host memory, in byte 
   * @param width width of host memory, in element 
   * @param height height of host memory, in element 
   */ 
  void readDevice(void* hostmem, size_t hostpitch_bytes, int width, int height);
  
  /*!
   * @brief Write Device Texture Memory from host 
   * @param hostmem pointer of host memory 
   * @param hostpitch pitch of host memory, in byte 
   * @param width width of host memory, in element 
   * @param height height of host memory, in element 
   */
  void writeDeviceToArray(void* hostmem, size_t host_pitch_bytes, int width, int height);
  
  /*!
   * @brief Read Device data from Texture Memory to Host memory  
   * @param hostmem pointer of host memory 
   * @param hostpitch pitch of host memory, in byte 
   * @param width width of host memory, in element 
   * @param height height of host memory, in element 
   * @note this function is not meant to be used frequently since Cuda Array is meant to be read fast and compute result
   *       and hence the cuda array usually doesn't contain the result data
   */ 
  void readDeviceFromArray(void* hostmem, size_t host_pitch_bytes, int width, int height);
  
  
   
  /*!
   * @brief Wraper function that perform memory copy from Host Mat object to Device CudaMat gpu memory
   * @param mat The Mat object that stores the data 
   * @note The Mat object should have the same type as type_
   */
  void copyFromMat(const Mat& mat);
  
  /*!
   * @brief Wraper function that perform memory copy from Device CudaMat to Host Mat object memory 
   * @param mat The Mat object that the data will copy to 
   * @note The Mat object should have the same type as type_
   */
   void copyToMat(Mat& mat);
   
   template<typename T>
   __device__ T* ptr(int row){
     if(row>=0 && row<rows_){
       return (T*)(data + row * pitch_bytes_);
     } else{
       //can assert here?
       printf("[CUDA] Invalid row address");
       return NULL;
     }
   }
   
   
public:
  //TODO: store data in smart pointer 
  unsigned char* data;
  __host__ __device__ const int rows() const;
  __host__ __device__ const int cols() const;
  __host__ __device__ const size_t pitch_bytes() const;
  __host__ __device__ const int type() const;
  __host__ __device__ const int depth() const;
  __host__ __device__ const int elemSize() const;
  __host__ __device__ const cudaTextureObject_t texture_object() const;
  __host__ __device__ const cudaChannelFormatDesc channel_desc() const;
  __host__ __device__ const cudaResourceDesc resource_desc() const;
  __host__ __device__ const cudaTextureDesc texture_desc() const;
private:
  //number of rows and columns 
  int cols_;//column number
  int rows_;//row number 
  size_t pitch_bytes_;
  int type_;
  int depth_;
  int elemSize_;
  //set true if allocated using allocate(), and will deallocate when destruct
  bool internalAllocated_{false};
  
  //CUDA Array texture memory 
  cudaArray* cuda_array_;
  cudaTextureObject_t tex_obj_;
  cudaChannelFormatDesc channel_desc_;
  cudaResourceDesc res_desc_;
  cudaTextureDesc tex_desc_;
  bool internalAllocatedArray_{false};
  bool valid_data_in_tex{false};
  bool valid_texture_obj_{false};
  
  
};
}


#endif /* CUDA_MAT_H_*/