#ifndef CUDA_MAT_H_
#define CUDA_MAT_H_
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"

namespace surf_cuda{
class CudaMat{
public:
  /*!
   * @brief default constructor
   */
  CudaMat():rows_(0),cols_(0),pitch_bytes_(0),type_(0), depth_(0), elemSize_(0), cuda_mem_(nullptr),cuda_array_(nullptr){}
  /*!
   * @brief constructor that initializes the rows, cols and type for allocation
   * @param rows number of rows for Global Memory and Cuda Array 
   * @param cols number of columns for Global Memory and Cuda Array
   * @param type OpenCV type that is used to indicate the type of Memory   
   */
  CudaMat(const int& rows, const int& cols, int type):rows_(rows),cols_(cols),type_(type), cuda_mem_(nullptr),cuda_array_(nullptr){}
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
   * @brief Wraper function that perform memory copy from Host Mat object to Device CudaMat Texture memory
   * @param mat The Mat object that stores the data 
   * @note The Mat object should have the same type as type_
   */
  void copyFromMatToArray(const Mat& mat);
  
  /*!
   * @brief Wraper function that perform memory copy from Device CudaMat to Host Mat object memory 
   * @param mat The Mat object that the data will copy to 
   * @note The Mat object should have the same type as type_
   */
   void copyToMat(Mat& mat);
   
  /*!
   * @brief Wraper function that perform memory copy from Device CudaMat Texture Memory to Host Mat object memory 
   * @param mat The Mat object that the data will copy to 
   * @note The Mat object should have the same type as type_
   */
   void copyToMatFromArray(Mat& mat);
   
public:
  unsigned char* data() const;
  const int rows() const;
  const int cols() const;
  const size_t pitch_bytes() const;
  const int type() const;
  const int depth() const;
  const int elemSize() const;
  const cudaTextureObject_t texture_object() const;
  const cudaChannelFormatDesc channel_desc() const;
  const cudaResourceDesc resource_desc() const;
  const cudaTextureDesc texture_desc() const;
private:
  
  void checkMemAllocation() const;
  void checkArrayAllocation() const;
  
  //number of rows and columns 
  int cols_;//column number
  int rows_;//row number 
  int type_;
  int depth_;
  int elemSize_;
  //CUDA Global Memory on GPU 
  shared_ptr<unsigned char> cuda_mem_;
  size_t pitch_bytes_;
  
  //CUDA Array texture memory 
  shared_ptr<cudaArray> cuda_array_;
  //texture object interface to fetch texture memory 
  cudaTextureObject_t tex_obj_;
  //channel format descriptor to create cuda array 
  cudaChannelFormatDesc channel_desc_;
  //cuda resource descriptor that is used to create texture object 
  cudaResourceDesc res_desc_;
  //cuda texture descriptor that is used to create texture object
  cudaTextureDesc tex_desc_;
  bool valid_data_in_tex{false};
  bool valid_texture_obj_{false};
};
}


#endif /* CUDA_MAT_H_*/