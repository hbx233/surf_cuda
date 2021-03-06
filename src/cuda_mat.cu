#include "surf_cuda/cuda_mat.h"

namespace surf_cuda{

void CudaMat::allocate()
{
  cudaError_t err;
  bool leagal_type=true;
  unsigned char* data;
  switch(type_){
    //8bit unsigned char 
    case CV_8U:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(unsigned char),rows_));
      depth_ = sizeof(unsigned char);
      elemSize_ = depth_;
      break;
    case CV_32S:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(int),rows_));
      depth_ = sizeof(int);
      elemSize_ = depth_;
      break;
    case CV_32F:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(float),rows_));
      depth_ = sizeof(float);
      elemSize_ = depth_;
      break;
    case CV_64F:
      err = CudaSafeCall(cudaMallocPitch((void**)&data,&pitch_bytes_,cols_*sizeof(double),rows_));
      depth_ = sizeof(double);
      elemSize_ = depth_;
      break;
    default:
      //TODO: Error handling 
      fprintf(stderr,"Unsupported depth");
      leagal_type=false;
      exit(-1);
      break;
  }
  if(err==cudaSuccess && leagal_type==true){
    //store the pointer to shared_ptr
    cuda_mem_ = shared_ptr<unsigned char>(data,cudaFree);
  } else{
    fprintf(stderr,"[CUDA] CUDA Allocation Error");
  }
}

void CudaMat::allocateArray(){
  cudaError_t err;
  bool leagal_type=true;
  cudaChannelFormatDesc channelDesc;
  cudaArray* arrayPtr;
  switch(type_){
    //8bit unsigned char 
    case CV_8U:
      channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
      err = CudaSafeCall(cudaMallocArray(&arrayPtr, &channelDesc, cols_, rows_));
      channel_desc_ = channelDesc;
      break;
    case CV_32S:
      channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
      err = CudaSafeCall(cudaMallocArray(&arrayPtr, &channelDesc, cols_, rows_));
      channel_desc_ = channelDesc;
      break;
    case CV_32F:
      channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
      err = CudaSafeCall(cudaMallocArray(&arrayPtr, &channelDesc, cols_, rows_));
      channel_desc_ = channelDesc;
      break;
    case CV_64F:
      fprintf(stderr, "[CUDA] 64bit Float type is not supported for CUDA Texture Memory");
      leagal_type = false;
      exit(-1);
      break;
    default:
      //TODO: Error handling 
      fprintf(stderr,"Unsupported type");
      leagal_type=false;
      exit(-1);
      break;
  }
  if(err==cudaSuccess && leagal_type==true){
    //set resourse descriptor
    memset(&res_desc_,0,sizeof(res_desc_));
    res_desc_.resType = cudaResourceTypeArray;
    res_desc_.res.array.array = arrayPtr;
    cuda_array_ = shared_ptr<cudaArray>(arrayPtr,cudaFreeArray);
  } else{
    fprintf(stderr,"[CUDA] CUDA Array Texture Memory Allocation Error");
  }
  
}

void CudaMat::setTextureObjectInterface(cudaTextureDesc tex_desc){
  checkArrayAllocation();
  tex_obj_ = 0;
  //store texture descriptor
  tex_desc_ = tex_desc;
  //create texture object 
  cudaError_t err = CudaSafeCall(cudaCreateTextureObject(&tex_obj_, &res_desc_, &tex_desc_, NULL));
  if(err==cudaSuccess){
    valid_texture_obj_=true;
  }
}

void CudaMat::copyToArray(){
  checkArrayAllocation();
  //copy internal data in Global Memory to Texture Memory
  CudaSafeCall(cudaMemcpy2DToArray(cuda_array_.get() , 0, 0, (void*)cuda_mem_.get(), pitch_bytes_, cols_ * depth_, rows_, cudaMemcpyDeviceToDevice));
}

void CudaMat::writeDevice(void* hostmem, size_t hostpitch_bytes, int width, int height)
{
  checkMemAllocation();
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D((void*)cuda_mem_.get(), pitch_bytes_, (void*)hostmem, hostpitch_bytes, width*elemSize_, height, cudaMemcpyHostToDevice));
    if(err==cudaSuccess){
      printf("[CUDA] Wrote %i bytes data to Device\n", width*height*elemSize_);
    } 
  } else{
    fprintf(stderr,"[CUDA] [Write Device] Dimension of Host Source Memory and Device Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
  }
}


void CudaMat::writeDeviceToArray(void* hostmem, size_t hostpitch_bytes, int width, int height)
{
  checkArrayAllocation();
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2DToArray(cuda_array_.get(), 0, 0, hostmem, hostpitch_bytes, width * elemSize_, height, cudaMemcpyHostToDevice));
    if(err==cudaSuccess){
      printf("[CUDA] Wrote %i bytes data to Device\n", width*height*elemSize_);
    } 
  } else{
    fprintf(stderr,"[CUDA] [Write Device] Dimension of Host Source Memory and Device Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
  }
}

void CudaMat::readDevice(void* hostmem, size_t hostpitch_bytes, int width, int height){
  checkMemAllocation();
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2D((void*)hostmem,hostpitch_bytes, (void*)cuda_mem_.get(), pitch_bytes_, width*elemSize_,height, cudaMemcpyDeviceToHost));
    if(err == cudaSuccess){
      printf("[CUDA] Read %i bytes data from Device\n", width*height*elemSize_);
    }
  } else{
    fprintf(stderr,"[CUDA] [Read Device] Dimension of Device Source Memory and Source Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
    exit(-1);
  }
}


void CudaMat::readDeviceFromArray(void* hostmem, size_t hostpitch_bytes, int width, int height){
  checkArrayAllocation();
  if(width==cols_ && height==rows_){
    cudaError_t err = CudaSafeCall(cudaMemcpy2DFromArray((void*)hostmem, hostpitch_bytes, cuda_array_.get(), 0, 0, width*elemSize_,height, cudaMemcpyDeviceToHost));
    if(err == cudaSuccess){
      printf("[CUDA] Read %i bytes data from Device\n", width*height*elemSize_);
    }
  } else{
    fprintf(stderr,"[CUDA] [Read Device] Dimension of Device Source Memory and Source Memory do not Match\n");
    fprintf(stderr,"[CUDA] Host   Memory width (in element): %i, height(in element): %i \n",width, height);
    fprintf(stderr,"[CUDA] Device Memory width (in element): %i, height(in element): %i \n",cols_, rows_);
    exit(-1);
  }
}

void CudaMat::copyFromMat(const Mat& mat)
{
  if(mat.type()==type_){
    writeDevice((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Write Device] Mat type not compatible");
    exit(-1);
  }
}

void CudaMat::copyFromMatToArray(const Mat& mat)
{
  if(mat.type()==type_){
    writeDeviceToArray((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Write Device] Mat type not compatible");
    exit(-1);
  }
}

void CudaMat::copyToMat(Mat& mat)
{
  if(mat.type()==type_){
    mat = Mat(rows_, cols_, type_);
    readDevice((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Read device] Mat type not compatible");
  }
}

void CudaMat::copyToMatFromArray(Mat& mat)
{
  if(mat.type()==type_){
    mat = Mat(rows_, cols_, type_);
    readDeviceFromArray((void*)mat.data, mat.step[0], mat.cols, mat.rows);
  } else{
    fprintf(stderr, "[CUDA] [Read device] Mat type not compatible");
  }
}


void CudaMat::checkMemAllocation() const{
  if(cuda_mem_==nullptr){
    fprintf(stderr, "[CUDA] CUDA Memory is Not Allocated");
    exit(-1);
  } 
}

void CudaMat::checkArrayAllocation() const{
  if(cuda_array_ == nullptr){
    fprintf(stderr, "[CUDA] CUDA Array Texture Memory is Not Allocated");
    exit(-1);
  }
}
unsigned char* CudaMat::data() const{
  checkMemAllocation();
  return cuda_mem_.get();
}
const int CudaMat::rows() const{
  return rows_;
}
const int CudaMat::cols() const{
  return cols_;
}
const size_t CudaMat::pitch_bytes() const{
  checkMemAllocation();
  return pitch_bytes_;
}
const int CudaMat::depth() const{
  return depth_;
}
const int CudaMat::type() const{
  return type_;
}
const int CudaMat::elemSize() const{
  return elemSize_;
}
const cudaTextureObject_t CudaMat::texture_object() const{
  checkArrayAllocation();
  if(valid_texture_obj_==false){
    fprintf(stderr,"[CUDA] Invalid texture object");
    exit(-1);
  }
  return tex_obj_;
}
const cudaChannelFormatDesc CudaMat::channel_desc() const{
  checkArrayAllocation();
  return channel_desc_;
}
const cudaResourceDesc CudaMat::resource_desc() const{
  checkArrayAllocation();
  return res_desc_;
}
const cudaTextureDesc CudaMat::texture_desc() const{
  checkArrayAllocation();
  return tex_desc_;
}
}