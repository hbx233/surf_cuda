#include "surf_cuda/DoH_filter.cuh"

namespace surf_cuda {
  
//kernel function 
__global__ void kernel_DoH_Filter(unsigned char* integral_mat, size_t integral_pitch_bytes, int integral_rows, int integral_cols, unsigned char* response_mat, size_t response_pitch_bytes, int response_rows, int response_cols, int stride, DoHFilter doh_filter){
  //first check if the output response map with current stride parameter can fit in the memory of input response map
  //will not use the fast few columns for sub sampling 
  if(response_rows==integral_rows/stride || response_cols==integral_cols/stride){
    //Just one kernel per row's computation 
    int row_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      float* row_response_addr = (float*)(response_mat + row_response_idx * response_pitch_bytes);
      int row_integral_idx = row_response_idx * stride;
      //loop through columns 
//#pragma unroll 4
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = doh_filter(integral_mat, integral_pitch_bytes, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
  }
}
void compDoHResponseMap(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter ,const int& stride){
  //check CudaMat type 
  if(img_integral.type()==CV_32S && img_doh_response.type() == CV_32F){
    size_t block_dim_x = 128;
    dim3 block(block_dim_x,1,1);
    dim3 grid(img_doh_response.rows()/block_dim_x + 1,1,1);
    kernel_DoH_Filter<<<grid,block>>> (img_integral.data,img_integral.pitch_bytes(),img_integral.rows(),img_integral.cols(),img_doh_response.data,img_doh_response.pitch_bytes(),img_doh_response.rows(),img_doh_response.cols(),stride, doh_filter);
    CudaCheckError();
    cudaDeviceSynchronize();
  } else{
    fprintf(stderr,"[CUDA] [DoH Response Map] The CudaMat type should be CV_32S for inpute, CV_32F for output");
  }
}

__global__ void kernel_DoH_Filter_texture(cudaTextureObject_t integral_tex, int integral_rows, int integral_cols, unsigned char* response_mat, size_t response_pitch_bytes, int response_rows, int response_cols, int stride, DoHFilter doh_filter){
    //Just one kernel per row's computation 
    int row_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      float* row_response_addr = (float*)(response_mat + row_response_idx * response_pitch_bytes);
      int row_integral_idx = row_response_idx * stride;
      //loop through columns 
//#pragma unroll 4
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = doh_filter(integral_tex, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
}


void compDoHResponseMap_texture(cudaTextureObject_t integral_tex, const CudaMat& img_doh_response, const DoHFilter& doh_filter, const int& stride){
  //check texture object type
  //texture object should be:
  //1. In border addressing mode 
  //2. Filter mode should be Point mode
  //3. Unnormalized coordinate 
  //4. cudaReadModeElementType Read mode, no type conversion 
  //get CUDA texture descriptor 
  cudaTextureDesc tex_desc;
  CudaSafeCall(cudaGetTextureObjectTextureDesc(&tex_desc, integral_tex));
  //get CUDA texture object's resource descriptor
  cudaResourceDesc res_desc;
  CudaSafeCall(cudaGetTextureObjectResourceDesc(&res_desc, integral_tex));
  if(
    tex_desc.addressMode[0] == cudaAddressModeBorder &&
    tex_desc.addressMode[1] == cudaAddressModeBorder &&
    tex_desc.filterMode == cudaFilterModePoint &&
    tex_desc.readMode == cudaReadModeElementType && 
    tex_desc.normalizedCoords == 0 &&
    res_desc.resType == cudaResourceTypeArray &&
    res_desc.desc.f == cudaChannelFormatKindSigned &&
    res_desc.desc.x == 32 &&
    res_desc.desc.y == 0 && 
    res_desc.desc.z == 0 &&
    res_desc.desc.w == 0 ){
      //check if texture height and width are compatible with stride and response rows and cols 
      int integral_cols = res_desc.width;
      int integral_rows = res_desc.height;
      if(integral_rows/stride == img_doh_response.rows() && integral_cols/stride == img_doh_response.cols()){
	
      } else{
	fprintf(stderr,"[CUDA] [DoH Filter] Incompatible integral image size, response map size and stride");
	fprintf(stderr,"Integral Image in texture memory has %i rows, %i columns",integral_rows, integral_cols);
	fprintf(stderr,"Response Map )
      }
    }
  } else{
    fprintf(stderr,"[CUDA] [DoH Filter] Incorrect Texture Object Type");
  }
}
}