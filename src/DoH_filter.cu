#include "surf_cuda/DoH_filter.cuh"

namespace surf_cuda {
#if USE_GLOBAL
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
#pragma unroll 2
#if 0
      for(int c=0; c<response_cols; c++){
	row_response_addr[c] = doh_filter(integral_mat, integral_pitch_bytes, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
#endif
      for(int c=threadIdx.y; c<response_cols; c+=blockDim.y){
	row_response_addr[c] = doh_filter(integral_mat, integral_pitch_bytes, row_integral_idx, c*stride, integral_rows, integral_cols); 
      }
    }
  }
}
void compDoHResponseMap(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter ,const int& stride){
  //check CudaMat type 
  if(img_integral.type()==CV_32S && img_doh_response.type() == CV_32F){
    size_t block_dim_x = 128;
    size_t block_dim_y = 4;
    dim3 block(block_dim_x,block_dim_y,1);
    dim3 grid(img_doh_response.rows()/block_dim_x + 1,1,1);
    kernel_DoH_Filter<<<grid,block>>> (img_integral.data(),img_integral.pitch_bytes(),img_integral.rows(),img_integral.cols(),img_doh_response.data(),img_doh_response.pitch_bytes(),img_doh_response.rows(),img_doh_response.cols(),stride, doh_filter);
    CudaCheckError();
    cudaDeviceSynchronize();
  } else{
    fprintf(stderr,"[CUDA] [DoH Response Map] The CudaMat type should be CV_32S for inpute, CV_32F for output");
  }
}
#endif

#if USE_TEXTURE
__global__ void kernel_DoH_Filter_texture(cudaTextureObject_t integral_tex, int integral_rows, int integral_cols, unsigned char* response_mat, size_t response_pitch_bytes, int response_rows, int response_cols, int stride, DoHFilter doh_filter){
    //Just one kernel per row's computation 
    int row_response_idx = threadIdx.y + blockDim.y * blockIdx.y;
    if(row_response_idx<response_rows){
      //response map's row pointer 
      int col_response_idx = threadIdx.x + blockDim.x * blockIdx.x;
      if(col_response_idx<response_cols){
        float* row_response_addr = (float*)(response_mat + row_response_idx * response_pitch_bytes);
        row_response_addr[col_response_idx] = doh_filter(integral_tex, row_response_idx*stride, col_response_idx*stride, integral_rows, integral_cols); 
      }
    }
}


void compDoHResponseMap_texture(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter, const int& stride){
  //check texture object type
  //texture object should be:
  //1. In border addressing mode 
  //2. Filter mode should be Point mode
  //3. Unnormalized coordinate 
  //4. cudaReadModeElementType Read mode, no type conversion 
  //get CUDA texture descriptor 
  cudaTextureDesc tex_desc = img_integral.texture_desc();
  //get CUDA texture object's resource descriptor
  cudaResourceDesc res_desc = img_integral.resource_desc();
  cudaChannelFormatDesc channel_desc = img_integral.channel_desc();
  if(
    tex_desc.addressMode[0] == cudaAddressModeBorder &&
    tex_desc.addressMode[1] == cudaAddressModeBorder &&
    tex_desc.filterMode == cudaFilterModePoint &&
    tex_desc.readMode == cudaReadModeElementType && 
    tex_desc.normalizedCoords == 0 &&
    res_desc.resType == cudaResourceTypeArray &&
    channel_desc.f == cudaChannelFormatKindSigned &&
    channel_desc.x == 32 &&
    channel_desc.y == 0 && 
    channel_desc.z == 0 &&
    channel_desc.w == 0 )
    {
      //check if texture height and width are compatible with stride and response rows and cols 
      if(img_integral.rows()/stride == img_doh_response.rows() && img_integral.cols()/stride == img_doh_response.cols()){
	size_t block_dim_x = 64;
	size_t block_dim_y = 10;
        dim3 block(block_dim_x,block_dim_y,1);
        dim3 grid(img_doh_response.cols()/block_dim_x+1,img_doh_response.rows()/block_dim_y + 1,1);
        kernel_DoH_Filter_texture<<<grid,block>>> (img_integral.texture_object(),img_integral.rows(),img_integral.cols(),img_doh_response.data(), img_doh_response.pitch_bytes(),img_doh_response.rows(),img_doh_response.cols(),stride, doh_filter);
        CudaCheckError();
        cudaDeviceSynchronize();
      } else{
	fprintf(stderr,"[CUDA] [DoH Filter] Incompatible integral image size, response map size and stride");
	fprintf(stderr,"Integral Image in texture memory has %i rows, %i columns",img_integral.rows(), img_integral.cols());
	fprintf(stderr,"Response Map has %i ros, %i columns",img_doh_response.rows(),img_doh_response.cols());
      }
  } else{
    fprintf(stderr,"[CUDA] [DoH Filter] Incorrect Texture Object Type");
  }
}
#endif
}