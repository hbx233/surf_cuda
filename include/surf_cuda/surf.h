#ifndef SURF_H_
#define SURF_H_

#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.cuh"
#include "surf_cuda/octave.h"


namespace surf_cuda{
class SURF{
public:
  SURF(){};
public:
  /*!
   * @brief Launch CUDA kernels to compute integral image 
   * @param img_in Reference to inpute img whose data is already allocated on device memory 
   * @param img_integral Reference to output integral image whose data is already allocated on device memory 
   * @note Only compute integral image one time, and use integral image for all DoH filters 
   */
  void compIntegralImage(const CudaMat& img_in, const CudaMat& img_integral);
  /*!
   * @brief Launch CUDA kernels to compute Determinant of Gaussian Blob response map from integral image 
   * @param img_integral Reference to input integral image whose data is already allocated on device memory
   * @param img_doh_response Reference to output integral image whose data is already allocated on device memory 
   */
  void compDoHBlobResponseMap(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter, const int& stride);

public:
  Octave octave[3];//Three octaves used in the surf 
};  
}

#endif
