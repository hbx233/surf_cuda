#ifndef SURF_H_
#define SURF_H_

#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.h"
#include "surf_cuda/octave.h"


namespace surf_cuda{
class SURF{
public:
  SURF(){};
public:
  void compIntegralImage(const CudaMat& img_in, const CudaMat& img_out, const size_t& block_size_row, const size_t& block_size_col);
};  
}

#endif
