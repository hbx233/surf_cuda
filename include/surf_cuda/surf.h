#ifndef SURF_H_
#define SURF_H_

#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.cuh"
#include "surf_cuda/octave.h"

#define NUM_OCTAVE
namespace surf_cuda{
class SURF{
public:
  using Ptr = shared_ptr<SURF>;
  SURF(){};
  SURF(const int& rows, const int& cols);  
public:
  /*!
   * @brief Copy input image data from Mat object on host to CudaMat object whose memory is on Device 
   */
  void copyInputImageToDevice(const Mat& img);
  /*!
   * @brief Launch CUDA kernels to compute integral image 
   * @param img_in Reference to inpute img whose data is already allocated on device memory 
   * @param img_integral Reference to output integral image whose data is already allocated on device memory 
   * @note Only compute integral image one time, and use integral image for all DoH filters 
   */
  void compIntegralImage(const CudaMat& img_in, const CudaMat& img_integral);
  
  void extractKeyPoints(Mat img_input);
private:
  void allocateInputAndIntegralImage();
  void allocateOctaves(const vector<vector<int>>& filter_sizes, const vector<int>& strides);
private:
  int rows_;
  int cols_;
  float threshold_{500};
  CudaMat cuda_img_in_;
  CudaMat cuda_img_integral_;
  vector<Octave> octaves_;//Three octaves used in the surf 
  vector<cv::Point2f> keypoints_;
};  
}

#endif
