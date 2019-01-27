#ifndef OCTAVE_H_
#define OCTAVE_H_

#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.cuh"

namespace surf_cuda{
class Octave{
public:
  Octave(){};
  /*!
   * @brief Constructor of Octave 
   * @param rows number of rows for images in the Octave 
   * @param cols number of columns for images in the Octave 
   * @param stride stride number for DoH response map sub-sampling 
   * @param filters_size size of all filters in the Octave, the size of filters_size should be the same as level_num 
   */
  Octave(const int& rows, const int& cols, const int& stride, const std::vector<int>& filters_size);
public:
  /*!
   * @brief allocate all the image levels in octave, only allocate once
   */
  void allocate();
  /*!
   * @brief fill the octave with images calculate from DoH Filter with different size 
   * @param integral_mat input integral image
   */
  void fill(const CudaMat& integral_mat);
  
  /*!
   * @brief helper function that read gpu DoH result to cpu mat
   */
  void readDoHResponseMap(vector<Mat>& images_cpu);
  
  void thresholdAndNonMaxSuppression();
  
  void findKeyPoints();
  
  //number of levels(images) in the Octave
  int level_num_; 
  //number of rows and columns of response maps, 
  //all levels in same octave have the same rows and columns size
  int rows_;
  int cols_;
  int stride_;
public:
  vector<CudaMat> response_maps;//image levels
  vector<DoHFilter> filters;//DoH filters for different level images
};
}
#endif