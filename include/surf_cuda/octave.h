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
   * @param level_num number of levels of image in the Octave
   * @param rows number of rows for images in the Octave 
   * @param cols number of columns for images in the Octave 
   * @param filters_size size of all filters in the Octave, the size of filters_size should be the same as level_num 
   */
  Octave(const int& level_num, const int& rows, const int& cols, const std::vector<int>& filters_size);
public:
  /*!
   * @brief aallocate all the image levels in octave, only allocate once
   */
  void allocateGpu();
  /*!
   * @brief fill the octave with images calculate from DoH Filter with different size 
   * @param integral_mat input integral image
   * @param stride filter convolution stride on the integral_mat, stride need to be compatible \ 
   *               with rows and cols change between integral_mat and  images in the Octave
   */
  void fill(const CudaMat& integral_mat, int stride);
  
  /*!
   * @brief helper function that read gpu DoH result to cpu mat
   */
  void readDoHResponseMap(vector<Mat>& images_cpu);
  
  void findKeyPoints();
  
  void nonMaxSupression();
  //number of levels(images) in the Octave
  int level_num_; 
  //number of rows and columns, all levels in octave have the same rows and columns size
  int rows_;
  int cols_;
public:
  vector<CudaMat> images;//image levels
  vector<DoHFilter> filters;//DoH filters for different level images
};
}
#endif