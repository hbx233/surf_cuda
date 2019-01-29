#ifndef OCTAVE_H_
#define OCTAVE_H_

#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#include "surf_cuda/DoH_filter.cuh"

#define MAX_NUM_KEY_POINTS 200

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
  void allocateMemAndArray();
  /*!
   * @brief fill the octave with images calculate from DoH Filter with different size 
   * @param integral_mat input integral image
   */
  void fill(const CudaMat& integral_mat);
  
  /*!
   * @brief copy the response maps to their Texture memory and set cuda texture object interface for all the response maps 
   * for future thresholding and non-max supression 
   */
  void copyResponseMapsToArray();
  
  /*!
   * @brief helper function that read gpu DoH result to cpu mat
   */
  void readDoHResponseMap(vector<Mat>& images_cpu);
  
  /*!
   * @brief helper function taht read gpu DoH result after thresholding and non-max supression to cpu mat 
   */
  void readDoHResponseMapAfterSupression(vector<Mat>& images_cpu);
  
  /*!
   * @brief do thresholding and non max suppression on DoH response maps 
   */
  void thresholdAndNonMaxSuppression();
  
  void findKeyPoints();
  
  //number of levels(images) in the Octave
  int level_num_; 
  //number of rows and columns of response maps, 
  //all levels in same octave have the same rows and columns size
  int rows_;
  int cols_;
  int stride_;
  float threshold_ = 400;
public:
  vector<CudaMat> response_maps;//image levels
  vector<float> scales;//scales of every response map
  vector<DoHFilter> filters;//DoH filters for different level images
  shared_ptr<float> cuda_keypoints_x;
  shared_ptr<float> cuda_keypoints_y;
  shared_ptr<int> cuda_curr_idx;
  int keypoints_num;
  float keypoints_x[MAX_NUM_KEY_POINTS];
  float keypoints_y[MAX_NUM_KEY_POINTS];
  
};
}
#endif