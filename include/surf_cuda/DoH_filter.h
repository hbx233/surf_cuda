#ifndef DOH_FILTER_H_
#define DOH_FILTER_H_
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"

namespace surf_cuda{

/*!@brief: Fast computation of convolution with integral image \
 * unified weight in the whole region \
 * This is the basic component of DoH filter 
 */
struct WeightedRegionIntegral{
  //integral region's top-left and bottom-right pixel's row and column offset 
  //convolution weights are the same number for one region
  //only contains few offset variables, no need for constant memory 
  int row_offset_1;//top-left
  int col_offset_1;
  int row_offset_2;//bottom-right
  int col_offset_2;
  //integral weight 
  float weight;
  //constructor
  WeightedRegionIntegral(int row_ofs_1, int col_ofs_1, int row_ofs_2, int col_ofs_2, float w)
  : row_offset_1(row_ofs_1),col_offset_1(col_ofs_1), row_offset_2(row_ofs_2), col_offset_2(col_ofs_2), weight(w)
  {};
  WeightedRegionIntegral(float w):weight(w),row_offset_1(0),col_offset_1(0),row_offset_2(0),col_offset_2(0){};
  WeightedRegionIntegral():weight(0),row_offset_1(0),col_offset_1(0),row_offset_2(0),col_offset_2(0){};
  void setOffsets(int row_ofs_1, int col_ofs_1, int row_ofs_2, int col_ofs_2){
    row_offset_1 = row_ofs_1; col_offset_1 = col_ofs_1;
    row_offset_2 = row_ofs_2; col_offset_2 = col_ofs_2;
  };
  /**
   *    ________________________________
   *   |(r1,c1)| ...            |       |    
   *   |       | ...            |       |
   *   |       | ...            |       |
   *   |_____  |________________|(r2,c2)|
   * 
   *   I(rigion(r1,c1,r2,c2)) = I(r2,c2) - I(r1-1,c2) - I(r2,c1-1) + I(r1-1,c1-1)
   */
  /*!@brief: device function which calculate weighted Integral from integral image, 
   * @param integral_mat integral img pointer 
   * @param pitch integral img's 2D allocation pitch in byte
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the weighted sum (convolution with uniform weight)
   */
  __device__ float operator()(float* integral_mat, const size_t& pitch, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    //compute row address
    //get intersection of union to avoid exceeding the image region
    //top-left corner of intersection of integral image and region
    //use max
    int r1 = max(0,row_c+row_offset_1);
    int c1 = max(0,col_c+col_offset_1);
    //bottom-right, use min
    //Note: rows and cols are provided in 1-base, whereas row_c and col_c are 0-base
    int r2 = min(rows-1, row_c+row_offset_2);
    int c2 = min(cols-1, col_c+col_offset_2);
    //check if there are any intersection area
    if(r1<=r2 && c1<=c2){
      //printf("[CUDA] [Integral CONV] Has intesection\n");
      //bottom-right pixel's row address
      float* row_addr_2 = (float*)((char*)integral_mat + r2*pitch);
      //top-left pixel's previous pixel's row address
      float* row_addr_1=NULL;
      if(r1==0){
        //row_addr_1 = integral_mat;
        if(c1==0){
	  //top-left is (0,0)
	  //no prev pixel 
	  return weight * row_addr_2[c2];
        } else{
	  //top-left is (0,c1) where c1>=1
	  //top-left's prev pixel is (0,c1-1)
	  return weight * (row_addr_2[c2] - row_addr_2[c1-1]);
        }
      } else{
        row_addr_1 = (float*)((char*)integral_mat + (r1-1)*pitch);
        if(c1==0){
	  //top-left is (r1,0)
	  return weight * (row_addr_2[c2] - row_addr_1[c2]);
        } else{
	  //both r1 c1 >=0
	  return weight * (row_addr_2[c2]-row_addr_2[c1-1]-row_addr_1[c2] + row_addr_1[c1-1]);
        }
      }
    } else{
      //printf("[CUDA] [Integral CONV] Doesn't have intesection\n");
      return 0;
    }
  }
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in x direction
 */
//Question: Can the Following BoxFilters be implemented with polymorphysim?
struct BoxFilter_xx{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  /*!@brief initialize the BoxFilter with certain size, then use the size to determain the location and size \
   * of WeightedRegionIntegral that being used to approximate Gaussian second derivative filter,
   * For mory information, please check the original SURF paper 
   */
  BoxFilter_xx(int s):size(s),wri_1(1),wri_2(-2),wri_3(1){
    if(size<9){
      fprintf(stderr,"[CUDA] [Filter] Filter size must greater than 9");
      exit(-1);
    } else{
      //configure all the WeightedRegionIntegral
      wri_1.setOffsets(-(size-1)/2, -(size/3-1), -(size+3)/6, size/3-1 );
      wri_2.setOffsets(-(size-3)/6, -(size/3-1),  (size-3)/6, size/3-1 );
      wri_3.setOffsets( (size+3)/6, -(size/3-1),  (size-1)/2, size/3-1 );
    }
  };
  /*!
   * @brief device function that caculate the convolution of Approximated Gaussian sencond derivative along x direction\
   * @param integral_mat integral img pointer 
   * @param pitch integral img's 2D allocation pitch in byte
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the convolution of Approximated Gaussian second derivative with image, at pixel (row_c,col_c)
   */
  __device__ float operator()(float* integral_mat, size_t pitch, int row_c, int col_c, int rows, int cols){
    return wri_1(integral_mat,pitch,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch,row_c,col_c,rows,cols) +
	       wri_3(integral_mat,pitch,row_c,col_c,rows,cols);  
  };
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in y direction
 */
struct BoxFilter_yy{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  BoxFilter_yy(int s):size(s),wri_1(1),wri_2(-2),wri_3(1){
    if(size<9){
      fprintf(stderr,"[CUDA] [Filter] Filter size must greater than 9");
      exit(-1);
    } else{
      //configure all the WeightedRegionIntegral
      wri_1.setOffsets(-(size/3-1), -(size-1)/2, size/3-1, -(size+3)/6 );
      wri_2.setOffsets(-(size/3-1), -(size-3)/6, size/3-1,  (size-3)/6 );
      wri_3.setOffsets(-(size/3-1),  (size+3)/6, size/3-1,  (size-1)/2 );
    }
  };
  __device__ float operator()(float* integral_mat, size_t pitch, int row_c, int col_c, int rows, int cols){
    return wri_1(integral_mat,pitch,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch,row_c,col_c,rows,cols) +
	       wri_3(integral_mat,pitch,row_c,col_c,rows,cols);  
  };
  
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in xy direction, use the same one  for yx direction
 */
struct BoxFilter_xy{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  WeightedRegionIntegral wri_4;
  BoxFilter_xy(int s):size(s),wri_1(1),wri_2(-1),wri_3(-1), wri_4(1){
    if(size<9){
      fprintf(stderr,"[CUDA] [Filter] Filter size must greater than 9");
      exit(-1);
    }  else{
      wri_1.setOffsets(-(size/3), -(size/3),  -1     ,  -1     );
      wri_2.setOffsets(-(size/3),  1       ,  -1     , (size/3));
      wri_3.setOffsets( 1       , -(size/3), (size/3),  -1     );
      wri_4.setOffsets( 1       ,  1       , (size/3), (size/3));
    }
  }
  __device__ float operator()(float* integral_mat, size_t pitch, int row_c, int col_c, int rows, int cols){
    return wri_1(integral_mat,pitch,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch,row_c,col_c,rows,cols) +
		wri_3(integral_mat,pitch,row_c,col_c,rows,cols) +  
		  wri_4(integral_mat,pitch,row_c,col_c,rows,cols);  
  };
};
/*!
 * @brief The wrapper for Determinant of Hessian filter \
 * contains 3 BoxFilter, first compute each BoxFilter's convolution reasult as Hessian matrix's four entries, \
 * then compute the weighted determinant of Hessian matrix
 */
struct DoHFilter{
  float weight{0.9};
  //size of all the BoxFilters 
  int size;
  BoxFilter_xx box_filter_xx;
  BoxFilter_yy box_filter_yy;
  BoxFilter_xy box_filter_xy;
  DoHFilter(int s): size(s),box_filter_xx(s),box_filter_yy(s),box_filter_xy(s){}
  
  /*!
   * @brief device function that use BoxFilters' conv result to calculate the final Determinant of Hessian response for pixel (row_c, col_c)
   * @param integral_mat integral img pointer 
   * @param pitch integral img's 2D allocation pitch in byte
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the Determinant of Hessian response of pixel at (row_c, col_c)
   */
  __device__ float operator()(float* integral_mat, size_t pitch, int row_c, int col_c, int rows, int cols){
    return (box_filter_xx(integral_mat, pitch, row_c, col_c, rows, cols) * 
	      box_filter_yy(integral_mat, pitch, row_c, col_c, rows, cols) - 
		 powf(weight *  box_filter_xy(integral_mat, pitch, row_c, col_c, rows, cols),2))/(float)(size*size);
  };
};

}
#endif