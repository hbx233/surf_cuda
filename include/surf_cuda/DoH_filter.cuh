#ifndef DOH_FILTER_H_
#define DOH_FILTER_H_
#include "surf_cuda/common.h"
#include "surf_cuda/cuda_util.cuh"
#include "surf_cuda/cuda_mat.h"
namespace surf_cuda{

/*!@brief: Fast computation of convolution with integral image \
 * unified weight in the whole region \
 * This is the basic component of DoH filter 
 */
template <typename T>
struct WeightedRegionIntegral{
  //integral region's top-left and bottom-right pixel's row and column offset 
  //convolution weights are the same number for one region
  //only contains few offset variables, no need for constant memory 
  int row_offset_1;//top-left
  int col_offset_1;
  int row_offset_2;//bottom-right
  int col_offset_2;
  //integral weight 
  T weight;
  //constructor
  WeightedRegionIntegral(int row_ofs_1, int col_ofs_1, int row_ofs_2, int col_ofs_2, T w)
  : row_offset_1(row_ofs_1),col_offset_1(col_ofs_1), row_offset_2(row_ofs_2), col_offset_2(col_ofs_2), weight(w)
  {}
  WeightedRegionIntegral(T w):weight(w),row_offset_1(0),col_offset_1(0),row_offset_2(0),col_offset_2(0){};
  WeightedRegionIntegral():weight(0),row_offset_1(0),col_offset_1(0),row_offset_2(0),col_offset_2(0){};
  void setOffsets(int row_ofs_1, int col_ofs_1, int row_ofs_2, int col_ofs_2){
    row_offset_1 = row_ofs_1; col_offset_1 = col_ofs_1;
    row_offset_2 = row_ofs_2; col_offset_2 = col_ofs_2;
  }
  /**
   *    ________________________________
   *   |(r1,c1)| ...            |       |    
   *   |       | ...            |       |
   *   |       | ...            |       |
   *   |_____  |________________|(r2,c2)|
   * 
   *   I(rigion(r1,c1,r2,c2)) = I(r2,c2) - I(r1-1,c2) - I(r2,c1-1) + I(r1-1,c1-1)
   */
  /*!@brief: device function which calculate weighted Integral from integral image in Global Memory  
   * @param integral_mat integral img pointer 
   * @param pitch integral img's 2D allocation pitch in byte
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the weighted sum (convolution with uniform weight)
   * @note: Returned value has the same type as elements in integral_mat
   */
  __device__ T operator()(unsigned char* integral_mat, const size_t& pitch_bytes, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
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
    //Handle warp divergence 
    //check if there are any intersection area
    //NOTE: Instead of using if(r1<=r2 && c1<=c2) which will cause warp divergence 
    //I use one int number to indicate if the weighted region intersect with integral image
    //and multiply intersect to the final result to handle the out-of-range problem
    //performance inproved about 2ms(12.5%) for global memory implementation  with 1920*1080 image 
    int intersect = (int)(r1<=r2 && c1<=c2);
    //bottom-right pixel's row address
    T* row_addr_2 = (T*)(integral_mat + r2*pitch_bytes);
    //top-left pixel's previous pixel's row address
    T* row_addr_1=NULL;
    if(r1==0){
      //row_addr_1 = integral_mat;
      if(c1==0){
	//top-left is (0,0)
	//no prev pixel 
	return intersect * weight * row_addr_2[c2];
      } else{
	//top-left is (0,c1) where c1>=1
	//top-left's prev pixel is (0,c1-1)
	return intersect * weight * (row_addr_2[c2] - row_addr_2[c1-1]);
      }
    } else{
      row_addr_1 = (T*)(integral_mat + (r1-1)*pitch_bytes);
      if(c1==0){
	//top-left is (r1,0)
	return intersect * weight * (row_addr_2[c2] - row_addr_1[c2]);
      } else{
	//both r1 c1 >=0
	return intersect * weight * (row_addr_2[c2]-row_addr_2[c1-1]-row_addr_1[c2] + row_addr_1[c1-1]);
      }
    }
  }
  
  /*!@brief: Overload device function which calculate weighted Integral from integral image in Texture Memory to Improve performance  
   * @param integral_tex integral img texture object  
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the weighted sum (convolution with uniform weight)
   * @note: Returned value has the same type as elements in integral_mat
   * @note: Assume the texture object is in border addressing mode, and point filtering mode, check the texture object's type in wrapper function 
   */
  __device__ T operator()(cudaTextureObject_t integral_tex, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    int r1 = max(0,row_c+row_offset_1);
    int c1 = max(0,col_c+col_offset_1);
    //bottom-right, use min
    //Note: rows and cols are provided in 1-base, whereas row_c and col_c are 0-base
    int r2 = min(rows-1, row_c+row_offset_2);
    int c2 = min(cols-1, col_c+col_offset_2);
    //eliminate warp divergence as described in 
    int intersect = (int)(r1<=r2 && c1<=c2);
    //assume texture array is in border addressing mode
    //no need to handle out-of-range problem
    //address coordinate in tex2D<> fetching function 
    // x - column
    // y - row
    //tex2D<T>(integral_tex,col,row)
    // I(rigion(r1,c1,r2,c2)) = I(r2,c2) - I(r1-1,c2) - I(r2,c1-1) + I(r1-1,c1-1)
    return intersect * weight * 
	     (tex2D<T>(integral_tex,c2,r2) 
	        - tex2D<T>(integral_tex,c2,r1-1) 
		  - tex2D<T>(integral_tex,c1-1,r2) 
		    + tex2D<T>(integral_tex,c1-1,r1-1)
	     );
  }
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in x direction
 */
//Question: Can the Following BoxFilters be implemented with polymorphysim?
template <typename T>
struct BoxFilter_xx{
  int size;
  WeightedRegionIntegral<T> wri_1;
  WeightedRegionIntegral<T> wri_2;
  WeightedRegionIntegral<T> wri_3;
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
  }
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
  __device__ T operator()(unsigned char* integral_mat, size_t pitch_bytes, int row_c, int col_c, int rows, int cols) const{
    return wri_1(integral_mat,pitch_bytes,row_c,col_c,rows,cols) 
	     + wri_2(integral_mat,pitch_bytes,row_c,col_c,rows,cols)
	       + wri_3(integral_mat,pitch_bytes,row_c,col_c,rows,cols);  
  }
  /*!
   * @brief Overload device function that caculate the convolution of Approximated Gaussian sencond derivative along x direction
   * from integral image in Texture Memory
   * @param integral_tex integral image's texture object  
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the convolution of Approximated Gaussian second derivative with image, at pixel (row_c,col_c)
   */
  __device__ T operator()(cudaTextureObject_t integral_tex, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    return wri_1(integral_tex, row_c, col_c, rows, cols)
	     + wri_2(integral_tex, row_c, col_c, rows, cols)
	       + wri_3(integral_tex, row_c, col_c, rows, cols);
  }
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in y direction
 */
template <typename T>
struct BoxFilter_yy{
  int size;
  WeightedRegionIntegral<T> wri_1;
  WeightedRegionIntegral<T> wri_2;
  WeightedRegionIntegral<T> wri_3;
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
  }
  __device__ T operator()(unsigned char* integral_mat, size_t pitch_bytes, int row_c, int col_c, int rows, int cols) const{
    return wri_1(integral_mat,pitch_bytes,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch_bytes,row_c,col_c,rows,cols) +
	       wri_3(integral_mat,pitch_bytes,row_c,col_c,rows,cols);  
  }
  __device__ T operator()(cudaTextureObject_t integral_tex, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    return wri_1(integral_tex, row_c, col_c, rows, cols)
	     + wri_2(integral_tex, row_c, col_c, rows, cols)
	       + wri_3(integral_tex, row_c, col_c, rows, cols);
  }
  
};

/*!
 *@brief The Box Filter which approximate second order Gaussian derivative in xy direction, use the same one  for yx direction
 */
template <typename T>
struct BoxFilter_xy{
  int size;
  WeightedRegionIntegral<T> wri_1;
  WeightedRegionIntegral<T> wri_2;
  WeightedRegionIntegral<T> wri_3;
  WeightedRegionIntegral<T> wri_4;
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
  __device__ T operator()(unsigned char* integral_mat, size_t pitch_bytes, int row_c, int col_c, int rows, int cols) const{
    return wri_1(integral_mat,pitch_bytes,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch_bytes,row_c,col_c,rows,cols) +
		wri_3(integral_mat,pitch_bytes,row_c,col_c,rows,cols) +  
		  wri_4(integral_mat,pitch_bytes,row_c,col_c,rows,cols);  
  }
  __device__ T operator()(cudaTextureObject_t integral_tex, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    return wri_1(integral_tex, row_c, col_c, rows, cols)
	     + wri_2(integral_tex, row_c, col_c, rows, cols)
	       + wri_3(integral_tex, row_c, col_c, rows, cols)
	         + wri_4(integral_tex, row_c, col_c, rows, cols);
  }
};
/*!
 * @brief The wrapper for Determinant of Hessian filter \
 * contains 3 BoxFilter, first compute each BoxFilter's convolution reasult as Hessian matrix's four entries, \
 * then compute the weighted determinant of Hessian matrix
 */
struct DoHFilter{
  //weight to calculate determinant 
  float weight{0.9};
  //size of all the BoxFilters 
  int size;
  //Calculate and operate integral image in int type
  //NOTE: Using float as integral image type will cause large error since float type cannot represent 
  //large integer value exactly 
  BoxFilter_xx<int> box_filter_xx;
  BoxFilter_yy<int> box_filter_yy;
  BoxFilter_xy<int> box_filter_xy;
  DoHFilter(int s): size(s),box_filter_xx(s),box_filter_yy(s),box_filter_xy(s){}
  
  /*!
   * @brief Top level device function that use Three BoxFilters' conv result to calculate the final Determinant of Hessian response for pixel (row_c, col_c)
   * @param integral_mat integral img pointer 
   * @param pitch integral img's 2D allocation pitch in byte
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the Determinant of Hessian response of pixel at (row_c, col_c)
   */
  __device__ float operator()(unsigned char* integral_mat, size_t pitch_bytes, int row_c, int col_c, int rows, int cols) const{
    return ((float)box_filter_xx(integral_mat, pitch_bytes, row_c, col_c, rows, cols) 
	      * (float)box_filter_yy(integral_mat, pitch_bytes, row_c, col_c, rows, cols) 
		 - powf(weight * (float)box_filter_xy(integral_mat, pitch_bytes, row_c, col_c, rows, cols),2))/(float)(size*size);
  }
  /*!
   * @brief Overload top level device function that use Three BoxFilters' conv result to calculate the final Determinant of Hessian response for pixel (row_c, col_c)
   *        from integral image in Texture Memory 
   * @param integral_tex integral image data that stored in texture memory 
   * @param row_c central pixel's row coordinate 
   * @param col_c central pixel's column coordinate
   * @param rows total rows of integral_mat
   * @param cols total columns of integral_mat
   * @return: the Determinant of Hessian response of pixel at (row_c, col_c)
   */
  __device__ float operator()(cudaTextureObject_t integral_tex, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    return ((float)box_filter_xx(integral_tex, row_c, col_c, rows, cols)
              * (float)box_filter_yy(integral_tex, row_c, col_c, rows, cols)
		 - powf(weight * (float)box_filter_xy(integral_tex, row_c, col_c, rows, cols),2))/(float)(size*size);
  }
};

  /*!
   * @brief Launch CUDA kernels to compute Determinant of Gaussian Blob response map from integral image 
   * @param img_integral Reference to input integral image whose data is already allocated on device memory
   * @param img_doh_response Reference to output integral image whose data is already allocated on device memory 
   */
  void compDoHResponseMap(const CudaMat& img_integral, const CudaMat& img_doh_response, const DoHFilter& doh_filter ,const int& stride);
  /*!
   * @brief Launch CUDA kernels to compute Determinant of Gaussian Blob response map from integral image in Texture Memory 
   * @param img_integral Reference to input integral image whose data is already allocated on device memory
   * @param img_doh_response Reference to output integral image whose data is already allocated on device memory 
   */
  void compDoHResponseMap_texture(cudaTextureObject_t integral_tex, const int& integral_rows, const int& integral_cols, const CudaMat& img_doh_response, const DoHFilter& doh_filter, const int& stride);
}
#endif