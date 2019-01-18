#ifndef DOH_FILTER_H_
#define DOH_FILTER_H_
#include "surf_cuda/common.h"



namespace surf_cuda{

//Fast computation of convolution with integral image 
//unified weight in the whole region 
//This is the basic component of DoH filter 
struct WeightedRegionIntegral{
  //integral region's top-left and bottom-right pixel's row and column offset 
  int row_offset_1;//top-left
  int col_offset_1;
  int row_offset_2;//bottom-right
  int col_offset_2;
  //integral weight 
  float weight;
  WeightedRegionIntegral(int row_ofs_1, int col_ofs_1, int row_ofs_2, int col_ofs_2, float w)
  : row_offset_1(row_ofs_1),col_offset_1(col_ofs_1), row_offset_2(row_ofs_2), col_offset_2(col_ofs_2), weight(w)
  {};
  WeightedRegionIntegral(float w):weight(w),row_offset_1(0),col_offset_1(0),row_offset_2(0),col_offset_2(0){};
  WeightedRegionIntegral(){};
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
  //device function which calculate weighted Integral from integral image, 
  //param:
  //integral_mat: integral img pointer 
  //pitch: integral img's 2D allocation pitch in byte
  //row_c: central pixel's row coordinate 
  //col_c: central pixel's column coordinate
  __device__ float operator()(float* integral_mat, const size_t& pitch, const int& row_c, const int& col_c, const int& rows, const int& cols) const{
    //compute row address
    //get intersection of union to avoid exceeding the image region
    //top-left corner of intersection of integral image and region
    //use max
    int r1 = max(0,row_c+row_offset_1);
    int c1 = max(0,col_c+col_offset_1);
    //bottom-right, use min 
    int r2 = min(rows, row_c+row_offset_2);
    int c2 = min(cols, col_c+col_offset_2);
    //check if there are any intersection area
    if(r1<=r2 && c1<=c2){
      printf("[CUDA] [Integral CONV] Has intesection\n");
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
      printf("[CUDA] [Integral CONV] Doesn't have intesection\n");
      return 0;
    }
  }
};

struct DoHFilter_xx{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  DoHFilter_xx(int s):size(s),wri_1(1),wri_2(-2),wri_3(1){
    if(size<9){
      printf("[CUDA] [Filter] Filter size must greater than 9");
      
    } else{
      //configure all the WeightedRegionIntegral
      wri_1.setOffsets(-(size-1)/2, -(size/3-1), -(size+3)/6, size/3-1 );
      wri_2.setOffsets(-(size-3)/6, -(size/3-1),  (size-3)/6, size/3-1 );
      wri_3.setOffsets( (size+3)/6, -(size/3-1),  (size-1)/2, size/3-1 );
    }
  };
  //TODO: Normalize with respect to filter size
  __device__ float operator()(float* integral_mat, size_t pitch, int row_c, int col_c, int rows, int cols){
    return wri_1(integral_mat,pitch,row_c,col_c,rows,cols) + 
	     wri_2(integral_mat,pitch,row_c,col_c,rows,cols) +
	       wri_3(integral_mat,pitch,row_c,col_c,rows,cols);  
  };
};

struct DoHFilter_yy{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  DoHFilter_yy(int s):size(s),wri_1(1),wri_2(-2),wri_3(1){
    if(size<9){
      printf("[CUDA] [Filter] Filter size must greater than 9");
      
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

struct DoHFilter_xy{
  int size;
  WeightedRegionIntegral wri_1;
  WeightedRegionIntegral wri_2;
  WeightedRegionIntegral wri_3;
  WeightedRegionIntegral wri_4;
  DoHFilter_xy(int s):size(s),wri_1(1),wri_2(-1),wri_3(-1), wri_4(1){
    if(size<9){
      printf("[CUDA] [Filter] Filter size must greater than 9");
    }  else{
      wri_1.setOffsets();
      wri_2.setOffsets();
      wri_3.setOffsets();
      wri_4.setOffsets();
    }
  }
};

struct DoHFilter{

  
};

}
#endif