#ifndef KEYPOINTS_H_
#define KEYPOINTS_H_

//KeyPoints in Both CPU Host and GPU Device Memory, use Stucture of arrays
#define MAX_NUM_KEYPOINTS 200
class KeyPoints{
public:
  __device__ addKeyPoint(float x, float y){
    curr_idx_ = atomicAdd(&curr_idx_,1);
    key_points_x
  }
private:
  __device__ int curr_idx_;
  __device__ float key_points_x[MAX_NUM_KEYPOINTS];
  __device__ float key_points_y[MAX_NUM_KEYPOINTS];
};
#endif 