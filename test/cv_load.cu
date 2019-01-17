#include "surf_cuda/common.h"
#include "surf_cuda/cuda_mat.h"
#if 1
//cuda kernel 
//compute integral image along rows
//aligned, uncoalesced read/write
template <typename T>
__global__ void integral_row(T* img, T* integral_img_row, int rows, int cols){
    //get thread id
    int t_idx = (threadIdx.x + blockDim.x * blockIdx.x);
    //check the thread validity 
    if(t_idx<rows){
        int row_addr = t_idx*cols;
        //compute integral along the row 
        T integral_cache;
        for(int i=0;i<cols; i++){
            if(i==0){
                //initial value
                integral_cache=img[row_addr];
                integral_img_row[row_addr]=img[row_addr];
            } else{
                integral_cache = img[row_addr+i] + integral_cache;
                integral_img_row[row_addr+i] = integral_cache;
            }
        }
    }
}
//compute integral image along columns
//aligned coalesced read/write
template <typename T>
__global__ void integral_column(T* img, T* integral_img_col, int rows, int cols){
    //get thread id 
    int t_idx = (threadIdx.x + blockDim.x * blockIdx.x);
    //ckeck thread validity 
    if(t_idx<cols){
        T integral_cache;
        for(int i=0; i<rows; i++){
            if(i==0){
                integral_cache = img[t_idx];
                integral_img_col[t_idx] = img[t_idx];
            } else{
                integral_cache = img[i*cols+t_idx]+integral_cache;
                integral_img_col[i*cols+t_idx] = integral_cache; 
            }
        }
    }
} 
#endif
//validation host code 
template <typename T>
void integral_row_host(T* img, T* integral_img, int row, int col){
    for(int i=0; i<row; i++){
        T integral_cache;
        for(int j=0; j<col; j++){
            if(j==0){
                integral_cache = img[i*col];
                integral_img[i*col] = integral_cache;
            } else{
                integral_cache = integral_cache + img[i*col+j];
                integral_img[i*col+j] = integral_cache;
            }
        }
    }
}
template <typename T>
void integral_column_host(T* img, T* integral_img_col, int rows, int cols){
    for(int i=0;i<cols;i++){
        T integral_cache;
        for(int j=0; j<rows; j++){
            if(j==0){
                integral_cache = img[i];
                integral_img_col[i] = integral_cache;
            } else{
                integral_cache = integral_cache + img[j*cols+i];
                integral_img_col[j*cols+i] = integral_cache;
            }
        }
    }
}
template <typename T> 
bool compare(T* data1, T* data2, int size, char* msg){
    printf("%s\n",msg);
    for(int i=0; i<size; i++){
        if((data1[i]-data2[i])>=0.00001){
            printf("[COMPARE] Different\n");
            return false;
        }
        //printf("%f, %f\n",data1[i], data2[i]);
    }
    printf("[COMPARE] Same\n");
    return true;
}
template <typename T>
bool compare1(T* data, int row, int col){
    for(int i=0;i<row;i++){
        for(int j=0; j<col; j++){
            if((data[i*col+j]-(j+1))>=0.00001){
                return false;
            }
        }
    }
    return true;
}
int main(){
    cv::Mat img;// = cv::Mat::ones(1080,1920,CV_8U);
    img = cv::imread("./data/img1.png");
    //convert image to gray scale image
    cv::Mat gray_img = cv::Mat::ones(1080,1920,CV_8U);
    //cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    //convert image type from CV_8U to CV_32F
    cv::Mat gray_img_float;
    gray_img.convertTo(gray_img_float, CV_32F);
    //get image size 
    const int rows = gray_img_float.rows;
    const int cols = gray_img_float.cols;
    cout<<"rows: "<<rows<<endl;
    cout<<"cols: "<<cols<<endl;
    //get image data
    float* data_float = (float*)gray_img_float.data;
    float* data_gpu = (float*)malloc(rows*cols*sizeof(float));
    float* integral_row_gpu = (float*)malloc(rows*cols*sizeof(float));
    float* integral_row_cpu = (float*)malloc(rows*cols*sizeof(float));
    float* integral_col_gpu = (float*)malloc(rows*cols*sizeof(float));
    float* integral_col_cpu = (float*)malloc(rows*cols*sizeof(float));
    #if 0
    //verification of data conversion 
    for(int i = 0; i<gray_img.cols*gray_img.rows; i++){a
        cout<<"unsigned char: "<<(int)(data_uc[i])<<endl;
        cout<<"float: "<<data_float[i]<<endl;
    }
    #endif
    //copy float image data to GPU
#if 1
    float* cuda_gray_img_float;
    float* cuda_integral_img_row;
    float* cuda_integral_img_col;
    cudaMalloc((float**)&cuda_gray_img_float,rows*cols*sizeof(float)); 
    cudaMalloc((float**)&cuda_integral_img_row,rows*cols*sizeof(float));
    cudaMalloc((float**)&cuda_integral_img_col,rows*cols*sizeof(float));
#endif
    
    
    cout<<"Copy data to GPU"<<endl;
    cudaMemcpy(cuda_gray_img_float, data_float, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    //copy back to host to varify 
    cudaMemcpy(data_gpu,cuda_gray_img_float,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);
    //compare 
    compare<float>(data_gpu, data_float,rows*cols,"Compare copy data");
    //compute row integral
    int block_size = 128;
    int grid_size = rows/block_size + 1;
    dim3 block_row(block_size,1,1);
    dim3 grid_row(grid_size,1,1);
    //launch kernel
    cout<<"launch integral_row kernel: "<<endl;
    integral_row<float> <<<block_row, grid_row>>>(cuda_gray_img_float, cuda_integral_img_row,rows,cols);
    //synchronization of all thread 
    cudaDeviceSynchronize();
    cout<<"Copy result to Host"<<endl;
    cudaMemcpy(integral_row_gpu, cuda_integral_img_row,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);
    //calculate the ground truth 
    integral_row_host<float>(data_float, integral_row_cpu, rows, cols);
    cout<<"Compare result"<<endl;
    compare<float>(integral_row_gpu, integral_row_cpu,rows*cols,"Compare row integral results from GPU and CPU");
    //compute column integral 
    int block_size_col = 128;
    int grid_size_col = cols/block_size_col+1;
    dim3 block_col(block_size_col,1,1);
    dim3 grid_col(grid_size_col,1,1);
    //launch kernel 
    cout<<"launch integral_column kernel: "<<endl;
    integral_column<float> <<<block_col, grid_col>>>(cuda_integral_img_row, cuda_integral_img_col, rows, cols);
    //synchronization 
    cudaDeviceSynchronize();
    cout<<"Copy result to Host"<<endl;
    cudaMemcpy(integral_col_gpu, cuda_integral_img_col, rows*cols*sizeof(float),cudaMemcpyDeviceToHost);
    //compute cpu result
    integral_column_host<float>(integral_row_cpu, integral_col_cpu, rows, cols);
    compare<float>(integral_col_gpu, integral_col_cpu, rows*cols, "Compare col integral results from GPU and CPU");
    int j=1000;
    for(int i=0; i<rows; i++){
        //printf("%f, %f\n", integral_col_gpu[i*cols+j],integral_col_cpu[i*cols+j]);
        //printf("%f\n", integral_row_gt[i]);
    }
    #if 0
    cv::namedWindow("Display");
    imshow("Display",gray_img);
    cv::waitKey(0);
    #endif
    //free memory

    return 0;
}
