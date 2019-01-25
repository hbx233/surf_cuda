add_library(surf_cuda STATIC
	    cuda_mat.cu
	    surf.cu
	    DoH_filter.cu
            octave.cu
)
target_link_libraries(surf_cuda ${OpenCV_LIBRARIES})
target_compile_features(surf_cuda PUBLIC cxx_std_11)
set_target_properties(surf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
