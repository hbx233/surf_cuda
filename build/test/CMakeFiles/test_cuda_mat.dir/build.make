# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hbx/git/surf_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hbx/git/surf_cuda/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_cuda_mat.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_cuda_mat.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_cuda_mat.dir/flags.make

test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o: test/CMakeFiles/test_cuda_mat.dir/flags.make
test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o: ../test/test_cuda_mat.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hbx/git/surf_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o"
	cd /home/hbx/git/surf_cuda/build/test && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hbx/git/surf_cuda/test/test_cuda_mat.cu -o CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o

test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_cuda_mat
test_cuda_mat_OBJECTS = \
"CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o"

# External object files for target test_cuda_mat
test_cuda_mat_EXTERNAL_OBJECTS =

test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: test/CMakeFiles/test_cuda_mat.dir/build.make
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: src/libsurf_cuda.a
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o: test/CMakeFiles/test_cuda_mat.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hbx/git/surf_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_cuda_mat.dir/cmake_device_link.o"
	cd /home/hbx/git/surf_cuda/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cuda_mat.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_cuda_mat.dir/build: test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o

.PHONY : test/CMakeFiles/test_cuda_mat.dir/build

# Object files for target test_cuda_mat
test_cuda_mat_OBJECTS = \
"CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o"

# External object files for target test_cuda_mat
test_cuda_mat_EXTERNAL_OBJECTS =

../bin/test_cuda_mat: test/CMakeFiles/test_cuda_mat.dir/test_cuda_mat.cu.o
../bin/test_cuda_mat: test/CMakeFiles/test_cuda_mat.dir/build.make
../bin/test_cuda_mat: src/libsurf_cuda.a
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
../bin/test_cuda_mat: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
../bin/test_cuda_mat: test/CMakeFiles/test_cuda_mat.dir/cmake_device_link.o
../bin/test_cuda_mat: test/CMakeFiles/test_cuda_mat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hbx/git/surf_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../../bin/test_cuda_mat"
	cd /home/hbx/git/surf_cuda/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cuda_mat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_cuda_mat.dir/build: ../bin/test_cuda_mat

.PHONY : test/CMakeFiles/test_cuda_mat.dir/build

test/CMakeFiles/test_cuda_mat.dir/clean:
	cd /home/hbx/git/surf_cuda/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_cuda_mat.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_cuda_mat.dir/clean

test/CMakeFiles/test_cuda_mat.dir/depend:
	cd /home/hbx/git/surf_cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hbx/git/surf_cuda /home/hbx/git/surf_cuda/test /home/hbx/git/surf_cuda/build /home/hbx/git/surf_cuda/build/test /home/hbx/git/surf_cuda/build/test/CMakeFiles/test_cuda_mat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_cuda_mat.dir/depend

