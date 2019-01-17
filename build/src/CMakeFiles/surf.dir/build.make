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
include src/CMakeFiles/surf.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/surf.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/surf.dir/flags.make

src/CMakeFiles/surf.dir/cuda_mat.cu.o: src/CMakeFiles/surf.dir/flags.make
src/CMakeFiles/surf.dir/cuda_mat.cu.o: ../src/cuda_mat.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hbx/git/surf_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/CMakeFiles/surf.dir/cuda_mat.cu.o"
	cd /home/hbx/git/surf_cuda/build/src && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hbx/git/surf_cuda/src/cuda_mat.cu -o CMakeFiles/surf.dir/cuda_mat.cu.o

src/CMakeFiles/surf.dir/cuda_mat.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/surf.dir/cuda_mat.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/surf.dir/cuda_mat.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/surf.dir/cuda_mat.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target surf
surf_OBJECTS = \
"CMakeFiles/surf.dir/cuda_mat.cu.o"

# External object files for target surf
surf_EXTERNAL_OBJECTS =

src/libsurf.a: src/CMakeFiles/surf.dir/cuda_mat.cu.o
src/libsurf.a: src/CMakeFiles/surf.dir/build.make
src/libsurf.a: src/CMakeFiles/surf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hbx/git/surf_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libsurf.a"
	cd /home/hbx/git/surf_cuda/build/src && $(CMAKE_COMMAND) -P CMakeFiles/surf.dir/cmake_clean_target.cmake
	cd /home/hbx/git/surf_cuda/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/surf.dir/build: src/libsurf.a

.PHONY : src/CMakeFiles/surf.dir/build

src/CMakeFiles/surf.dir/clean:
	cd /home/hbx/git/surf_cuda/build/src && $(CMAKE_COMMAND) -P CMakeFiles/surf.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/surf.dir/clean

src/CMakeFiles/surf.dir/depend:
	cd /home/hbx/git/surf_cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hbx/git/surf_cuda /home/hbx/git/surf_cuda/src /home/hbx/git/surf_cuda/build /home/hbx/git/surf_cuda/build/src /home/hbx/git/surf_cuda/build/src/CMakeFiles/surf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/surf.dir/depend

