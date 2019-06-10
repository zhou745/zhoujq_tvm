CCFLAGS=-std=c++11
LDFLAGS=-lcudnn

all: conv2d_nchwc conv2d_nhwc

conv2d_nchwc: conv2d_nchwc.cu
	nvcc $? -o $@ ${CCFLAGS} ${LDFLAGS}

conv2d_nhwc: conv2d_nhwc.cu
	nvcc $? -o $@ ${CCFLAGS} ${LDFLAGS}

