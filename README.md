# Resource Aware Hawk Eye
This repository contains the code used in my masters thesis " Resource-Aware Convolutional Nerual Networks for Object Detection using Video Temporal Data" at TUM LIS.

## Requires:

	1. c compiler
	2. c++11 compiler
	3. OpenCV3
	4. OpenMP	'Most compilers are shipped with it'
	5. cmake
	6. Darknet configuartion and weight files

Darknet weight files can be downloaded from [here](http://pjreddie.com/media/files/yolo.weights). Differenct weights can be found on Darknet [website](http://pjreddie.com/darknet/)

## Compilation

**Linux:** To setup the envirnment for linux user with apt-get based package system follow these steps:
	
	sudo apt-get install build-essentials cmake opencv3 
	
**OSX:** To setip the enviornemtn for OSX (macOS) users follow these steps:

	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	brew update
	brew install clang-omp
	brew install cmake
	brew install opencv3
	brew ln opencv3 --force
To compile, copy the repo to your machine and using the terminal navgiagte to repo directory

	mkdir build && cd build
	cmake .. #Linux Users
	CC=clang-omp CXX=clang-omp++ cmake .. #OSX Users
	make
	
## Usage

RAHE support 4 modes of operation:

**1. Normal:** This a passthrough to YOLO's original detector and processes and single image. Usage:

	./RAHE yolo [cfg] [weight] [image-file]

**2. YOLO-Stripe:** Benchmarks YOLO-Stripe method over the VOC2012 validation set. Describtion in [Benchmarking](#benchmark)

**3. YOLO-Occur:** Ouputs the occurence of the number of processed stipes using Stripe-YOLO over VOC2012 validation set. Describtion in [Benchmarking](#benchmark)

**4. Demo:** A proof-of-concept real-time object detector. Usage:

	./RAHE demo [cfg] [weights]
	press S to toggle Sripped mode.
	
## <a name="benchmark"></a> Benchmarking

1. Download the VOC2012 validation dataset and the development kit code from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar), then extract the archives.

2. Copy the val.txt from VOC2012 directory to RAHE root directory. The following commands creates the necessary filenames to be used by Darknet:

		cd ${RAHE_ROOT}
		cp ${VOCdevkit}/VOC2012/ImageSets/Main/val.txt .
		sed 's?^?'`pwd`'${VOCdevkit}/VOC2012/JPEGImages/?; s?$?.jpg?' val > voc.2012.val

3. Place **voc.2012.val** inside **build/data/** directory.

4. Benchmarck are designed to be run on multipe PCs to fasten the compuatations.
	
	a. **Strip-YOLO:** To becnhmark the accuracy of Stripe-YOLO run the following command:
		
		./RAHE stripped [cfg] [weights] [output-base-file] [offset] [range]
	b. **Occurance:** To calculate the occurance rate of the number of stripes in Stripe-YOLO, run the following commands:
	
		./RAHE occur [cfg] [weights] [output-base-file] [offset] [range]
		
		
 **output-base-file** is the output directory + the base file name for the results e.g. results/comp4_det_val_stripped
	
 **offset** is the number of images to be skipped from VOC2012 set.
	
 **range** is the number of images to be processed by this PC.

The resulting txt files are to be processed by the scripts avialble in **scripts** directory. Then to produce **mAP** and the **recall-precession curve**, use the MATLAB scripts available in **${VOCdevkit}/VOCcode/** specifically the **VOCevaldet.m** script.

## Libraries
This code uses three main external libraries:

1. [Darknet](https://github.com/pjreddie/darknet)
2. [CppMT](https://github.com/gnebehay/CppMT)
3. [RectangleBinPack](https://github.com/juj/RectangleBinPack)