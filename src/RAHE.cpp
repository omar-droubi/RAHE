extern "C"{
	#include "yolo.h"
	#include "network.h"
	#include "parser.h"
	#include "image.h"
	#include "box.h"
	#include "data.h"
	#include "list.h"
	#include "layer.h"
	#include "utils.h"
}
#include <omp.h>
#include "classifier.hpp"
#include "trackerP.hpp"
#include "MaxRectsBinPack.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CMT.h"
#include "gui.h"
#include <thread>
#include <atomic>
using namespace cv;

void YOLOStripped(char* cfg, char* weights, char* base, int offset, int range)
{
	network net = parse_network_cfg(cfg);
	load_weights(&net, weights);
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

	//char *base = "results/comp4_det_test_stripped";
	list *plist = get_paths("data/voc.2012.val");
	char **paths = (char **)list_to_array(plist);
	int classes = 20;
	if(offset > plist->size)    return;
	int m;
	if((offset + range) >= plist->size)
		m = plist->size;
	else
		m = offset + range;

	int j;
	FILE **fps = (FILE**)calloc(20, sizeof(FILE *));
	for(j = 0; j < classes; ++j){
		char buff[1024];
		snprintf(buff, 1024, "%s%d-%d_%s.txt", base, offset, m, rvoc_names[j].c_str());
		fps[j] = fopen(buff, "w");
	}
	char buff[1024];
	snprintf(buff, 1024, "%s%d-%d_%s.txt", base, offset, m, "timing");
	FILE *time_file = fopen(buff, "w");

	int netOutSize = get_yolo_output_size(net);
	ext_box *ext_boxes1st = (ext_box*)calloc(netOutSize, sizeof(ext_box));
	ext_box *ext_boxes2nd = (ext_box*)calloc(netOutSize*20, sizeof(ext_box));
	Classifier ClassifierObj = Classifier();

	int i;
	for(i = offset ; i < m ; ++i)
	{
		char* path = paths[i];
		char* id = basecfg(path);
		image im = load_image_color(path, 0, 0);
		image resized = resize_image(im, net.w, net.h);
		run_yolo_external(net, im, ext_boxes1st,1);
		ClassifierObj.ProcessYOLOOut1stRun(netOutSize, ext_boxes1st, -1.0);
		ClassifierObj.SetImageDim(im.w, im.h);
		ClassifierObj.TransformDimToCNN(ClassifierObj.objectRects);
		ClassifierObj.FillGrid();
		ClassifierObj.GenerateRectsFromGrid();
		int nW = ClassifierObj.PackBins(ClassifierObj.genObjectRects);
		image packed = make_image(net.w, net.h, 3);
		ClassifierObj.ModifyImage(resized, packed, ClassifierObj.genObjectRects);
		float t = validate_yolo_external_stripped(net,packed, id, base, ext_boxes2nd, i, offset, nW);
		ClassifierObj.ProcessYOLOOut2ndRun(netOutSize*20, ext_boxes2nd, true);
		fprintf(time_file, "%s %f %d\n", id, t, nW);
		for(j = 0 ; j < ClassifierObj.objectRects.size() ; ++j)
		{
			int clsIndex = ClassifierObj.objectRects[j].labelID;
			int xmin = ClassifierObj.objectRects[j].org_x;
			int xmax = xmin + ClassifierObj.objectRects[j].width;
			int ymin = ClassifierObj.objectRects[j].org_y;
			int ymax = ymin + ClassifierObj.objectRects[j].height;
			float prob = ClassifierObj.objectRects[j].prob;
			fprintf(fps[clsIndex], "%s %f %d %d %d %d\n", id,prob,xmin,ymin,xmax,ymax);
		}
		//im = load_image_color(path, 0, 0);
		//ClassifierObj.ColorImageBlocks(im, ClassifierObj.objectRects);
		//save_image(im, id);
		free(resized.data);
		ClassifierObj.genObjectRects.clear();
		ClassifierObj.objectRects.clear();
	}
	//run_yolo_external(net,"data/2008_000879.jpg", ext_boxes);
	/*
	image im = load_image_color("data/2008_000879.jpg",0 ,0);
	save_image(imPacked, "data/packed");
	run_yolo_external_stripped(net,"data/packed.png", ext_boxes, nW);
	ClassifierObj.ProcessYOLOOut2ndRun(netOutSize, ext_boxes);
	ClassifierObj.PrintObjects();
	ClassifierObj.ColorImageBlocks(im, ClassifierObj.objectRects);
	save_image(im, "data/Final");*/
}

void YOLONormal(char* cfg, char* weights, char* filename)
{
	network net = parse_network_cfg(cfg);
	load_weights(&net, weights);
	image im = load_image_color(filename, 0, 0);
	auto start = std::chrono::steady_clock::now();
	run_yolo_external(net, im, NULL, 0);
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	auto diff_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
	printf("Timing from Classifier is %.5f\n",diff_sec/1000000000.0 );
}

image MatToImage(Mat im)
{
	uchar *data = im.data;
	int h = im.rows;
	int w = im.cols;
	int c = im.channels();
	int step = im.step;
	image out = make_image(w, h, c);
	int i, j, k;
	int count = 0;
	for(k = 0 ; k < c ; ++k)
	{
		for(i = 0 ; i < h ; ++i)
		{
			for(j = 0 ; j < w ; ++j)
			{
				out.data[count++] = data[i*step + j*c + k]/255.0;
			}
		}
	}
	for(i = 0 ; i < w*h ; ++i)
	{
		float swap = out.data[i];
		out.data[i] = out.data[i + w*h*2];
		out.data[i + w*h*2] = swap;
	}
	return out;
}

void YoloThread(network &net, image &dIm, ext_box* ext_boxes , std::atomic<bool> &classifying, Classifier &ClassifierObj, int netOutSize, bool stripped, int nThreads)
{
	if(stripped)
	{
		ClassifierObj.TransformDimToCNN(ClassifierObj.objectRects);
		ClassifierObj.FillGrid();
		ClassifierObj.GenerateRectsFromGrid();
		int nW = ClassifierObj.PackBins(ClassifierObj.genObjectRects);
		printf("nW is %d\n", nW);
		image packed = make_image(net.w, net.h, 3);
		ClassifierObj.ModifyImage(dIm, packed, ClassifierObj.genObjectRects);
		run_yolo_external_stripped_mp(net, packed, ext_boxes, 1, nW, nThreads);
		ClassifierObj.ProcessYOLOOut2ndRun(netOutSize, ext_boxes, false);
		ClassifierObj.genObjectRects.clear();
		classifying = false;
	}
	else
	{
		ClassifierObj.objectRects.clear();
		run_yolo_external_mp(net, dIm, ext_boxes,1, nThreads);
		ClassifierObj.ProcessYOLOOutExtBoxes(netOutSize, ext_boxes, 0.2);
		classifying = false;
	}
}

void YOLOAware(char* cfg, char* weights)
{
	//OpenCV Stuff
	int fontFace = FONT_HERSHEY_DUPLEX;
	double fontScale = 0.75;
	int thickness = 1;
	int baseline = 0;
	Size textSize;
	Point textOrg;
	int fps = 30;
	int delay = 1000/fps;
	std::string WIN_NAME = "CMT";
	bool show_preview=true;
	namedWindow("Control", WINDOW_AUTOSIZE);
	namedWindow(WIN_NAME);
	VideoCapture cap;
	Mat im;
	Mat pIm;
	Mat im_resized;
	Mat cIm = Mat(350,350, CV_8UC3);
	cap.open(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
	char key;

	//Darknet Stuff
	network net = parse_network_cfg(cfg);
	load_weights(&net, weights);
	int netOutSize = get_yolo_output_size(net);
	ext_box *ext_boxes = (ext_box*)calloc(netOutSize, sizeof(ext_box));
	Classifier ClassifierObj = Classifier();
	ClassifierObj.SetImageDim(640, 360);
	std::array<bool, 10> cType = {true,true,false,true,false,false,true,false,false,true};
	int cTypeCounter = 0;
	bool stripped = false;
	
	//Threading
	int nCores = omp_get_num_threads();
	int cCores = 1;
	int tCores = nCores - 1;
	std::atomic<bool> classifying(false);
	std::thread t;
	t = std::thread([]{std::this_thread::sleep_for(std::chrono::seconds(0));});	//Dummpy Thread
	
	//Timing
	auto cStart = std::chrono::steady_clock::now();
	auto cEnd = std::chrono::steady_clock::now();
	auto cDiff = std::chrono::duration_cast<std::chrono::milliseconds>(cEnd - cStart);

	//Misc
	TrackerP TrackerPObj = TrackerP();
	std::vector<rRect> trackerRects;

	cap >> im;
	im.copyTo(pIm);
	while(true)
	{
		auto tStart = std::chrono::steady_clock::now();
		int b;
		cap >> im;
		key = waitKey(2);
		if(key == 's')
			stripped = !stripped;

		if(!classifying)
		{
			cEnd = std::chrono::steady_clock::now();
			cDiff =  std::chrono::duration_cast<std::chrono::milliseconds>(cEnd - cStart);
			cStart = std::chrono::steady_clock::now();
			t.join();
			classifying = true;
			if(stripped)
			{
				resize(im, im_resized,Size(448,448));
				printf("Mode is Striped\n");
				//if(trackerRects.size() != 0)
				//	ClassifierObj.objectRects = trackerRects;
			}
			else
			{
				im.copyTo(im_resized);
				printf("Mode is Normal\n");
			}
			trackerRects = ClassifierObj.objectRects;
			TrackerPObj.ProcessFrame(pIm, trackerRects);
			image dIm = MatToImage(im_resized);
			t = std::thread(YoloThread,std::ref(net), std::ref(dIm), std::ref(ext_boxes),
							 std::ref(classifying), std::ref(ClassifierObj), netOutSize, stripped, cCores);
			im.copyTo(pIm);
			//cDiff =  std::chrono::duration_cast<std::chrono::milliseconds>(cEnd - cStart);
		}
	
		
		trackerRects.clear();
		TrackerPObj.Track(im);
		TrackerPObj.Export(trackerRects);
		for(int i = 0 ; i < trackerRects.size() ; ++i)
		{
			Point pt1 = Point(trackerRects[i].org_x, trackerRects[i].org_y);
			Point pt2 = Point(pt1.x+trackerRects[i].width, pt1.y+trackerRects[i].height);
			rectangle(im, pt1, pt2, Scalar( 0, 55, 255 ), 1, 4);
			textSize = getTextSize(trackerRects[i].label, fontFace, fontScale, thickness, &baseline);
			baseline += thickness;
			textOrg = Point(pt1.x, pt1.y + textSize.height);
			rectangle(im, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0,0,255), -1);
			putText(im, trackerRects[i].label, textOrg, fontFace, fontScale, Scalar::all(255), thickness, LINE_AA);
		}
		resize(im, im_resized, Size(640, 360));
		imshow(WIN_NAME, im_resized);
		auto tEnd = std::chrono::steady_clock::now();
		auto tDiff = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);

		//Status Drawing
		cIm = Scalar(0,0,0);
		if(stripped)
			putText(cIm, "Mode: Stripped", Point(10,(2*textSize.height)), fontFace, fontScale, Scalar::all(255), thickness, LINE_AA);
		else
			putText(cIm, "Mode: Normal", Point(10,(2*textSize.height)), fontFace, fontScale, Scalar::all(255), thickness, LINE_AA);
		 
		putText(cIm, "Detection Delay: " + std::to_string(cDiff.count()/1000) + "." + std::to_string(cDiff.count()%1000) + " s"
			, Point(10,(4*textSize.height)), fontFace, fontScale, Scalar::all(255), thickness, LINE_AA);
		imshow("Control",cIm);

	}
}

void YOLOOccur(char* cfg, char* weights, char* base, int offset, int range)
{
	network net = parse_network_cfg(cfg);
	load_weights(&net, weights);
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

	//char *base = "results/comp4_det_test_occur_";
	list *plist = get_paths("data/voc.2012.val");
	char **paths = (char **)list_to_array(plist);
	int classes = 20;
	if(offset > plist->size)    return;
	int m;
	if((offset + range) >= plist->size)
		m = plist->size;
	else
		m = offset + range;

	int j;

	char buff[1024];
	snprintf(buff, 1024, "%s%d-%d_%s.txt", base, offset, m, "occur");
	FILE *time_file = fopen(buff, "w");

	int netOutSize = get_yolo_output_size(net);
	ext_box *ext_boxes1st = (ext_box*)calloc(netOutSize, sizeof(ext_box));
	ext_box *ext_boxes2nd = (ext_box*)calloc(netOutSize*20, sizeof(ext_box));
	Classifier ClassifierObj = Classifier();
	ClassifierObj.searchID = -2;
	int i;
	for(i = offset ; i < m ; ++i)
	{
		char* path = paths[i];
		char* id = basecfg(path);
		image im = load_image_color(path, 0, 0);
		run_yolo_external(net, im, ext_boxes1st,1);
		ClassifierObj.ProcessYOLOOut1stRun(netOutSize, ext_boxes1st, -1.0);
		ClassifierObj.SetImageDim(im.w, im.h);
		ClassifierObj.TransformDimToCNN(ClassifierObj.objectRects);
		ClassifierObj.FillGrid();
		ClassifierObj.GenerateRectsFromGrid();
		int nW = ClassifierObj.PackBins(ClassifierObj.genObjectRects);
		fprintf(time_file, "%s %d\n", id, nW);
		//im = load_image_color(path, 0, 0);
		//ClassifierObj.ColorImageBlocks(im, ClassifierObj.objectRects);
		//save_image(im, id);
		ClassifierObj.genObjectRects.clear();
		ClassifierObj.objectRects.clear();
	}
}
int display(Mat im, cmt::CMT & cmt)
{
	//Visualize the output
	//It is ok to draw on im itself, as CMT only uses the grayscale image
	for(size_t i = 0; i < cmt.points_active.size(); i++)
	{
		circle(im, cmt.points_active[i], 2, Scalar(0, 255, 0));
	}

	Scalar color;
	if (cmt.confidence < 0.3) {
	  color = Scalar(0, 0, 255);
	} else if (cmt.confidence < 0.4) {
	  color = Scalar(0, 255, 255);
	} else {
	  color = Scalar(255, 0, 0);
	}
	Point2f vertices[4];
	cmt.bb_rot.points(vertices);
	for (int i = 0; i < 4; i++)
	{
		line(im, vertices[i], vertices[(i+1)%4], color);
	}

	imshow("CMT", im);

	return waitKey(5);
}

int main(int argc, char* argv[])
{
	char* cfg = argv[2];
	char* weights = argv[3];
	if(strcmp(argv[1],"yolo") == 0)
		validate_yolo_external(cfg, weights,atoi(argv[4]), atoi(argv[5]) );
	else if(strcmp(argv[1],"stripped") == 0)
		YOLOStripped(cfg, weights, argv[4], atoi(argv[5]), atoi(argv[6]));
	else if(strcmp(argv[1],"normal") == 0)
		YOLONormal(cfg, weights, argv[4]);
	else if(strcmp(argv[1],"occur") == 0)
		YOLOOccur(cfg, weights, argv[4], atoi(argv[5]), atoi(argv[6]));
	else if(strcmp(argv[1],"demo") == 0)
		YOLOAware(cfg, weights);
	else
		printf("Option not recognized, please check you input\n");
	return 0;
}