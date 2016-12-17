#ifndef trackerP_H
#define trackerP_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CMT.h"
#include "gui.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "classifier.hpp"
#include "trackerC.hpp"

using namespace cv;

class TrackerP
{
public:
	std::vector<TrackerC> TrackerCObj;
	float CalculateIR(int left1, int right1, int top1, int bottom1, int left2, int right2, int top2, int bottom2);
	void Clean();
	void Track(Mat im);
	void Export(std::vector<rRect> & objectRects);
	void ProcessObjects(Mat im_grey, std::vector<rRect> const& objectRects);
	void ProcessFrame(Mat const& im, std::vector<rRect> const& objectRects);
};

#endif