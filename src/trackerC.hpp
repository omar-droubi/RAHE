#ifndef trackerC_H
#define trackerC_H
#include <array>
#include <string>
#include <numeric>
#include "CMT.h"
#include "Classifier.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TrackerC
{
public:
	static const int FILTERLENGTH = 25;
	static const int NCLASSES = 20;
	int x;
	int y;
	int width;
	int height;
	int px;
	int py;
	int pwidth;
	int pheight;
	float prob;
	int label;
	std::array<std::array<float, FILTERLENGTH>, NCLASSES> probs;
	cmt::CMT cmt;
	bool updated;
	bool draw;
	TrackerC();
	TrackerC(Mat im_grey, int _x, int _y, int _width, int _height, std::string _label, float _prob);
	int GetIndex(std::string _label);
	void MaxProb();
	void Track(Mat im_grey);
	void UpdateFilter(float _prob, std::string _label);
	void Update(Mat im_grey, int _x, int _y, int _width, int _height, std::string _label, float _prob);
};
#endif