#include "trackerC.hpp"

TrackerC::TrackerC()
{
	updated = false;
	draw = false;
	for (int i = 0; i < NCLASSES; ++i)
	{
		for (int j = 0; j < FILTERLENGTH; ++j)
		{
			probs[i][j] = 0.0;
		}
	}
}
TrackerC::TrackerC(Mat im_grey, int _x, int _y, int _width, int _height, std::string _label, float _prob)
{
	x = _x;
	y = _y;
	width = _width;
	height = _height;
	Rect rect = Rect(_x, _y, _width, _height);
	updated = false;
	draw = false;
	for (int i = 0; i < NCLASSES; ++i)
	{
		for (int j = 0; j < FILTERLENGTH; ++j)
		{
			probs[i][j] = 0.0;
		}
	}
	cmt.initialize(im_grey, rect);
	UpdateFilter(_prob, _label);
}
int TrackerC::GetIndex(std::string _label)
{
	int i;
	for(i = 0 ; i < NCLASSES ; ++i)
	{
		if(!_label.compare(rvoc_names[i]))
			return i;
	}
	return -1;
}
void TrackerC::MaxProb()
{
	float maxProb = 0.0;
	float tmpProb;
	int i, maxI;
	for(i = 0 ; i < NCLASSES ; ++i)
	{
		tmpProb =  std::accumulate(probs[i].begin(), probs[i].end(), 0.0) / FILTERLENGTH;
		if(tmpProb >= maxProb)
			{
				maxProb = tmpProb;
				maxI = i;
			}
	}
	prob = std::accumulate(probs[maxI].begin(), probs[maxI].end(), 0.0) / FILTERLENGTH;
	label = maxI;
}
void TrackerC::Track(Mat im_grey)
{
	try{
		cmt.processFrame(im_grey);
		Rect rect = cmt.bb_rot.boundingRect();
		x = rect.x;
		y = rect.y;
		width = rect.width;
		height = rect.height;
		if(cmt.confidence < 0.4)
		{
			draw = false;
			UpdateFilter(0.0, rvoc_names[label]);
		}
		else
		{
			draw = true;
			//UpdateFilter(0, rvoc_names[label]);
			UpdateFilter(probs[label][0], rvoc_names[label]);
		}
	}
	catch(...){
		printf("Caught the Exception\n");
		prob = 0.0;
	}
}
void TrackerC::UpdateFilter(float _prob, std::string _label)
{
	updated = true;
	int i;
	int index = GetIndex(_label);
	for(i = 0 ; i < NCLASSES ; ++i)
	{	
		std::rotate(probs[i].rbegin(), probs[i].rbegin() + 1, probs[i].rend());
		if(i == index)
			probs[i][0] = _prob;
		else
			probs[i][0] = 0;
	}
	MaxProb();

}
void TrackerC::Update(Mat im_grey, int _x, int _y, int _width, int _height, std::string _label, float _prob)
{
	try{
		x = _x;
		y = _y;
		width = _width;
		height = _height;
		px = _x;
		py = _y;
		pwidth = _width;
		pheight = _height;
		updated = false;
		draw = true;
		Rect rect = Rect(_x, _y, _width, _height);
		cmt = cmt::CMT();
		cmt.initialize(im_grey, rect);
		UpdateFilter(_prob, _label);
	}
	catch(...){
		printf("Caught the Exception\n");
		prob = 0.0;
	}
}

