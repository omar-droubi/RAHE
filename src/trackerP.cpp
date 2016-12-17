#include "trackerP.hpp"

float TrackerP::CalculateIR(int left1, int right1, int top1, int bottom1, int left2, int right2, int top2, int bottom2)
{
	int left, right, top, bottom, S1, S2, SI, SU;
	float IR;
	left = std::max(left1, left2);
	right = std::min(right1, right2);
	top = std::max(top1, top2);
	bottom = std::min(bottom1, bottom2);
	bottom = std::min(bottom1, bottom2);
	S1 = (right1 - left1) * (bottom1 - top1);
	S2 = (right2 - left2) * (bottom2 - top2);
	SI = (right - left) * (bottom - top);
	SU = S1 + S2 - SI;
	if((left<right) && (top < bottom))	//Holds if there is an intersection
	{
		if(S1 >= S2)	//S2 is larger in area >> check if S2 is contained withing S1
		{
			return (float)SI/S2;
		}
		else
		{
			return (float)SI/S1;
		}
	}
	return 0.0;
}
void TrackerP::Clean()
{
	TrackerCObj.erase(std::remove_if(TrackerCObj.begin(), TrackerCObj.end(), 
			[](const TrackerC & o) -> bool {
				if(o.prob==0.0)
					return true;
				else
					return false;}),
			TrackerCObj.end());
	
	int i;
}
void TrackerP::Track(Mat im)
{
	Mat im_grey;
	if(im.channels() > 1)
	{
		cvtColor(im, im_grey, CV_BGR2GRAY);
	}
	else
	{
		im_grey = im;
	}
	int i;
	for(i = 0 ; i < TrackerCObj.size() ; ++i)
	{
		TrackerCObj[i].draw = false;
		if(!TrackerCObj[i].updated)
		{
			TrackerCObj[i].Track(im_grey);
		}
	}
	Clean();
	for(i = 0 ; i < TrackerCObj.size() ; ++i)
	{
		TrackerCObj[i].updated = false;
	}
}
void TrackerP::ProcessObjects(Mat im_grey, std::vector<rRect> const& objectRects)
{
	//Scan if there is a tracker for an object
	int i, j, maxJ;
	int left1, left2, right1, right2, top1, top2, bottom1, bottom2;
	float IR, maxIR;
	for (i = 0 ; i < objectRects.size() ; ++i)
	{	
		left1 = objectRects[i].org_x;
		right1 = left1  + objectRects[i].width;
		top1 = objectRects[i].org_y;
		bottom1 = top1 + objectRects[i].height;
		maxIR = 0.0;
		for(j = 0 ; j < TrackerCObj.size() ; ++j)
		{
			if(!TrackerCObj[j].updated)
			{
				left2 = TrackerCObj[j].px;
				right2 = left2 +  TrackerCObj[j].pwidth;
				top2 = TrackerCObj[j].py;
				bottom2 = top2 + TrackerCObj[j].pheight;
				IR = CalculateIR(left1, right1, top1, bottom1, left2, right2, top2, bottom2);
				if(IR > maxIR)
				{
					maxIR = IR;
					maxJ = j;
				}
			}
		}
		if(maxIR < 0.75)	//No previous objects were found, This is the overlappig threshold
		{
			// Add a new CMT for this object :D
			int width = objectRects[i].width;
			int height = objectRects[i].height;
			std::string label = objectRects[i].label;
			float prob = objectRects[i].prob;
			//TrackerCObj.push_back(TrackerC(im_grey, left1, right1, width, height, label, prob));
			TrackerCObj.push_back(TrackerC());
			TrackerCObj.back().Update(im_grey, left1, top1, width, height, label, prob);
		}
		else	//A previous CMT was found >> update it with the new frame >> basically create a new object, bnut update the filter weights
		{	
			int width = objectRects[i].width;
			int height = objectRects[i].height;
			std::string label = objectRects[i].label;
			float prob = objectRects[i].prob;
			TrackerCObj[maxJ].Update(im_grey, left1, top1, width, height, label, prob);
		}
	}
}
void TrackerP::ProcessFrame(Mat const& im, std::vector<rRect> const& objectRects)
{
	Mat im_grey;
	if(im.channels() > 1)
	{
		cvtColor(im, im_grey, CV_BGR2GRAY);
	}
	else
	{
		im_grey = im;
	}
	ProcessObjects(im_grey, objectRects);
	Clean();
	Track(im_grey);
}
void TrackerP::Export(std::vector<rRect> & objectRects)
{
	int i;
	for(i = 0 ; i < TrackerCObj.size() ; ++i)
	{
		if(TrackerCObj[i].draw)
			objectRects.push_back(rRect(TrackerCObj[i].x, TrackerCObj[i].y, TrackerCObj[i].width, TrackerCObj[i].height, TrackerCObj[i].prob, rvoc_names[TrackerCObj[i].label]));
	}
}