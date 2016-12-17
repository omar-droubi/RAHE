#include "classifier.hpp"

std::string rvoc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

Classifier::Classifier()
{
	minNGrids = std::sqrt(nGridW/2);
	searchID = -2;
}

void Classifier::FillGrid()
{
	int i,j;
	int gridLeft, gridRight, gridTop,gridBottom;
	int objLeft, objRight, objTop, objBottom;
	float IR, preIR;	//Intersection area ration between two rectangles
	for(i = 0 ; i < gridSize ; ++i)
	{
		gridLeft = (i % nGridW) * gridW;
		gridRight = gridLeft + gridW;
		gridTop = (i / nGridH) * gridH;
		gridBottom = gridTop + gridH;
		grid[i] = -2;
		IR = 0.0;
		preIR = 0.0;
		for(j = 0 ; j < objectRects.size() ; ++j)
		{
			objLeft = objectRects[j].org_x;
			objRight = objLeft + objectRects[j].width;
			objTop = objectRects[j].org_y;
			objBottom = objTop + objectRects[j].height;
			IR = CalculateAOI(gridLeft, gridRight, gridTop, gridBottom, objLeft, objRight, objTop, objBottom);
			if(IR > preIR)
			{
				grid[i] = -1;
			}
			preIR = IR;
		}
	}
}

void Classifier::ColorImageBlocks(image &im, std::vector<rRect> const& rects)
{
	int i,j,k,m,n;
	int src_x, src_y, dest_x, dest_y;
	int objectWidth, objectHeight;
	for(i = 0 ; i < rects.size() ; ++i)
	{
		objectWidth = rects[i].width;
		objectHeight = rects[i].height;
		src_x = rects[i].org_x;
		src_y = rects[i].org_y;
		for(j = 0 ; j < objectWidth ; ++j)
		{
			for(k = 0 ; k < objectHeight ; ++k)
			{
				im.data[((k+src_y)*im.w) + (j+src_x)] = 0;
				im.data[(im.h*im.w) + ((k+src_y)*im.w) + (j+src_x)] = 0;
				im.data[(2*im.h*im.w) + ((k+src_y)*im.w) + (j+src_x)] = 0;
			}
		}
	}
}

void Classifier::GenerateRectsFromGrid()
{
	int objectID = 0;
	int rectW, rectH;
	bool res, hGrow, vGrow;
	int i,j;
	for(i = 0 ; i < gridSize ; ++i)
	{
		res = FindMinimumGridRect(i, objectID);
		if(!res)
		{
			continue;
		}
		rectW = minNGrids;
		rectH = minNGrids;
		for(j = i ; j < gridSize ; ++j)
		{
			hGrow = GrowRectHorizontal(i, objectID, rectW, rectH);
			if(hGrow)
				rectW++;
			vGrow = GrowRectVertical(i, objectID, rectW, rectH);
			if(vGrow)
				rectH++;
			if(rectH >= maxNGridH)
				break;
			if(!hGrow && !vGrow)
				break;
			
		}
		genObjectRects.push_back(rRect((i%nGridW) * gridW, (i/nGridW) * gridH, rectW * gridW, rectH * gridH, "WhatEver"));
		/*genObjectRects[objectID].org_x = (i%nGridW) * gridW;
		genObjectRects[objectID].org_y = (i/nGridW) * gridH;
		genObjectRects[objectID].width = rectW * gridW;
		genObjectRects[objectID].height = rectH * gridH; */
		objectID++;
	}
}

bool Classifier::GrowRectVertical(int startI, int objectID, int rectW, int rectH)
{
	if(((startI/nGridH) + rectH) >= nGridH)
		return false;
	int i;
	for(i = 0 ; i < rectW ; ++i)
	{
		if(grid[startI + i + rectH*nGridW] != searchID)
			return false;
	}
	for(i = 0 ; i < rectW ; ++i)
	{
		grid[startI + i + rectH*nGridW] = objectID;
	}
	return true;
}

bool Classifier::GrowRectHorizontal(int startI, int objectID, int rectW, int rectH)
{
	if(((startI%nGridW) + rectW) >= nGridW)
		return false;
	int i;
	for(i = 0 ; i < rectH ; ++i)
	{
		if(grid[startI + rectW + i*nGridW] != searchID)
			return false;
	}
	for(i = 0 ; i < rectH ; ++i)
	{
		grid[startI + rectW + i*nGridW] = objectID;
	}
	return true;
}

bool Classifier::FindMinimumGridRect(int startI, int objectID)
{
	if(((startI/nGridH) + minNGrids) >= nGridH)
		return false;
	if(((startI%nGridW) + minNGrids) >= nGridW)
		return false;
	int i,j;
	for(i = startI ; i < startI + minNGrids ; ++i)
	{
		for(j = 0 ; j < minNGrids ; ++j)
		{
			if(grid[i + j*nGridW] != searchID)
				return false;
		}
	}
	for(i = startI ; i < startI + minNGrids ; ++i)
	{
		for(j = 0 ; j < minNGrids ; ++j)
		{
			grid[i + j*nGridW] = objectID;
		}
	}
	return true;
}

void Classifier::PrintGrid()
{
	int i,j;
	for(i = 0 ; i < nGridH ; ++i)
	{
		for(j = 0 ; j < nGridW ; ++j)
		{
			if(grid[(i*nGridH) + j] < 0)
				printf("%d ", grid[(i*nGridH) + j]);
			else
				printf(" %d ", grid[(i*nGridH) + j]);
		}	
		printf("\n");
	}
	printf("\n");
}

float Classifier::CalculateAOI(int left1, int right1, int top1, int bottom1, int left2, int right2, int top2, int bottom2)
{
	int left, right, top, bottom, S1, S2, SI, SU;
	float IR;
	left = std::max(left1, left2);
	right = std::min(right1, right2);
	top = std::max(top1, top2);
	bottom = std::min(bottom1, bottom2);
	S1 = (right1 - left1) * (bottom1 - top1);
	S2 = (right2 - left2) * (bottom2 - top2);
	SI = (right - left) * (bottom - top);
	SU = S1 + S2 - SI;
	if((left<right) && (top < bottom))	//Holds if there is an intersection
	{
		//printf("Intersection Cord: left %d right %d top %d bottom %d\n",left, right, top, bottom);
		//printf("Intersection Area: %d\n", SI);
		//printf("Intersection Ratio: %.2f\n",(float)SI/SU );
		IR = (float)SI/SU;
		if(IR >= iThresh)
		{
			return IR;
		}
	}
	return 0.0;
}

int Classifier::CalculateIA(int left1, int right1, int top1, int bottom1, int left2, int right2, int top2, int bottom2)
{
	int left, right, top, bottom, S1, S2, SI, SU;
	float IR;
	left = std::max(left1, left2);
	right = std::min(right1, right2);
	top = std::max(top1, top2);
	bottom = std::min(bottom1, bottom2);
	S1 = (right1 - left1) * (bottom1 - top1);
	S2 = (right2 - left2) * (bottom2 - top2);
	SI = (right - left) * (bottom - top);
	SU = S1 + S2 - SI;
	if((left<right) && (top < bottom))	//Holds if there is an intersection
	{
		//printf("Intersection Cord: left %d right %d top %d bottom %d\n",left, right, top, bottom);
		//printf("Intersection Area: %d\n", SI);
		//printf("Intersection Ratio: %.2f\n",(float)SI/SU );
		return SI;
	}
	return 0;
}

float Classifier::CalculateMergerRatio(int left1, int right1, int top1, int bottom1, int left2, int right2, int top2, int bottom2)
{
	int left, right, top, bottom, S1, S2, SI, SU;
	float IR;
	left = std::max(left1, left2);
	right = std::min(right1, right2);
	top = std::max(top1, top2);
	bottom = std::min(bottom1, bottom2);
	S1 = (right1 - left1) * (bottom1 - top1);
	S2 = (right2 - left2) * (bottom2 - top2);
	SI = (right - left) * (bottom - top);
	SU = S1 + S2 - SI;
	if((left<right) && (top < bottom))	//Holds if there is an intersection
	{
		//printf("Intersection Cord: left %d right %d top %d bottom %d\n",left, right, top, bottom);
		//printf("Intersection Area: %d\n", SI);
		//printf("Intersection Ratio: %.2f\n",(float)SI/SU );
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
void Classifier::ModifyImage(image const& im, image &imPacked, std::vector<rRect> const& rects)
{
	int i,j,k,m,n;
	int src_x, src_y, dest_x, dest_y;
	int objectWidth, objectHeight;
	for(i = 0 ; i < rects.size() ; ++i)
	{
		objectWidth = rects[i].width;
		objectHeight = rects[i].height;
		src_x = rects[i].org_x;
		src_y = rects[i].org_y;
		dest_x = rects[i].packed_x;
		dest_y = rects[i].packed_y;

		for(j = 0 ; j < objectWidth ; ++j)	//The width Loop
		{
			for(k = 0 ; k < objectHeight ; ++k)
			{
				imPacked.data[((k+dest_y)*im.w) + (j+dest_x)] = im.data[((k+src_y)*im.w) + (j+src_x)];
				imPacked.data[(im.h*im.w) + ((k+dest_y)*im.w) + (j+dest_x)] = im.data[(im.h*im.w) + ((k+src_y)*im.w) + (j+src_x)];
				imPacked.data[(2*im.h*im.w) + ((k+dest_y)*im.w) + (j+dest_x)] = im.data[(2*im.h*im.w) + ((k+src_y)*im.w) + (j+src_x)];
			}
		}

	}
}

int Classifier::PackBins(std::vector<rRect> &rects)
{
	rbp::MaxRectsBinPack bin;
	int packed = 0;
	int binWidth = w;
	int binHeight;
	for(int j = 1 ; j < 7 ; ++j)
	{
		packed = 0;
		binHeight = j * 64;
		bin.Init(binWidth, binHeight);
		for(int i = 0; i < rects.size(); ++i)
		{
			// Read next rectangle to pack.
			int rectWidth = rects[i].width;
			int rectHeight = rects[i].height;
			//printf("Packing rectangle of size %dx%d: \n", rectWidth, rectHeight);

			// Perform the packing.
			rbp::MaxRectsBinPack::FreeRectChoiceHeuristic heuristic = rbp::MaxRectsBinPack::RectBestAreaFit; // This can be changed individually even for each rectangle packed.
			rbp::Rect packedRect = bin.Insert(rectWidth, rectHeight, heuristic);

			// Test success or failure.
			if (packedRect.height > 0)
			{
				packed++;
				rects[i].packed_x = packedRect.x;
				rects[i].packed_y = packedRect.y;
				//printf("Packed to (x,y)=(%d,%d), (w,h)=(%d,%d). Free space left: %.2f%%\n", packedRect.x, packedRect.y, packedRect.width, packedRect.height, 100.f - bin.Occupancy()*100.f);
			}
			else
			{	
				///printf("Failed! Could not find a proper position to pack this rectangle into. Skipping this one.\n");
			}
		}
		//printf("We Findiehs one iTeration %d\n", j);
		//printf("Packed = %d\n",binHeight );
		if(packed == rects.size()){
			//printf("This What we have retured %d\n", j);
			return j;
		}
	}
	//If we reached here, the packing has failed
	for(int i = 0; i < rects.size(); ++i)
	{
		rects[i].packed_x = rects[i].org_x;
		rects[i].packed_y = rects[i].org_y;
	}
	return 7;
}

void Classifier::AddRect(std::vector<rRect> &rects, int x, int y, int width, int height, std::string label)
{
	//Blindly Add a Rect to the ObjectRects Vector
	//int i = ScanForRect(x, y, width, height);
	//if(i < 0)	//No match found
	//{
		rects.push_back(rRect(x, y, width, height, label));
	//}
	//else
	//{	
	//	objectRects[i].org_x = x;
	//	objectRects[i].org_y = y;
	//	objectRects[i].width = width;
	//	objectRects[i].height = height;
	//	objectRects[i].label = label;
	//}

}

int Classifier::ScanForRect(int x, int y, int width, int height)
{
	//TODO: needs to be modified for the tracking algorithm
	int i;
	float IR;
	float thresh = 0.9;
	int left1, right1, top1, bottom1;
	int left2, right2, top2, bottom2;
	left1 = x;
	right1 = x + width;
	top1 = y;
	bottom1 = y + height;
	for(i = 0 ; i < objectRects.size() ; ++i)
	{
		left2 = objectRects[i].org_x;
		right2 = left2 + objectRects[i].width;
		top2 = objectRects[i].org_y;
		bottom2 = top2 + objectRects[i].height;
		IR = CalculateAOI(left1, right1, top1, bottom1, left2, right2, top2, bottom2);
		if(IR >= thresh)
		{
			return i;
		}
	}
	return -1;
}

void Classifier::PrintObjects()
{
	int i;
	for(i = 0 ; i < objectRects.size() ; ++i)
	{
		printf("Object %d: Label: %s\n", i, objectRects[i].label.c_str());
		printf("Org X: %d Org Y: %d ", objectRects[i].org_x, objectRects[i].org_y);
		printf("Width: %d Height: %d ", objectRects[i].width, objectRects[i].height);
		printf("Packed X: %d Packed Y: %d\n", objectRects[i].packed_x, objectRects[i].packed_y);
	}
}

void Classifier::PrintGenObjects()
{
	int i;
	for(i = 0 ; i < genObjectRects.size() ; ++i)
	{
		printf("Object %d: Label: %s\n", i, genObjectRects[i].label.c_str());
		printf("Org X: %d Org Y: %d ", genObjectRects[i].org_x, genObjectRects[i].org_y);
		printf("Width: %d Height: %d ", genObjectRects[i].width, genObjectRects[i].height);
		printf("Packed X: %d Packed Y: %d\n", genObjectRects[i].packed_x, genObjectRects[i].packed_y);
	}
}
void Classifier::TransformDimToCNN(std::vector<rRect> &rects)
{
	int i;
	for(i = 0 ; i < rects.size() ; ++i)
	{
		rects[i].org_x = rects[i].org_x * w / imW;
		rects[i].org_y = rects[i].org_y * h / imH;
		rects[i].width = rects[i].width * w / imW;
		rects[i].height = rects[i].height * h / imH;
		if(rects[i].org_x > w)	rects[i].org_x = w;
		if(rects[i].org_y > h)	rects[i].org_y = h;
		if(rects[i].width + rects[i].org_x > w)	rects[i].width = w - rects[i].org_x;
		if(rects[i].height + rects[i].org_y > h)	rects[i].width = h - rects[i].org_y;
	}
}

void Classifier::TransformDimFromCNN(std::vector<rRect> &rects)
{
	int i;
	for(i = 0 ; i < rects.size() ; ++i)
	{
		rects[i].org_x = rects[i].org_x * imW / w;
		rects[i].org_y = rects[i].org_y * imH / h;
		rects[i].width = rects[i].width * imW / w;
		rects[i].height = rects[i].height * imH / h;
		if(rects[i].org_x > imW)	rects[i].org_x = imW;
		if(rects[i].org_y > imH)	rects[i].org_y = imH;
		if(rects[i].width + rects[i].org_x > imW)	rects[i].width = imW - rects[i].org_x;
		if(rects[i].height + rects[i].org_y > imH)	rects[i].width = imH - rects[i].org_y;
	}
}

void Classifier::SetImageDim(int w, int h)
{
	imW = w;
	imH = h;
}

void Classifier::ProcessYOLOOutExtBoxes(int n, ext_box *ext_boxes, float thresh)
{	
	int i;
	for(i = 0 ; i < n ; ++i)
	{
		if(ext_boxes[i].prob >= thresh)
		{
			int x = ext_boxes[i].left;
			int y = ext_boxes[i].top;
			int width = ext_boxes[i].right - ext_boxes[i].left;
			int height = ext_boxes[i].bottom - ext_boxes[i].top;
			objectRects.push_back(rRect(x, y, width, height, ext_boxes[i].prob, rvoc_names[ext_boxes[i].voc_index]));
		}
	}
}
void Classifier::MergeOverLappingRects()
{
	int i,j;
	int left1, right1, top1, bottom1, left2, right2, top2, bottom2;
	int area1, area2;
	float IR;
	for(i = 0 ; i < objectRects.size() ; ++i)	//Merge Overlapping Rects
	{
		if(objectRects[i].merged == 1)
				continue;
		left1 = objectRects[i].org_x;
		right1 = left1 + objectRects[i].width;
		top1 = objectRects[i].org_y;
		bottom1 = top1 +  objectRects[i].height;
		area1 = objectRects[i].width * objectRects[i].height;
		for(j = i + 1 ; j < objectRects.size() ; ++j)
		{
			if(objectRects[j].merged == 1)
				continue;
			left2 = objectRects[j].org_x;
			right2 = left2 + objectRects[j].width;
			top2 = objectRects[j].org_y;
			bottom2 = top2 +  objectRects[j].height;
			IR = CalculateMergerRatio(left1, right1, top1, bottom1, left2, right2, top2, bottom2);
			//printf("%.2f\n", IR);
			if(IR >= 0.75)
			{
				//printf("there is an intersection i:%d j:%d %.2f\n",i ,j,IR);
				area2 = objectRects[j].width * objectRects[j].height;
				if(area1 >= area2)
				{
					objectRects[j].merged = 1;
				}
				else
				{
					objectRects[i].merged = 1;
					break;
				}
			}
		}
	}
	objectRects.erase(std::remove_if(objectRects.begin(), objectRects.end(), 
			[](const rRect & o) -> bool {
				if(o.merged)
					return true;
				else
					return false;}),
			objectRects.end());
}
void Classifier::ProcessYOLOOut1stRun(int n, ext_box *ext_boxes, float thresh)
{
	ProcessYOLOOutExtBoxes(n, ext_boxes, thresh);
	MergeOverLappingRects();
}

void Classifier::ProcessYOLOOut2ndRun(int n, ext_box *ext_boxes, bool clear)
{
	if(clear)
		objectRects.clear();
	int i, j, maxIRi;
	int left, right, top, bottom, x, y, width, height;
	int IR, preIR;
	int hShift, vShift;
	for(i = 0 ; i < n ; ++i)
	{
		preIR = 0;
		IR = 0;
		maxIRi = -1;
		if(ext_boxes[i].prob != -1.0)
		{
			for(j = 0 ; j < genObjectRects.size() ; ++j)
			{
				left = genObjectRects[j].packed_x;
				right = left + genObjectRects[j].width;
				top = genObjectRects[j].packed_y;
				bottom = top + genObjectRects[j].height;
				IR = CalculateIA(ext_boxes[i].left, ext_boxes[i].right, ext_boxes[i].top, ext_boxes[i].bottom, left, right, top, bottom);
				if(IR > preIR)
					maxIRi = j;
				preIR = IR;
			}
			if(maxIRi != -1)
			{
				hShift = genObjectRects[maxIRi].org_x - genObjectRects[maxIRi].packed_x;
				vShift = genObjectRects[maxIRi].org_y - genObjectRects[maxIRi].packed_y;
				width = ext_boxes[i].right - ext_boxes[i].left;
				height = ext_boxes[i].bottom - ext_boxes[i].top;
				x = hShift + ext_boxes[i].left;
				y = vShift + ext_boxes[i].top;
				objectRects.push_back(rRect(x, y, width, height, rvoc_names[ext_boxes[i].voc_index]));
				objectRects.back().labelID = ext_boxes[i].voc_index;
				objectRects.back().prob = ext_boxes[i].prob;

			}
		}
	}
	TransformDimFromCNN(objectRects);
}