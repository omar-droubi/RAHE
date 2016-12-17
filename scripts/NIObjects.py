import numpy as np
import glob
from xml.etree import ElementTree
import scipy.io as sio

annonpath = "/Volumes/External-Mac/VOC/TrainVal/VOCdevkit/VOC2012/Annotations/"

annofiles = glob.glob(annonpath+"*")

occur = {"aeroplane":[], "bicycle":[], "bird":[], "boat":[], "bottle":[], "bus":[], "car":[], "cat":[],\
		"chair":[], "cow":[], "diningtable":[], "dog":[], "horse":[], "motorbike":[], "person":[], \
		"pottedplant":[], "sheep":[], "sofa":[], "train":[], "tvmonitor":[]}

for file in annofiles:
	with open(file, 'rt') as f:
		tree = ElementTree.parse(f)

	for node1 in tree.iter('object'):
		name = node1.find('name').text
		bndbox1 = node1.find('bndbox')
		xmin1 = float(bndbox1.find('xmin').text)
		ymin1 = float(bndbox1.find('ymin').text)
		xmax1 = float(bndbox1.find('xmax').text)
		ymax1 = float(bndbox1.find('ymax').text)
		occur[name].append(0)
		for node2 in tree.iter('object'):
			bndbox2 = node2.find('bndbox')
			xmin2 = float(bndbox2.find('xmin').text)
			ymin2 = float(bndbox2.find('ymin').text)
			xmax2 = float(bndbox2.find('xmax').text)
			ymax2 = float(bndbox2.find('ymax').text)
			if( xmin1 == xmin2) and (xmax1 == xmax2) and (ymin1 == ymin2) and (ymax1 == ymax2):
				continue
			xmin = max(xmin1, xmin2)
			xmax = min(xmax1, xmax2)
			ymin = max(ymin1, ymin2)
			ymax = min(ymax1, ymax2)
			if (xmin < xmax) and (ymin < ymax):
				occur[name][-1] = occur[name][-1] + 1

sio.savemat("NIObjects",occur)
