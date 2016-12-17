import numpy as np
import glob
from xml.etree import ElementTree
import scipy.io as sio

annonpath = "/Volumes/External-Mac/VOC/TrainVal/VOCdevkit/VOC2012/Annotations/"

annofiles = glob.glob(annonpath+"*")

ratios = {"aeroplane":[], "bicycle":[], "bird":[], "boat":[], "bottle":[], "bus":[], "car":[], "cat":[],\
		"chair":[], "cow":[], "diningtable":[], "dog":[], "horse":[], "motorbike":[], "person":[], \
		"pottedplant":[], "sheep":[], "sofa":[], "train":[], "tvmonitor":[]}
for file in annofiles:
	with open(file, 'rt') as f:
		tree = ElementTree.parse(f)
	imArea = 0
	imWidth = 0
	imHeight = 0
	for node in tree.iter('size'):
		imWidth = int(node.find('width').text)
		imHeight = int(node.find('height').text)
		imArea = imWidth*imHeight
	for node in tree.iter('object'):
		name = node.find('name').text;
		bndbox = node.find('bndbox')
		xmin = float(bndbox.find('xmin').text)
		ymin = float(bndbox.find('ymin').text)
		xmax = float(bndbox.find('xmax').text)
		ymax = float(bndbox.find('ymax').text)
		area = (xmax - xmin) * (ymax - ymin)
		ratio = area/imArea;
		ratios[name].append(ratio)

sio.savemat("areaRatios",ratios)
#for key in ratios.keys():
#	avg = np.mean(ratios[key])
#	print "%s: %.6f" % (key,avg)