import numpy as np
import glob
from xml.etree import ElementTree

#Initialize VOC operands
VOCopts = {}
VOCopts['dataset'] = 'VOC2012'
devkitroot = "/Volumes/External-Mac/VOC/VOCdevkit"
VOCopts['datadir'] = devkitroot + '/'
VOCopts['resdir'] = devkitroot + '/results/' + VOCopts['dataset'] + '/'
VOCopts['localdir'] = devkitroot + '/local/' + VOCopts['dataset'] + '/'
VOCopts['testset'] = 'test'
VOCopts['annopath'] = VOCopts['datadir'] + VOCopts['dataset'] + '/Annotations/%s.xml'
VOCopts['imgpath'] = VOCopts['datadir'] + VOCopts['dataset'] + '/JPEGImages/%s.jpg'
VOCopts['imgsetpath'] = VOCopts['datadir'] + VOCopts['dataset'] + '/ImageSets/Main/%s.txt'
VOCopts['detrespath'] = VOCopts['resdir'] + 'Main/comp4_det_' + VOCopts['testset'] + '_%s.txt'
VOCopts['classes'] = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",\
					 "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", \
					 "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
VOCopts['nclasses'] = len(VOCopts['classes'])
VOCopts['annocachepath'] = VOCopts['localdir'] + '%s_anno.mat'

cls = 'person'
npos=0;
cp = VOCopts['annocachepath'] % VOCopts['testset']	#Ground truth matrix
gt = {}
annofiles = glob.glob(VOCopts['annopath']%'*')

for file in annofiles:
	x = len(VOCopts['datadir'] + VOCopts['dataset'] + '/Annotations/')
	id = file[x:x+11]
	gt[id] = {'BB':[], 'diff':[], 'det':[]}

#Read the Ground truth
for key in gt:
	with open(VOCopts['annopath']%key, 'rt') as f:
		tree = ElementTree.parse(f)
	for node in tree.iter('object'):
		name = node.find('name').text
		if(name == cls):
			bndbox = node.find('bndbox')
			diff = node.find('difficult')
			if(diff is not None):
				if diff.text == '1':
					gt[key]['diff'].append(1)
				else:
					gt[key]['diff'].append(0)
					npos = npos + 1
			xmin = bndbox.find('xmin').text
			ymin = bndbox.find('ymin').text
			xmax = bndbox.find('xmax').text
			ymax = bndbox.find('ymax').text
			gt[key]['BB'].append([xmax, xmin, ymax, ymin])
			gt[key]['det'].append(0)

print gt['2011_003268']
#Read the Results
detres = {}
filename = VOCopts['detrespath'] % cls;
filer = open(filename, 'r')
for line in filer:
	splitline = line.split()