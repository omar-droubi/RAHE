import glob
import os
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',\
			'chair','cow','diningtable','dog','horse','motorbike','person',\
			'pottedplant','sheep','sofa','train','tvmonitor','timing'];

start = 0
end = 5823
step = 2000
base_dir = '../build/results/stripped/'
base_keyword = 'comp4_det_test_stripped_'
base_keyword2 = 'comp4_det_val_'

offsets = range(start,end,step);
if (offsets<end):
	offsets.append(end)
if not os.path.isdir(base_dir+'merged'):
	os.mkdir(base_dir+'merged', 0777)
for cls in classes:
	filew = open(base_dir+'merged/'+base_keyword2+cls+'.txt','w')
	for offset in offsets:
		search_phrase = base_dir+base_keyword+str(offset)+'-'+'*'+cls+'.txt';
		tmpfilename = glob.glob(search_phrase);
		if len(tmpfilename)!=0:
			filer = open(tmpfilename[0]);
			filew.write(filer.read());
		else:
			print offset
	filew.close()




