base_dir = '../build/results/vanilla/merged/'
base_keyword = 'comp4_det_test_timing.txt'
file = open(base_dir+base_keyword);
lines = file.readlines()
dic = {}
for line in lines:
	key = line[0:11];
	if dic.has_key(key):
		print 'found pre-existing key :('
	else:
		dic[key] = 0