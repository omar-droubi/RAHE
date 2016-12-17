#ifndef YOLO_H
#define YOLO_H
#include "box.h"
#include "network.h"

void train_yolo(char *cfgfile, char *weightfile);
void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
//void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h);
//void validate_yolo(char *cfgfile, char *weightfile);
void validate_yolo_recall(char *cfgfile, char *weightfile);
void validate_yolo(char *cfgfile, char *weightfile);
float validate_yolo_external_stripped(network net,image sized, char *id, char*base, ext_box *ext_boxes, int m, int offset, int nW);
void validate_yolo_external(char *cfgfile, char *weightfile, int offset, int range);
void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh);
void test_yolo_seq(char *cfgfile, char *weightfile, char *filename, float thresh, int start_i, int end_i, char *output_file);
void run_yolo(int argc, char **argv);
void run_yolo_external(network net,image im, ext_box *ext_boxes, int val);
void run_yolo_external_mp(network net,image im, ext_box *ext_boxes,int val, int nThreads);
void run_yolo_external_stripped(network net, image im, ext_box *ext_boxes,int val, int nW);
void run_yolo_external_stripped_mp(network net,image im, ext_box *ext_boxes,int val, int nW, int nThreads);
void write_ext_boxes(image im, int num, float thresh, box *boxes, float **probs, int classes, ext_box *ext_boxes);
int get_yolo_output_size(network net);
#endif
