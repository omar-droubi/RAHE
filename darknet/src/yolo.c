#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image voc_labels[20];

void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_yolo_detections_stripped(ext_box* ext_boxes, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    int k =0;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j])
            {
            	ext_boxes[k].left = xmin;
            	ext_boxes[k].right = xmax;
            	ext_boxes[k].top = ymin;
            	ext_boxes[k].bottom = ymax;
            	ext_boxes[k].prob = probs[i][j];
            	ext_boxes[k].voc_index = j;
            }
            else{
            	ext_boxes[k].prob = -1.0;
            }
            k++;
        }
    }
}


float validate_yolo_external_stripped(network net,image sized, char *id, char*base, ext_box *ext_boxes, int m, int offset, int nW)
{
    set_batch_network(&net, 1);
    srand(time(0));

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;
    //time_t start = time(0);
    time_t start, end;
	float *X = sized.data;
	start = clock();
	float *predictions = network_predict_stripped(net, X, nW);
	end = clock();
	int w = sized.w;
	int h = sized.h;
	fprintf(stderr, "Image %d Total Detection Time: %f Seconds\n",m, sec(end-start));
	convert_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
	print_yolo_detections_stripped(ext_boxes, id, boxes, probs, side*side*l.n, classes, w, h);
	free_image(sized);
	return sec(end-start);
}

void validate_yolo_external(char *cfgfile, char *weightfile, int offset, int range)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2012.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    //int m = plist->size;
    if(offset > plist->size)    return;
    int m;
    if((offset + range) >= plist->size)
        m = plist->size;
    else
        m = offset + range;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%d-%d_%s.txt", base, offset, m, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    char buff[1024];
    snprintf(buff, 1024, "%s%d-%d_%s.txt", base, offset, m, "timing");
    FILE *time_file = fopen(buff, "w");
    //time_t start = time(0);
    time_t start;
    for(i = offset; i < m; ++i){
            char *path = paths[i];
            char *id = basecfg(path);
            image im = load_image_color(path,0,0);
            image sized = resize_image(im, net.w, net.h);
            float *X = sized.data;
            start = clock();
            float *predictions = network_predict(net, X);
            fprintf(stderr, "Image %d from %d Total Detection Time: %f Seconds\n",i,m, sec(clock() - start));
            fprintf(time_file, "%s %f\n", id,sec(clock() - start));
            int w = im.w;
            int h = im.h;
            convert_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(im);
            free_image(sized);
    }

}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2012.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    //int m = plist->size;
    int m = 1;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 1;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    //time_t start = time(0);
    time_t start = clock();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", sec(clock() - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

int get_yolo_output_size(network net)
{
	detection_layer l = net.layers[net.n-1];
	return (l.side*l.side*l.n);
}
void write_ext_boxes(image im, int num, float thresh, box *boxes, float **probs, int classes, ext_box *ext_boxes)
{
	int i;

	    for(i = 0; i < num; ++i){
	        int class = max_index(probs[i], classes);
	        float prob = probs[i][class];
	        if(prob > thresh){
	            int width = pow(prob, 1./2.)*10+1;
	            width = 8;
	            box b = boxes[i];

	            int left  = (b.x-b.w/2.)*im.w;
	            int right = (b.x+b.w/2.)*im.w;
	            int top   = (b.y-b.h/2.)*im.h;
	            int bot   = (b.y+b.h/2.)*im.h;

	            if(left < 0) left = 0;
	            if(right > im.w-1) right = im.w-1;
	            if(top < 0) top = 0;
	            if(bot > im.h-1) bot = im.h-1;
	            ext_boxes[i].left = left;
	            ext_boxes[i].right = right;
	            ext_boxes[i].top = top;
	            ext_boxes[i].bottom = bot;
	            ext_boxes[i].prob = prob;
	            ext_boxes[i].voc_index = class;
	        }
	        else
	        {
	        	ext_boxes[i].prob = -1.0;
	        }
	    }
}

void run_yolo_external_stripped_mp(network net,image im, ext_box *ext_boxes,int val, int nW, int nThreads)
{
    float thresh = 0.2;
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
	time=clock();
	float *predictions = network_predict_stripped_mp(net, X, nW, nThreads);
	//printf("Predicted in %f seconds.\n", sec(clock()-time));
	convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	if(val==0){
		draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		save_image(im, "predictions");
		show_image(im, "predictions");
		show_image(sized, "resized");
	}
	else{
		write_ext_boxes(im, l.side*l.side*l.n, thresh, boxes, probs, 20, ext_boxes);
	}
	free_image(im);
	free_image(sized);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif

}

void run_yolo_external_stripped(network net, image im, ext_box *ext_boxes,int val, int nW)
{
    float thresh = 0.2;
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
	time=clock();
	float *predictions = network_predict_stripped(net, X, nW);
	printf("Predicted in %f seconds.\n", sec(clock()-time));
	convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	if(val==0){
		draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		save_image(im, "predictions");
		show_image(im, "predictions");
		show_image(sized, "resized");
	}
	else{
		write_ext_boxes(im, l.side*l.side*l.n, thresh, boxes, probs, 20, ext_boxes);
	}
	free_image(im);
	free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
}

void run_yolo_external_mp(network net,image im, ext_box *ext_boxes,int val, int nThreads)
{
    float thresh = 0.2;
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
	time=clock();
	float *predictions = network_predict_mp(net, X, nThreads);
	//printf("Predicted in %f seconds.\n", sec(clock()-time));
	convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	if(val==0){
		draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		save_image(im, "predictions");
		show_image(im, "predictions");
		show_image(sized, "resized");
	}
	else{
		write_ext_boxes(im, l.side*l.side*l.n, thresh, boxes, probs, 20, ext_boxes);
	}
	free_image(im);
	free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
}

void run_yolo_external(network net,image im, ext_box *ext_boxes,int val)
{
    float thresh = 0.2;
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
	time=clock();
	float *predictions = network_predict(net, X);
	printf("Predicted in %f seconds.\n", sec(clock()-time));
	convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	if(val==0){
		draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		save_image(im, "predictions");
		show_image(im, "predictions");
		show_image(sized, "resized");
	}
	else{
		write_ext_boxes(im, l.side*l.side*l.n, thresh, boxes, probs, 20, ext_boxes);
	}
	free_image(im);
	free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif

}
void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *pre_boxes = calloc(2,sizeof(box));
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        save_image(im, "predictions");
        show_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void test_yolo_seq(char *cfgfile, char *weightfile, char *filename, float thresh, int start_i, int end_i, char *output_file)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    j=start_i;
    for(j=start_i; j < end_i ; j++){
        char ibuff[256];
        char *input = ibuff;
        char jobuff[256];
        char *jpeg_output = jobuff;
        char cobuff[256];
        char *csv_output = cobuff;
        char t_buff[13];

        strncpy(input, filename, 256);
        strncpy(jpeg_output, output_file, 256);
        strncpy(csv_output, output_file, 256);

        sprintf(t_buff, "%06d.JPEG",j);
        strcat(input,t_buff);
        strcat(jpeg_output,t_buff);
        sprintf(t_buff, "%06d.csv",j);
        strcat(csv_output,t_buff);

        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        write_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20, csv_output);
        show_image(im, jpeg_output);
        save_image(im, jpeg_output);
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
    }
}

void run_yolo(int argc, char **argv)
{
    int i;
    for(i = 0; i < 20; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    int start_i = (argc > 6) ? atof(argv[6]) : 0;
    int end_i = (argc > 7) ? atof(argv[7]) : 0;
    char *output_file = (argc > 8) ? argv[8] : 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    //else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 20, frame_skip);
    else if(0==strcmp(argv[2], "seq")) test_yolo_seq(cfg, weights, filename, thresh,start_i,end_i, output_file);
}
