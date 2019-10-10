#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif


#ifdef OPENCV

#include "http_stream.h"

#define MAXSTREAM 4

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes = 0;
static detection *dets = NULL;

static network net;
static image in_s[MAXSTREAM] ;
static image det_s[MAXSTREAM];

static cap_cv *cap[MAXSTREAM];
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;

// I am not sure if this has some functionality
//#define NFRAMES 1
//static float* predictions[NFRAMES];
//static int demo_index = 0;
//static image images[NFRAMES];
//static mat_cv* cv_images[NFRAMES];
//static float *avg;

mat_cv* in_img[MAXSTREAM];
mat_cv* det_img[MAXSTREAM];
mat_cv* show_img[MAXSTREAM];

static volatile int flag_exit;
static int letter_box = 0;

void *fetch_in_thread(void *ptr)
{
    int i = *((int*) ptr);
    free(ptr);
    int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
    if(letter_box)
        in_s[i] = get_image_from_stream_letterbox(cap[i], net.w, net.h, net.c, &in_img[i], dont_close_stream);
    else
        in_s[i] = get_image_from_stream_resize(cap[i], net.w, net.h, net.c, &in_img[i], dont_close_stream);
    if(!in_s[i].data){
        printf("Stream closed.\n");
        flag_exit = 1;
        //exit(EXIT_FAILURE);
        return 0;
    }

    return 0;
}

void *detect(int resourceID)
{
    int j = resourceID;
    //layer l = net.layers[net.n-1];
    float *X = det_s[j].data;
    //float *prediction = network_predict(net, X);
    network_predict(net, X);

    //memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    //mean_arrays(predictions, NFRAMES, l.outputs, avg);
    //l.output = avg;

    free_image(det_s[j]);

    //cv_images[demo_index] = det_img[j];
    //det_img[j] = cv_images[(demo_index + NFRAMES / 2 + 1) % NFRAMES];
    //demo_index = (demo_index + 1) % NFRAMES;

    if (letter_box)
        dets = get_network_boxes(&net, get_width_mat(in_img[j]), get_height_mat(in_img[j]), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
    else
        dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

    return 0;
}

double get_wall_time()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename,
          char **names, int classes, int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port,
          int dont_show, int ext_output, char *outfile, int letter_box_in)
{
    int resourceCount = 3;

    letter_box = letter_box_in;
    in_img[0] = det_img[0] = show_img[0] = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    char* filename0 = "rtsp://test:test1234@192.168.25.20:554/stream0/mobotix.mjpeg";
    char* filename1 = "/home/birgit/Downloads/source1.avi";
    char* filename2 = "/home/birgit/Downloads/source0.avi";
    char* filename3 = "/home/birgit/Downloads/source2.avi";
    char* filename4 = "/home/birgit/Downloads/source3.avi";

    for(int i=0; i<resourceCount; i++){
        if (i == 0) filename = filename0;
        if (i == 1) filename = filename1;
        if (i == 2) filename = filename2;
        if (i == 3) filename = filename3;
        if (i == 4) filename = filename4;

        if(filename){
            printf("video file: %s\n", filename);
            cap[i] = get_capture_video_stream(filename);
        }else{
            printf("Webcam index: %d\n", cam_index);
            cap[i] = get_capture_webcam(cam_index);
        }
        if (!cap[i]) {
            #ifdef WIN32
                printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
            #endif
            error("Couldn't connect to webcam.\n");
        }
    }

    layer l = net.layers[net.n-1];
    int j;

    //avg = (float *) calloc(l.outputs, sizeof(float));
    //for(j = 0; j < NFRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    //for(j = 0; j < NFRAMES; ++j) images[j] = make_image(1,1,3);

    if (l.classes != demo_classes) {
        printf("Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }

    flag_exit = 0;


    for(int i=0; i<resourceCount; i++){

        int* resourceID = calloc(1,sizeof(int));
        *resourceID = i;
        fetch_in_thread((void*)resourceID);
        det_img[i] = in_img[i];
        det_s[i] = in_s[i];
    }

    for(int i=0; i<resourceCount; i++){
        int* fetch_resource = calloc(1,sizeof(int));
        *fetch_resource = i;
        fetch_in_thread((void*)fetch_resource);
        detect(i);
        det_img[i] = in_img[i];
        det_s[i] = in_s[i];
    }

    for(int i=0; i<resourceCount; i++){
        //for (j = 0; j < NFRAMES / 2; ++j) {
            int* resource_fetch = calloc(1,sizeof(int));
            *resource_fetch = i;
            fetch_in_thread((void*)resource_fetch);
            detect(i);
            det_img[i] = in_img[i];
            det_s[i] = in_s[i];
        //}

        if(!prefix && !dont_show){
            int full_screen = 0;
            if (i == 0)
                create_window_cv(filename0, full_screen, 640, 480);
            if (i == 1)
                create_window_cv(filename1, full_screen, 640, 480);
            if (i == 2)
                create_window_cv(filename2, full_screen, 640, 480);
            if (i == 3)
                create_window_cv(filename3, full_screen, 640, 480);
            if (i == 4)
                create_window_cv(filename4, full_screen, 640, 480);
        }
    }

    int count = 0;
//    FILE* json_file = NULL;
//    if (outfile) {
//        int src_fps = get_stream_fps_cpp_cv(cap);
//        json_file = fopen(outfile, "wb");
//        int width = get_width_mat(det_img[0]);
//        int height = get_height_mat(det_img[0]);
//        char *tmp;
//        int now = time(NULL);
//        sprintf(tmp, "{\n \"Mediasource\":\"%s\",\n \"FPS\":\"%d\",\n \"Width\":%d,\n \"Height\":%d,\n \"Time\":\"%d-%02d-%02d %02d:%02d:%02d\",\n \"Frames\":[\n",
//                filename, src_fps, width, height, now/31556926+1970, (now%31556926)/2629743+1, (now%2629743)/86400+1, (now%86400)/3600+2, (now%3600)/60, now%60);
//        fwrite(tmp, sizeof(char),  strlen(tmp), json_file);
//    }


//    write_cv* output_video_writer = NULL;
//    if (out_filename && !flag_exit)
//    {
//        int src_fps = 25;
//        src_fps = get_stream_fps_cpp_cv(cap);
//        output_video_writer =
//            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img[0]), get_height_mat(det_img[0]), 1);
//
//        //'H', '2', '6', '4'
//        //'D', 'I', 'V', 'X'
//        //'M', 'J', 'P', 'G'
//        //'M', 'P', '4', 'V'
//        //'M', 'P', '4', '2'
//        //'X', 'V', 'I', 'D'
//        //'W', 'M', 'V', '2'
//    }

    pthread_t fetch_thread[MAXSTREAM];

    double before = get_wall_time();

    while(1){
        ++count;
        {

            // fetch treads
            for (int i = 0; i<resourceCount; i++){
                int* resource_fetch = calloc(1,sizeof(int));
                *resource_fetch = i;
                if(pthread_create(&fetch_thread[i], 0, fetch_in_thread, (void*)resource_fetch)) error("Thread creation failed");
            }
            // detect
            for(int i=0; i<resourceCount; i++){
                detect(i);

                //printf("\033[2J");
                //printf("\033[1;1H");
                //printf("\nFPS:%.1f\n", fps);
                printf("Objects:\n\n");

                ++frame_id;
//            if(outfile) {
//                if(frame_id >1) {
//                    fwrite(",\n", sizeof(char), strlen(",\n"), json_file);
//                }
//                char *json_buf = detection_to_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, NULL);
//                fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
//                free(json_buf);
//            }
//            if (demo_json_port > 0) {
//                int timeout = 400000;
//                send_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, demo_json_port, timeout);
//            }

                float nms = .45;    // 0.4F
                int local_nboxes = nboxes;
                detection *local_dets = dets;

                if (nms) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                draw_detections_cv_v3(show_img[i], local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
                free_detections(local_dets, local_nboxes);
                printf("\n-------------------\nSource %d: FPS %.1f\n", i, fps);

                if(!prefix){
                    if (!dont_show) {
                        if (i == 0)
                            show_image_mat(show_img[i], filename0);
                        if (i == 1)
                            show_image_mat(show_img[i], filename1);
                        if (i == 2)
                            show_image_mat(show_img[i], filename2);
                        if (i == 3)
                            show_image_mat(show_img[i], filename3);
                        if (i == 4)
                            show_image_mat(show_img[i], filename4);

                        int c = wait_key_cv(1);
                        if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                        {
                            flag_exit = 1;
                        }
                    }
                }
            }

//            else{
//                char buff[256];
//                sprintf(buff, "%s_%08d.jpg", prefix, count);
//                if(show_img[0]) save_cv_jpg(show_img[0], buff);
//            }

            // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
//            if (mjpeg_port > 0 && show_img[0]) {
//                int port = mjpeg_port;
//                int timeout = 400000;
//                int jpeg_quality = 40;    // 1 - 100
//                send_mjpeg(show_img[0], port, timeout, jpeg_quality);
//            }

            // save video file
//            if (output_video_writer && show_img[0]) {
//                write_frame_cv(output_video_writer, show_img[0]);
//                printf("\n cvWriteFrame \n");
//            }


            // next frames
            for(int i=0; i<resourceCount; i++){
                release_mat(&show_img[i]);
                pthread_join(fetch_thread[i], 0);
                show_img[i] = det_img[i];
                det_img[i] = in_img[i];
                det_s[i] = in_s[i];
            }

            if (flag_exit == 1) break;

        }

        // measure time and calcualte fps
        double after = get_time_point();    // more accurate time measurements
        float curr = 1000000. / (after - before);
        fps = curr;
        before = after;
    } //while(1)

    printf("input video stream closed. \n");
//    if (output_video_writer) {
//        release_video_writer(&output_video_writer);
//        printf("output_video_writer closed. \n");
//    }

//    if(outfile) {
//        fwrite("\n]\n}", sizeof(char), strlen("\n]\n}"), json_file);
//        fclose(json_file);
//    }

    // free memory
    for (int i=0; i<resourceCount; i++){
        release_mat(&show_img[i]);
        release_mat(&in_img[i]);
        free_image(in_s[i]);
        free(cap[i]);
    }

    //free(avg);
    //for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
    //for (j = 0; j < NFRAMES; ++j) free_image(images[j]);

    free_ptrs((void **)names, net.layers[net.n - 1].classes);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);

    //cudaProfilerStop();
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
