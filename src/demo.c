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

#define MAXCAMERAS 10

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes[MAXCAMERAS] = {0};
static detection* dets[MAXCAMERAS]; //= NULL;

static network net;
static image in_s[MAXCAMERAS];
static image det_s[MAXCAMERAS];

static cap_cv *cap[MAXCAMERAS];
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id[MAXCAMERAS] = {0};
static int demo_json_port = -1;

#define NFRAMES 3

static float* predictions[NFRAMES];
static int demo_index = 0;
static image images[NFRAMES];
static mat_cv* cv_images[NFRAMES];
static float *avg;

mat_cv* in_img[MAXCAMERAS];
mat_cv* det_img[MAXCAMERAS];
mat_cv* show_img[MAXCAMERAS];

static volatile int flag_exit;
static int letter_box = 0;

struct arg_struct{
    int resource;
};

void *fetch_in_thread(void *ptr)
{
    struct arg_struct *args = (struct arg_struct*) ptr;
    int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
    if(letter_box)
        in_s[args->resource] = get_image_from_stream_letterbox(cap[args->resource], net.w, net.h, net.c, &in_img[args->resource], dont_close_stream);
    else
        in_s[args->resource] = get_image_from_stream_resize(cap[args->resource], net.w, net.h, net.c, &in_img[args->resource], dont_close_stream);
    if(!in_s[args->resource].data){
        printf("Stream closed.\n");
        flag_exit = 1;
        //exit(EXIT_FAILURE);
        return 0;
    }
    //in_s = resize_image(in, net.w, net.h);

    return 0;
}

void *detect_in_thread(void *ptr)
{
    struct arg_struct *args = (struct arg_struct*) ptr;
    layer l = net.layers[net.n-1];
    float *X = det_s[args->resource].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, NFRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s[args->resource]);

    cv_images[demo_index] = det_img[args->resource];
    det_img[args->resource] = cv_images[(demo_index + NFRAMES / 2 + 1) % NFRAMES];
    demo_index = (demo_index + 1) % NFRAMES;

    if (letter_box)
        dets[args->resource] = get_network_boxes(&net, get_width_mat(in_img[args->resource]), get_height_mat(in_img[args->resource]), demo_thresh, demo_thresh, 0, 1, &nboxes[args->resource], 1); // letter box
    else
        dets[args->resource] = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes[args->resource], 0); // resized

    return 0;
}

void *process_results_in_thread(void *ptr){

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
    letter_box = letter_box_in;

    int resource_count = 2;
    char* files[resource_count];
    files[0] = "/home/birgit/Downloads/source0.avi";
    files[1] = "/home/birgit/Downloads/source1.avi";
    for(int i = 0; i < resource_count; i++){
        in_img[i] = det_img[i] = show_img[i] = NULL;

        if(filename){
            printf("video file: %s\n", filename);
            cap[i] = get_capture_video_stream(files[i]);
        }else{
            printf("Webcam index: %d\n", cam_index);
            cap[i] = get_capture_webcam(cam_index);
        }

        if (!cap[i]) ;{
            #ifdef WIN32
            printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
            #endif
            error("Couldn't connect to webcam or open video file.\n");
        }
        if(!prefix && !dont_show){
            int full_screen = 0;
            char windowName[256];
            sprintf(windowName,"Source %d", i);
            create_window_cv(windowName, full_screen, 640, 360);
        }
    }

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


    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < NFRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < NFRAMES; ++j) images[j] = make_image(1,1,3);

    if (l.classes != demo_classes) {
        printf("Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }

    flag_exit = 0;

    for(int i =0; i < resource_count; i++){

        struct arg_struct args;
        args.resource = i;

        fetch_in_thread((void*)&args);
        det_img[i] = in_img[i];
        det_s[i] = in_s[i];

        fetch_in_thread((void*)&args);
        detect_in_thread((void*)&args);
        det_img[i] = in_img[i];
        det_s[i] = in_s[i];

        for (j = 0; j < NFRAMES / 2; ++j) {
            fetch_in_thread((void*)&args);
            detect_in_thread((void*)&args);
            det_img[i] = in_img[i];
            det_s[i] = in_s[i];
        }
    }
    pthread_t fetch_thread[resource_count];
    pthread_t detect_thread[resource_count];


    int count = 0;
//    FILE* json_file = NULL;
//    if (outfile) {
//        int src_fps = get_stream_fps_cpp_cv(cap);
//        json_file = fopen(outfile, "wb");
//        int width = get_width_mat(det_img);
//        int height = get_height_mat(det_img);
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
//            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);
//
//        //'H', '2', '6', '4'
//        //'D', 'I', 'V', 'X'
//        //'M', 'J', 'P', 'G'
//        //'M', 'P', '4', 'V'
//        //'M', 'P', '4', '2'
//        //'X', 'V', 'I', 'D'
//        //'W', 'M', 'V', '2'
//    }

    double before = get_wall_time();

    while(1){
        ++count;
        {
            // start all treads, for each video one fetch and one detect thread
            for (int i = 0; i < resource_count; i++){
                struct arg_struct args;
                args.resource = i;
                if(pthread_create(&fetch_thread[i], 0, fetch_in_thread, (void*)&args)) error("Thread creation failed");
//                if(pthread_create(&detect_thread[i], 0, detect_in_thread, (void*)&args)) error("Thread creation failed");
            }

// --------------------------------------------------------------------------------------------------------------------
            // draw results
            for (int i = 0; i < resource_count; i++){

//                float nms = .45;    // 0.4F
//                int local_nboxes = nboxes[i];
//                detection *local_dets = dets[i];
//                if (nms) do_nms_sort(local_dets, local_nboxes, l.classes, nms);

//            printf("Objects:\n\n");

                ++frame_id[i];
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

//                draw_detections_cv_v3(show_img[i], local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
//                free_detections(local_dets, local_nboxes);

//            printf("\nFPS:%.1f\n", fps);

//            if(!prefix){
                if (!dont_show) {
                    char windowName[256];
                    sprintf(windowName,"Source %d", i);
                    show_image_mat(show_img[i], windowName);
                    int c = wait_key_cv(1);
                    // if enter is pressed toggle between different frame skips
//                    if (c == 10) {
//                        if (frame_skip == 0) frame_skip = 60;
//                        else if (frame_skip == 4) frame_skip = 0;
//                        else if (frame_skip == 60) frame_skip = 4;
//                        else frame_skip = 0;
//                    } else
                    if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }
                }
//            }
//            else{
//                char buff[256];
//                sprintf(buff, "%s_%08d.jpg", prefix, count);
//                if(show_img) save_cv_jpg(show_img, buff);
//            }

                // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
                //           if (mjpeg_port > 0 && show_img) {
                //               int port = mjpeg_port;
                //               int timeout = 400000;
                //               int jpeg_quality = 40;    // 1 - 100
                //               send_mjpeg(show_img, port, timeout, jpeg_quality);
                //           }

                // save video file
//            if (output_video_writer && show_img) {
//                write_frame_cv(output_video_writer, show_img);
//                printf("\n cvWriteFrame \n");
//            }

                release_mat(&show_img[i]);
            }
// --------------------------------------------------------------------------------------------------------------------


            // wait for all threads to finish their job
            for (int i = 0; i < resource_count; i++){
                pthread_join(fetch_thread[i], 0);
//                pthread_join(detect_thread[i], 0);
            }

            if (flag_exit == 1) break;

            // if frameskip is > 0 show image only after delay = frameskip
            for(int i =0; i < resource_count; i++){

                if(delay == 0){
                    show_img[i] = det_img[i];
                }
                det_img[i] = in_img[i];
                det_s[i] = in_s[i];
            }
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            // measure time and calcualte fps
            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            float curr = 1000000. / (after - before);
            fps = curr;
            before = after;
        }
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
    for(int i = 0; i < resource_count; i++){
        release_mat(&show_img[i]);
        release_mat(&in_img[i]);
        free_image(in_s[i]);
    }

    free(avg);
    for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
    for (j = 0; j < NFRAMES; ++j) free_image(images[j]);

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
