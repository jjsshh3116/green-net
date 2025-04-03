#include "avgpool_layer.h"
#include "cuda.h"
#include <arm_neon.h>
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    // for(b = 0; b < l.batch; ++b){
    //     for(k = 0; k < l.c; ++k){
    //         int out_index = k + b*l.c;
    //         l.output[out_index] = 0;
    //         for(i = 0; i < l.h*l.w; ++i){
    //             int in_index = i + l.h*l.w*(k + b*l.c);
    //             l.output[out_index] += net.input[in_index];
    //         }
    //         l.output[out_index] /= l.h*l.w;
    //     }
    // }

    for (b = 0; b < l.batch; ++b) {
        for (k = 0; k < l.c; ++k) {
            int out_index = k + b * l.c;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (i = 0; i < l.h * l.w - 3; i += 4) { // 4개씩 처리
                int in_index = i + l.h * l.w * (k + b * l.c);
                float32x4_t input_vec = vld1q_f32(&net.input[in_index]); // 4개의 float 로드
                sum_vec = vaddq_f32(sum_vec, input_vec); // 네 요소의 합산
            }
             // 벡터의 요소들을 합산하여 하나의 값으로 만듦
            float sum = vaddvq_f32(sum_vec);
            
            // 남은 요소들 처리
            for (; i < l.h * l.w; ++i) {
                int in_index = i + l.h * l.w * (k + b * l.c);
                sum += net.input[in_index];
            }
            
            l.output[out_index] = sum / (l.h * l.w);
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

