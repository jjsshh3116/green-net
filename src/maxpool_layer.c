#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>
#include <arm_neon.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    int32x4_t l_h_vec = vdupq_n_s32(l.h); 
    int32x4_t l_w_vec = vdupq_n_s32(l.w);
    int32x4_t zero = vdupq_n_s32(0);

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;

                    int32_t cur_h_data[4] = {h_offset + i*l.stride + 0, h_offset + i*l.stride + 0, h_offset + i*l.stride + 1, h_offset + i*l.stride + 1};
                    int32_t cur_w_data[4] = {w_offset + j*l.stride + 0, w_offset + j*l.stride + 1, w_offset + j*l.stride + 0, w_offset + j*l.stride + 1};
                    int32_t index[4] = {cur_w_data[0] + l.w*(cur_h_data[0] + l.h*(k + b*l.c)), cur_w_data[1] + l.w*(cur_h_data[1] + l.h*(k + b*l.c)), cur_w_data[2] + l.w*(cur_h_data[2] + l.h*(k + b*l.c)), cur_w_data[3] + l.w*(cur_h_data[3] + l.h*(k + b*l.c))};

                    int32x4_t cur_h = vld1q_s32(cur_h_data);
                    int32x4_t cur_w = vld1q_s32(cur_w_data);

                    // cur_h >= 0 and cur_h < l.h
                    uint32x4_t ge_h = vcgeq_s32(cur_h, zero); // cur_h >= 0
                    uint32x4_t lt_h = vcltq_s32(cur_h, l_h_vec); // cur_h < l_h

                    // cur_w >= 0 and cur_w < l.w
                    uint32x4_t ge_w = vcgeq_s32(cur_w, zero); // cur_w >= 0
                    uint32x4_t lt_w = vcltq_s32(cur_w, l_w_vec); // cur_w < l_w

                    // Combine the results using AND operations
                    uint32x4_t valid_h = vandq_u32(ge_h, lt_h); // (cur_h >= 0 && cur_h < l.h)
                    uint32x4_t valid_w = vandq_u32(ge_w, lt_w); // (cur_w >= 0 && cur_w < l.w)

                    uint32x4_t valid = vandq_u32(valid_h, valid_w); // final validity: (cur_h valid) && (cur_w valid)

                    // float val = (valid != 0) ? net.input[index] : -FLT_MAX; -> neon code 
                    float32x4_t negative_max = vdupq_n_f32(-FLT_MAX);
                    float32x4_t values = {net.input[index[0]], net.input[index[1]], net.input[index[2]], net.input[index[3]]};
                    float32x4_t output = vbslq_f32(valid, values, negative_max);

                    float32x2_t max_half = vmax_f32(vget_low_f32(output), vget_high_f32(output));

                    l.output[out_index] = vget_lane_f32(vpmax_f32(max_half, max_half), 0);

                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

