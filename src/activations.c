#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{

    if(a == RELU){
        float32x4_t vzero = vmovq_n_f32(0.0f);

        int i;

        for (i = 0; i <= n - 32; i += 32) {
            float32x4_t A0 = vld1q_f32(x + i);
            float32x4_t A1 = vld1q_f32(x + i + 4);
            float32x4_t A2 = vld1q_f32(x + i + 8);
            float32x4_t A3 = vld1q_f32(x + i + 12);
            float32x4_t A4 = vld1q_f32(x + i + 16);
            float32x4_t A5 = vld1q_f32(x + i + 20);
            float32x4_t A6 = vld1q_f32(x + i + 24);
            float32x4_t A7 = vld1q_f32(x + i + 28);

            A0 = vmaxq_f32(A0, vzero);
            A1 = vmaxq_f32(A1, vzero);
            A2 = vmaxq_f32(A2, vzero);
            A3 = vmaxq_f32(A3, vzero);
            A4 = vmaxq_f32(A4, vzero);
            A5 = vmaxq_f32(A5, vzero);
            A6 = vmaxq_f32(A6, vzero);
            A7 = vmaxq_f32(A7, vzero);

            vst1q_f32(x + i, A0);
            vst1q_f32(x + i + 4, A1);
            vst1q_f32(x + i + 8, A2);
            vst1q_f32(x + i + 12, A3);
            vst1q_f32(x + i + 16, A4);
            vst1q_f32(x + i + 20, A5);
            vst1q_f32(x + i + 24, A6);
            vst1q_f32(x + i + 28, A7);
        }

        for (; i < n; ++i) {
            x[i] = x[i] > 0 ? x[i] : 0;
        }

        // for (i = 0; i <= n - 16; i += 16) {
        //     float32x4_t A0 = vld1q_f32(x + i);
        //     float32x4_t A1 = vld1q_f32(x + i + 4);
        //     float32x4_t A2 = vld1q_f32(x + i + 8);
        //     float32x4_t A3 = vld1q_f32(x + i + 12);
            

        //     A0 = vmaxq_f32(A0, vzero);
        //     A1 = vmaxq_f32(A1, vzero);
        //     A2 = vmaxq_f32(A2, vzero);
        //     A3 = vmaxq_f32(A3, vzero);

        //     vst1q_f32(x + i, A0);
        //     vst1q_f32(x + i + 4, A1);
        //     vst1q_f32(x + i + 8, A2);
        //     vst1q_f32(x + i + 12, A3);
        // }
        // for (; i < n; ++i) {
        //     x[i] = x[i] > 0 ? x[i] : 0;
        // }

    }
    else if(a == LEAKY){
        int i;
        float alpha = 0.1;
        float32x4_t vzero = vmovq_n_f32(0.0f);
        float32x4_t valpha = vmovq_n_f32(alpha);

        for (i = 0; i < n; i += 16) {
            float32x4_t A0 = vld1q_f32(x + i);
            float32x4_t A1 = vld1q_f32(x + i + 4);
            float32x4_t A2 = vld1q_f32(x + i + 8);
            float32x4_t A3 = vld1q_f32(x + i + 12);
           

            uint32x4_t mask0 = vcltq_f32(A0, vzero);
            uint32x4_t mask1 = vcltq_f32(A1, vzero);
            uint32x4_t mask2 = vcltq_f32(A2, vzero);
            uint32x4_t mask3 = vcltq_f32(A3, vzero);

            A0 = vbslq_f32(mask0, vmulq_f32(A0, valpha), A0);
            A1 = vbslq_f32(mask1, vmulq_f32(A1, valpha), A1);
            A2 = vbslq_f32(mask2, vmulq_f32(A2, valpha), A2);
            A3 = vbslq_f32(mask3, vmulq_f32(A3, valpha), A3);

            vst1q_f32(x + i, A0);
            vst1q_f32(x + i + 4, A1);
            vst1q_f32(x + i + 8, A2);
            vst1q_f32(x + i + 12, A3);
        }

        for (; i < n; ++i) {
            x[i] = (x[i] < 0) ? x[i] * alpha : x[i];
        }

    }
    else{
        int i;
        for(i = 0; i < n; ++i){
            x[i] = activate(x[i], a);
        }
    }

    
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

