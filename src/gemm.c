#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <arm_neon.h>
#include <time.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    clock_t time;

    // printf("==== register 3 ====\n");
    // time = clock();

    // for(i = 0; i < M; ++i){ //레지스터 3개
    //     for(k = 0; k < K; ++k){
    //         float32x4_t a_part = vdupq_n_f32(ALPHA*A[i*lda+k]);
    //         for(j = 0; j <= N-4; j+=4){
    //            float32x4_t b = vld1q_f32(B + k * ldb + j);
    //            float32x4_t c = vld1q_f32(C + i * ldc + j);
    //            c = vmlaq_f32(c, a_part, b);
    //            vst1q_f32(C + i * ldc + j, c);
    //         }
    //         for(; j < N; j++){
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));


    // printf("==== register 9 ====\n");
    // time = clock();
    // for(i = 0; i < M; ++i) { // 레지스터 9개
    //     for(k = 0; k < K; ++k) {
    //         float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
    //         // 레지스터를 한 번에 더 많이 사용하기 위해 루프 언롤링
    //         for(j = 0; j <= N - 16; j += 16) {
    //             // B 벡터를 4개 로드
    //             float32x4_t b0 = vld1q_f32(B + k * ldb + j);
    //             float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
    //             float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
    //             float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);

    //             // C 벡터도 4개 로드
    //             float32x4_t c0 = vld1q_f32(C + i * ldc + j);
    //             float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
    //             float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
    //             float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);

    //             // 병렬로 곱셈 및 누적 연산
    //             c0 = vmlaq_f32(c0, a_part, b0);
    //             c1 = vmlaq_f32(c1, a_part, b1);
    //             c2 = vmlaq_f32(c2, a_part, b2);
    //             c3 = vmlaq_f32(c3, a_part, b3);

    //             // 결과를 다시 C로 저장
    //             vst1q_f32(C + i * ldc + j, c0);
    //             vst1q_f32(C + i * ldc + j + 4, c1);
    //             vst1q_f32(C + i * ldc + j + 8, c2);
    //             vst1q_f32(C + i * ldc + j + 12, c3);
    //         }

    //         // 남은 요소 처리
    //         for(; j < N; j++) {
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

    // printf("==== register 11 ====\n");
    // time = clock();
    for(i = 0; i < M; ++i) { // 레지스터 11개
        for(k = 0; k < K; ++k) {
            float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
            // 레지스터를 한 번에 더 많이 사용하기 위해 루프 언롤링
            for(j = 0; j <= N - 20; j += 20) {
                // B 벡터를 5개 로드
                float32x4_t b0 = vld1q_f32(B + k * ldb + j);
                float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
                float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
                float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);
                float32x4_t b4 = vld1q_f32(B + k * ldb + j + 16);

                // C 벡터도 5개 로드
                float32x4_t c0 = vld1q_f32(C + i * ldc + j);
                float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
                float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
                float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);
                float32x4_t c4 = vld1q_f32(C + i * ldc + j + 16);

                // 병렬로 곱셈 및 누적 연산
                c0 = vmlaq_f32(c0, a_part, b0);
                c1 = vmlaq_f32(c1, a_part, b1);
                c2 = vmlaq_f32(c2, a_part, b2);
                c3 = vmlaq_f32(c3, a_part, b3);
                c4 = vmlaq_f32(c4, a_part, b4);

                // 결과를 다시 C로 저장
                vst1q_f32(C + i * ldc + j, c0);
                vst1q_f32(C + i * ldc + j + 4, c1);
                vst1q_f32(C + i * ldc + j + 8, c2);
                vst1q_f32(C + i * ldc + j + 12, c3);
                vst1q_f32(C + i * ldc + j + 16, c4);
            }

            // 남은 요소 처리
            for(; j < N; j++) {
                C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));
    

    // printf("==== register 13 ====\n");
    // time = clock();
    //  for(i = 0; i < M; ++i) { // 레지스터 13개 -> 여기서부터 값이 이상해짐.
    //     for(k = 0; k < K; ++k) {
    //         float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
    //         // 레지스터를 한 번에 더 많이 사용하기 위해 루프 언롤링
    //         for(j = 0; j <= N - 24; j += 24) {
    //             // B 벡터를 6개 로드
    //             float32x4_t b0 = vld1q_f32(B + k * ldb + j);
    //             float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
    //             float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
    //             float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);
    //             float32x4_t b4 = vld1q_f32(B + k * ldb + j + 16);
    //             float32x4_t b5 = vld1q_f32(B + k * ldb + j + 20);
                

    //             // C 벡터도 6개 로드
    //             float32x4_t c0 = vld1q_f32(C + i * ldc + j);
    //             float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
    //             float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
    //             float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);
    //             float32x4_t c4 = vld1q_f32(C + i * ldc + j + 16);
    //             float32x4_t c5 = vld1q_f32(B + k * ldb + j + 20);
                
    //             // 병렬로 곱셈 및 누적 연산
    //             c0 = vmlaq_f32(c0, a_part, b0);
    //             c1 = vmlaq_f32(c1, a_part, b1);
    //             c2 = vmlaq_f32(c2, a_part, b2);
    //             c3 = vmlaq_f32(c3, a_part, b3);
    //             c4 = vmlaq_f32(c4, a_part, b4);
    //             c5 = vmlaq_f32(c5, a_part, b5);
               

    //             // 결과를 다시 C로 저장
    //             vst1q_f32(C + i * ldc + j, c0);
    //             vst1q_f32(C + i * ldc + j + 4, c1);
    //             vst1q_f32(C + i * ldc + j + 8, c2);
    //             vst1q_f32(C + i * ldc + j + 12, c3);
    //             vst1q_f32(C + i * ldc + j + 16, c4);
    //             vst1q_f32(C + i * ldc + j + 20, c5);
               
    //         }

    //         // 남은 요소 처리
    //         for(; j < N; j++) {
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

    // printf("==== register 15 ====\n");
    // time = clock();
    //  for(i = 0; i < M; ++i) { // 레지스터 15개  여기까지는 acceptable함.
    //     for(k = 0; k < K; ++k) {
    //         float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
    //         // 레지스터를 한 번에 더 많이 사용하기 위해 루프 언롤링
    //         for(j = 0; j <= N - 28; j += 28) {
    //             // B 벡터를 7개 로드
    //             float32x4_t b0 = vld1q_f32(B + k * ldb + j);
    //             float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
    //             float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
    //             float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);
    //             float32x4_t b4 = vld1q_f32(B + k * ldb + j + 16);
    //             float32x4_t b5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t b6 = vld1q_f32(B + k * ldb + j + 24);
                

    //             // C 벡터도 7개 로드
    //             float32x4_t c0 = vld1q_f32(C + i * ldc + j);
    //             float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
    //             float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
    //             float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);
    //             float32x4_t c4 = vld1q_f32(C + i * ldc + j + 16);
    //             float32x4_t c5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t c6 = vld1q_f32(B + k * ldb + j + 24);
                
    //             // 병렬로 곱셈 및 누적 연산
    //             c0 = vmlaq_f32(c0, a_part, b0);
    //             c1 = vmlaq_f32(c1, a_part, b1);
    //             c2 = vmlaq_f32(c2, a_part, b2);
    //             c3 = vmlaq_f32(c3, a_part, b3);
    //             c4 = vmlaq_f32(c4, a_part, b4);
    //             c5 = vmlaq_f32(c5, a_part, b5);
    //             c6 = vmlaq_f32(c6, a_part, b6);
               

    //             // 결과를 다시 C로 저장
    //             vst1q_f32(C + i * ldc + j, c0);
    //             vst1q_f32(C + i * ldc + j + 4, c1);
    //             vst1q_f32(C + i * ldc + j + 8, c2);
    //             vst1q_f32(C + i * ldc + j + 12, c3);
    //             vst1q_f32(C + i * ldc + j + 16, c4);
    //             vst1q_f32(C + i * ldc + j + 20, c5);
    //             vst1q_f32(C + i * ldc + j + 24, c6);
               
    //         }

    //         // 남은 요소 처리
    //         for(; j < N; j++) {
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

    // printf("==== register 17 ====\n");
    // time = clock();
    // for(i = 0; i < M; ++i) { // 레지스터 17개
    //     for(k = 0; k < K; ++k) {
    //         float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
    //         // 레지스터를 한 번에 더 많이 사용하기 위해 루프 언롤링
    //         for(j = 0; j <= N - 32; j += 32) {
    //             // B 벡터를 8개 로드
    //             float32x4_t b0 = vld1q_f32(B + k * ldb + j);
    //             float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
    //             float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
    //             float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);
    //             float32x4_t b4 = vld1q_f32(B + k * ldb + j + 16);
    //             float32x4_t b5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t b6 = vld1q_f32(B + k * ldb + j + 24);
    //             float32x4_t b7 = vld1q_f32(B + k * ldb + j + 28);

    //             // C 벡터도 8개 로드
    //             float32x4_t c0 = vld1q_f32(C + i * ldc + j);
    //             float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
    //             float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
    //             float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);
    //             float32x4_t c4 = vld1q_f32(C + i * ldc + j + 16);
    //             float32x4_t c5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t c6 = vld1q_f32(B + k * ldb + j + 24);
    //             float32x4_t c7 = vld1q_f32(B + k * ldb + j + 28);

    //             // 병렬로 곱셈 및 누적 연산
    //             c0 = vmlaq_f32(c0, a_part, b0);
    //             c1 = vmlaq_f32(c1, a_part, b1);
    //             c2 = vmlaq_f32(c2, a_part, b2);
    //             c3 = vmlaq_f32(c3, a_part, b3);
    //             c4 = vmlaq_f32(c4, a_part, b4);
    //             c5 = vmlaq_f32(c5, a_part, b5);
    //             c6 = vmlaq_f32(c6, a_part, b6);
    //             c7 = vmlaq_f32(c7, a_part, b7);

    //             // 결과를 다시 C로 저장
    //             vst1q_f32(C + i * ldc + j, c0);
    //             vst1q_f32(C + i * ldc + j + 4, c1);
    //             vst1q_f32(C + i * ldc + j + 8, c2);
    //             vst1q_f32(C + i * ldc + j + 12, c3);
    //             vst1q_f32(C + i * ldc + j + 16, c4);
    //             vst1q_f32(C + i * ldc + j + 20, c5);
    //             vst1q_f32(C + i * ldc + j + 24, c6);
    //             vst1q_f32(C + i * ldc + j + 28, c7);
    //         }

    //         // 남은 요소 처리
    //         for(; j < N; j++) {
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

    // printf("==== register 31 ====\n");
    // time = clock();

    // for(i = 0; i < M; ++i) { // 레지스터 31개
    //     for(k = 0; k < K; ++k) {
    //         float32x4_t a_part = vdupq_n_f32(ALPHA * A[i * lda + k]);
            
    //         for(j = 0; j <= N - 60; j += 60) {
    //             // B 벡터를 15개 로드
    //             float32x4_t b0 = vld1q_f32(B + k * ldb + j);
    //             float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
    //             float32x4_t b2 = vld1q_f32(B + k * ldb + j + 8);
    //             float32x4_t b3 = vld1q_f32(B + k * ldb + j + 12);
    //             float32x4_t b4 = vld1q_f32(B + k * ldb + j + 16);
    //             float32x4_t b5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t b6 = vld1q_f32(B + k * ldb + j + 24);
    //             float32x4_t b7 = vld1q_f32(B + k * ldb + j + 28);
    //             float32x4_t b8 = vld1q_f32(B + k * ldb + j + 32);
    //             float32x4_t b9 = vld1q_f32(B + k * ldb + j + 36);
    //             float32x4_t b10 = vld1q_f32(B + k * ldb + j + 40);
    //             float32x4_t b11 = vld1q_f32(B + k * ldb + j + 44);
    //             float32x4_t b12 = vld1q_f32(B + k * ldb + j + 48);
    //             float32x4_t b13 = vld1q_f32(B + k * ldb + j + 52);
    //             float32x4_t b14 = vld1q_f32(B + k * ldb + j + 56);

    //             // C 벡터도 15개 로드
    //             float32x4_t c0 = vld1q_f32(C + i * ldc + j);
    //             float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
    //             float32x4_t c2 = vld1q_f32(C + i * ldc + j + 8);
    //             float32x4_t c3 = vld1q_f32(C + i * ldc + j + 12);
    //             float32x4_t c4 = vld1q_f32(C + i * ldc + j + 16);
    //             float32x4_t c5 = vld1q_f32(B + k * ldb + j + 20);
    //             float32x4_t c6 = vld1q_f32(B + k * ldb + j + 24);
    //             float32x4_t c7 = vld1q_f32(B + k * ldb + j + 28);
    //             float32x4_t c8 = vld1q_f32(C + i * ldc + j + 32);
    //             float32x4_t c9 = vld1q_f32(C + i * ldc + j + 36);
    //             float32x4_t c10 = vld1q_f32(C + i * ldc + j + 40);
    //             float32x4_t c11 = vld1q_f32(C + i * ldc + j + 44);
    //             float32x4_t c12 = vld1q_f32(B + k * ldb + j + 48);
    //             float32x4_t c13 = vld1q_f32(B + k * ldb + j + 52);
    //             float32x4_t c14 = vld1q_f32(B + k * ldb + j + 56);

    //             // 병렬로 곱셈 및 누적 연산
    //             c0 = vmlaq_f32(c0, a_part, b0);
    //             c1 = vmlaq_f32(c1, a_part, b1);
    //             c2 = vmlaq_f32(c2, a_part, b2);
    //             c3 = vmlaq_f32(c3, a_part, b3);
    //             c4 = vmlaq_f32(c4, a_part, b4);
    //             c5 = vmlaq_f32(c5, a_part, b5);
    //             c6 = vmlaq_f32(c6, a_part, b6);
    //             c7 = vmlaq_f32(c7, a_part, b7);
    //             c8 = vmlaq_f32(c8, a_part, b8);
    //             c9 = vmlaq_f32(c9, a_part, b9);
    //             c10 = vmlaq_f32(c10, a_part, b10);
    //             c11 = vmlaq_f32(c11, a_part, b11);
    //             c12 = vmlaq_f32(c12, a_part, b12);
    //             c13 = vmlaq_f32(c13, a_part, b13);
    //             c14 = vmlaq_f32(c14, a_part, b14);

    //             // 결과를 다시 C로 저장
    //             vst1q_f32(C + i * ldc + j, c0);
    //             vst1q_f32(C + i * ldc + j + 4, c1);
    //             vst1q_f32(C + i * ldc + j + 8, c2);
    //             vst1q_f32(C + i * ldc + j + 12, c3);
    //             vst1q_f32(C + i * ldc + j + 16, c4);
    //             vst1q_f32(C + i * ldc + j + 20, c5);
    //             vst1q_f32(C + i * ldc + j + 24, c6);
    //             vst1q_f32(C + i * ldc + j + 28, c7);
    //             vst1q_f32(C + i * ldc + j + 32, c8);
    //             vst1q_f32(C + i * ldc + j + 36, c9);
    //             vst1q_f32(C + i * ldc + j + 40, c10);
    //             vst1q_f32(C + i * ldc + j + 44, c11);
    //             vst1q_f32(C + i * ldc + j + 48, c12);
    //             vst1q_f32(C + i * ldc + j + 52, c13);
    //             vst1q_f32(C + i * ldc + j + 56, c14);
    //         }

    //         // 남은 요소 처리
    //         for(; j < N; j++) {
    //             C[i * ldc + j] += ALPHA * A[i * lda + k] * B[k * ldb + j];
    //         }
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
     int i,j,k;
    //  clock_t time;

    // printf("==== register 3 ====\n");
    // time = clock();
    // for(i = 0; i < M; ++i){ // register: 3
    //     for(j = 0; j < N; ++j){
    //         float32x4_t sum_vec = vdupq_n_f32(0.0f);
    //         float sum = 0.0f;

    //         for(k = 0; k <= K - 4; k += 4){
    //             float32x4_t a_vec = vld1q_f32(A + i*lda+k);
    //             float32x4_t b_vec = vld1q_f32(B + j*ldb+k);

    //             sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
    //         }

    //         sum += ALPHA * (vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) + vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3));

    //         for(; k < K; k++){
    //             sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
    //         }

    //         C[i*ldc+j] += sum;
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));


    // printf("==== register 12 ====\n");
    // time = clock();
    // for(i = 0; i < M; ++i){
    //     for(j = 0; j < N; ++j){ // register 12
    //         float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec3 = vdupq_n_f32(0.0f);
    //         float sum = 0.0f;

    //         for(k = 0; k <= K - 16; k += 16){
    //             float32x4_t a_vec0 = vld1q_f32(A + i*lda+k);
    //             float32x4_t a_vec1 = vld1q_f32(A + i*lda+k+4);
    //             float32x4_t a_vec2 = vld1q_f32(A + i*lda+k+8);
    //             float32x4_t a_vec3 = vld1q_f32(A + i*lda+k+12);

    //             float32x4_t b_vec0 = vld1q_f32(B + j*ldb+k);
    //             float32x4_t b_vec1 = vld1q_f32(B + j*ldb+k+4);
    //             float32x4_t b_vec2 = vld1q_f32(B + j*ldb+k+8);
    //             float32x4_t b_vec3 = vld1q_f32(B + j*ldb+k+12);

    //             sum_vec0 = vmlaq_f32(sum_vec0, a_vec0, b_vec0);
    //             sum_vec1 = vmlaq_f32(sum_vec1, a_vec1, b_vec1);
    //             sum_vec2 = vmlaq_f32(sum_vec2, a_vec2, b_vec2);
    //             sum_vec3 = vmlaq_f32(sum_vec3, a_vec3, b_vec3);
    //         }

    //         sum += ALPHA * (vgetq_lane_f32(sum_vec0, 0) + vgetq_lane_f32(sum_vec0, 1) + vgetq_lane_f32(sum_vec0, 2) + vgetq_lane_f32(sum_vec0, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec1, 0) + vgetq_lane_f32(sum_vec1, 1) + vgetq_lane_f32(sum_vec1, 2) + vgetq_lane_f32(sum_vec1, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec2, 0) + vgetq_lane_f32(sum_vec2, 1) + vgetq_lane_f32(sum_vec2, 2) + vgetq_lane_f32(sum_vec2, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec3, 0) + vgetq_lane_f32(sum_vec3, 1) + vgetq_lane_f32(sum_vec3, 2) + vgetq_lane_f32(sum_vec3, 3));

    //         for(; k < K; k++){
    //             sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
    //         }

    //         C[i*ldc+j] += sum;
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));


    // printf("==== register 18 ====\n");
    // time = clock();
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){ // register 18
            float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec3 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec4 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec5 = vdupq_n_f32(0.0f);
            float sum = 0.0f;

            for(k = 0; k <= K - 24; k += 24){
                float32x4_t a_vec0 = vld1q_f32(A + i*lda+k);
                float32x4_t a_vec1 = vld1q_f32(A + i*lda+k+4);
                float32x4_t a_vec2 = vld1q_f32(A + i*lda+k+8);
                float32x4_t a_vec3 = vld1q_f32(A + i*lda+k+12);
                float32x4_t a_vec4 = vld1q_f32(A + i*lda+k+16);
                float32x4_t a_vec5 = vld1q_f32(A + i*lda+k+20);

                float32x4_t b_vec0 = vld1q_f32(B + j*ldb+k);
                float32x4_t b_vec1 = vld1q_f32(B + j*ldb+k+4);
                float32x4_t b_vec2 = vld1q_f32(B + j*ldb+k+8);
                float32x4_t b_vec3 = vld1q_f32(B + j*ldb+k+12);
                float32x4_t b_vec4 = vld1q_f32(B + j*ldb+k+16);
                float32x4_t b_vec5 = vld1q_f32(B + j*ldb+k+20);

                sum_vec0 = vmlaq_f32(sum_vec0, a_vec0, b_vec0);
                sum_vec1 = vmlaq_f32(sum_vec1, a_vec1, b_vec1);
                sum_vec2 = vmlaq_f32(sum_vec2, a_vec2, b_vec2);
                sum_vec3 = vmlaq_f32(sum_vec3, a_vec3, b_vec3);
                sum_vec4 = vmlaq_f32(sum_vec4, a_vec4, b_vec4);
                sum_vec5 = vmlaq_f32(sum_vec5, a_vec5, b_vec5);
            }

            sum += ALPHA * (vgetq_lane_f32(sum_vec0, 0) + vgetq_lane_f32(sum_vec0, 1) + vgetq_lane_f32(sum_vec0, 2) + vgetq_lane_f32(sum_vec0, 3));
            sum += ALPHA * (vgetq_lane_f32(sum_vec1, 0) + vgetq_lane_f32(sum_vec1, 1) + vgetq_lane_f32(sum_vec1, 2) + vgetq_lane_f32(sum_vec1, 3));
            sum += ALPHA * (vgetq_lane_f32(sum_vec2, 0) + vgetq_lane_f32(sum_vec2, 1) + vgetq_lane_f32(sum_vec2, 2) + vgetq_lane_f32(sum_vec2, 3));
            sum += ALPHA * (vgetq_lane_f32(sum_vec3, 0) + vgetq_lane_f32(sum_vec3, 1) + vgetq_lane_f32(sum_vec3, 2) + vgetq_lane_f32(sum_vec3, 3));
            sum += ALPHA * (vgetq_lane_f32(sum_vec4, 0) + vgetq_lane_f32(sum_vec4, 1) + vgetq_lane_f32(sum_vec4, 2) + vgetq_lane_f32(sum_vec4, 3));
            sum += ALPHA * (vgetq_lane_f32(sum_vec5, 0) + vgetq_lane_f32(sum_vec5, 1) + vgetq_lane_f32(sum_vec5, 2) + vgetq_lane_f32(sum_vec5, 3));

            for(; k < K; k++){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }

            C[i*ldc+j] += sum;
        }
    }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));
    

    // printf("==== register 30 ====\n");
    // time = clock();
    // for(i = 0; i < M; ++i){
    //     for(j = 0; j < N; ++j){ // register 18
    //         float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec3 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec4 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec5 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec6 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec7 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec8 = vdupq_n_f32(0.0f);
    //         float32x4_t sum_vec9 = vdupq_n_f32(0.0f);
    //         float sum = 0.0f;

    //         for(k = 0; k <= K - 40; k += 40){
    //             float32x4_t a_vec0 = vld1q_f32(A + i*lda+k);
    //             float32x4_t a_vec1 = vld1q_f32(A + i*lda+k+4);
    //             float32x4_t a_vec2 = vld1q_f32(A + i*lda+k+8);
    //             float32x4_t a_vec3 = vld1q_f32(A + i*lda+k+12);
    //             float32x4_t a_vec4 = vld1q_f32(A + i*lda+k+16);
    //             float32x4_t a_vec5 = vld1q_f32(A + i*lda+k+20);
    //             float32x4_t a_vec6 = vld1q_f32(A + i*lda+k+24);
    //             float32x4_t a_vec7 = vld1q_f32(A + i*lda+k+28);
    //             float32x4_t a_vec8 = vld1q_f32(A + i*lda+k+32);
    //             float32x4_t a_vec9 = vld1q_f32(A + i*lda+k+36);

    //             float32x4_t b_vec0 = vld1q_f32(B + j*ldb+k);
    //             float32x4_t b_vec1 = vld1q_f32(B + j*ldb+k+4);
    //             float32x4_t b_vec2 = vld1q_f32(B + j*ldb+k+8);
    //             float32x4_t b_vec3 = vld1q_f32(B + j*ldb+k+12);
    //             float32x4_t b_vec4 = vld1q_f32(B + j*ldb+k+16);
    //             float32x4_t b_vec5 = vld1q_f32(B + j*ldb+k+20);
    //             float32x4_t b_vec6 = vld1q_f32(B + j*ldb+k+24);
    //             float32x4_t b_vec7 = vld1q_f32(B + j*ldb+k+28);
    //             float32x4_t b_vec8 = vld1q_f32(B + j*ldb+k+32);
    //             float32x4_t b_vec9 = vld1q_f32(B + j*ldb+k+36);

    //             sum_vec0 = vmlaq_f32(sum_vec0, a_vec0, b_vec0);
    //             sum_vec1 = vmlaq_f32(sum_vec1, a_vec1, b_vec1);
    //             sum_vec2 = vmlaq_f32(sum_vec2, a_vec2, b_vec2);
    //             sum_vec3 = vmlaq_f32(sum_vec3, a_vec3, b_vec3);
    //             sum_vec4 = vmlaq_f32(sum_vec4, a_vec4, b_vec4);
    //             sum_vec5 = vmlaq_f32(sum_vec5, a_vec5, b_vec5);
    //             sum_vec6 = vmlaq_f32(sum_vec6, a_vec6, b_vec6);
    //             sum_vec7 = vmlaq_f32(sum_vec7, a_vec7, b_vec7);
    //             sum_vec8 = vmlaq_f32(sum_vec8, a_vec8, b_vec8);
    //             sum_vec9 = vmlaq_f32(sum_vec9, a_vec9, b_vec9);
    //         }

    //         sum += ALPHA * (vgetq_lane_f32(sum_vec0, 0) + vgetq_lane_f32(sum_vec0, 1) + vgetq_lane_f32(sum_vec0, 2) + vgetq_lane_f32(sum_vec0, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec1, 0) + vgetq_lane_f32(sum_vec1, 1) + vgetq_lane_f32(sum_vec1, 2) + vgetq_lane_f32(sum_vec1, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec2, 0) + vgetq_lane_f32(sum_vec2, 1) + vgetq_lane_f32(sum_vec2, 2) + vgetq_lane_f32(sum_vec2, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec3, 0) + vgetq_lane_f32(sum_vec3, 1) + vgetq_lane_f32(sum_vec3, 2) + vgetq_lane_f32(sum_vec3, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec4, 0) + vgetq_lane_f32(sum_vec4, 1) + vgetq_lane_f32(sum_vec4, 2) + vgetq_lane_f32(sum_vec4, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec5, 0) + vgetq_lane_f32(sum_vec5, 1) + vgetq_lane_f32(sum_vec5, 2) + vgetq_lane_f32(sum_vec5, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec6, 0) + vgetq_lane_f32(sum_vec6, 1) + vgetq_lane_f32(sum_vec6, 2) + vgetq_lane_f32(sum_vec6, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec7, 0) + vgetq_lane_f32(sum_vec7, 1) + vgetq_lane_f32(sum_vec7, 2) + vgetq_lane_f32(sum_vec7, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec8, 0) + vgetq_lane_f32(sum_vec8, 1) + vgetq_lane_f32(sum_vec8, 2) + vgetq_lane_f32(sum_vec8, 3));
    //         sum += ALPHA * (vgetq_lane_f32(sum_vec9, 0) + vgetq_lane_f32(sum_vec9, 1) + vgetq_lane_f32(sum_vec9, 2) + vgetq_lane_f32(sum_vec9, 3));

    //         for(; k < K; k++){
    //             sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
    //         }

    //         C[i*ldc+j] += sum;
    //     }
    // }
    // fprintf(stderr, "Predicted in %f seconds.\n", sec(clock()-time));

}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            float32x4_t a_part = vdupq_n_f32(ALPHA*A[k*lda+i]);
            for(j = 0; j <= N - 4; j += 4){
                float32x4_t b = vld1q_f32(B + k*ldb+j);
                float32x4_t c = vld1q_f32(C + i*ldc+j);
                c = vmlaq_f32(c, a_part, b);
                vst1q_f32(C + i*ldc+j, c);
            }
            for(; j < N; j++){
                C[i*ldc+j] += ALPHA*A[k*lda+i]*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float sum = 0.0f;

            for(k = 0; k <= K - 4; k += 4){
                float32x4_t a_vec = vld1q_f32(A + i+k*lda);
                float32x4_t b_vec = vld1q_f32(B + k+j*ldb);

                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }

            sum += ALPHA * (vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) + vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3));

            for(; k < K; k++){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb]; 
            }

            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

