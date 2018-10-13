// mnist_conv_nn_tb.cpp
// 2017/06/14 by marsee
// 畳み込み層のカーネル数 10
//

#include <stdio.h>
#include <ap_fixed.h>

#include "conv1_weight.h"
#include "conv1_bias.h"
#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"

#include "mnist_data.h"

int mnist_conv_nn(ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> in[784], ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> out[10]);
int mnist_conv_nn_float(float in[784], float out[10]);
int max_ap_fixed(ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> out[10]);
int max_float(float out[10]);

#define NUM_ITERATIONS	100 // C Simulation
//#define NUM_ITERATIONS	2 // C/RTL CoSimulation

int main(){
	float t_tran_float[NUM_ITERATIONS][784];
	ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> result_ap_fixed[NUM_ITERATIONS][10];
	float result_float[NUM_ITERATIONS][10];
	int max_id_hw, max_id_sw, max_id_ref;

	for(int i=0; i<NUM_ITERATIONS; i++)
		for(int j=0; j<784; j++)
			t_tran_float[i][j] = (float)t_train[i][j];

	for(int i=0; i<NUM_ITERATIONS; i++){
		mnist_conv_nn(&t_train[i][0], &result_ap_fixed[i][0]);
		mnist_conv_nn_float(&t_tran_float[i][0], &result_float[i][0]);
	}

	int errflag=0;
	for(int i=0; i<NUM_ITERATIONS; i++){
		max_id_hw = max_ap_fixed(&result_ap_fixed[i][0]);
		max_id_sw = max_float(&result_float[i][0]);
		max_id_ref = max_float(&t_test[i][0]);

		if(max_id_ref != max_id_hw){
			printf("id = %d, max_id_ref = %d, max_id_hw = %d\n", i, max_id_ref, max_id_hw);
			errflag = 1;
		}
		if(max_id_ref != max_id_sw){
			printf("id = %d, max_id_ref = %d, max_id_sw = %d\n", i, max_id_ref, max_id_sw);
			errflag = 1;
		}
	}
	if(errflag == 0)
		printf("No Error\n");

	return(0);
}

int mnist_conv_nn_float(float in[784], float out[10]){
	float buf[28][28];
	float conv_out[10][24][24];
	float pool_out[10][12][12];
	float dot1[100];
	float dot2[10];

	buf_copy1: for(int i=0; i<28; i++)
		buf_copy2: for(int j=0; j<28; j++)
			buf[i][j] = in[i*28+j];

	// Convolutional Neural Network 5x5 kernel, Stride = 1, Padding = 0
	// + ReLU
	CONV1: for(int i=0; i<10; i++){	// カーネルの個数
		CONV2: for(int j=0; j<24; j++){
			CONV3: for(int k=0; k<24; k++){
				conv_out[i][j][k] = 0;
				CONV4: for(int m=0; m<5; m++){
					CONV5: for(int n=0; n<5; n++){
						conv_out[i][j][k] += buf[j+m][k+n] * conv1_fweight[i][0][m][n];
					}
				}
				conv_out[i][j][k] += conv1_fbias[i];

				if(conv_out[i][j][k]<0)	// ReLU
					conv_out[i][j][k] = 0;
			}
		}
	}

	// Pooling Kernel = 2 x 2, Stride = 2
	POOL1: for(int i=0; i<10; i++){
		POOL2: for(int j=0; j<24; j += 2){
			POOL3: for(int k=0; k<24; k += 2){
				POOL4: for(int m=0; m<2; m++){
					POOL5: for(int n=0; n<2; n++){
						if(m==0 && n==0){
							pool_out[i][j/2][k/2] = conv_out[i][j][k];
						} else if(pool_out[i][j/2][k/2] < conv_out[i][j+m][k+n]){
							pool_out[i][j/2][k/2] = conv_out[i][j+m][k+n];
						}
					}
				}
			}
		}
	}

	af1_dot1: for(int col=0; col<100; col++){
		dot1[col] = 0;
		af1_dot2: for(int i=0; i<10; i++){
			af1_dot3: for(int j=0; j<12; j++){
				af1_dot4: for(int k=0; k<12; k++){
					dot1[col] += pool_out[i][j][k]*af1_fweight[i*12*12+j*12+k][col];
				}
			}
		}
		dot1[col] += af1_fbias[col];

		if(dot1[col] < 0)	// ReLU
			dot1[col] = 0;
	}

	af2_dot1: for(int col=0; col<10; col++){
		dot2[col] = 0;
		af2_dot2: for(int row=0; row<100; row++){
			dot2[col] += dot1[row]*af2_fweight[row][col];
		}
		dot2[col] += af2_fbias[col];

		out[col] = dot2[col];
	}

	return(0);
}

int max_ap_fixed(ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> out[10]){
	int max_id;
	ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> max;

	for(int i=0; i<10; i++){
		if(i == 0){
			max = out[0];
			max_id = 0;
		}else if(out[i]>max){
			max = out[i];
			max_id = i;
		}
	}
	return(max_id);
}

int max_float(float out[10]){
	int max_id;
	float max;

	for(int i=0; i<10; i++){
		if(i == 0){
			max = out[0];
			max_id = 0;
		}else if(out[i]>max){
			max = out[i];
			max_id = i;
		}
	}
	return(max_id);
}
