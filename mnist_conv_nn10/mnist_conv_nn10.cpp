// mnist_conv_nn10.cpp
// 2017/06/12 by marsee
// 畳み込み層のカーネル数 10
//

#include <ap_fixed.h>

#include "conv1_weight.h"
#include "conv1_bias.h"
#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"

int mnist_conv_nn(ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> in[784], ap_fixed<12, 7, AP_TRN_ZERO, AP_SAT> out[10]){
	ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> buf[28][28];
	ap_fixed<10, 3, AP_TRN_ZERO, AP_SAT> conv_out[10][24][24];
	ap_fixed<10, 3, AP_TRN_ZERO, AP_SAT> pool_out[10][12][12];
	ap_fixed<13, 7, AP_TRN_ZERO, AP_SAT> dot1[100];
	ap_fixed<13, 7, AP_TRN_ZERO, AP_SAT> dot2[10];

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
						conv_out[i][j][k] += buf[j+m][k+n] * conv1_weight[i][0][m][n];
					}
				}
				conv_out[i][j][k] += conv1_bias[i];

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
					dot1[col] += pool_out[i][j][k]*af1_weight[i*12*12+j*12+k][col];
				}
			}
		}
		dot1[col] += af1_bias[col];

		if(dot1[col] < 0)	// ReLU
			dot1[col] = 0;
	}

	af2_dot1: for(int col=0; col<10; col++){
		dot2[col] = 0;
		af2_dot2: for(int row=0; row<100; row++){
			dot2[col] += dot1[row]*af2_weight[row][col];
		}
		dot2[col] += af2_bias[col];

		out[col] = dot2[col];
	}

	return(0);
}
