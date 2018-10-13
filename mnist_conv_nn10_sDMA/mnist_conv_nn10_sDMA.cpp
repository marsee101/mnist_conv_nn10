// mnist_conv_nn10_sDMA.cpp
// 2017/06/12 by marsee
// 畳み込み層のカーネル数 10
// 2017/06/29 : アドレスオフセット導入　800x600 画像中の 28x28 を切り取ってDMAする
// |      アドレスオフセット      |
// *************************-手書き数字1行目-****************
// *************************-手書き数字2行目-****************
//

#include <ap_fixed.h>

#include "conv1_weight.h"
#include "conv1_bias.h"
#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"

ap_ufixed<8, 0, AP_TRN, AP_SAT> conv_rgb2y(int rgb);

int mnist_conv_nn(int in[22400], int addr_offset, ap_fixed<12, 7, AP_TRN, AP_SAT> out[10]){
#pragma HLS INTERFACE s_axilite port=addr_offset
#pragma HLS INTERFACE s_axilite register port=out
#pragma HLS INTERFACE m_axi depth=22400 port=in offset=slave
#pragma HLS INTERFACE s_axilite port=return
	ap_ufixed<8, 0, AP_TRN, AP_SAT> buf[28][28];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
	ap_fixed<10, 3, AP_TRN, AP_SAT> conv_out[10][24][24];
	ap_fixed<10, 3, AP_TRN, AP_SAT> pool_out[10][12][12];
	ap_fixed<13, 7, AP_TRN, AP_SAT> dot1[100];
#pragma HLS ARRAY_PARTITION variable=dot1 complete dim=1
	ap_fixed<13, 7, AP_TRN, AP_SAT> dot2[10];

#pragma HLS ARRAY_PARTITION variable=dot2 complete dim=1
	buf_copy1: for(int i=0; i<28; i++){
		buf_copy2: for(int j=addr_offset; j<addr_offset+28; j++){
#pragma HLS PIPELINE II=1
			buf[i][j-addr_offset] = (ap_ufixed<8, 0, AP_TRN, AP_SAT>)0.99609375 - conv_rgb2y(in[i*800+j]);
			// 1.0 にならないように 1/256を引いておく
		}
	}

	// Convolutional Neural Network 5x5 kernel, Stride = 1, Padding = 0
	// + ReLU
	CONV1: for(int i=0; i<10; i++){	// カーネルの個数
		CONV2: for(int j=0; j<24; j++){
			CONV3: for(int k=0; k<24; k++){
				conv_out[i][j][k] = 0;
				CONV4: for(int m=0; m<5; m++){
#pragma HLS PIPELINE II=1
					conv_out[i][j][k] = buf[j+m][k] * conv1_weight[i][0][m][0]+
							buf[j+m][k+1] * conv1_weight[i][0][m][1] +
							buf[j+m][k+2] * conv1_weight[i][0][m][2] +
							buf[j+m][k+3] * conv1_weight[i][0][m][3] +
							buf[j+m][k+4] * conv1_weight[i][0][m][4];
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
#pragma HLS PIPELINE II=2
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
#pragma HLS PIPELINE II=1
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
#pragma HLS PIPELINE II=1
			dot2[col] += dot1[row]*af2_weight[row][col];
		}
		dot2[col] += af2_bias[col];

		out[col] = dot2[col];
	}

	return(0);
}

// RGBからYへの変換
// RGBのフォーマットは、{8'd0, R(8bits), G(8bits), B(8bits)}, 1pixel = 32bits
// 輝度信号Yのみに変換する。変換式は、Y =  0.299R + 0.587G + 0.114B
// "YUVフォーマット及び YUV<->RGB変換"を参考にした。http://vision.kuee.kyoto-u.ac.jp/~hiroaki/firewire/yuv.html
//　2013/09/27 : float を止めて、すべてint にした
// 2017/06/30 : ap_ufixed<8, 0, AP_TRN, AP_SAT> 出力とした
ap_ufixed<8, 0, AP_TRN, AP_SAT> conv_rgb2y(int rgb){
    int r, g, b, y_f;
    int y;
    ap_ufixed<16, 8, AP_TRN, AP_SAT> y_ap_ufixed;

    b = rgb & 0xff;
    g = (rgb>>8) & 0xff;
    r = (rgb>>16) & 0xff;

    y_f = 77*r + 150*g + 29*b; //y_f = 0.299*r + 0.587*g + 0.114*b;の係数に256倍した
    y = y_f >> 8; // 256で割る

    if (y >= 256)
    	y = 255;

    y_ap_ufixed = (ap_ufixed<16, 8, AP_TRN, AP_SAT>)y / 256;

    return((ap_ufixed<8, 0, AP_TRN, AP_SAT>)y_ap_ufixed);
}
