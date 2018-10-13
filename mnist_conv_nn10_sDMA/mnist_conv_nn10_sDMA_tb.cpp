// mnist_conv_nn_sDMA_tb.cpp
// 2017/06/14 by marsee
// 畳み込み層のカーネル数 10
// 2017/06/29 : ストライドDMAのためのテストベンチ
//

#include <stdio.h>
#include <ap_fixed.h>

#include "conv1_weight.h"
#include "conv1_bias.h"
#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"

#include "bmp_header.h"

int mnist_conv_nn(int in[22400], int addr_offset, ap_fixed<12, 7, AP_TRN, AP_SAT> out[10]);
int mnist_conv_nn_float(int in[22400], int addr_offset, float out[10]);
int max_ap_fixed(ap_fixed<12, 7, AP_TRN, AP_SAT> out[10]);
int max_float(float out[10]);
float conv_rgb2y_soft(int rgb);

#define READ_BMP_FILE_NAME	"bmp_file0.bmp"

// 8
#define X_POS	560
#define Y_POS	183
// 7
//#define X_POS	504
//#define Y_POS	184
// 5
//#define X_POS	390
//#define Y_POS	138
// 0
//#define X_POS	390
//#define Y_POS	70
#define WIDTH	28
#define HEIGHT	28

int main(){
	ap_fixed<12, 7, AP_TRN, AP_SAT> result_ap_fixed[10];
	float result_float[10];
	int max_id_hw, max_id_sw, max_id_ref;
	int *in;
	int *inf;

    BITMAPFILEHEADER bmpfhr; // BMPファイルのファイルヘッダ(for Read)
    BITMAPINFOHEADER bmpihr; // BMPファイルのINFOヘッダ(for Read)
    FILE *fbmpr;
    int *rd_bmp;
    int blue, green, red;

    if ((fbmpr = fopen(READ_BMP_FILE_NAME, "rb")) == NULL){ // test.bmp をオープン
        fprintf(stderr, "Can't open ");
        fprintf(stderr, READ_BMP_FILE_NAME);
        fprintf(stderr, " by binary read mode\n");
        exit(1);
    }
    // bmpヘッダの読み出し
    fread(&bmpfhr.bfType, sizeof(uint16_t), 1, fbmpr);
    fread(&bmpfhr.bfSize, sizeof(uint32_t), 1, fbmpr);
    fread(&bmpfhr.bfReserved1, sizeof(uint16_t), 1, fbmpr);
    fread(&bmpfhr.bfReserved2, sizeof(uint16_t), 1, fbmpr);
    fread(&bmpfhr.bfOffBits, sizeof(uint32_t), 1, fbmpr);
    fread(&bmpihr, sizeof(BITMAPINFOHEADER), 1, fbmpr);

    // ピクセルを入れるメモリをアロケートする
    if ((rd_bmp =(int *)malloc(sizeof(int) * (bmpihr.biWidth * bmpihr.biHeight))) == NULL){
        fprintf(stderr, "Can't allocate rd_bmp memory\n");
        exit(1);
    }

    if ((in =(int *)malloc(sizeof(int) * (800 * 28))) == NULL){
        fprintf(stderr, "Can't allocate (ap_ufixed<8, 0, AP_TRN, AP_SAT>)in memory\n");
        exit(1);
    }

    if ((inf =(int *)malloc(sizeof(int) * (800 * 28))) == NULL){
        fprintf(stderr, "Can't allocate (float)inf memory\n");
        exit(1);
    }

    // rd_bmp にBMPのピクセルを代入。その際に、行を逆転する必要がある
    for (int y=0; y<bmpihr.biHeight; y++){
        for (int x=0; x<bmpihr.biWidth; x++){
            blue = fgetc(fbmpr);
            green = fgetc(fbmpr);
            red = fgetc(fbmpr);
            rd_bmp[((bmpihr.biHeight-1)-y)*bmpihr.biWidth+x] = (blue & 0xff) | ((green & 0xff)<<8) | ((red & 0xff)<<16);
        }
    }
    fclose(fbmpr);

    // rd_bmp を in と inf に入力
    for (int y=Y_POS; y<Y_POS+HEIGHT; y++){
        for (int x=0; x<bmpihr.biWidth; x++){
        	in[(y-Y_POS)*bmpihr.biWidth+x] = rd_bmp[y*bmpihr.biWidth+x];
        	inf[(y-Y_POS)*bmpihr.biWidth+x] = rd_bmp[y*bmpihr.biWidth+x];
        }
    }

    mnist_conv_nn(in, X_POS, result_ap_fixed);
    mnist_conv_nn_float(inf, X_POS, result_float);

	max_id_hw = max_ap_fixed(result_ap_fixed);
	max_id_sw = max_float(result_float);

	printf("max_id_hw = %d\n", max_id_hw);
	printf("max_id_sw = %d\n", max_id_sw);

	return(0);
}

int mnist_conv_nn_float(int in[22400], int addr_offset, float out[10]){
	float buf[28][28];
	float conv_out[10][24][24];
	float pool_out[10][12][12];
	float dot1[100];
	float dot2[10];

    // 手書き数字の値を表示
    for (int i=0; i<28; i++){
    	for (int j=0; j<800; j++){
    		if (j>=addr_offset && j<addr_offset+28)
    			printf("%2x, ", (int)(conv_rgb2y_soft(in[i*800+j])*256.0));
    	}
    	printf("\n");
    }

	buf_copy1: for(int i=0; i<28; i++){
		buf_copy2: for(int j=0; j<800; j++){
			if (j>=addr_offset && j<addr_offset+28)
				buf[i][j-addr_offset] = (float)0.99609375 - (float)conv_rgb2y_soft(in[i*800+j]);
		}
	}

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

int max_ap_fixed(ap_fixed<12, 7, AP_TRN, AP_SAT> out[10]){
	int max_id;
	ap_fixed<12, 7, AP_TRN, AP_SAT> max;

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


// RGBからYへの変換
// RGBのフォーマットは、{8'd0, R(8bits), G(8bits), B(8bits)}, 1pixel = 32bits
// 輝度信号Yのみに変換する。変換式は、Y =  0.299R + 0.587G + 0.114B
// "YUVフォーマット及び YUV<->RGB変換"を参考にした。http://vision.kuee.kyoto-u.ac.jp/~hiroaki/firewire/yuv.html
//　2013/09/27 : float を止めて、すべてint にした
// 2017/06/30 : retval を float にした
float conv_rgb2y_soft(int rgb){
    int r, g, b, y_f;
    int y;
    float y_float;

    b = rgb & 0xff;
    g = (rgb>>8) & 0xff;
    r = (rgb>>16) & 0xff;

    y_f = 77*r + 150*g + 29*b; //y_f = 0.299*r + 0.587*g + 0.114*b;の係数に256倍した
    y = y_f >> 8; // 256で割る

    if (y >= 256)
    	y = 255;

    y_float = (float)y/256.0;

    return(y_float);
}
