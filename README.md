# mnist_conv_nn10
Source code of the period of Chapter 4 of the November 2018 issue of "Toragi"

トランジスタ技術2018年11月号の第4章「手書き数字認識用FPGAニューラル・ネットワーク・システムの製作」のVivado HLSのソースコードです。
http://toragi.cqpub.co.jp/tabid/852/Default.aspx
http://toragi.cqpub.co.jp/Portals/0/backnumber/2018/11/p074.pdf

2つのVivado HLSプロジェクト用ソースコードが入っています。<br>
mnist_conv_nn10プロジェクト<br>
&nbsp;&nbsp;mnist_conv_nn10.cpp　（source）<br>
&nbsp;&nbsp;af1_bias.h　（source）<br>
&nbsp;&nbsp;af1_weight.h　（source）<br>
&nbsp;&nbsp;af2.bias.h　（source）<br>
&nbsp;&nbsp;af2.weight.h　（source）<br>
&nbsp;&nbsp;conv1_bias.h　（source）<br>
&nbsp;&nbsp;conv1_weight.h　（source）<br>
&nbsp;&nbsp;mnist_data.h　（testbench）<br>
&nbsp;&nbsp;mnist_conv_nn10_tb.cpp　（testbench）<br>
 <br>
mnist_conv_nn10_sDMAプロジェクト<br>
&nbsp;&nbsp;mnist_conv_nn10_sDMA.cpp　（source）<br>
&nbsp;&nbsp;af1_bias.h　（source）<br>
&nbsp;&nbsp;af1_weight.h　（source）<br>
&nbsp;&nbsp;af2.bias.h　（source）<br>
&nbsp;&nbsp;af2.weight.h　（source）<br>
&nbsp;&nbsp;conv1_bias.h　（source）<br>
&nbsp;&nbsp;conv1_weight.h　（source）<br>
&nbsp;&nbsp;bmp_file0.bmp　（testbench）<br>
&nbsp;&nbsp;bmp_header.h　（testbench）<br>
&nbsp;&nbsp;mnist_conv_nn10_sDMA_tb.cpp　（testbench）<br>
 
Vivado HLSの使い方については、「FPGAの部屋」のブログをご覧ください。<br>
http://marsee101.web.fc2.com/vivado_hls.html
