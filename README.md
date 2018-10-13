# mnist_conv_nn10
Source code of the period of Chapter 4 of the November 2018 issue of "Toragi"

トランジスタ技術2018年11月号の第4章「手書き数字認識用FPGAニューラル・ネットワーク・システムの製作」のVivado HLSのソースコードです。<br>
http://toragi.cqpub.co.jp/tabid/852/Default.aspx<br>
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

なお、日昇テクノロジーのMT9D111メガピクセルカメラモジュール&nbsp;http://www.csun.co.jp/SHOP/2010020301.html &nbsp;は在庫切れですが、ディスコンぽいです。
FPGAの部屋のブログの「2 Mega pixel Camera Module MT9D111 JPEG Out + HQ lens」&nbsp; http://marsee101.blog19.fc2.com/blog-entry-2587.html &nbsp;の 2 Mega pixel Camera Module MT9D111 JPEG Out + HQ lens &nbsp;https://www.ebay.com/itm/2-Mega-pixel-Camera-Module-MT9D111-JPEG-Out-HQ-lens-/280572739197?ssPageName=ADME:L:OC:US:3160 &nbsp;はまだ在庫があるようです。
