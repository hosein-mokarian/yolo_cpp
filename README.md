# YOLO

YOLO (You Only Look Once) is an algorithm to detect objects in images or videos frames. It is really simple and also faster than many other famous object detection algorithms such as: RNN and CNN. It was firstly proposed in 2015 and the update version of it has been being released during recent years. Every version has a yolo-x.cfg or yolo-tiny-x.cfg file which indicate how the network is implemented. X is version of YOLO. Those files can be found on YOLO's Github page.

The main point of YOLO is its speed. Also it uses some simple operation in every step. These points make it suitable for hardware implementation. 2D convolution, down sampling and Leaky-Relu algorithm are used. 

Here we have decided to implement YOLO 2 tiny. It has less steps, but it is still powerful. The Vivado HLS software is used to implement  YOLO on a FPGA. 
