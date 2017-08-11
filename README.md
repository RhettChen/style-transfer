# style-transfer
This is an implementation of style transfer based on VGGnet.

This work is based on an assignment of cs20si of stanford.

Platform configuration:

	tensorflow 1.1.0
	
	python 3.6.1

You can enter "python train.py" to training with default. 

Or enter "python train.py -c [context image route] -s [style image route]".

Firstly, I resize the input image to 250 x 333 (height x width)

/src/utils.py:
	define functions about download VGGnet19, resizing image and saving image.

/src/style_trainsfer.py:
	define the training process

/src/vgg_model.py:
	define the variable and create the convolutional layers for VGGnet19

And I use the context image:
![image](https://github.com/RhettChen/style-transfer/raw/master/content/deadpool.jpg)

the style inmage:
![image](https://github.com/RhettChen/style-transfer/raw/master/styles/guernica.jpg)

get the image during the training:
![image](https://github.com/RhettChen/style-transfer/raw/master/training_process.png)
