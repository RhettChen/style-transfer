import argparse
import os
from style_transfer import training






if __name__=='__main__':
	class_ = argparse.ArgumentDefaultsHelpFormatter
	parser = argparse.ArgumentParser(description=__doc__,formatter_class=class_)

	parser.add_argument('-lr',help="learning rate",default=2.0, dest='learning_rate',type=float)
	parser.add_argument('-c',dest='content',default='../content/deadpool.jpg',help="content image")
	parser.add_argument('-s',dest='style',default='../styles/guernica.jpg',help="style image")
	parser.add_argument('-n',help='percentage of weight of the noise for intermixing with the content image',
						default=0.6,dest='noise', type=float)
	parser.add_argument('-e', help='Number of epochs', default=300,
                        dest='num_epochs', type=int)
	parser.add_argument('-load',dest='model_load', default='../Checkpoints',help='the directory of the model you want to load')
	args = parser.parse_args()
	
	training(args.style,args.content,args.learning_rate,args.num_epochs,args.noise,args.model_load,True)



