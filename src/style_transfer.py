from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils

# parameters to manage experiments

IMAGE_HEIGHT = 250
IMAGE_WIDTH = 333

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    _,a,b,c = p.shape
    co_eff = 1/(4*a*b*c)
    content_loss = co_eff*tf.reduce_sum(tf.square(f-p))
    return content_loss

def _gram_matrix(F, N, M):
    F = tf.reshape(F,[M,N])
    G = tf.matmul(F,tf.transpose(F))
    return G

def _single_style_loss(a, g):
    _, d1, d2,d3 = a.shape
    co_eff = 1/(4*d1*d2*d3*d1*d2*d3)
    s_style_loss = co_eff * tf.reduce_sum(tf.square(g-a))
    return s_style_loss

def _create_style_loss(A, model):
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]

    style_loss = 0.0
    for i in range(n_layers):
        style_loss += W[i]*E[i]
    return style_loss

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        total_loss = 0.01*content_loss + 1*style_loss

    return content_loss, style_loss, total_loss

def _create_summary(model):
    with tf.name_scope('summary'):
        tf.summary.scalar('content_loss',model['content_loss'])
        tf.summary.scalar('style_loss',model['style_loss'])
        tf.summary.scalar('total_loss',model['total_loss'])
        tf.summary.histogram('content_loss',model['content_loss'])
        tf.summary.histogram('style_loss',model['style_loss'])
        tf.summary.histogram('total_loss',model['total_loss'])
        return tf.summary.merge_all()

def train(model, generated_image, initial_image, ITERS,model_load):
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
       
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs",sess.graph)
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_load+'/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:        
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], 
                                                             model['summary_op']])
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = '../outputs/%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, model_load+'/style_transfer', index)
    writer.close()

def training(STYLE_IMAGE,CONTENT_IMAGE,LR,ITERS,NOISE_RATIO,model_load,trainable):
    with tf.variable_scope('input') as scope:
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)
    
    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                    input_image, content_image, style_image)
  
    model['optimizer'] = tf.train.AdamOptimizer(learning_rate=LR).minimize(model['total_loss'],global_step=model['global_step'])
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)

    if(trainable):
        train(model, input_image, initial_image,ITERS,model_load)
    else:
        if(os.path.exists(file_name)):
            test(model,input_image,initial_image,model_load)
        else:
            raise IOError("there is no model for test existing")

def test(model, intput_image, initial_image,model_load):
    with tf.Session() as sess:
        saver = tf.train.Saver()
       
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs",sess.graph)
        sess.run(input_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_load+'/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            gen_image= sess.run([input_image])
            gen_image = gen_image + MEAN_PIXELS
            filename = '../outputs/test.png'
            utils.save_image(filename, gen_image)





