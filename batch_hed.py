
import numpy as np
import scipy.misc
import Image
import scipy.io
import os
import cv2
import argparse

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm


def parse_args():
    parser = argparse.ArgumentParser(description='batch proccesing: photos->edges')
    parser.add_argument('--caffe_root', dest='caffe_root', help='caffe root', default='/home/pxm/caffe/', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel', default='examples/EdgeDetection/hed/hed_pretrained_bsds.caffemodel', type=str)
    parser.add_argument('--prototxt', dest='prototxt', help='caffe prototxt file', default='examples/EdgeDetection/hed/hed_test.pt', type=str)
    parser.add_argument('--images_dir', dest='images_dir', help='directory to store input photos', type=str)
    parser.add_argument('--hed_mat_dir', dest='hed_mat_dir', help='directory to store output hed edges in mat file',  type=str)
    parser.add_argument('--border', dest='border', help='padding border', type=int, default=128)
    parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id', type=int, default=1)
    args = parser.parse_args()
    return args


def plot_single_scale(scale_lst, size,image_name):
	pylab.rcParams['figure.figsize'] = size, size/2
	plt.figure()
	for i in range(0, len(scale_lst)):
	    s=plt.subplot(1,5,i+1)
	    plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
	    s.set_xticklabels([])
	    s.set_yticklabels([])
	    s.yaxis.set_ticks_position('none')
	    s.xaxis.set_ticks_position('none')
	plt.tight_layout()
        plt.draw()
        plt.savefig('examples/EdgeDetection/hed/test/'+image_name+'_'+str(size)+'.png', bbox_inches='tight')
        #plt.close(fig)



args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
# Make sure that caffe is on the python path:  
caffe_root = args.caffe_root   # this file is expected to be in {caffe_root}/examples/hed/
import sys
print caffe_root+'python'
sys.path.insert(0, caffe_root + 'python')

import caffe
import scipy.io as sio

#if not os.path.exists(args.hed_mat_dir):
#    print('create output directory %s' % args.hed_mat_dir)
#    os.makedirs(args.hed_mat_dir)

imgList = os.listdir(args.images_dir)
nImgs = len(imgList)
print args.images_dir
print('#images = %d' % nImgs)


caffe.set_mode_cpu()
#caffe.set_device(args.gpu_id)
# load net
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
# pad border
border = args.border    

for i in range(nImgs):
    if i % 500 == 0:
        print('processing image %d/%d' % (i, nImgs))
    im = Image.open(os.path.join(args.images_dir, imgList[i]))
    im = im.resize((500,500), Image.ANTIALIAS)

    in_ = np.array(im, dtype=np.float32)

    #in_ = np.pad(in_,((border, border),(border,border),(0,0)),'reflect')

    in_ = in_[:,:,::-1]
    print in_.shape
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2, 0, 1))
    # remove the following two lines if testing with cpu

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()

    out1 = net.blobs['sigmoid_dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid_dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid_dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid_dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid_dsn5'].data[0][0,:,:]
    fuse = net.blobs['sigmoid_fuse'].data[0][0,:,:]
    
    scale_lst = [fuse]
    plot_single_scale(scale_lst, 22 , imgList[i])
    scale_lst = [out1, out2, out3, out4, out5]
    plot_single_scale(scale_lst, 10, imgList[i].split('.')[0])


    # get rid of the border
    #fuse = fuse[border:-border, border:-border]
    # save hed file to the disk
    #name, ext = os.path.splitext(imgList[i])

    #sio.savemat(os.path.join(args.hed_mat_dir, name + '.mat'), {'predict':fuse})




 
