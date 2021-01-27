import os
import glob
import argparse
import matplotlib
import cv2
import numpy as np

from skimage.transform import resize

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='Input Handling')
parser.add_argument('--input', default='samples/dev/input/in.png', type=str, help='Input filepath.')
parser.add_argument('--output', default='samples/dev/out-1', type=str, help='Output folder name.')
args = parser.parse_args()

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, _images

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model("nyu.h5", custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format("nyu.h5"))

# Input images
inputs = load_images([args.input])
image = cv2.imread(args.input)

print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
outputs = outputs.reshape((inputs.shape[1] // 2, inputs.shape[2] // 2, 1))
os.mkdir(args.output)
np.save(f"{args.output}/depth.npy", outputs)
cv2.imwrite(f"{args.output}/depth-out.png", outputs * 255)
cv2.imwrite(f"{args.output}/input.png", image)

print("Output Saved")
