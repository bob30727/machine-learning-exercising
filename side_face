# https://github.com/thnkim/OpenFacePytorch
# https://github.com/AlfredXiangWu/LightCNN

import json
import os
from skimage import io, transform, color, util
from xml.dom import minidom as md

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
import numpy as np
from xml.dom import minidom as md
import json
from collections import OrderedDict
from skimage import io, transform, color, util
from torch.utils.data import Dataset

USE_CUDA = True
USE_PRETRAINED = True


    return padded_image[(padding+top):(padding+bottom), (padding+left):(padding+right), :], top, left, width, height


def predict_face_landmark(image,
                          face_top, face_left, face_width, face_height,
                          model_file_name, normalization_file_name, n_landmark=21):
    target_width = 227
    target_height = 227

    normalization = np.load(normalization_file_name)
    image_mean = normalization['mean']
    image_stddev = normalization['stddev']

    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    _model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    net = HyperFace(n_landmark)
    net.load_state_dict(_model['state_dict'])

    face_image, top, left, width, height = crop_face(
        image,
        face_top, face_left,
        face_width, face_height,
        buffer_portion=1/4.)

    width_scale = 1. * target_width / width
    height_scale = 1. * target_width / height
    face_image = transform.resize(face_image, (target_height, target_width))
    face_image = np.rollaxis(face_image, 2)
    face_image = 1.0*(face_image - image_mean) / image_stddev
    face_image = face_image[np.newaxis, :, :, :]

    landmarks = net(torch.autograd.Variable(torch.from_numpy(face_image).type(torch.FloatTensor)))
    landmarks = landmarks.data.numpy().reshape((-1, 2))
    landmarks = landmarks / np.array([[width_scale, height_scale]]) * np.array([[target_width, target_height]]) + np.array([left, top])

    return landmarks


def predict(_image, model_file_name, normalization_file_name, n_landmark=21, create_plot=True):
    import dlib
    import matplotlib.pyplot as plt
