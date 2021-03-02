import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

from add_pieces_mosaic import *
from parameters import *


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i

    filenames = os.listdir(params.small_images_dir)

    images = []
    for image_name in filenames:
        img_current = cv.imread(params.small_images_dir + image_name)
        images.append(img_current)

    images = np.array(images)


    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# def load_pieces(params: Parameters):
#     # citeste toate cele N piese folosite la mozaic din directorul corespunzator
#     # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
#     # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
#     # functia intoarce pieseMozaic = matrice N x H x W x C in params
#     # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
#
#
#     dict = unpickle(params.small_images_dir + "/data_batch_1")
#     dict_labels = unpickle(params.small_images_dir + "/batches.meta")
#
#     print(dict_labels)
#     for x in dict:
#         print(x)
#
#     images = []
#     searched_label = ""
#     if "airplane" in params.image_path:
#         searched_label = b'airplane'
#     elif "automobile" in params.image_path:
#         searched_label = b'automobile'
#     elif "bird" in params.image_path:
#         searched_label = b'bird'
#     elif "cat" in params.image_path:
#         searched_label = b'cat'
#     elif "dog" in params.image_path:
#         searched_label = b'dog'
#
#     print(searched_label)
#
#     nr = 0
#     for image_values in dict[b'data']:
#         if dict_labels[b'label_names'][dict[b'labels'][nr]] == searched_label:
#             image = np.zeros((32, 32, 3), dtype=int)
#             for i in range(0, 1024):
#                 image[int(i / 32)][i % 32][0] = image_values[i]
#                 image[int(i / 32)][i % 32][1] = image_values[i + 1024]
#                 image[int(i / 32)][i % 32][2] = image_values[i + 2048]
#             images.append(image)
#         nr += 1
#
#     images = np.array(images)
#     print(len(images))
#
#     if params.show_small_images:
#         for i in range(10):
#             for j in range(10):
#                 plt.subplot(10, 10, i * 10 + j + 1)
#                 # OpenCV reads images in BGR format, matplotlib reads images in RBG format
#                 im = images[i * 10 + j].copy()
#                 plt.imshow(im)
#         plt.show()
#
#     params.small_images = images
#


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    N, H, W, C = params.small_images.shape
    h, w, c = params.image.shape
    params.num_pieces_vertical = int(((params.num_pieces_horizontal * W) * (h / w) / H))

    # redimensioneaza imaginea
    new_h = params.num_pieces_vertical * H
    new_w = params.num_pieces_horizontal * W
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
