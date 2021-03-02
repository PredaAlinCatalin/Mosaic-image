from parameters import *
import numpy as np
import pdb
import timeit
import math


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    small_images_indexes = np.zeros((params.num_pieces_vertical, params.num_pieces_horizontal))

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = np.mean(params.small_images, axis=(1, 2))
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]
                mean_patch = np.mean(patch, axis=(0, 1))

                index = get_sorted_indices(mean_color_pieces, mean_patch)
                for ii in index:
                    if i > 0:
                        if small_images_indexes[i - 1, j] == ii:
                            continue
                    if j > 0:
                        if small_images_indexes[i, j - 1] == ii:
                            continue
                    break

                index = ii

                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                small_images_indexes[i, j] = index
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def get_sorted_indices(mean_color_pieces, mean_patch):
    distances = np.sum((mean_color_pieces - mean_patch) ** 2, axis=1)
    return distances.argsort()


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    bigger_image = np.zeros((h + H, w + W, c))
    img_mosaic = np.zeros((h + H, w + W, c), np.uint8)
    for i in range(h + H):
        for j in range(w + W):
            for k in range(c):
                if i < h and j < w:
                    bigger_image[i, j, k] = params.image_resized[i, j, k]
                else:
                    bigger_image[i, j, k] = 0

    free_pixels = np.zeros((h + H, w + W), dtype=int)
    nr = 0
    for i in range(h + H):
        for j in range(w + W):
            if i < h and j < w:
                free_pixels[i, j] = nr

            else:
                free_pixels[i, j] = -1
            nr += 1

    if params.criterion == 'aleator':
       None

    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = np.mean(params.small_images, axis=(1, 2))
        while True:
            free_ = free_pixels[free_pixels > -1]

            if len(free_) == 0:
                break
            index = np.random.randint(low=0, high=len(free_), size=1)

            row = math.floor(free_[index] / free_pixels.shape[1])
            col = math.floor(free_[index] % free_pixels.shape[1])

            patch = bigger_image[row:row + H, col:col + W]
            mean_patch = np.mean(patch, axis=(0, 1))

            index = get_sorted_indices(mean_color_pieces, mean_patch)[0]

            img_mosaic[row:row + H, col:col + W] = params.small_images[index]
            free_pixels[row:row + H, col:col + W] = -1
        img_mosaic = img_mosaic[0:h, 0:w, :]
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    bigger_image = np.zeros((h + 2 * H, w + 2 * W, C))
    img_mosaic = np.zeros((h + 2 * H, w + 2 * W, c), np.uint8)

    nr_vertical = h + 2 * H
    nr_horizontal = w + 2 * W
    small_images_indexes = np.zeros((nr_vertical, nr_horizontal))
    nr = 0


    mask = np.zeros((H, W, C), dtype=int)
    for i in range(H):
        for j in range(W):
            for k in range(C):
                mask[i, j, k] = 1

    width_divided_by_3 = int(W / 3)
    for i in range(H):
        for j in range(W):
            for k in range(C):
                if i < H / 2 - 1 and j < width_divided_by_3 and i + j <= H / 2 - 2:
                    mask[i, j, k] = 0
                if i < H / 2 - 1 and j > 2 * width_divided_by_3 and i <= j - 2 * width_divided_by_3 - 1:
                    mask[i, j, k] = 0
                if i > H / 2 and j < width_divided_by_3 and i - H / 2 > j:
                    mask[i, j, k] = 0
                if i > H / 2 and j > 2 * width_divided_by_3 and i - H / 2 - 1 + j - 2 * width_divided_by_3 - 1 >= H / 2 - 2:
                    mask[i, j, k] = 0

    with open('masca.txt', 'w') as f:
        for row in mask:
            for elem in row:
                f.write("%s " % elem)
            f.write("\n")

    mean_color_pieces = np.mean(params.small_images, axis=(1, 2))
    print(mean_color_pieces.shape)
    for i in range(H, h + H):
        for j in range(W, w + W):
            for k in range(c):
                bigger_image[i, j, k] = params.image_resized[i - H, j - W, k]

    first_row_start = 14
    row_index = 1

    for i in range(first_row_start, bigger_image.shape[0] - H, H):
        col_index = 0
        for j in range(0, bigger_image.shape[1] - W, W + width_divided_by_3):
            patch = bigger_image[i: i + H, j: j + W]
            mean_patch = np.mean(patch, axis=(0, 1))
            index = get_sorted_indices(mean_color_pieces, mean_patch)

            for ii in index:
                if row_index > 1:
                    if small_images_indexes[row_index - 2, col_index] == ii:
                        continue

                if col_index > 0:
                    if small_images_indexes[row_index, col_index - 2] == ii:
                        continue

                if row_index < small_images_indexes.shape[0] - 2 and col_index > 1:
                    if small_images_indexes[row_index + 2, col_index - 2] == ii:
                        continue
                break

            index = ii

            img_mosaic[i: i + H, j: j + W] = (1 - mask) * img_mosaic[i: i + H, j: j + W] + mask * params.small_images[index]
            small_images_indexes[row_index, col_index] = index
            col_index += 2

        row_index += 2

    row_index = 0
    for i in range(0, bigger_image.shape[0] - H, H):
        col_index = 0
        for j in range(2 * width_divided_by_3, bigger_image.shape[1] - W, W + width_divided_by_3):
            patch = bigger_image[i: i + H, j: j + W]
            mean_patch = np.mean(patch, axis=(0, 1))
            index = get_sorted_indices(mean_color_pieces, mean_patch)

            for ii in index:
                if row_index > 1 and col_index > 1:
                    if small_images_indexes[row_index - 2, col_index - 2] == ii:
                        continue

                if col_index > 1:
                    if small_images_indexes[row_index, col_index - 2] == ii:
                        continue

                if row_index > 1:
                    if small_images_indexes[row_index - 2, col_index] == ii:
                        continue

                break

            index = ii

            img_mosaic[i: i + H, j: j + W] = (1 - mask) * img_mosaic[i: i + H, j: j + W] + mask * params.small_images[index]
            small_images_indexes[row_index, col_index] = index
            col_index += 2

        row_index += 2

    img_mosaic = img_mosaic[H: H + h, W: W + w, ]

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic
