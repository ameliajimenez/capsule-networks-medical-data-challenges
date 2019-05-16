import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import matplotlib.ticker as plticker
import pandas as pd
from collections import OrderedDict
from random import randint
from PIL import Image, ImageChops
from scipy.ndimage.measurements import center_of_mass


def visualize_grid(img, width, height):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    my_intervalx = width  # width
    my_intervaly = height  # height
    locx = plticker.MultipleLocator(base=my_intervalx)
    locy = plticker.MultipleLocator(base=my_intervaly)
    ax.xaxis.set_major_locator(locx)
    ax.yaxis.set_major_locator(locy)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-')

    # Add the image
    ax.imshow(img)

    # Find number of gridsquares in x and y direction
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(my_intervalx)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(my_intervaly)))
    plt.show()


def preprocess_img(img, fmask, all_masks, clipLimit=3.0):
    # convert to PIL
    pil_image = Image.fromarray(img)
    bg = Image.new(pil_image.mode, pil_image.size, pil_image.getpixel((0, 0)))
    diff = ImageChops.difference(pil_image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    del pil_image
    # crop image if bbox is found
    if bbox:
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        crop_mask = all_masks[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        crop = img
        crop_mask = all_masks

    # onverting image to LAB Color model
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    # splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    # applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    cl = clahe.apply(l)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    # converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # normalize to float32 and 0-1 range
    all_masks = cv2.normalize(all_masks.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    img_final = cv2.normalize(final.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    return img_final[:, :, 1], all_masks


def extract_patches(img, centers, patch_size=50, n_patches=30, augment=False):
    n_centers = centers.shape[0]
    n_patches_per_center = np.ceil(n_patches / n_centers).astype(np.uint8)
    if n_patches_per_center == 0:
        n_patches_per_center = 1

    tx_max = 30
    tx_min = -30
    tx_range = np.arange(tx_min, tx_max+1, 1).astype(np.int8)
    ty_range = np.arange(tx_min, tx_max+1, 1).astype(np.int8)

    patches = []
    rows, cols = img.shape

    for m in range(0, n_centers):
        cx = centers[m, 0]
        cy = centers[m, 1]

        for npat in range(0, n_patches_per_center):
            offx = np.random.choice(tx_range, 1)[0]
            offy = np.random.choice(ty_range, 1)[0]

            row_start = cx - patch_size + offx
            row_end = cx + patch_size + offx
            col_start = cy - patch_size + offy
            col_end = cy + patch_size + offy

            if row_start < 0:
                small_offset = np.random.randint(1, 20)
                row_start = 0 + small_offset
                row_end = 2*patch_size + small_offset

            if row_end > rows:
                small_offset = np.random.randint(1, 20)
                row_end = rows - small_offset
                row_start = row_end-2*patch_size - small_offset

            if col_start < 0:
                small_offset = np.random.randint(1, 20)
                col_start = 0 + small_offset
                col_end = 2 * patch_size + small_offset

            if col_end > cols:
                small_offset = np.random.randint(1, 20)
                col_end = cols - small_offset
                col_start = col_end - 2 * patch_size - small_offset

            if row_start < 0 or col_start < 0:
                print('error')

            patch = img[row_start:row_end, col_start:col_end]
            # augmentations
            if augment:
                if randint(1, 2) == 1:  # <------------ rotation (50% prob)
                    [rows_patch, cols_patch] = patch.shape
                    rot = randint(-10, 10)
                    M = cv2.getRotationMatrix2D((cols_patch / 2, rows_patch / 2), rot, 1)
                    patch = cv2.warpAffine(patch, M, (cols_patch, rows_patch))

                if randint(1, 2) == 1:  # <------------- horizontal flip : left / right (50% prob)
                    patch = np.fliplr(patch)

                if randint(1, 2) == 1:  # <-------------- vertical flip : up / down (50% prob)
                    patch = np.flipud(patch)

            patch = cv2.resize(patch, (28, 28))  # INTER_LINEAR interpolation by default
            patches.append(patch)

    return np.asarray(patches)


def binarize_grid(img, mask, grid_size):
    rows, cols = mask.shape
    nrows = rows // grid_size
    ncols = cols // grid_size

    ccols_grid = np.arange(grid_size/2, ncols*grid_size, grid_size).astype(np.uint32)
    crows_grid = np.arange(grid_size/2, nrows*grid_size, grid_size).astype(np.uint32)

    # don't take patches from corners of image
    corners = []
    corners.append([crows_grid[0], ccols_grid[0]])
    corners.append([crows_grid[0], ccols_grid[-1]])
    corners.append([crows_grid[-1], ccols_grid[0]])
    corners.append([crows_grid[-1], ccols_grid[-1]])

    grid = np.zeros(shape=(nrows, ncols), dtype=np.uint8)

    centers1 = []
    centers2 = []

    for r in range(0, nrows):
        for c in range(0, ncols):
            patch_mask = mask[r*grid_size:r*grid_size+grid_size, c*grid_size:c*grid_size+grid_size]
            patch_img = img[r*grid_size:r*grid_size+grid_size, c*grid_size:c*grid_size+grid_size]

            cc = [crows_grid[r], ccols_grid[c]]
            #if cc not in corners:
            if np.count_nonzero(patch_mask) > 0:
                cc_mass = center_of_mass(patch_mask)
                cc_mass = np.asarray(cc_mass).astype(np.uint32)
                grid[r, c] = 1
                centers1.append(cc_mass)
            else:
                if np.count_nonzero(patch_img) > 0:
                    centers2.append(cc)

    if len(centers1) != 0:
        centers1 = np.asarray(centers1)

    if len(centers2) != 0:
        centers2 = np.asarray(centers2)

        # if centers2.shape[0] > 80:  # this was 60
        #     random_idcs = np.random.randint(0, centers2.shape[0], 10)
        #     centers2 = centers2[random_idcs]

    return centers1, centers2


def convert_list_to_array(mylist):
    n_out_patches = len(mylist)
    newlist = []
    for nout in range(0, n_out_patches):
        multipatches = mylist[nout]
        n_in_patches = multipatches.shape[0]
        for nin in range(0, n_in_patches):
            newlist.append(multipatches[nin])
    return np.asarray(newlist)


# all images are 1152 x 1500 pixels
raw_data_path = './raw_data/diaretdb1/'

grid_size = 300
patch_size = 100
augment = True
np.random.seed(10)

if augment:
    saving_path = './data/diaret/'
    n_patches_train = [480, 480]
else:
    saving_path = './data/diaret-noaug/'
    n_patches_train = [40, 36]
n_patches_test = [60, 70]

train_files = pd.read_csv(os.path.join(raw_data_path, 'train_images.txt')).values.tolist()
test_files = pd.read_csv(os.path.join(raw_data_path, 'test_images.txt')).values.tolist()

splits = OrderedDict()
splits['train'] = train_files
splits['test'] = test_files

data = OrderedDict()
data['train'] = OrderedDict()
data['test'] = OrderedDict()

train_patches = []
train_labels = []

test_patches = []
test_labels = []

images_path = os.path.join(raw_data_path, 'ddb1_fundusimages/')
gt_path = os.path.join(raw_data_path, 'ddb1_groundtruth/')
path_to_mask = os.path.join(raw_data_path, 'ddb1_fundusmask/')
fmask = cv2.imread(os.path.join(path_to_mask, 'fmask.tif'))[:, :, 0]

for set_split in splits:
    print(set_split)

    for bname in splits[set_split]:
        patches1 = []
        patches2 = []
        bname = bname[0]
        print(bname)
        # load image and masks
        im = cv2.imread(os.path.join(images_path, bname))  # BGR img

        mask_soft = cv2.imread(os.path.join(gt_path, 'softexudates/', bname))[:, :, 0]
        mask_hard = cv2.imread(os.path.join(gt_path, 'hardexudates/', bname))[:, :, 0]
        mask_hemo = cv2.imread(os.path.join(gt_path, 'hemorrhages/', bname))[:, :, 0]
        mask_red = cv2.imread(os.path.join(gt_path, 'redsmalldots/', bname))[:, :, 0]

        mask1 = cv2.add(mask_soft, mask_hard)  # cv2 add takes care of overflow
        mask2 = cv2.add(mask_hemo, mask_red)
        all_masks = cv2.add(mask1, mask2)

        im, all_masks = preprocess_img(im, fmask, all_masks)  # preprocess image

        centers1, centers2 = binarize_grid(im, all_masks, grid_size)
        if not isinstance(centers1, list):
            if set_split == 'train':
                patches1 = extract_patches(im, centers1, patch_size=patch_size, n_patches=n_patches_train[0],
                                           augment=augment)
            else:
                patches1 = extract_patches(im, centers1, patch_size=patch_size, n_patches=n_patches_test[0])  # test
            labels1 = np.repeat(1, patches1.shape[0]).astype(np.uint8)
        else:
            print('no anomaly in image')

        if not isinstance(centers2, list):
            if set_split == 'train':
                patches2 = extract_patches(im, centers2, patch_size=patch_size, n_patches=n_patches_train[1],
                                           augment=augment)
            else:
                patches2 = extract_patches(im, centers2, patch_size=patch_size, n_patches=n_patches_test[1])  # test

            labels2 = np.repeat(2, patches2.shape[0]).astype(np.uint8)
        else:
            print('no healthy in image')

        pp = np.asarray(patches1)
        # concatenate patches
        if not isinstance(patches1, list):
            if not isinstance(patches2, list):
                patches = np.concatenate((patches1, patches2), axis=0)
                labels = np.concatenate((labels1, labels2), axis=0)
            else:
                patches = patches1
                labels = labels1
        else:
            if not isinstance(patches2, list):
                patches = patches2
                labels = labels2

        if set_split == 'train':
            train_patches.append(patches)
            train_labels.append(labels)
        else:
            test_patches.append(patches)
            test_labels.append(labels)

    # create directory if it doesn't exist
    print('Saving...')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    if set_split == 'train':
        data[set_split]['images'] = convert_list_to_array(train_patches)
        data[set_split]['labels'] = convert_list_to_array(train_labels)

        np.save(saving_path + 'train_images.npy', data[set_split]['images'])
        np.save(saving_path + 'train_labels.npy', data[set_split]['labels'])
        print('Saved train-patches')

    else:
        data[set_split]['images'] = convert_list_to_array(test_patches)
        data[set_split]['labels'] = convert_list_to_array(test_labels)

        np.save(saving_path + 'test_images.npy', data[set_split]['images'])
        np.save(saving_path + 'test_labels.npy', data[set_split]['labels'])
        print('Saved test-patches')
