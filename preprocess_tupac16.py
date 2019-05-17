import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2
import numpy as np
from random import randint
from collections import OrderedDict
from stain_normalization import stainNorm_Vahadane

def find_grid(centers):
    grid = np.ones(shape=(4, 4), dtype=np.uint8)
    ccols_grid = [250, 750, 1250, 1750]
    crows_grid = [250, 750, 1250, 1750]

    for m in range(centers.shape[0]):  # number of mitotic figures
        crows = centers[m, 0]
        ccols = centers[m, 1]

        idxrows = np.argmin(np.abs(crows-crows_grid))
        idxcols = np.argmin(np.abs(ccols-ccols_grid))

        grid[idxrows, idxcols] = 0

    grid_centers = []  # center of grid where there are NO mitotic figures
    n_centers = 3
    for row in range(0, 4):
        for col in range(0, 4):
            if grid[row, col] == 1:
                grid_centers.append([crows_grid[row], ccols_grid[col]])
    grid_centers = np.asarray(grid_centers)
    random_idcs = np.random.randint(0, grid_centers.shape[0], n_centers)
    return grid_centers[random_idcs]


def extract_patches(img, centers, augment=False):
    n_patches = 30
    psize = 50

    n_mitoses = centers.shape[0]
    n_patches_per_mitosis = n_patches // n_mitoses
    if n_patches_per_mitosis == 0:
        n_patches_per_mitosis = 1

    tx_max = 30
    tx_min = -30
    step_size = 2*(tx_max - tx_min) / (n_patches_per_mitosis)
    tx_range = np.arange(tx_min, tx_max+1, step_size).astype(np.int8)
    ty_range = np.arange(tx_min, tx_max+1, step_size).astype(np.int8)

    patches = []

    # extract mitotic patches
    for m in range(0, n_mitoses):
        cx = centers[m, 0]
        cy = centers[m, 1]

        if cx < 80:
            tx_max = 10
            tx_min = -10
            if cx < 60:
                tx_max = 4
                tx_min = -4
            if cx < 55:
                tx_max = 3
                tx_min = -3
            if cx-psize < 0:
                cx = 50
                tx_max = 5
                tx_min = 0
            step_size = 2 * (tx_max - tx_min) / (n_patches_per_mitosis)
            tx_range = np.arange(tx_min, tx_max + 1, step_size).astype(np.int8)

        if cy < 80:
            tx_max = 10
            tx_min = -10
            if cy < 60:
                tx_max = 10
                tx_min = 0
            if cy-psize < 0:
                cy = 50
                tx_max = 5
                tx_min = 0
            step_size = 2 * (tx_max - tx_min) / (n_patches_per_mitosis)
            ty_range = np.arange(tx_min, tx_max + 1, step_size).astype(np.int8)

        tx_range = np.unique(tx_range)
        ty_range = np.unique(ty_range)

        # translations in x-axis
        for offx in tx_range:
            offy = 0
            if (cx-psize+offx) < 0:
                print('problem')
            if (cy-psize+offy) < 0:
                print('problem')

            patch = img[cx - psize + offx:cx + psize + offx, cy - psize + offy:cy + psize + offy, ...]
            if augment:
                if randint(1, 2) == 1:  # <------------- horizontal flip : left / right (50% prob)
                    patch = np.fliplr(patch)

                if randint(1, 2) == 1:  # <-------------- vertical flip : up / down (50% prob)
                    patch = np.flipud(patch)
            patch = cv2.resize(patch, (28, 28))
            patches.append(patch)

        # translations in y-axis
        for offy in ty_range:
            offx = 0
            if (cx-psize+offx) < 0:
                print('problem')
            if (cy-psize+offy) < 0:
                print('problem')

            patch = img[cx - psize + offx:cx + psize + offx, cy - psize + offy:cy + psize + offy, ...]
            if augment:
                if randint(1, 2) == 1:  # <------------- horizontal flip : left / right (50% prob)
                    patch = np.fliplr(patch)

                if randint(1, 2) == 1:  # <-------------- vertical flip : up / down (50% prob)
                    patch = np.flipud(patch)
            patch = cv2.resize(patch, (28, 28))
            patches.append(patch)
    return np.asarray(patches)


def convert_list_to_array(my_list):
    n_out_patches = len(my_list)
    new_list = []
    for nout in range(0, n_out_patches):
        multipatches = my_list[nout]
        n_in_patches = multipatches.shape[0]
        for nin in range(0, n_in_patches):
            new_list.append(multipatches[nin])
    return np.asarray(new_list)


# Split the slides from the dataset into: train, validation and test
raw_data_path = './raw_data/tupac16/'
augment = False
if augment:
    saving_path = './data/tupac16/'
else:
    saving_path = './data/tupac16-noaug'
chs = 1

df = pd.read_csv(os.path.join(raw_data_path, 'train_slides.csv'), dtype=object).values.tolist()
train_slides = [df[k][0] for k in range(0, len(df))]

df = pd.read_csv(os.path.join(raw_data_path, 'val_slides.csv'), dtype=object).values.tolist()
val_slides = [df[k][0] for k in range(0, len(df))]

df = pd.read_csv(os.path.join(raw_data_path, 'test_slides.csv'),  dtype=object).values.tolist()
test_slides = [df[k][0] for k in range(0, len(df))]

splits = OrderedDict()
splits['test'] = test_slides
splits['val'] = val_slides
splits['train'] = train_slides

data = OrderedDict()
data['test'] = OrderedDict()
data['val'] = OrderedDict()
data['train'] = OrderedDict()

# Stain normalization
n = stainNorm_Vahadane.normalizer()
np.random.seed(10)

# Load images and extract patches
for set_split in splits.keys():
    slides = splits[set_split]
    patches = []
    labels = []

    print(set_split)
    for slide in slides:
        print('Slide: {} Number of patches: {}' .format(slide, len(slides)))
        slide_path = os.path.join(raw_data_path, slide)
        filenames = os.listdir(slide_path)
        for name in filenames:
            if name.endswith('.tif'):
                # get filename
                bname = name.split('.tif')[0]
                # load image
                img = cv2.imread(os.path.join(slide_path, name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # swap channels because of cv2
                # stain normalization
                n.fit(img)
                normalized = n.transform(img)
                # normalize image to range 0-1
                img = cv2.normalize(normalized.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
                hemo = n.hematoxylin(img).astype(np.float32)  # keep hematoxylin channel

                # get ground truth
                centers_list = []
                with open(os.path.join(slide_path, bname + '.csv')) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        centers_list.append(row)
                centers = np.asarray(centers_list).astype(np.uint32)

                # extract mitotic figures patches
                mitotic_patches = extract_patches(hemo, centers, augment=augment)
                n_mitotic = mitotic_patches.shape[0]

                # extract NO-mitotic figures patches
                new_centers = find_grid(centers)
                nomitotic_patches = extract_patches(hemo, new_centers, augment=augment)
                n_nomitotic = nomitotic_patches.shape[0]

                patches.append(mitotic_patches)
                labels.append(np.repeat(1, n_mitotic).astype(np.uint8))  # mitosis: 1
                patches.append(nomitotic_patches)
                labels.append(np.repeat(2, n_nomitotic).astype(np.uint8))  # no-mitosis: 2

    print('Saving...')
    data[set_split]['images'] = convert_list_to_array(patches)
    data[set_split]['labels'] = convert_list_to_array(labels)

    # create directory if it doesn't exist
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # save data into different variables
    if set_split == 'train':
        np.save(saving_path+'train_images.npy', data[set_split]['images'])
        np.save(saving_path+'train_labels.npy', data[set_split]['labels'])
        mean_value = np.mean(np.mean(data[set_split]['images']))
        np.save(saving_path+'mean_value.npy', mean_value)
        print('Saved train-patches')

    elif set_split == 'val':
        np.save(saving_path+'val_images.npy', data[set_split]['images'])
        np.save(saving_path+'val_labels.npy', data[set_split]['labels'])
        print('Saved validation-patches')

    else:
        np.save(saving_path+'test_images.npy', data[set_split]['images'])
        np.save(saving_path+'test_labels.npy', data[set_split]['labels'])
        print('Saved test-patches')
