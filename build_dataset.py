# Build dataset
import numpy as np
from scipy.misc import imresize
from util import *
import random
import os
from tqdm import trange
import argparse

data_dir = './data/raw_GTSDB/'
output_dir = './data/GTSDB/'
image_resize = 224
num_grid = 7

train_size = 700
val_size = 100
test_size = 100

raw_data = np.loadtxt(data_dir + 'gt.txt', delimiter = ';', dtype= str)
image_names = raw_data[:, 0]
box_coords = raw_data[:, 1:5].astype(float)

X = []
Y = []
conflict_count = np.zeros(900)

for i in trange(900):
	# Load and resize ith image
	name = get_image_name(i)
	image = plt.imread(data_dir + name)
	resized_image = imresize(image, (image_resize, image_resize))
	X.append(resized_image)

	# Load bounding boxes
	y = np.zeros((num_grid, num_grid, 5))
	orig_hw = image.shape[0:2]
	resized_hw = resized_image.shape[0:2]
	indices = np.argwhere(image_names == name).reshape(-1,)
	
	for index in indices:
		box_xy = box_coords[index]
		resized_box_xy = resize_box_xy(orig_hw, resized_hw, box_xy)
		box_cwh = xy_to_cwh(resized_box_xy)
		normalized_cwh, position = normalize_box_cwh(resized_hw, num_grid, box_cwh)
		row, col = position
		xc, yc, w, h = normalized_cwh
		if y[row, col, 0] == 1:
			conflict_count[i] += 1
		y[row, col, :] = [1, xc, yc, w, h]
	Y.append(y)

X = np.array(X)
Y = np.array(Y)

permutation = np.random.permutation(900)
train = permutation[0:train_size]
val = permutation[train_size : train_size + val_size]
test = permutation[train_size + val_size:train_size + val_size + test_size]

X_train = X[train]
X_val = X[val]
X_test = X[test]
Y_train = Y[train]
Y_val = Y[val]
Y_test = Y[test]

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

np.save(output_dir + 'X_train', X_train)
np.save(output_dir + 'Y_train', Y_train)
np.save(output_dir + 'X_val', X_val)
np.save(output_dir + 'Y_val', Y_val)
np.save(output_dir + 'X_test', X_test)
np.save(output_dir + 'Y_test', Y_test)

train_conflict = conflict_count[train]
train_conflict_images = train[train_conflict == 1]
train_conflict_indices = np.argwhere(train_conflict == 1)

np.savetxt(output_dir + 'train_conflict_indices.txt', train_conflict_indices, fmt='%.18g')
np.savetxt(output_dir + 'train_images.txt', train, fmt='%.18g')
np.savetxt(output_dir + 'train_conflict_images.txt', train_conflict_images, fmt='%.18g')

print('Build dataset done.')
print('Train shape:', X_train.shape, Y_train.shape)
print('Val shape:', X_val.shape, Y_val.shape)
print('Test shape:', X_test.shape, Y_test.shape)
print('Conflict count:', np.sum(conflict_count))