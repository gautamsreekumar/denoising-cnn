import numpy as np
import scipy.misc as sp

def load(batch_size, file_list, stddev=1.0):
	assert batch_size == len(file_list), "[! ASSERTION] Length of file_list not equal to batch_size"
	img_x = []
	img_y = []
	max_val = np.load('../datasets/urban_and_natural_images/max_val.npy')
	min_val = np.load('../datasets/urban_and_natural_images/min_val.npy')
	for i in range(batch_size):
		temp_img = sp.imread(file_list[i])
		temp_noisy = temp_img+np.random.normal(scale=stddev, size=temp_img.shape)
		temp_img = (temp_img-min_val)/(max_val-min_val)
		temp_noisy = (temp_noisy-min_val)/(max_val-min_val)
		img_x.append(temp_noisy)
		img_y.append(temp_img)
	return img_x, img_y