# import keras
import keras

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

PATH_TO_DATASET = input()
PATH_TO_MODEL = 'resnet50_final.h5'

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

def create_mask_file(draw, mask, b, prefix):
	# resize to fit the box
	mask = mask.astype(np.float32)
	mask = cv2.resize(mask, (b[2] - b[0], b[3] - b[1]))
	# binarize the mask
	binarize_threshold = 0.5
	mask = (mask > binarize_threshold).astype(np.uint8)
	# apply color to the mask and border
	mask = (np.stack([mask] * 3, axis=2) * (255, 255, 255)).astype(np.uint8)
	# draw the mask
	indices = np.where(mask == [0, 0, 0])
	local_draw = draw[b[1]:b[3], b[0]:b[2]].copy()
	local_draw[indices[0], indices[1], :] = [255, 255, 255]
	# save
	file_output_path = prefix + '_mask.JPG'
	cv2.imwrite(file_output_path, cv2.cvtColor(local_draw, cv2.COLOR_RGB2BGR))

def create_box_file(draw, b, prefix):
	# create croped draw
	local_draw = draw[b[1]:b[3], b[0]:b[2]].copy()
	# save
	file_output_path = prefix + '_box.JPG'
	cv2.imwrite(file_output_path, cv2.cvtColor(local_draw, cv2.COLOR_RGB2BGR))

def create_info_file(sign_name, score, prefix):
    file_path = prefix + '_info.json'
    f = open(file_path, 'w')
    f.write('[{\n')
    f.write('  "SignName": "'+sign_name+'",\n')
    f.write('  "Score": '+'{:.3f}'.format(score)+'\n')
    f.write('}]\n')
    f.close()

	

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join(PATH_TO_MODEL)

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# load label to names mapping
labels_to_names = {0: 'B3', 1: 'C8', 2: 'C12', 3: 'C13', 4: 'C18', 5: 'E16b', 6: 'E16c', 7: 'A16', 8: 'IP7', 9: 'E12', 10: 'C24a', 11: 'C24b', 12: 'B11', 13: 'IS40', 14: 'C16', 15: 'E16d', 16: 'E16a'}

# open temporary file for storing info about names of created files
tmp_file = open(PATH_TO_DATASET+'/tmp.txt', 'w')

# create the list of all input files (all JPG files in directory PATH_TO_DATASET/inputs)
input_path = PATH_TO_DATASET+'/inputs/*.JPG'
path_list = glob.glob(input_path)
path_list.sort()

for file_input_path in path_list:
	#create file name without prefix (PATH_TO_DATASET/inputs/) and extension (.JPG)
	file_name = file_input_path[len(PATH_TO_DATASET)+8:-4]

	try:
		try:	
			# load 
			image = read_image_bgr(file_input_path)
	
			# copy to draw on
			draw = image.copy()
			draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	
			# preprocess image for network
			image = preprocess_image(image)
			image, scale = resize_image(image)
	
			# process image
			start = time.time()
			outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
			print("Processing file:", file_name, "time:", time.time() - start)

			boxes  = outputs[-4][0]
			scores = outputs[-3][0]
			labels = outputs[-2][0]
			masks  = outputs[-1][0]

			# correct for image scale
			boxes /= scale
	   
			# process detections
			counter = 0
			for box, score, label, mask in zip(boxes, scores, labels, masks):
				if score < 0.5:
					break
				b = box.astype(int)
				mask = mask[:, :, label]
				output_prefix = PATH_TO_DATASET + '/outputs/' + file_name + '_' + str(counter)
				create_mask_file(draw, mask, b, output_prefix)
				create_box_file(draw, b, output_prefix)
				sign_name = labels_to_names[label]
				create_info_file(sign_name, score, output_prefix)
				tmp_file.write(file_name + ' ' + str(counter) + '\n')
				counter += 1
		except ValueError:
			print("Oops, there has been a problem with:", file_name)
	except OSError:
		print("Oops, there has been a problem with:", file_name) 

# close the tmp_file
tmp_file.close() 
