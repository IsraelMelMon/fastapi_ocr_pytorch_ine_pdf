import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse






def clustering(image):

	# convert to RGB
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# reshape the image to a 2D array of pixels and 3 color values (RGB)
	pixel_values = image.reshape((-1, 3))
	# convert to float
	pixel_values = np.float32(pixel_values)

	#print(pixel_values.shape)

	# define stopping criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

	# number of clusters (K)
	k = 2
	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# convert back to 8 bit values
	centers = np.uint8(centers)

	# flatten the labels array
	labels = labels.flatten()

	# convert all pixels to the color of the centroids
	segmented_image = centers[labels.flatten()]


	# reshape back to the original image dimension
	segmented_image = segmented_image.reshape(image.shape)
	#cv2.imshow("gu", segmented_image)
	#cv2.waitKey(0)
	# show the image
	return segmented_image
	"""#cv2.imwrite("output.png", segmented_image)
	#plt.imshow(segmented_image)
	#plt.show()

	# disable only the cluster number 2 (turn the pixel into black)
	masked_image = np.copy(image)
	# convert to the shape of a vector of pixel values
	masked_image = masked_image.reshape((-1, 3))
	# color (i.e cluster) to disable
	cluster = 2
	masked_image[labels == cluster] = [0, 0, 0]
	# convert back to original shape
	masked_image = masked_image.reshape(image.shape)
	# show the image
	#plt.imshow(masked_image)
	#plt.show()
	"""

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image to be OCR'd")
	#ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	#	help="type of preprocessing to be done")
	args = vars(ap.parse_args())
