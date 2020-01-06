from PIL import Image
import cv2
import argparse
import math
import progressbar
from pointillism import *
from collections import OrderedDict
import numpy as np
import dlib
import imutils

def black_and_white(input_image_path,
                    output_image_path):
	color_image = Image.open(input_image_path)
	bw = color_image.convert('L')
	bw.save(output_image_path)
	bw.show()

def make_sepia_palette(color):
	palette = []
	r, g, b = color
	for i in range(255):
		palette.extend(((r * i) / 255, (g * i) / 255, (b * i) / 255))

	return palette

def create_sepia(input_image_path, output_image_path):
	whitish = (255, 240, 192)
	sepia = make_sepia_palette(whitish)
	sepia = [round(x) for x in sepia]

	color_image = Image.open(input_image_path)

	# convert our image to gray scale
	bw = color_image.convert('L')

	# add the sepia toning
	bw.putpalette(sepia)

	# convert to RGB for easier saving
	sepia_image = bw.convert('RGB')

	sepia_image.save(output_image_path)
	sepia_image.show()

def create_cartoon(input_image_path, output_image_path):
	num_down = 2       # number of downsampling steps
	num_bilateral = 7  # number of bilateral filtering steps

	img_rgb = cv2.imread(input_image_path)

	# downsample image using Gaussian pyramid
	img_color = img_rgb
	for _ in range(num_down):
		img_color = cv2.pyrDown(img_color)

	# repeatedly apply small bilateral filter instead of
	# applying one large filter
	for _ in range(num_bilateral):
		img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

	# upsample image to original size
	for _ in range(num_down):
		img_color = cv2.pyrUp(img_color)

	# convert to grayscale and apply median blur
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	img_blur = cv2.medianBlur(img_gray, 7)

	# detect and enhance edges
	img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

	# convert back to color, bit-AND with color image
	img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

	if (np.shape(img_color) > np.shape(img_edge)):
		img_color = cv2.resize(img_color, (np.shape(img_edge)[1], np.shape(img_edge)[0]))
	else:
		img_edge = cv2.resize(img_edge, (np.shape(img_color)[1], np.shape(img_color)[0]))

	img_cartoon = cv2.bitwise_and(img_color, img_edge)

	cv2.imwrite(output_image_path, img_cartoon)

def create_drawing(input_image, output_image, mode):
	palette_size_value = 20 #"Number of colors of the base palette"
	stroke_scale_value = 0 #"Scale of the brush strokes (0 = automatic)"
	gradient_smoothing_radius_value = 0 #"Radius of the smooth filter applied to the gradient (0 = automatic)"
	limit_image_size_value = 0 #"Limit the image size (0 = no limits)"
	img_path_value = input_image

	res_path = output_image
	img = cv2.imread(img_path_value)

	if limit_image_size_value > 0:
		img = limit_size(img, limit_image_size_value)

	if stroke_scale_value == 0:
		stroke_scale = int(math.ceil(max(img.shape) / 1000))
		print("Automatically chosen stroke scale: %d" % stroke_scale)
	else:
		stroke_scale = stroke_scale_value

	if gradient_smoothing_radius_value == 0:
		gradient_smoothing_radius = int(round(max(img.shape) / 50))
		print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
	else:
		gradient_smoothing_radius = gradient_smoothing_radius_value

	# convert the image to grayscale to compute the gradient
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	print("Computing color palette...")
	palette = ColorPalette.from_image(img, palette_size_value)

	print("Extending color palette...")
	palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

	# display the color palette
	#cv2.imshow("palette", palette.to_image())
	cv2.waitKey(200)

	print("Computing gradient...")
	gradient = VectorField.from_gradient(gray)

	print("Smoothing gradient...")
	gradient.smooth(gradient_smoothing_radius)

	print("Drawing image...")
	# create a "cartonized" version of the image to use as a base for the painting
	res = cv2.medianBlur(img, 11)
	# define a randomized grid of locations for the brush strokes
	grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
	batch_size = 10000

	bar = progressbar.ProgressBar()
	for h in bar(range(0, len(grid), batch_size)):
		# get the pixel colors at each point of the grid
		pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
		# precompute the probabilities for each color in the palette
		# lower values of k means more randomnes
		color_probabilities = compute_color_probabilities(pixels, palette, k=9)

		for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
			color = color_select(color_probabilities[i], palette)
			angle = math.degrees(gradient.direction(y, x)) + 90
			length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

			# draw the brush stroke
			cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

	if (mode == 0):
		cv2.imshow("res", limit_size(res, 1080))
	cv2.imwrite(res_path, res)
	cv2.waitKey(0)
    
def create_sketch(input_image, output_image, mode):
    create_cartoon(input_image, output_image)
    create_drawing(output_image, output_image, mode)