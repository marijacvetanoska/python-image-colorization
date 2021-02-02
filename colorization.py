import numpy as np
import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
args = parser.parse_args()

# Specify the paths for the 2 model file and cluster center points
# caffemodel: Path to the model weights trained in Caffe.
# prototxt: Caffe specific file which defines the network.
# kernel: Path to cluster center points stored in numpy format.
prototxtFile = "./model/colorization_deploy_v2.prototxt"
caffeModel = "./model/colorization_release_v2.caffemodel"
kernel = "pts_in_hull.npy"

# Load the pre-trained network
net = cv.dnn.readNetFromCaffe(prototxtFile, caffeModel)

# Load cluster centers
pts_in_hull = np.load(kernel)

# Add the cluster centers as 1x1 convolutions to the model
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Read the input image in BGR format
image = cv.imread(args.image)
# Convert to RGB
image = image[:,:,[2, 1, 0]]
# Scale the pixel intensities to range [0, 1]
img_rgb = (image * 1.0 / 255).astype(np.float32)
# Convert the image to Lab color space
img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
# Take the L channel
img_L = img_lab[:, :, 0]

# Original image size
(H_orig, W_orig) = image.shape[:2]

# Resize the image to 224x224 (the dimensions the colorization network accepts)
img_resized = cv.resize(img_rgb, (224, 224))
img_lab_resized = cv.cvtColor(img_resized, cv.COLOR_RGB2Lab)
# Extract the L channel
img_L_resized = img_lab_resized[:, :, 0]

# Subtract 50 for mean-centering
img_L_resized -= 50

# Pass the L channel through the network which will predict the 'a' and 'b' channel values
net.setInput(cv.dnn.blobFromImage(img_L_resized))
ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0)) # these are the predicted a and b channels
(H_out, W_out) = ab_channels.shape[:2]

# Resize the predicted 'ab' image to the same dimensions as our input image
ab_channels_orig = cv.resize(ab_channels, (W_orig, H_orig))

# concatenate with original image i.e. L channel
colorized = np.concatenate((img_L[:, :, np.newaxis], ab_channels_orig), axis=2)

# Convert the output image from the Lab color space to RGB
colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
# Clip any values that fall outside the range [0, 1] and rescale to 0-255
colorized = 255 * np.clip(colorized, 0, 1)
colorized = np.uint8(colorized)

# Concatenate input and colorized image to display
concatenated = np.hstack([image, colorized])
cv.imwrite("colorized_" + args.image, concatenated)
