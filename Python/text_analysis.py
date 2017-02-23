########################################
# 2. Data And Features > Lecture: Feature Representation
# text analysis
#######################################

from sklearn.feature_extraction.text import CountVectorizer

# read in the text
corpus = [
  "Authman ran faster than Harry because he is an athlete.",
  "Authman and Harry ran faster and faster.",
 ]

# tokinise it
bow = CountVectorizer()
X = bow.fit_transform(corpus) # Sparse Matrix

# matrix of words (cols) and appearance cnt                     
print bow.get_feature_names()

print X.toarray()

#############################################
# image analysis
############################################

# Uses the Image module (PIL)
from scipy import misc

# Load the image up
img = misc.imread('E://R/Photos/test.png')

# Is the image too big? Resample it down by an order of magnitude
img = img[::2, ::2]

# Scale colors from (0-255) to (0-1), then reshape to 1D array per pixel, e.g. grayscale
# If you had color images and wanted to preserve all color channels, use .reshape(-1,3)
X = (img / 255.0).reshape(-1)

print X

#  You can create a dataset of images by simply adding them to a regular
# Python list and then converting the whole thing in one shot:

# Uses the Image module (PIL)
from scipy import misc

# Load the image up
dset = []
for fname in files:
  img = misc.imread(fname)
  dset.append(  (img[::2, ::2] / 255.0).reshape(-1)  )

dset = pd.DataFrame( dset )

#################################################
# audio
# make sure at same sample rate or scaling issues
#################################################

import scipy.io.wavfile as wavfile

sample_rate, audio_data = wavfile.read('sound.wav')
print audio_data

# To-Do: Machine Learning with audio_data!
#