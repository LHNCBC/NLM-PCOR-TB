# Global Constants File

# Image dimensions
IMAGE_SIZE = 256  # IMAGE DIMENSION (Assumed equal height/width)
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE

SEGMENTATION_INPUT_SIZE = 256, 256
SEGMENTATION_BACKBONE = 'resnet50'

# Cross validation folds
TOTAL_FOLDS = 5  # For n-fold cross validation experiment

# Number of classes
NUM_CLASSES: int = 1  # 1 for BCE for 2 classes

# Augmentation Parameters
AUG_FACTOR = 2  # Number of times to increase data by transformations
MAX_ROTATION_DEGREES = 10  # Maximum rotation for augmentation
MAX_TRANSLATION_PIXELS = 5  # Maximum translation pixels
INTENSITY_CHANGE_PERCENT = .25  # .25 means new intensity is between .75*original and 1.25 original)
NOISE_FACTOR = 1.25  # Maximum sigma of gaussian noise to add to the image
RESIZE_MIN = -5
RESIZE_MAX = 10

MODEL_NAME = 'inceptionv3'

VALIDATION_RATIO = .075   # Uses a % of training set. validation set is mostly for early stopping.