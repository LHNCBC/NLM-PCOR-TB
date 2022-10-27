import os
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from skimage import exposure
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from global_constants import *

# def balance(*args):
#     """
#     Creates equal number of samples based on their class
#     Samples from the larger class are excluded.
#     NEW number of samples <= OLD number of samples
#     Input :
#       args[0] : (array) should always be (R/S) Labels, N X 1 (Number of samples X label)
#       args[1-n] : (arrays) can be X-ray images, masks, bounding boxes co-ordinates etc. that correspond to that patient
#     Note: First axis represents number of samples (patients)
#     Returns :
#       balanced: (arrays) same arguments as input, however equal samples
#               from each class are present and extra samples are excluded
#       extra: (arrays) same arguments as input, includes the data not included in the balanced set from the larger class
#     """
#     Y = args[0]
#     balanced = []
#     extra = []
#     for group in args:
#         r = group[Y == 1]
#         s = group[Y == 0]
#         if len(r) > len(s):
#             e = r[len(s):]
#             r = r[:len(s)]
#         else:
#             e = s[len(r):]
#             s = s[:len(r)]
#         group = np.concatenate((r, s), axis=0)
#         balanced.append(group)
#         extra.append(e)
#     return balanced, extra

def balance(Y,X,random_seed):
    """
    Creates equal number of samples based on their class
    Samples from the larger class are excluded.
    Input :
      Y(np.ndarray): should always be (1/0) Labels, N X 1 (Number of samples X label)
      X(np.ndarray): preprocessed chest x-ray image array
      random_seed(int): Random seed
    Returns :
      Y_balanced(np.ndarray):  Equal samples from each class are present in this balanced labels.
      X_balanced(np.ndarray):  Preprocessed chest x ray image array is filtered to contain
                  balanced array
    """

    X_R = X[Y == 1]
    X_S = X[Y == 0]

    np.random.seed(random_seed)
    if len(X_R) > len(X_S):
        idxs = np.random.choice(range(len(X_R)),len(X_S))
        X_R_balanced = X_R[idxs]
        Y_R_balanced = np.ones(len(X_R_balanced))
        Y_S = np.zeros(len(X_S))
        X_balanced = np.concatenate([X_R_balanced,X_S])
        Y_balanced = np.concatenate([Y_R_balanced,Y_S])
    else:
        idxs = np.random.choice(range(len(X_S)),len(X_R))
        X_S_balanced = X_S[idxs]
        Y_S_balanced = np.zeros(len(X_S_balanced))
        Y_R = np.ones(len(X_R))
        X_balanced = np.concatenate([X_R,X_S_balanced])
        Y_balanced = np.concatenate([Y_R,Y_S_balanced])

    return Y_balanced,X_balanced


def hist_equal(X):
    """
    Min-max normalize input x-ray images array and perform adaptive histogram equalization
    Args:
        X(np.ndarray): preprocessed chest x-ray image array
    Returns:
      X(np.ndarray): Equalized images array
    """
    X = X - np.min(X)
    X = X / np.max(X)
    for k in range(X.shape[0]):
        X[k, ...] = exposure.equalize_adapthist(X[k, ...])
    return X


# Min-max normalize input x-ray images array and perform adaptive histogram equalization
# Input Params:
#    X: Images array
#    mean_X: (int or array), if int, mean of the input images is computed.
# Returns:
#   X: Modified images array
def hist_match(X, mean_X):
    if isinstance(mean_X, np.int):
        mean_X = np.mean(X, axis=0)
    for k in range(X.shape[0]):
        X[k, ...] = exposure.match_histograms(X[k, ...], mean_X)
    return X


# parse script parameters
# def parse_args():
#     parser = argparse.ArgumentParser(description="Drug Resistance Detection.")
#     parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
#     parser.add_argument('--country', type=str, default='all', help='Country')
#     parser.add_argument('--custom', type=int, default=0, help='Custom or Pretrained')
#     parser.add_argument('--cv', type=int, default=0, help='Cross Validation Set')
#     parser.add_argument('--data_path', type=str, default='../data/all_data256.h5', help='HDF5 data file path')
#     parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
#     parser.add_argument('--exclude', type=int, default=0,
#                         help='to exclude specified country [1] or only include that country [0]')
#     parser.add_argument('--exp_desc', type=str, default='FIRST',
#                         help='# Experiment Suffix (string) description to help user'
#                              ' track of the experiments and corresponding models')
#     parser.add_argument('--gpu_id', type=str, default='4', help='GPU ID')
#     parser.add_argument('--image_path', type=str, default='')
#     parser.add_argument('--lr', type=float, default=.001, help='Learning Rate')
#     parser.add_argument('--model_path', type=str, default='../dr_ds_generalization/weights/sample_trained_model.h5', help='trained model for inference')
#     parser.add_argument('--seg_path', type=str, default='../lung_segmentation/weights/Segmentation_resnet50_UNet.h5',
#                         help='trained model for segmentation')
#     parser.add_argument('--mode', type=str, default='train', help='train/test/inference')
#     return parser.parse_args()



def sextant_masks(bb, S, shape):
    """
    Convert sextant labels to masks of size of image
    These are still rectangular blocks. (Lungs mask is multiplied later)
    Input:
    bb(np.ndarray): N x 8 bounding boxes measurments [computed for each image separtely],
                  sextant labels and size of images
    S(np.ndarray): Sextant labels N x 6 indicates if   abnromaties are present in the 6 sextant (0: no, 1: yes)\
    shape(tuple): shape of the images array
    Return:
    S_mask(np.ndarray): sextant masks based on sextant labels
    """
    bb = np.int32(bb)
    l_y = bb[:, 1]
    l_h = bb[:, 3]
    l_w = bb[:, 2]
    r_y = bb[:, 5]
    r_x = bb[:, 4]
    r_h = bb[:, 7]
    S_mask = np.zeros(shape)
    for k in range(shape[0]):
        # Creating sextant masks, assigning 1 or 0 to entire sextant ; based on radiologists annotation
        S_mask[k, l_y[k]:l_y[k] + l_h[k] // 3, :l_w[k]] = S[k, 0]
        S_mask[k, r_y[k]:r_y[k] + r_h[k] // 3, r_x[k]:] = S[k, 1]
        S_mask[k, l_y[k] + l_h[k] // 3:l_y[k] + 2 * l_h[k] // 3, :l_w[k]] = S[k, 2]
        S_mask[k, r_y[k] + r_h[k] // 3:r_y[k] + 2 * r_h[k] // 3, r_x[k]:] = S[k, 3]
        S_mask[k, l_y[k] + 2 * l_h[k] // 3:l_y[k] + l_h[k], :l_w[k]] = S[k, 4]
        S_mask[k, r_y[k] + 2 * r_h[k] // 3:r_y[k] + r_h[k], r_x[k]:] = S[k, 5]
    return S_mask


#   Stratified Cross validation fold ids generation
# Input params:
#   num_cv : (int) total folds,
#   cv : (int) fold number (0,.. num_cv-1),
#   X: (array) images array (e.g. N X image_width X image_height)
#   Y : (array) labels (N x 1)
# Returns:
#   train_idx: ids of training samples
#   test_idx: ids of testing samples
def cross_val_idx_strat(num_cv, cv, X, Y):
    kf = StratifiedKFold(n_splits=num_cv)
    tr = [[] for _ in range(num_cv)]
    ts = [[] for _ in range(num_cv)]
    i = 0
    for train_idx, test_idx in kf.split(X, Y):
        tr[i] = train_idx
        ts[i] = test_idx
        i += 1
    train_idx = tr[cv]
    test_idx = ts[cv]
    return train_idx, test_idx


def cross_val_idx(num_cv, cv, X):
    """
    Cross validation fold generation
    Args:
      num_cv : (int) total folds,
      cv : (int) fold number (0,.. num_cv-1),
      X: (array) images array (e.g. N X image_width X image_height)
      Y : (array) labels (N x 1)
    Returns:
      train_idx: ids of training samples
      test_idx: ids of testing samples
    """
    kf = KFold(n_splits=num_cv)  # 10 = 10 folds/splits
    tr = [[] for _ in range(num_cv)]
    ts = [[] for _ in range(num_cv)]
    i = 0
    for train_idx, test_idx in kf.split(X):  # np.argmax(Y,axis=1)):
        tr[i] = train_idx
        ts[i] = test_idx
        i += 1
    train_idx = tr[cv]
    test_idx = ts[cv]
    return train_idx, test_idx
