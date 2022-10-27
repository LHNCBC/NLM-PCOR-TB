import SimpleITK as sitk
import cv2
import numpy as np
import pydicom
from tensorflow.keras.models import load_model


def _srgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]
    channels = [sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32) for i in
                range(image.GetNumberOfComponentsPerPixel())]
    # linear mapping
    I = 1 / 255.0 * (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
    # nonlinear gamma correction
    I = I * sitk.Cast(I <= 0.0031308, sitk.sitkFloat32) * 12.92 + I ** (1 / 2.4) * sitk.Cast(I > 0.0031308,
                                                                                             sitk.sitkFloat32) * 1.055 - 0.55
    return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)


def _normalize_image(image):
    '''
    Normalizes image so that image array has an overall mean = 0, std deviation=1
    Args:
        image(SimpleITK.Image): Image to normalize. Pixel type must be sitkFloat32 or
                                sitkFloat64.
    Returns:
        SimpleITK.Image: Normalized, zero mean unit standard deviation, image.
        '''
    img = sitk.GetArrayViewFromImage(image)
    return (image - np.mean(img)) / (np.std(img))


def _resample_cxr_for_lung_segmentation_cnn(new_size, gaussian_sigma, file):
    """
    Downsample the input image to the given new_size. To avoid aliasing artifacts
    you may want to blur the image before the downsampling operation. This is important
    if your image contains high frequency data.
    Args:
        new_size: The size of the resampled image in pixels.
        gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard deviation(s)
               before resampling.
        file (str): File path to image we want to resample.
    Returns:
        Tuple (SimpleITK.Image, SimpleITK.Image): Original image and it's resampled image
        """
    try:
        original_image = sitk.ReadImage(file)
    except:
        ds = pydicom.dcmread(file)
        original_image = sitk.GetImageFromArray(ds.pixel_array, isVector=(len(ds.pixel_array.shape) == 3))
        # Some images have a 3rd dimension of size 1, get rid of it.
    if original_image.GetDimension() != 2 and original_image.GetSize()[2] == 1:
        original_image = original_image[:, :, 0]

    img0 = sitk.GetArrayFromImage(original_image)
    if len(img0.shape) == 3:
        img0 = img0[:, :, 0]
    img0 = img0 / img0.max()
    gray = 255 * (img0 < 1).astype(np.uint8)  # To invert the xray
    coords = cv2.findNonZero(gray)  # Find all non-zero points
    _, _, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    original_image = original_image[:w, :h]

    # Some images are grayscale but the channel is repeated three times (gray RGB image).
    if original_image.GetNumberOfComponentsPerPixel() > 1:
        original_image = _srgb2gray(original_image)
    new_spacing = [sz * spc / nsz for nsz, sz, spc in
                   zip(new_size, original_image.GetSize(), original_image.GetSpacing())]
    smoothed_image = sitk.SmoothingRecursiveGaussian(original_image, gaussian_sigma)
    # smoothed_image= original_image
    resampled_for_seg = sitk.Resample(smoothed_image, new_size, sitk.Transform(), sitk.sitkLinear,
                                      original_image.GetOrigin(), new_spacing, original_image.GetDirection(),
                                      0, sitk.sitkFloat32)
    return original_image, resampled_for_seg


def _predict_mask(resampled_image_arr, model_path, batch_size=10):
    """
    Predict the lung mask from the resampled image array. This uses UNet
    (pretrained model: https://github.com/imlab-uiip/lung-segmentation-2d) to segment
    lungs for a given image with size equal to model input size.

    Args:
        resampled_image_arr (numpy array): Numpy array obtained from resampled images
                                           to provide input for segmentation network in the
                                           shape of (num_images,segmentation_input_size_x,
                                           segmentation_input_size_y)
        model_path: File path of the trained model(UNet)
        batch_size: Batch size for the model. Performing inference in batch mode is faster than
                    image by image.
    Returns:
        Numpy array: Prediction masks of segmented lungs with same size as input array.
    """
    UNet = load_model(model_path)
    pred = UNet.predict(resampled_image_arr, batch_size=batch_size, verbose=0).reshape(resampled_image_arr.shape[0:3])
    pr = pred > 0.5
    return pr.astype(int)


def _segmented_lung_2_tb_cnn(new_size, resampled_image, np_lung_segmentation_mask,
                             original_image, gaussian_sigma):
    """
    Resample a subregion of a given image based on a mask.
    Args:
       original_image (SimpleITK.Image): Original Image
       resampled_image (SimpleITK.Image): Image that was resampled to create sitk_lung_segmentation_image.
                                         Both images occupy the same physical space, but they differ in
                                         size (pixel count) and spacing.
       np_lung_segmentation_mask (numpy.array): Array denoting segmentation mask.
                                                Array size matches the sitk_lung_segmentation_image
                                                image size.
       new_size (list like): The region in the original_image defined by the segmentation mask is resampled
                             to this size.
       gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard deviation(s)
               before resampling.
    Returns:
        SimpleITK.Image and Confidence: Returns the lung segmented region in the
                                        original image which is resampled to the
                                        given size and returns segmentation confidence
                                        if the segmentation process of the CXR
                                        was done well.

    """
    lung_labels = [1, 2]
    segmentation_image = sitk.GetImageFromArray(np_lung_segmentation_mask)
    segmentation_image.CopyInformation(resampled_image)
    # relabel the segmentation so that the labels 1, and 2 correspond to the largest
    # components, assumed to be the lungs
    disjointed_segmentation_image = sitk.RelabelComponent(sitk.ConnectedComponent(segmentation_image),
                                                          sortByObjectSize=True)
    if gaussian_sigma:
        original_image = sitk.SmoothingRecursiveGaussian(original_image, gaussian_sigma)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(disjointed_segmentation_image)
    # Assign segmentation confidence variable to 'High' whenever segmentation process
    # "succeeds" and 'Low' whenever segmentation process fails
    seg_confidence = 'High'
    perimeter_threshold = 300  # For successful segmentation of lungs
    try:
        # Some of the images which are successfully segmented do not contain lungs
        # in the images.So we threshold the perimeter of the 2 lung regions.Area
        # is not taken as a measure becuase some of the successfully segmented images
        # have a major area  difference. E.g: 'CHNCXR_0361_1.png'(Due to the existence
        # of large airspaces in the lungs). So area is not taken as a measure for
        # filtering out the "non-lung" containing images
        perimeter_left = label_shape_filter.GetPerimeter(lung_labels[0])
        perimeter_right = label_shape_filter.GetPerimeter(lung_labels[1])
        if not (perimeter_left > perimeter_threshold and perimeter_right > perimeter_threshold):  # Threshold
            np_lung_segmentation_mask = np.ones(resampled_image.GetSize(), dtype=np.int64)
            seg_confidence = 'Low'
    except:
        # Some predicted masks contain no lung masks(no '1's in 'np_lung_segmentation_mask' array)
        # Assigning lung predicted masks to all ones would essentially select all the
        # area of the original image in the end
        np_lung_segmentation_mask = np.ones(resampled_image.GetSize(), dtype=np.int64)
        seg_confidence = 'Low'

    # Replace all the noise objects(apart from 2 lung regions) with zeros
    noise_labels = [x for x in label_shape_filter.GetLabels() if x not in lung_labels]
    for ns_label in noise_labels:
        ns_bb = label_shape_filter.GetBoundingBox(ns_label)
        np_lung_segmentation_mask[ns_bb[1]:ns_bb[1] + ns_bb[3], ns_bb[0]:ns_bb[0] + ns_bb[2]] = np.zeros(
            (ns_bb[3], ns_bb[2]))
    segmentation_image = sitk.GetImageFromArray(np_lung_segmentation_mask)
    segmentation_image.CopyInformation(resampled_image)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segmentation_image)
    # The bounding box's first two entries are the starting index and last
    # two entries the size
    bounding_box = label_shape_filter.GetBoundingBox(lung_labels[0])
    new_origin = segmentation_image.TransformIndexToPhysicalPoint(bounding_box[0:2])
    new_spacing = [(sz - 1) * spc / (new_sz - 1) for sz, spc, new_sz in
                   zip(bounding_box[2:4], segmentation_image.GetSpacing(), new_size)]
    original_image_norm = _normalize_image(original_image)
    arr = sitk.GetArrayFromImage(sitk.Resample(original_image_norm, new_size, sitk.Transform(), sitk.sitkLinear,
                                               new_origin, new_spacing, original_image.GetDirection(),
                                               0, original_image.GetPixelIDValue()))
    return np.reshape(arr, arr.shape + (1,)), seg_confidence
