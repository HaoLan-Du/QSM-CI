"""
metrics.py

This module provides functions to compute various error metrics between 3D predicted and reference 
data arrays. The primary function, `all_metrics()`, returns a dictionary of all computed metrics.

Metrics included: RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE.

Example:
    >>> import numpy as np
    >>> import metrics
    >>> pred_data = np.random.rand(100, 100, 100)
    >>> ref_data = np.random.rand(100, 100, 100)
    >>> roi = np.random.randint(0, 2, size=(100, 100, 100), dtype=bool)
    >>> metrics = metrics.all_metrics(pred_data, ref_data, roi)

Authors: Boyi Du <boyi.du@uq.net.au>, Ashley Stewart <ashley.stewart@uq.edu.au>

"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_mutual_information
from skimage.metrics import normalized_root_mse
from scipy.ndimage import gaussian_laplace
from scipy.stats import median_abs_deviation
from numpy.linalg import norm
from skimage.measure import pearson_corr_coeff
import pandas as pd 

def calculate_rmse(pred_data, ref_data):
    """
    Calculate the Root Mean Square Error (RMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated RMSE value.

    """
   mse = mean_squared_error(pred_data, ref_data)
    rmse = np.sqrt(mse)
    return rmse

def calculate_nrmse(pred_data, ref_data):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated NRMSE value.

    """
    nrmse = normalized_root_mse(ref_data, pred_data)
    return nrmse

def calculate_hfen(pred_data, ref_data):
    """
    Calculate the High Frequency Error Norm (HFEN) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated HFEN value.

    """
    LoG_pred = gaussian_laplace(pred_data, sigma = 1.5)
    LoG_ref = gaussian_laplace(ref_data, sigma = 1.5)
    hfen = norm(LoG_ref - LoG_pred)/norm(LoG_pred)
    return hfen

def calculate_xsim(pred_data, ref_data):
    """
    Calculate the structural similarity (XSIM) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated structural similarity value.

    """
    xsim = structural_similarity(pred_data,ref_data,win_size = 3, K1 = 0.01, K2 = 0.001, data_range = 1)
    return xsim

def calculate_mad(pred_data, ref_data):
    """
    Calculate the Mean Absolute Difference (MAD) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated MAD value.

    """
    mad = np.mean(np.abs(pred_data - ref_data))
    return mad

def calculate_gxe(pred_data, ref_data):
    """
    Calculate the gradient difference error (GXE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated GXE value.

    """
    gxe = normalized_root_mse(np.gradient(ref_data), np.gradient(pred_data))
    return gxe


def get_bounding_box(roi):
    """
    Calculate the bounding box of a 3D region of interest (ROI).

    Parameters
    ----------
    roi : numpy.ndarray
        A 3D numpy array representing a binary mask of the ROI,
        where 1 indicates an object of interest and 0 elsewhere.

    Returns
    -------
    bbox : tuple
        A tuple of slice objects representing the bounding box of the ROI. This can be 
        directly used to slice numpy arrays.

    Example
    -------
    >>> mask = np.random.randint(0, 2, size=(100, 100, 100))
    >>> bbox = get_bounding_box(mask)
    >>> sliced_data = data[bbox]

    Notes
    -----
    The function works by identifying the min and max coordinates of the ROI along 
    each axis. These values are used to generate a tuple of slice objects.
    The function will work for ROIs of arbitrary dimension, not just 3D.
    """
    coords = np.array(roi.nonzero())
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return tuple(slice(min_coords[d], max_coords[d]) for d in range(roi.ndim))


def all_metrics(pred_data, ref_data, roi=None):
    """
    Calculate various error metrics between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.
    roi : numpy.ndarray, optional
        A binary mask defining a region of interest within the data. If not provided,
        the full extent of pred_data and ref_data is used.

    Returns
    -------
    dict
        A dictionary of calculated error metrics, including RMSE, NRMSE, FHEN, XSIM, MAD, 
        CC (Pearson Correlation Coefficient), NMI (Normalized Mutual Information) and GXE 
        (Gradient difference error).

    """
    d = dict();
    if roi is None:
        # d['RMSE'] = calculate_rmse(pred_data, ref_data)
        d['NRMSE'] = calculate_nrmse(pred_data, ref_data)
        d['HFEN'] = calculate_hfen(pred_data, ref_data)
        d['XSIM'] = 0.5 * calculate_xsim(pred_data, ref_data)+1
        d['MAD'] = calculate_mad(pred_data, ref_data)
        d['CC'] = 0.5 * pearson_corr_coeff(pred_data, ref_data)[0] + 1
        d['NMI'] = normalized_mutual_information(pred_data, ref_data)
        d['GXE'] = calculate_gxe(pred_data, ref_data)    
    return d

if __name__ == "__main__":
    
    # get command-line arguments
    # ... - argparse (python module; e.g. import argparse)
    # If enter command line manually, check if the command line legal
    # get tested algorithm name
    # get file paths
    parser = argparse.ArgumentParser(description='Calculating metrics')
    parser.add_argument('file_paths', metavar='file', type=str, nargs='+',
                        help='Paths to input files')
    args = parser.parse_args()
    for file_path in args.file_paths:
        print(f'Processing file: {file_path}')
    
    
    # load nifti images (ground truth + recon) - should for all ground truth images and all reconstructions
    # Algorithms' output
    # Ground true
    # mask
    # seg

    # preprocessing data
    # overall result of tested algorithm and ground
    # Region of interest of tested algorithm and ground
    # make dataframe
    
    # do metrics
    # ...

    
    # generate figures
    # ...
    
    # save figures as pngs
    # ...
    