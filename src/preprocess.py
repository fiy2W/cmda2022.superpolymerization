from typing import List, Dict, Tuple
import numpy as np
import SimpleITK as sitk


def resampleDCM(itk_imgs, new_spacing: List[float]=[1,1,1], is_label: bool=False, new_size: List[int]=None, new_origin: List[float]=None):
    """ resample SimpleITK Image variable
    itk_imgs:    SimpleITK Image variable for input image to be resampled
    new_spacing: output spacing
    is_label:    whether to resample a image or a label
    new_size:    output size
    new_origin:  output origin
    """
    resample = sitk.ResampleImageFilter()
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    resample.SetOutputDirection(itk_imgs.GetDirection())
    if new_origin is None:
        new_origin = itk_imgs.GetOrigin()
    resample.SetOutputOrigin(new_origin)
    resample.SetOutputSpacing(new_spacing)

    ori_size = np.array(itk_imgs.GetSize(), dtype=np.int32)
    ori_spacing = np.asarray(itk_imgs.GetSpacing())
    if new_size is None:
        new_size = ori_size*(ori_spacing/new_spacing)
        new_size = np.ceil(new_size).astype(np.int32)
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    out = resample.Execute(itk_imgs)
    return out, new_size, new_origin


def histMatch(src, tgt, hist_level: int=1024, match_points: int=7):
    """ histogram matching from source image to target image
    src:          SimpleITK Image variable for source image
    tgt:          SimpleITK Image variable for target image
    hist_level:   number of histogram levels
    match_points: number of match points
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(hist_level)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.ThresholdAtMeanIntensityOn()
    dist = matcher.Execute(src, tgt)

    return dist


def ants_affine(affine_path, moving, fixed, is_label=False):
    Affine = sitk.ReadTransform(affine_path)
    resampler_affine = sitk.ResampleImageFilter()
    resampler_affine.SetReferenceImage(fixed)
    
    if is_label:
        resampler_affine.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler_affine.SetInterpolator(sitk.sitkLinear)

    resampler_affine.SetDefaultPixelValue(0)
    resampler_affine.SetTransform(Affine)

    out_affine = resampler_affine.Execute(moving)
    return out_affine