from os import path, walk
from nibabel import load as nibload
from numpy import array, max as npmax, min as npmin, ndarray




def separate_masks(files: list[str], key: str, normalize: bool = False, dtype: str = "float64") -> tuple[ndarray, ndarray]:
    '''It takes a list of files, a key to separate the masks from the images, and a boolean to normalize
    the images. It returns two arrays, one for the images and one for the masks
    
    Parameters
    ----------
    files : list[str]
        list[str] - list of paths to the files
    key : str
        the key that is used to separate the images from the masks.
    normalize : bool, optional
        if True, normalizes the image data to [0, 1]
    dtype : str, optional
        the data type of the images.
    
    Returns
    -------
        two arrays. The first one is an array of images, the second one is an array of masks.
    
    '''
    images = list(filter(lambda f: key not in f, files))
    masks = list(filter(lambda f: key in f, files))

    res_images = list()
    for image in images:
        data = array(nibload(image).get_fdata(), dtype=dtype)
        if normalize:
            data_max = npmax(data)
            data_min = npmin(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        data[i, j, k] = (data[i, j, k] - data_min) / (data_max - data_min)
        res_images.append(data)
    
    res_masks = list()
    for mask in masks:
        data = array(nibload(mask).get_fdata(), dtype="float16")
        for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        data[i, j, k] = 1 if data[i, j, k] > 0 else 0
        data = array(data, dtype="uint8")
        res_masks.append(mask)
    
    return array(res_images, dtype=dtype), array(res_masks, dtype="uint8")