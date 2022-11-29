from numpy import max as np_max
from numpy import min as np_min
from numpy import array, ndarray
from nibabel import load

# Generate a function to pre-process the images and normalize them
def nii_preprocess(filename: str, dtype: str = "float64") -> ndarray:
    img = array(load(filename), dtype=dtype)
    witdth, height, queue = img.shape
    for w in range(witdth):
        for h in range(height):
            for q in range(queue):
                img[w, h, q] = (img[w, h, q] - np_min(img)) / (np_max(img) - np_min(img))
    return img
