from os import path, walk
from numpy import array, ndarray, round as npround, pad as nppad, dtype, expand_dims, float16, max as npmax, min as npmin, uint8
from nibabel import load as nibload
import tables


def padding(array: ndarray, xx: int, yy: int) -> ndarray:
    '''It takes an array and pads it with zeros to make it a square
    
    Parameters
    ----------
    array : ndarray
        the array to be padded
    xx : int
        the desired width of the output image
    yy : int
        the height of the image
    
    Returns
    -------
        A padded array.
    
    '''

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return nppad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def prepare_data(dir: str) -> list[str]:
    '''It returns a list of all files in a directory and its subdirectories
    
    Parameters
    ----------
    dir : str
        the directory to search for files
    levels : int, optional
        The number of levels to go down in the directory tree.
    
    Returns
    -------
        A list of strings.
    
    '''
    r = list()
    subdirs = [x[0] for x in walk(dir)]
    for sub_dir in subdirs:
        files = walk(sub_dir).__next__()[2]
        if len(files) > 0:
            for file in files:
                r.append(path.join(sub_dir, file))
    return list(filter(lambda file: ".json" not in file, r))


# Retrieve the filenames from the ATLAS dataset.
files = prepare_data(r"Data\ATLAS_R2.0\Training")

# Filter the MRI file images and process them for generating a volume (M x N x K) normalized between 0 and 1 as float16

image_path = r"Data\image_data.h5"
with tables.open_file(image_path, mode="w") as mri_file:
    mri_atom = tables.Atom.from_dtype(dtype(uint8, (0, 256, 256)))
    mri_array = mri_file.create_earray(mri_file.root, "data", mri_atom, (0, 256, 256))
    for file in list(filter(lambda file: "_mask" not in file, files)):
        data = array(nibload(file).get_fdata(), dtype="float16")
        data_max = npmax(data)
        data_min = npmin(data)
        data = npround(255 * (data - data_min) / (data_max - data_min))
        for k in range(data.shape[2]):
            # Pad the image so it is square.
            img = padding(data[:, :, k], 256, 256).astype("uint8")
            mri_array.append(expand_dims(img, axis=0))
        print(f"Image added to the training set: {file}")
# Generate an array based on the retrieved data and update the .npy file containing the images.
print(f"--- {image_path} has been created. ---")
# Filter the mask file images and process them for generating a binary volume (M x N x K).
mask_path = r"Data\mask_data.h5"
with tables.open_file(mask_path, mode="w") as mask_file:
    mask_atom = tables.Atom.from_dtype(dtype(uint8, (0, 256, 256)))
    mask_array = mask_file.create_earray(mask_file.root, "data", mask_atom, (0, 256, 256))
    for file in list(filter(lambda file: "_mask" in file, files)):
        data = array(nibload(file).get_fdata(), dtype="float16")
        data_max = npmax(data)
        data_min = npmin(data)
        data[data > 0] = 1
        data[data == 0] = 0
        for k in range(data.shape[2]):
            mask = padding(data[:, :, k], 256, 256).astype("uint8")
            mask_array.append(expand_dims(mask, axis=0))
        print(f"Mask added to the training set: {file}")
# Generate an array based on the retrieved data and update the .npy file containing the masks.
print(f"--- {mask_path} has been created. ---")