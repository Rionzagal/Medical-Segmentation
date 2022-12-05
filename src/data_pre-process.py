from data_generation import prepare_data
from numpy import array, dtype, expand_dims, load, save, max as npmax, min as npmin, uint8
from nibabel import load as nibload
from os.path import exists as path_exists
import tables


N = 2000 # Prepare a maximum number of images and masks to process.

# Retrieve the filenames from the ATLAS dataset.
files = prepare_data(r"Data\ATLAS_R2.0\Training")
files = list(filter(lambda file: ".json" not in file, files))

# Retrieve the filenames that were already processed and remove them from the list of filenames.
ignored_files = list()
files_to_ignore = list()
with open("Data\\read_files.txt", "r") as f:
    ignored_files = [line for line in f]
for file in ignored_files:
    files.remove(file.strip("\n"))

# Filter the MRI file images and process them for generating a volume (M x N x K) normalized between 0 and 255
image_files = list(filter(lambda file: "_mask" not in file, files))
res_images = list()
image_path = r"Data\image_data.h5"

with tables.open_file(image_path, mode="w") as mri_file:
    mri_atom = tables.Atom.from_dtype(dtype(uint8, (0, 197, 233, 189)))
    mri_array = mri_file.create_earray(mri_file.root, "data", mri_atom, (0, 197, 233, 189))
    for n, file in enumerate(image_files):
        if n >= N:
            break
        data = array(nibload(file).get_fdata(), dtype="float16")
        data_max = npmax(data)
        data_min = npmin(data)
        data = 255 * (data - data_min) / (data_max - data_min)
        mri_array.append(expand_dims(data.astype("uint8"), axis=0))
        print(f"Image ({n+1}/{N}) added to the training set: {file}")
        files_to_ignore.append(file)
# Generate an array based on the retrieved data and update the .npy file containing the images.
print(f"---{image_path} has been updated. ---")
# Filter the mask file images and process them for generating a binary volume (M x N x K).
mask_path = r"Data\mask_data.h5"
mask_files = list(filter(lambda file: "_mask" in file, files))
res_masks = list()

with tables.open_file(mask_path, mode="w") as mask_file:
    mask_atom = tables.Atom.from_dtype(dtype(uint8, (0, 197, 233, 189)))
    mask_array = mask_file.create_earray(mask_file.root, "data", mask_atom, (0, 197, 233, 189))
    for n, file in enumerate(mask_files):
        if n >= N:
            break
        data = array(nibload(file).get_fdata(), dtype="float16")
        data_max = npmax(data)
        data_min = npmin(data)
        data[data > 0] = 1
        data[data == 0] = 0
        mask_array.append(expand_dims(data.astype("uint8"), axis=0))
        print(f"Mask ({n+1}/{N}) added to the training set: {file}")
        files_to_ignore.append(file)
# Generate an array based on the retrieved data and update the .npy file containing the masks.
print(f"--- {mask_path} has been updated. ---")
# Update the ignore file with the processed files' filenames.
with open("Data\\read_files.txt", "a") as f:
    for file in files_to_ignore:
        f.writelines(f"{file}\n")
