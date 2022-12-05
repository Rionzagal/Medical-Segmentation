from data_generation import prepare_data
from numpy import array, dtype, expand_dims, load, save, max as npmax, min as npmin, uint8
from nibabel import load as nibload
from os.path import exists as path_exists
import tables


# Retrieve the filenames from the ATLAS dataset.
files = prepare_data(r"Data\ATLAS_R2.0\Training")
files = list(filter(lambda file: ".json" not in file, files))

# Filter the MRI file images and process them for generating a volume (M x N x K) normalized between 0 and 255
# image_files = list(filter(lambda file: "_mask" not in file, files))
# res_images = list()
# image_path = r"Data\image_data.h5"

# with tables.open_file(image_path, mode="w") as mri_file:
#     mri_atom = tables.Atom.from_dtype(dtype(uint8, (197, 233, 0)))
#     mri_array = mri_file.create_earray(mri_file.root, "data", mri_atom, (197, 233, 0))
#     for file in image_files:
#         data = array(nibload(file).get_fdata(), dtype="float16")
#         data_max = npmax(data)
#         data_min = npmin(data)
#         data = (data - data_min) / (data_max - data_min)
#         for k in range(data.shape[2]):
#             mri_array.append(expand_dims(data[:, :, k], axis=2))
#         print(f"Image added to the training set: {file}")
# # Generate an array based on the retrieved data and update the .npy file containing the images.
# print(f"---{image_path} has been created. ---")
# # Filter the mask file images and process them for generating a binary volume (M x N x K).
mask_path = r"Data\mask_data.h5"
mask_files = list(filter(lambda file: "_mask" in file, files))
res_masks = list()

with tables.open_file(mask_path, mode="w") as mask_file:
    mask_atom = tables.Atom.from_dtype(dtype(uint8, (197, 233, 0)))
    mask_array = mask_file.create_earray(mask_file.root, "data", mask_atom, (197, 233, 0))
    for file in mask_files:
        data = array(nibload(file).get_fdata(), dtype="float16")
        data_max = npmax(data)
        data_min = npmin(data)
        data[data > 0] = 1
        data[data == 0] = 0
        for k in range(data.shape[2]):
            mask_array.append(expand_dims(data[:, :, k].astype("uint8"), axis=2))
        print(f"Mask added to the training set: {file}")
# Generate an array based on the retrieved data and update the .npy file containing the masks.
print(f"--- {mask_path} has been created. ---")