from data_generation import prepare_data
from numpy import array, load, save, max as npmax, min as npmin
from nibabel import load as nibload
from os.path import exists as path_exists


N = 50 # Prepare a maximum number of images and masks to process.

# Retrieve the filenames from the ATLAS dataset.
files = prepare_data(r"Data\ATLAS_R2.0\Training")
files = list(filter(lambda file: ".json" not in file, files))

# Retrieve the filenames that were already processed and remove them from the list of filenames.
ignored_files = list()
files_to_ignore = list()
with open("Data\\read_files.txt", "r") as f:
    ignored_files = [line for line in f]
for file in ignored_files:
    files.remove(file)

# Filter the MRI file images and process them for generating a volume (M x N x K) normalized between 0 and 1
image_files = list(filter(lambda file: "_mask" not in file, files))
res_images = list()
for n, file in enumerate(image_files):
    if n > N:
        break
    data = array(nibload(file).get_fdata(), dtype="float16")
    data_max = npmax(data)
    data_min = npmin(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i, j, k] = (data[i, j, k] - data_min) / (data_max - data_min)
    res_images.append(data)
    files_to_ignore.append(file)
    print(f"Image ({n}/{N}) added to the training set: {file}")
# Generate an array based on the retrieved data and update the .npy file containing the images.
res_images = array(res_images)
image_path = r"Data\image_data.npy"
if path_exists(image_path):
    existent_image_data = load(image_path).tolist()
    existent_image_data += res_images
    save(image_path, array(existent_image_data))
else:
    save(image_path, array(res_images))

# Filter the mask file images and process them for generating a binary volume (M x N x K).
mask_files = list(filter(lambda file: "_mask" in file), files)
res_masks = list()
for n, file in enumerate(mask_files):
    if n > N:
        break
    data = array(nibload(file).get_fdata(), dtype="float16")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i, j, k] = 1 if data[i, j, k] > 0 else 0
    res_masks.append(data)
    files_to_ignore.append(file)
    print(f"Mask added to the training set: {file}")
# Generate an array based on the retrieved data and update the .npy file containing the masks.
res_masks = array(res_masks, dtype="uint8")
mask_path = r"Data\mask_data.npy"
if path_exists(mask_path):
    existent_mask_data = load(mask_path).tolist()
    existent_mask_data += res_masks
    save(mask_path, array(existent_mask_data, dtype="uint8"))
else:
    save(mask_path, array(res_masks, dtype="uint8"))

# Update the ignore file with the processed files' filenames.
with open("Data\\read_files.txt", "a") as f:
    for file in files_to_ignore:
        f.writelines(file)
