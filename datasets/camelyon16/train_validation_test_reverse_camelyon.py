import os
import shutil

# define the main folder
main_folder = 'single'

# Define the fold folder
fold_number = '1'
fold_folder = f"single/fold{fold_number}"

# Define the subfolders
subfolders = ['train', 'validation', 'test']

# Define the target folders
target_folders = ['0_normal', '1_tumor']

# Iterate over the main folder and subfolders
for subfolder in subfolders:
    subfolder_path = os.path.join(fold_folder, subfolder)

    # Iterate over the target folders
    for target_folder in target_folders:
        target_folder_path = os.path.join(fold_folder, subfolder, target_folder)
        for wsi in os.listdir(target_folder_path):
            wsi_path = os.path.join(target_folder_path, wsi)
            print('wsi_path:', wsi_path)
            print('main_folder:', main_folder)
            print('target_folder:', target_folder)
            print(f'{main_folder}/{target_folder}')
            shutil.move(wsi_path, f'{main_folder}/{target_folder}')
