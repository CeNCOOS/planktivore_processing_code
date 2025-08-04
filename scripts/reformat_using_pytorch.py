# Gemeini created Code using pytorch from my code
# Removing all the dummy functions (which may be good, but make the code bloat)
# See if I can run without creating all the "dummy" stuff
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import multiprocessing as mp # Still useful for pre-scanning tar files if needed
from glob import glob
import tarfile
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time
from torchvision import transforms
import pdb # Uncomment for debugging
sys.path.append('c:/users/flbahr/Documents/GitHub/planktivore_processing_code/scripts/')
import cvtools_modified as cvtools
from check_dir import check_dir
from date_from_name import date_from_name
#
# Start of pytorch code set up
#
class TarAndTifDataset(Dataset):
#    def __init__(self, input_dir, maglev, settings, bayer_pattern_cv2, transform=None):
    def __init__(self, input_dir,output_dir, maglev,subdirmin, settings, bayer_pattern_cv2, transform=None):
        self.input_dir = Path(input_dir)
        self.output_dir= Path(output_dir)
        self.maglev = maglev
        self.subdirmin=subdirmin
        self.settings = settings
        self.bayer_pattern_cv2 = bayer_pattern_cv2
        self.transform = transform
        self.file_list = [] # Stores (original_file_path, index_within_tar_file_if_any)
        
        self._prepare_file_list()

    def _prepare_file_list(self):
        """
        Scans the input directory for .tar and .tif files and builds a list
        of (file_path, index_in_archive_if_tar) tuples.
        This is necessary because __len__ needs to know the total number of images.
        """
        print(f"Scanning directory {self.input_dir} for image files...")
        
        # Use pathlib.rglob for robust recursive search
        all_potential_files = []
        for ext in ['.tar', '.tif']:
            all_potential_files.extend(list(self.input_dir.rglob(f'*{ext}')))
        
        # Filter for actual files and maglev
        valid_files_for_processing = []
        for p in all_potential_files:
            if p.is_file() and str(p).find(self.maglev) >= 0:
                valid_files_for_processing.append(p)

        for filepath in valid_files_for_processing:
            if filepath.suffix == '.tar':
                try:
                    with tarfile.open(filepath, 'r') as tobj:
                        # Only consider actual files within the tar, not directories
                        # and potentially filter for images if needed (e.g., based on suffix)
                        image_members = [
                            m.name for m in tobj.getmembers() 
                            if m.isfile() and (m.name.lower().endswith('.tif') or m.name.lower().endswith('.tiff'))
                        ]
                        for member_name in image_members:
                            # Store the tar file path and the member name within the tar
                            self.file_list.append((str(filepath), member_name))
                except tarfile.ReadError:
                    print(f"Warning: Could not open tar file (possibly corrupted): {filepath}")
                except Exception as e:
                    print(f"Error processing tar file {filepath}: {e}")
            elif filepath.suffix == '.tif':
                # For .tif files, store the path and None for the tar member
                self.file_list.append((str(filepath), None))
                
        print(f"Found {len(self.file_list)} images across .tar and .tif files.")
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # I think this should also process the file
        original_file_path_str, tar_member_name = self.file_list[idx]
                
        try:
            if tar_member_name is None: # It's a direct .tif file
                # Original logic for tif files
                # Note: cv2.imread and PIL.Image.open might have different behaviors
                # and expectations for file paths.
                img_path = original_file_path_str
                # In this case get date from path
                filedate=date_from_name(img_path,self.maglev)
                subdirpath=check_dir(self.output_dir,filedate,self.subdirmin)
                #idslash=img_path.rfind('/')
                # The slash issue can we do a img_path.replace('\\','/') 
                # and use above code for idslash?
                idslash=img_path.rfind('\\')
                abpath=img_path[0:idslash+1]
                ftif=img_path[idslash+1:]
                fout=ftif[:-4]
                # Check if image can be read by cv2.imread (as in original code)
                # It's better to open with PIL for consistency with torchvision transforms
                #image = Image.open(img_path).convert("RGB")
                fsize=cv2.imread(img_path)
                if fsize is not None:
                    try:
                        image=cvtools.import_image(abpath,ftif,self.settings)
                    except Exception as e:
                        print('Failed for import_image\n')
                        print(repr(e))
                # Apply cvtools_modified specific steps if needed before transforms
                # For demonstration, directly pass PIL image to convert_to_8bit
                # assuming it handles PIL images.
                    try:
                        img_8bit = cvtools.convert_to_8bit(image, self.settings)
                    except Exception as e:
                        print('Failed to convert to 8bit\n')
                        print(repr(e))
                    #output=cvtools.extract_features(img_8bit,image,self.settings,save_to_disk=True,
                    #                            abs_path=os.path.join(subdirpath),file_prefix=fout)
                    try:
                        output=cvtools.extract_features(img_8bit,image,self.settings,save_to_disk=True,
                                                    abs_path=os.path.join(subdirpath),file_prefix=fout)
                    except Exception as e:
                        print('Failed on write/extract\n')
                        print(repr(e))
                    image_final = Image.fromarray(img_8bit) # For passing to transforms
                else:
                    image_final= Image.new('RGB', (128, 128), (0, 0, 0))
                
            else: # It's an image inside a .tar file
                filedate=date_from_name(tar_member_name,self.maglev)
                subdirpath=check_dir(self.output_dir,filedate,self.subdirmin)
                # again issue with slashes?
                idslash=tar_member_name.rfind('/')
                ftif=tar_member_name[idslash+1:]
                fout=ftif[:-4]
                # Original logic for tar files
                with tarfile.open(original_file_path_str, 'r') as tobj:
                    fileobj = tobj.extractfile(tar_member_name)
                    if fileobj is None:
                        raise FileNotFoundError(f"Member {tar_member_name} not found in {original_file_path_str}")
                    tiff_array = np.asarray(bytearray(fileobj.read()), dtype=np.uint8)
                    img = cv2.imdecode(tiff_array, cv2.IMREAD_UNCHANGED)
                    # Apply Bayer pattern conversion
                    if img.ndim == 2 and self.bayer_pattern_cv2: # Check if it's a single channel image that needs Bayer
                        imgc = cv2.cvtColor(img, self.bayer_pattern_cv2)
                    else:
                        imgc = img # Already color or not a Bayer pattern

                    # Convert to 8-bit using cvtools
                    img_8bit = cvtools.convert_to_8bit(imgc, self.settings)
                    output=cvtools.extract_features(img_8bit,img,self.settings,save_to_disk=True,
                                                    abs_path=os.path.join(subdirpath),file_prefix=fout)
                    

                    
                    # Convert to PIL Image for torchvision transforms
                    image_final = Image.fromarray(img_8bit)
            
            if self.transform:
                image_final = self.transform(image_final)
                
            return image_final # Return the processed image tensor
        
        except Exception as e:
            print(f"Error loading image {original_file_path_str} (member: {tar_member_name}): {repr(e)}")
            # Handle corrupted files by returning a dummy tensor or skipping.
            # Returning a dummy tensor (e.g., black image) is common practice to keep batch size consistent.
            # Define what a "dummy" tensor should look like based on your transforms and expected output.
            # For example, if your transform outputs a 3xHxC tensor of floats:
            if self.transform:
                # Create a dummy image (e.g., black 128x128 image)
                dummy_image = Image.new('RGB', (128, 128), (0, 0, 0)) 
                return self.transform(dummy_image)
            else:
                # If no transform, return a simple numpy array
                return np.zeros((128, 128, 3), dtype=np.uint8) # Or whatever expected raw format

if __name__ == "__main__":
    # --- Setup: Load settings from JSON (as in original code) ---
    # Create a dummy json file for testing if it doesn't exist
    json_file=input("Enter the path to the JSON file with settings: ")
    #json_path = 'c:/users/flbahr/setup_process_planktivore_octtest.json'
    #pvtr_settings_path = 'c:/users/flbahr/dummy_settings.json'

    # Load settings
    try: 
        fid=open(json_file,'r')
    except Exeception as e:
        print(f"Error open JSON file: {repr(e)}")
    
    #with open(json_path, 'r') as fid:
    plset = json.load(fid)
    fclose(fid)
    
    pvtr_settings_file = plset[0]['pvtr_setting']
    maglev = plset[0]['maglev']
    input_dir_root = plset[0]['inputdir']
    output_dir_root = plset[0]['outputdir']
    subdirmins = plset[0]['subdirmins']
    bayer_pattern_str = plset[0]['bayer_pattern']
    ncore=plset[0]['ncore']

    # Load actual pvtr settings
    with open(pvtr_settings_file, 'r') as jf:
        settings = json.load(jf)

    # Convert bayer pattern string to cv2 constant
    bayer_pattern_cv2 = getattr(cv2, bayer_pattern_str)
    
    # Define PyTorch Transforms
    # These transforms will be applied by the DataLoader's workers
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),      # Resize all images to a common size
        transforms.ToTensor(),              # Convert PIL Image to PyTorch Tensor (HWC to CHW, /255.0)
        # Add normalization based on your dataset if needed
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # Instantiate the Dataset
    dataset = TarAndTifDataset(
        input_dir=input_dir_root,
        output_dir=output_dir_root,
        maglev=maglev,
        subdirmin=subdirmins,
        settings=settings,
        bayer_pattern_cv2=bayer_pattern_cv2,
        transform=image_transforms
    )

    # Instantiate the DataLoader
    # Use a small batch size for demonstration
    # num_workers > 0 enables multiprocessing for data loading, speeding it up
    dataloader = DataLoader(
        dataset,
        batch_size=ncore,
        shuffle=True, # Shuffle the data each epoch
        num_workers=ncore, # Adjust based on your CPU cores
        pin_memory=True # For faster GPU transfers
    )

    print(f"\nTotal images found in dataset: {len(dataset)}")
    print(f"DataLoader batch size: {dataloader.batch_size}")

    # --- Iterate through the DataLoader ---
    print("\nStarting data loading (iterating through DataLoader):")
    for i, images_batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Batch shape: {images_batch.shape}") # Should be [batch_size, channels, height, width]
        print(f"  Batch dtype: {images_batch.dtype}")
        
        # You can now pass 'images_batch' to your PyTorch model
        # model(images_batch) 
        
        #if i >= 4: # Process a few batches for demonstration
        #    break
            
    print("\nData loading complete.")


        
