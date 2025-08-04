This repository contains code to process the images from the MBARI Planktivore device on one of MBARI's LRAUVs.
Data are currently placed in directories on the mass storage unit at MBARI called Thalassa.   The share name is Planktivore.  The directories on Thalassa/Planktivore are named with the convention
YYYY-MM-DD-LRAH-dd where YYYY is the year, MM is the month, DD is the start day, AH stands for Ahi and dd is the vehicle deployment number.

There are two versions of the code.  Both compute the same thing. file_format_refactor.py uses multiprocessing, the number of processors is now
read in from the JSON file.
The other code is reformat_using_pytorch.py.  This code uses the pytorch framework to do the same processing.  The number of processors is also
 refrenced from the JSON file.
The setup JSON files are now accessed by input from the prompt. setup_process_planktivore_refactor.json.  This file specifies the path to the pvtr_proc_settings.json file, the magnification level to process, the input directory, the upper directory for the output, the basis for creating subdirectories in minutes, and max number of files (not used), the bayer_pattern and the number of processors to use.  The example file has diretory paths using the windows format.  For use on a Linux machine the paths would need to be updated as appropriate.  Both versions of the code call a few other pieces of code.  These are: check_dir.py (create base directory names), date_from_name.py (get the date from the planktivore linux timestamp that is used in the file name), and cvtools_modified.py (This version of cvtools has been modified to only output the reformated image file and not all of the other aspects of the image such as masks etc.)
Due to the code using time to create the output directory name, a judicious choice for the number of minutes per subdirectory is necessary to not have too many files in a directory.  For the high_mag_cam setting 10 minutes seems to work.  For the low_mag_cam a much shorter time of 1 minute is appropriate.  Note the links to the json files are currently coded as windows style links.  An YML file base upon my conda python installation is included and contains way more than necessary for running the code but is included for completeness.

The original cvtools.py was from Paul Roberts repo https://github.com/mbari-org/rims-ptvr/tree/master/rois

NOTE: Even though the code processes all files, some images are "not valid" and are NOT written to disk.  Cvtools and cvtools_modified determine if an image is valid or not and if it should be processed.  This means that the total number of images processed may be less than the estimated number of images just using the tar files.

NOTE: morphocluster_files are legacy files and can be ignored.
