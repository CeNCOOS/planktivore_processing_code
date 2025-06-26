New updates:
The code now reads the raw tarfiles from the planktivore directory on Thalassa and outputs the converted images to directories under Thalass\DeepSea-AI\data\raw\Deployment_name\
The main code is file_reformat_refactor.py  This code uses multiprocessing and is set to only use 10 CPUs and loop through the files 10 at a time.
The code calls several other pieces: check_dir.py (create date base directory names), date_from_name.py (get the date from the planktivore linux timestamp in the filename), and cvtools_modified.py (This code only outputs the reformated image file and not all the other files that cvtools.py outputs).  Output parameters are set up in the "setup_process_planktivore_refactor.json" files.  Due to using time to output the directory names a judicious choice of the number of minutes per subdirectory is necessary.  Otherwise, you could get too many files in a directory.

NOTE: Even though all files get processed some images are "not valid" and those are NOT written to disk.  This is determined by Paul Roberson's code.  So the total number of images will be less than the number of images you would estimate from the tar files.  So for a small test of 5 tar files (which should be 5999 images, in reality there are 5969 valid images).

The repro is to make files availabe for collaboration on planktivore processing.  The repo contains the code to generate the zip file required by morphocluster to run.   The code for creating files that morphocluster wants are create_zip_file.py and index.csv.

The files dualmag.ini, ptvr_proc_settings.json, and test_read_tar.py are for processing the raw planktivore images to color output files and create the zip file of the steps used during each images processing.  The code uses cvtools.py and Paul Roberts repo.
https://github.com/mbari-org/rims-ptvr/tree/master/rois is the specific part of the repo where cvtoos.py is located.

The proposed pathway for images from planktivore are on the slide planktivore_data_pipeline.ppx

As the data stream is refined more files will be added along with notes and documentation.
