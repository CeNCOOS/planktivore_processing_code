The repro is to make files availabe for collaboration on planktivore processing.  The repo contains the code to generate the zip file required by morphocluster to run.   The code for creating files that morphocluster wants are create_zip_file.py and index.csv.

The file files dualmag.ini, ptvr_proc_settings.json, and test_read_tar.py are for processing the raw planktivore images to color output files and create the zip file of the steps used during each images processing.  The code uses cvtools.py and Paul Roberts repo.
https://github.com/mbari-org/rims-ptvr/tree/master/rois is the specific part of the repo where cvtoos.py is located.

The proposed pathway for images from planktivore are on the slide planktivore_data_pipeline.ppx

As the data stream is refined more files will be added along with notes and documentation.
