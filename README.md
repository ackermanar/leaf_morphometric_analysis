# Automated Analysis of Leaf Morphometrics
*Version 9, 03-20-25*

## Summary
Develop a model to measure leaf morphometrics from photos taken outside in field books and test if accuracy is comparable to images taken in a photo booth.
## Goals
Measure height, width, leaf area, serration count, and leaflet count. Once accuracy is found to be acceptable, experiment with other morphometrics such as individual leaflet widths. 
## Getting Started
Leaf morphometrics tools uses packages that can be installed by package manager Anaconda to accomplish analyses. For questions or help with Anaconda, read the following:
[Getting started with conda — conda 23.9.1.dev37 documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
### Installation
1. Create and activate a conda environment using leaf_morphometrics_tools: 
   1. This is best accomplished through Mamba rather than Conda, as Mamba is much quicker. If installing Mamba is difficult, conda works fine, just slower. Miniconda/Minimamba works as well and can save room on your local computer.
      1. Conda: [Installation — Anaconda documentation](https://docs.anaconda.com/free/anaconda/install/index.html)
      2. Mamba: [Installation — documentation](https://mamba.readthedocs.io/en/latest/installation.html)
         1. If using mamba, replace all ‘conda’ calls in your prompt with ‘mamba’
   2. Once Conda/Mamba/Miniconda/Minimamba is installed and operational, type the following to create the environment to create the environment:
   
```
conda env create -f /my/path/to/leaf_morphometrics_tools/leaf_morphometrics_tools.yml
```
*replace* /my/path/to/ *with the path to leaf_morphometrics_tools/leaf_mophometrics_tools.yml on your local computer*

3. Type the following to activate the leaf_morphometrics_tools environment:
   1. conda activate leaf_morphometrics-env
### Running in parallel over a collection of images - guided intro
1. The pipeline was made to use parallel computing to be run over a folder that includes all images taken within a study. To accomplish this, run the following from the command line prompt:

```
python /my/path/to/leaf_morphometrics_tools/leaf_morphoV6_12x16.py
```

*To use the pipeline for 12x12in templates type the following:*

```
python /my/path/to/leaf_morphometrics_tools/leaf_morphoV6_12x12.py
```

3. Entering the above code into a command line prompt will spawn a guided intro for running the pipeline over a collection of images.
4. An image with the word ‘calibrate’ in the filename is required for each pool of images. This image calibrates the expected pixels per inch in every image, helping to ID the red size markers on templates in various lighting conditions. This image should be of a blank template (no leaves), in good lighting, and should be taken by the camera the observer intends to use to capture images. 
   1. Examples of calibrate image filenames:
      1. calibrate.jpg
      2. NY1234_calibrate.jpg
      3. calibrate_IA999.png
      4. 1234_SC_calibrate.nef
### Running in parallel over a collection of images - manual argument input
Leaf morphometrics tools can take a variety of optional arguments for running leaf analyses. Most of these are requested for input in the guided intro, but can be manually input following the Python call:
```
-h, --help            show this help message and exit
-i INPUT_DIR, --input_dir INPUT_DIR
                        Path to input directory of images to be analyzed.
-t TOOLS_DIR, --tools_dir TOOLS_DIR
                        Path to leaf_morpho_tools file.
-o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output directory for resulting images.
 -r RESULTS_PATH, --results_path RESULTS_PATH
                        Desired path and file name for results.
 -w WORKERS, --workers WORKERS
                        Number of worker processes to use, if nothing is
                        specified half of all available workers will be used.
```
#### To enter all arguments manually, the Python call would resemble the following:
python /my/path/to/leaf_morphometrics_tools/leaf_morphoV7.py -i /my/path/to/inputImages -t my/path/to/leaf_morpho_tools -o /my/path/to/imageOutput -r /my/path/to/theseAreMyResults.csv -w 12
#### To display these options in the command line prompt, enter the following:
python /my/path/to/leaf_morphometrics_tools/leaf_morphoV7.py -h


