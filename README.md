# TLS

Code for automatic lung and airway segmentation and evaluation. [Visit Project Drive!](https://drive.google.com/drive/u/0/folders/0BwxxJ4jzIZrbSzRVcUZ2S1BSQlE)

## Segmentation Modules

### Pedro's new filters 
* AirwaySegmentationAuto_v2: Macaques version.
* Lungs_Segmentation: Include Hu, labeleize and morpholical filter for macaques lung segmentation.
* SpeedImageCalc: Calculation of Hessian features
* Checkers: Additional functions
* Externals: ImageJ/ITK Filters
* Libs: Dicom utilities
* Propagator: Fast Marching algorithm for airway segmentation
* Registration: Registration filters
* Utilities: Help's filters used by the rest of projects

## Utilities
1. R
  * Waterfalls code.
2. Java
  * Batch mode scripts.
  * Similitude evaluation
  * ImageJ plugings to open MMWKS images(.hdr, .img) and ITK files(.mhd,.raw). 
3. Python
  * Batch mode filters.
  * Manual thresholding tissue classification
  * Help's functions

## Clasification
 * Manual and automatic

## Compiling
In order to complile the code sucessfully the following packages must be installed before:
* g++/gcc (Compiler)
* git-core
* git-svan
* libfontconfig-dev
* libgl1-mesa-dev
* libglu1-mesa-dev
* libncurses5-dev
* libosmesa6-dev
* libx11-dev
* libxrender-dev
* libxt-dev
* make
* python-dev
* python-numpy
* subversion
* libbz2-dev

The libraries set could be installed in one step by running the following line:
~~~{.sh}
sudo apt-get install g++ gcc git-core git-svn libfontconfig-dev libgl1-mesa-dev libglu1-mesa-dev libncurses5-dev libosmesa6-dev libx11-dev libxrender-dev libxt-dev make python-dev python-numpy subversion libbz2-dev 
~~~

In this version is necessary to perform the compilation in two phases. A first compilation to install the dependecies ```CMake TLS_SECOND_STEP ``` (compilation option unchecked) and second one with ```CMake TLS_SECOND_STEP ``` checked.
