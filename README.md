# Prototype project of ALPR(Automatic License Plate Recognition) #

This is a prototype ALPR python program to extract vehicle license number out of Korea vehicle license plate
using tensorflow.  
Don't expect too much from this source code. This is just for my self study subject.

### Installation ###
Following instructions show how to setup a tensorflow with the Anconda on Windows 10.  
First of all, install the latest Anaconda.

* Setup tensorflow environment
```bash
(base) C:\> conda create -n tensorflow_env python
(base) C:\> conda activate tensorflow_env
(tensorflow_env) C:\> conda install -c conda-forge tensorflow
```

* Check to see installed tensorflow working fine.
```bash
>>> import tensorflow as tf
>>> tf.__version__
'1.13.1'
>>> quit()
```
* Install dependency packages
```bash
(tensorflow_env) C:\> pip install Pillow
(tensorflow_env) C:\> pip install matplotlib
```
* Install opencv package
```bash
(tensorflow_env) C:\> conda install anaconda-client
(tensorflow_env) C:\> conda install --channel https://conda.anaconda.org/menpo opencv
```

* Install Jupyter Anaconda extension
```bash
(tensorflow_env) C:\> conda install nb_conda
```

### Test number digit images recognition test ###

* Run "Anaconda Prompt"
* Enter "tensorflow_env" then goto a source directory
```bash
(tensorflow_env) C:\> conda activate tensorflow_env
(tensorflow_env) C:\> cd tensorflow_ALPR
```
* Train number digit images
```bash
(tensorflow_env) C:\\tensorflow_ALPR> python test_number_digits_recog.py train
```
* Test number digit images
```bash
(tensorflow_env) C:\\tensorflow_ALPR> python test_number_digits_recog.py test
```

### Test extracting vehicle license number from image file ###

* Run "Anaconda Prompt"
* Enter "tensorflow_env" then goto a source directory
```bash
(tensorflow_env) C:\> conda activate tensorflow_env
(tensorflow_env) C:\> cd tensorflow_ALPR
```
* Test license number extraction from a license plate image
```bash
(tensorflow_env) python test_LPR_digits_only.py my_images\vehicle_number_plates\1_0236.jpg
```
