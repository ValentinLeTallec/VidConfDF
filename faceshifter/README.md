# FaceShifter

This is the reproducing version of FaceShifter

Paper Reference: https://arxiv.org/pdf/1912.13457.pdf

Code Reference: https://github.com/taotaonice/FaceShifter and https://github.com/richarduuz/Research_Project/tree/master/ModelC

## Requirements
* Python 3.7
* torch 1.4.0
* torchvison 0.5.0
* opencv-python 4.2.0
* numpy 1.18.1
* Flask
* flask_cors
* requests

You can install these packages by pip
```
pip install -r requirements.txt
```
If you want to train the model, you have to install apex, you can download and install from the Nvidia/apex: https://github.com/NVIDIA/apex

## How to use FaceShifter
### Download Pre-trained Weights
* URL for download pre-trained weights of arcface: https://drive.google.com/open?id=15nZSJ2bAT3m-iCBqP3N_9gld5_EGv4kp , then put it to the directory "face_modules/"
* URL for download pre-trained weights of AEI-Net: https://drive.google.com/open?id=1iANX7oJoXCEECNzBEW1xOpac2tDOKeu9 , then put it to the directory "saved_models/"
### Apply FaceShifter to a folder of images
* run `python faceshifter/faceshifter_convert.py -i path/to/donor_image.png -f path/to/receiver_folder`
### Apply FaceShifter to the dataset's face images
* run `python faceshifter/execFaceShifter.py -f datasets/VideoConference/VideoConference-real/images -o datasets/VideoConference/VideoConference-synthesis/images/faceshifter/`
### Train
You can train the model
* set dataset_path in train.py to the actual path to your dataset
* run `python train.py`
* You can modify parameters of BatchSize, epoch, learning rate in train.py
