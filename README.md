### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This is the python scripts that can be run with Python versions 3.*<br>
Makesure you have following packages installed:
1. flask
2. OpenCV
3. ResNet50

## Project Motivation<a name="motivation"></a>

For this project, I build the web application which can predict the dog breed.<br>
The application accepts any user-supplied image as input.  <br>
If a dog is detected in the image, it will provide an estimate of the dog's breed.  <br>
If a human is detected, it will provide an estimate of the dog breed that is most resembling.<br>

## File Descriptions <a name="files"></a>

#### dog_app.ipynb
This notebook walks you through one of the most popular Udacity projects across machine learning and artificial intellegence nanodegree programs.<br>
You maynot run somepart of this notebook in your local computer as the script require the images files and other resources.

#### dog_app-Working.ipynb
Original detect dogs using ResNet50 was not working, I created it to investigated the issue

#### run.py
1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

#### master.html
The html template for home page when user access the web application

#### upload_form.html
The html template to display predicted dog breed after users choose the image and click on "Clasify Dog"

#### dog_breed.txt
List of 133 dog breed names the application can casify

#### extract_bottleneck_features.py
Extract output of bottleneck features

## Results<a name="results"></a>
1. The web app can predict dog breed from image using InceptionV3
2. I went through the process to build CNN architecture from scratch and using Transfer Learning to clasify dog breeds
3. Using Transfer Learning make the algorithm much more accurate and reduce a lot of training time.
4. Explored pre-trained model OpenCV, ResNet50, VGG16, VGG19 and InceptionV3
5. Have fun with dog breed classification app

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Feel free to use the code here as you would like! <br>
