from flask import Flask
from flask import render_template, request

from io import BytesIO
import base64

import numpy as np

import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image  
from keras.applications.resnet50 import preprocess_input                

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential


app = Flask(__name__)

# Get dog breed names from file
dog_names = []
with open('../dog_breed.txt', 'r') as file:
    dog_names = [line.strip() for line in file.readlines()]

# load model for face detect
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model for dog detect
ResNet50_model = ResNet50(weights='imagenet')

# define model architecture and load weights of dog breed detect
InceptionV3_model = Sequential()
InceptionV3_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
InceptionV3_model.add(Dense(133, activation='softmax'))
InceptionV3_model.load_weights('../saved_models/weights.best.InceptionV3.keras')

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def face_detector(file):
    """Detect faces in a image using OpenCV

    Args:
    file: image file that user uploaded

    Returns:
    returns "True" if human face is detected in image
    """
    file_bytes = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def file_to_tensor(file):
    """convert image file to array

    Args:
    file: image file that user uploaded

    Returns:
    returns array of the image that can be consumed by the model
    """
    try:
        img = image.load_img(BytesIO(file), target_size=(224, 224))
        x = image.img_to_array(img)
        # Normalize the image tensor - cause error
        return np.expand_dims(x, axis=0)#.astype('float32')/255
    except IOError:
        print(f"Warning: Skipping corrupted image {file.filename}")
        return None

def ResNet50_predict_labels(file):
    """Get the max value in predtion vector

    Args:
    file: image file that user uploaded

    Returns:
    returns max value in prediction vector, present the main content of the image
    """
    # returns prediction vector for image located at img file
    img = preprocess_input(file_to_tensor(file))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(file):
    """Detect if img file has a dog

    Args:
    file: image file that user uploaded

    Returns:
    returns "True" if a dog is detected in the image stored at img file
    """
    prediction = ResNet50_predict_labels(file)
    return ((prediction <= 268) & (prediction >= 151)) 

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def InceptionV3_predict_breed(file):
    """Detect dog breed in the img file
    if the img file has not dog, it will provide an estimate of the dog breed that is most resembling.

    Args:
    file: image file that user uploaded

    Returns:
    returns: string dog breed name
    """
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(file_to_tensor(file))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def pedict_dog_breed(file):
    """Determines whether the image contains a human, dog, or neither.  Then,
    - if a dog is detected in the image, return the predicted breed.
    - if a human is detected in the image, return the resembling dog breed.
    - if neither is detected in the image, provide output that indicates an error.

    Args:
    file: image file that user uploaded

    Returns:
    returns: string prediction result
    """
    is_human = face_detector(file)# 11% dog is detected as human
    is_dog = dog_detector(file)

    if is_dog:
        dog_breed = InceptionV3_predict_breed(file)
        return f'Hello Dog! Your predicted breed is {dog_breed}'
    elif is_human:
        dog_breed = InceptionV3_predict_breed(file)
        return f'Hello Human! Your predicted breed is {dog_breed}'
    else:
        return f'Hello, You are not Human nor dog'

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
        
    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request has the file part
    if 'file' not in request.files:
        return render_template(
            'upload_form.html',
            message = 'No file part'
        )

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty part
    if file.filename == '':
        return render_template(
            'upload_form.html',
            message = 'No selected file'
        )

    # predict dog breed 
    if file:
        image_data = file.read()
        encoded_img_data = base64.b64encode(image_data).decode('utf-8')
        message = pedict_dog_breed(image_data)
        return render_template(
            'upload_form.html',
            message = message,
            img_data=encoded_img_data, 
            mimetype=file.mimetype
        )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()