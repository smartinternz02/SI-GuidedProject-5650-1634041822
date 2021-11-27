
#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras import backend
from tensorflow import keras
import tensorflow as tf

# global graph
# # #graph=tf.get_default_graph()
# graph=tf.compat.v1.get_default_graph()

#global graph
#graph = tf.get_default_graph()


from skimage.transform import resize

 #Flask utils

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/crop_protection.h5'

# Load your trained model
model = load_model(r'C:\Users\chink\OneDrive\Desktop\FLASK1\models\gcp1.h5')
       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
#print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
	
@app.route('/Image',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
   if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        y=model.predict(x)
        preds=np.argmax(y,axis=1)
        
        #preds = model.predict_classes(x)
        index = ['cardboard','glass','metal','paper','plastic','trash']
        text = "The Predicted Garbage is : "+str(index[preds[0]])
        
               # ImageNet Decode
        
        return text
    
if __name__ == '__main__':
    app.run(debug=False,threaded = False,port=8000)


