import os
from flask import Flask, render_template, request
from flask import send_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as k
import warnings
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
warnings.filterwarnings('ignore')

app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'





# on thread 1
session = tf.compat.v1.Session(graph=tf.Graph())
with session.graph.as_default():
    set_session(session)
 #load model at very firs
    model = tf.keras.models.load_model(STATIC_FOLDER + '/' + 'weights.hdf5')


# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(224, 224, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
          # on thread 2
    with session.graph.as_default():
            set_session(session)
            predicted=model.predict(data)
   
    return predicted


# home page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/')
def index():
    return '<h1> this is my home page</h1>'



# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        
        result = api(full_name)

        pred_prob = np.asscalar(np.argmax(result, axis=1))


        if pred_prob==0:
            label = 'benign'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'malignant'
            accuracy = round((1 - pred_prob) * 100, 2)


    return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
    app.config[SERVER_NAME]='localhost:5000'
