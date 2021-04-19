from flask import Flask, render_template, request, send_file, make_response
import os
from flask_cors import CORS
import io
from PIL import Image

from backend.predict_classification import process_image, predict, init_model
from backend.predict_segmentation import *

app = Flask(__name__, template_folder='./public')
CORS(app)

model = init_model()


# # home endpoint, render html file
# @app.route('/')
# def render():
#     return render_template('index.html')

# classification endpoint
@app.route('/classify', methods=['POST'])
def classify():
    img_url = request.get_json(force=True)['image']
    process_image(img_url)
    print("predicting with model...")
    return {'classifications': predict(model).tolist()}


# segmentation endpoint
@app.route('/segment', methods=['POST'])
def segment():
    img_url = request.get_json(force=True)['image']
    process_image_segmentation(img_url)
    arr = readImage()

    # convert numpy array to PIL Image
    img = Image.fromarray(arr.astype('uint8'))

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    response = make_response(send_file(file_object, mimetype='image/PNG'))
    response.headers['Content-Transfer-Encoding'] = 'base64'

    return response


# run app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='', port=port)
