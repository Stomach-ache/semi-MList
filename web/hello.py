from werkzeug.utils import secure_filename
from flask import Flask,render_template,jsonify,request,send_from_directory,url_for,make_response
import time
import os
import base64
#from flask_bootstrap import Bootstrap
import cv2
import pickle

app = Flask(__name__)
#Bootstrap(app)

UPLOAD_FOLDER='upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['txt','png','jpg','xls','JPG','PNG','xlsx','gif','GIF'])

#app.run(port=40)

@app.route('/',methods=['GET'],strict_slashes=False)
def indexpage():
    return render_template('index.html')


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def return_img_stream(img_local_path):
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

@app.route('/',methods=['POST'],strict_slashes=False)
def upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['file']

    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print (fname)
        ext = fname.rsplit('.',1)[1]
        unix_time = int(time.time())
        save_path = os.path.join(file_dir,fname)
        if os.path.exists(basedir + 'static/test.jpg'):
            os.remove(basedir + 'static/test.jpg')
        f.save(os.path.join(basedir, 'static/', 'test.jpg'))
        #img = cv2.imread(fname)
        #cv2.imwrite(save_path, img)
        #token = base64.encode(fname)
        #print (token)
        #img_stream = return_img_stream(save_path)
        #response = make_response(image_data)
        #return response
        #img = cv2.imread(fname)
        #cv2.imwrite(save_path, img)

        return render_template('showPic.html', fname = fname)
    else:
        return jsonify({"errno":1001, "errmsg":u"failed"})



# load the isolation forest model
isolationForest = pickle.load(open('./isolationForest_model.sav', 'rb'))


@app.route('/predict', methods=['GET'],strict_slashes=False)
def predict():
    # fea, lbl = deep_model_predict('./static/test.jpg')
    val = 1
    # y = isolationForest.predict(fea)
    # if y == 0:
    #    msg = "Yeap! We\'ve got a new whale"
    # else:
    #    msg = 'The ID of this whale is: ' + str(lbl)
    return render_template('predictValue.html', pred_val = msg)

@app.route('/')
def hello_world():
        return 'Hello, World!'
if __name__ == '__main__':
            app.run()
