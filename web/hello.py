from werkzeug.utils import secure_filename
from flask import Flask,render_template,jsonify,request,send_from_directory,url_for,make_response
import time
import os
import base64
#from flask_bootstrap import Bootstrap
import cv2

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
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
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
    f = request.files['file']  # 从表单的file字段获取文件，file为该表单的name值

    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        print (fname)
        ext = fname.rsplit('.',1)[1]  # 获取文件后缀
        unix_time = int(time.time())
        # new_filename = str(unix_time)+'.'+ext  # 修改了上传的文件名
        #new_filename = '12'+'.'+ext  # 修改了上传的文件名
        save_path = os.path.join(file_dir,fname)
        if os.path.exists(basedir + 'static/test.jpg'):
            os.remove(basedir + 'static/test.jpg')
        f.save(os.path.join(basedir, 'static/', 'test.jpg'))  #保存文件到upload目录
        #img = cv2.imread(fname)
        #cv2.imwrite(save_path, img)
        #token = base64.encode(fname)
        #print (token)
        #<img src="{{ url_for('static', filename= './images/test.jpg') }}" width="400" height="400" alt="你的图片被外星人劫持了～～"/>
        #img_stream = return_img_stream(save_path)
        #response = make_response(image_data)
        #return response
        #img = cv2.imread(fname)
        #cv2.imwrite(save_path, img)
        
        return render_template('showPic.html', fname = fname)
    else:
        return jsonify({"errno":1001, "errmsg":u"failed"})


@app.route('/predict', methods=['GET'],strict_slashes=False)
def predict():
    val = 1
    return render_template('predictValue.html', pred_val = val)

@app.route('/')
def hello_world():
        return 'Hello, World!'
if __name__ == '__main__':
            app.run()
