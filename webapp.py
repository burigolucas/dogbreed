import os
from flask import Flask, flash, request, redirect, url_for
from flask import render_template, jsonify
from werkzeug.utils import secure_filename

import dogclf
from dogclf import dog_detector, dog_breed_detector, face_detector        

UPLOAD_FOLDER = 'tmp_files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# check for image extension
def allowed_file(filename):
    '''
    INPUT:
    filename     - (str) filename provided by the user 

    OUTPUT:
    allowed_file - (boolean) True if filename is of allowed extension
    
    Description:
    Check if provided file is an image with an allowed extension
    '''    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# index webpage with dog breed detector app
@app.route('/')
@app.route('/index')
def index():

    welcomeMessage = '''
<p>Hi there =)</p>
<p>I have been learning a few new things lately, 
    and my new game is to try to guess the breed of dogs!
    <p>Don't U wanna play with me?</p>
    <p>We can have fun together, let's do it!</p>
    <p>Just upload an image and let me guess what is it!</p>
    <p>But be patient if I get it wrong, I am just like a small child =)</p>
</p>
'''
    return render_template('master.html',message=welcomeMessage)

# add route to upload image for detection
@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():

    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'

    if success:

        print("Classify image: {:}".format(img_path))
        
        dog_breed = 'not a dog'
        isDog = dog_detector(img_path)
        
        if isDog:
            dog_breed = dog_breed_detector(img_path)
            response = {'message': 'We have a dog there and I think it is a <i> {:} </i><br/>'.format(dog_breed.replace('_',' ')) }
        else:
            if face_detector(img_path):
                dog_breed = dog_breed_detector(img_path)
                response = {'message': 'Not a dog hun! But isnt just like a <i> {:} </i><br/>'.format(dog_breed.replace('_',' ')) }

        resp = jsonify(response)
        resp.status_code = 201

        return resp

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
