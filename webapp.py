import os
from flask import Flask, flash, request, redirect, url_for
from flask import render_template, jsonify
from werkzeug.utils import secure_filename

# Import dog bree classifier
from dogclf.classifiers import DogBreedClassifier
dogBreedClassifier = DogBreedClassifier()

# Direction where uploaded images by user are stored for classification
UPLOAD_FOLDER = 'cache'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for image extension
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

# Helper function to classify dog bree from a file
def classify_image(file):
    '''
    INPUT:
    file - (FileStorage) file with image to be classified

    OUTPUT:
    message  - (str) output message
    
    Description:
    Store image in a temporary directory and classify the dog breed.
    Returns the message to be rendered in the front end.
    '''
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    (isDog, dog_breed) = dogBreedClassifier.classify_dog_breed(img_path)

    if isDog == 1:
        message = {'message': 'We have a dog there and I think it is a <i> {:} </i><br/>'.format(dog_breed.replace('_',' ')) }
    elif isDog == 0:
        message = {'message': 'Not a dog hun! But isnt just like a <i> {:} </i><br/>'.format(dog_breed.replace('_',' ')) }
    else:
        message = {'message': 'I dont know what to look at'}

    return message

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

    for file in files:
        if file and allowed_file(file.filename):
            message = classify_image(file)
            resp = jsonify(message)
            resp.status_code = 201
            return resp
        else:
            errors[file.filename] = 'File type is not allowed'

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()