from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '490d659d103985fe9c5b9ad14b4a6113'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the models
cancer_model = load_model('final_mobilenet_model.keras')  # Main model (no normalization)
skin_detector_model = load_model('skin_vs_nonskin_classifier.h5')  # Sigmoid model (needs normalization)

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess for skin vs non-skin (normalized)
def preprocess_for_skin_check(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Preprocess for cancer model (raw, no normalization)
def preprocess_for_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html', name=name, username=username)

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You will be redirected to login', 'success')
        return render_template('register.html', name=name, username=username, redirect=True)

    return render_template('register.html')

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Step 1: Check if the image is a skin image
            img_for_skin_check = preprocess_for_skin_check(filepath)
            skin_pred = skin_detector_model.predict(img_for_skin_check)
            is_skin = int(skin_pred[0][0] > 0.5)  # sigmoid output

            if is_skin != 1:
                flash('Please upload a skin lesion image.', 'danger')
                return redirect(url_for('upload'))

            # Step 2: Run the cancer classifier (no normalization)
            img_for_cancer = preprocess_for_cancer(filepath)
            predictions = cancer_model.predict(img_for_cancer)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            return redirect(url_for('show_results',
                                    diagnosis=predicted_class,
                                    confidence=round(confidence, 2),
                                    image_path=filename))

    return render_template('upload.html')

@app.route('/results')
def show_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    diagnosis = request.args.get('diagnosis')
    confidence = request.args.get('confidence')
    image_path = request.args.get('image_path')
    return render_template('results.html',
                           diagnosis=diagnosis,
                           confidence=confidence,
                           image_path=image_path)

@app.route('/info/<diagnosis>')
def show_info(diagnosis):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if diagnosis in class_names:
        return render_template(f'{diagnosis}.html', diagnosis=diagnosis)
    return "Information not found", 404

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
