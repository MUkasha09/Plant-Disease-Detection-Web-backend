from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import json
import uuid
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS for frontend
CORS(app, supports_credentials=True, origins=[
    'https://plant-disease-detection-web-frontend-cdjtx740g.vercel.app',
    'https://plant-disease-detection-web-fronten.vercel.app',
    'http://localhost:3000',
    'http://localhost:5173'
])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'api_login'

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

# History Model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Initialize the database
def init_db():
    with app.app_context():
        db.create_all()

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load AI model with error handling
def load_model():
    try:
        # Use a relative path or environment variable for model path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'plant_disease_recog_model_pwp.keras')
        
        # Try loading with Keras 3.x compatibility
        try:
            # First attempt: load with Keras 3.x (newer TensorFlow)
            import keras
            model = keras.saving.load_model(model_path)
            return model
        except Exception as e:
            print(f"Keras 3.x load failed: {e}")
            
            # Second attempt: try with custom objects for compatibility
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'Functional': tf.keras.Model}
                )
                return model
            except Exception as e2:
                print(f"Custom objects load failed: {e2}")
                return None

# Load disease labels with error handling
def load_disease_labels():
    try:
        labels_path = os.path.join(os.path.dirname(__file__), 'plant_disease.json')
        with open(labels_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading disease labels: {e}")
        return {}

# Load model and labels
model = load_model()
plant_disease = load_disease_labels()

# Function to extract features
def extract_features(image_path):
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        image_array = tf.keras.utils.img_to_array(image)
        return np.array([image_array])
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Model prediction function
def model_predict(image_path):
    if not model or not plant_disease:
        return {"error": "Model or labels not loaded"}
    
    img = extract_features(image_path)
    if img is None:
        return {"error": "Image processing error"}
    
    try:
        prediction = model.predict(img)
        predicted_label = plant_disease[prediction.argmax()]
        return {"disease": predicted_label, "confidence": float(prediction[0].max())}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": "Prediction failed"}

# ==================== API ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# --- Auth Endpoints ---

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user"""
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    
    # Check if user already exists
    existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        return jsonify({"error": "Username or email already exists"}), 400
    
    # Create new user
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "Registration successful", "user_id": new_user.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Registration failed"}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        session['user_id'] = user.id
        return jsonify({
            "message": "Login successful",
            "user": {"id": user.id, "username": user.username, "email": user.email}
        }), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def api_logout():
    """Logout user"""
    logout_user()
    return jsonify({"message": "Logout successful"}), 200

@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user info"""
    return jsonify({
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }), 200

# --- Prediction Endpoints ---

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """Upload image and get disease prediction"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if image and allowed_file(image.filename):
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(image.filename)}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            image.save(image_path)
            prediction = model_predict(image_path)
            
            if "error" in prediction:
                return jsonify(prediction), 500
            
            # Save history
            new_history = History(
                user_id=current_user.id,
                image_path=f"/static/uploads/{unique_filename}",
                disease_name=prediction['disease']
            )
            db.session.add(new_history)
            db.session.commit()
            
            return jsonify({
                "image_url": f"/static/uploads/{unique_filename}",
                "prediction": prediction['disease'],
                "confidence": prediction['confidence']
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

# --- History Endpoints ---

@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """Get user's prediction history"""
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    
    history_data = [{
        "id": h.id,
        "image_path": h.image_path,
        "disease_name": h.disease_name,
        "timestamp": h.timestamp.isoformat() if h.timestamp else None
    } for h in user_history]
    
    return jsonify({"history": history_data}), 200

# Initialize database when running the app
init_db()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
