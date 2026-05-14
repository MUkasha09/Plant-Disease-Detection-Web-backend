# Plant Disease Detection Backend API

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The API will be available at `https://your-render-app.onrender.com`

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/me` - Get current user info

### Prediction
- `POST /api/predict` - Upload image and get disease prediction

### History
- `GET /api/history` - Get user's prediction history

## Environment Variables
- `SECRET_KEY` - Flask secret key
- `DATABASE_URL` - PostgreSQL database URL (for production)
- `FLASK_ENV` - Set to "production" for deployment