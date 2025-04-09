# Face Recognition System with DeepFace and FastAPI

This project provides a web-based face recognition system using the DeepFace library and FastAPI. It allows users to register faces to a database and perform facial recognition against the registered faces.

## Features

- **Face Registration**: Add faces to the recognition database with associated names
- **Face Recognition**: Upload images to identify faces against the registered database
- **Web Interface**: Simple and intuitive web UI for interacting with the system
- **API Endpoints**: RESTful API for programmatic access

## Prerequisites

- Python 3.9+
- Git (for cloning the repository)
- Docker (optional for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-deepface.git
cd face-recognition-deepface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t face-recognition-deepface .
```

2. Run the container:
```bash
docker run -p 8000:8000 face-recognition-deepface
```

## Deployment to Railway

This project is configured for easy deployment to Railway:

1. Fork this repository to your GitHub account
2. Create a new project on Railway and connect it to your GitHub repository
3. Railway will automatically deploy the application using the included configuration

### Manual Railway Deployment

Alternatively, you can deploy using Railway CLI:

```bash
railway login
railway init
railway up
```

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

- `GET /`: Web interface for face registration and recognition
- `POST /register`: Register a new face to the database
  - Form data: `file` (image), `name` (person's name)
- `POST /recognize`: Recognize faces in an uploaded image
  - Form data: `file` (image), `threshold` (optional similarity threshold, default: 0.6)

## File Structure

```
├── main.py            # Main FastAPI application
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
├── railway.json       # Railway deployment configuration
├── templates/         # HTML templates
│   └── index.html     # Web interface
├── static/            # Static files (CSS, JS, etc.)
├── uploads/           # Temporary storage for uploaded images
└── database/          # Face database storage
```

## Customization

### Similarity Threshold

The recognition threshold can be adjusted in the UI or by modifying the `threshold` parameter in API calls. Lower values (closer to 0) will result in more matches but potentially more false positives.

### Face Detection Model

The system uses VGG-Face by default. You can modify the `model_name` parameter in the `DeepFace.find()` function in `main.py` to use different models:
- "VGG-Face" (default)
- "Facenet"
- "OpenFace"
- "DeepFace"
- "ArcFace"
- "Dlib"

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) library for face analysis
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
