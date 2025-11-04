import os
import io
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import torch
import torchvision.transforms as transforms

# Initialize FastAPI
app = FastAPI(title="Uopo Food Recognition API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_FILE_ID = "1FbSk1uRA_zz6ToxCwqNbb8wi8MU-LOnT"
MODEL_PATH = "/tmp/model.pth"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "/tmp/token.pickle"

model = None
device = torch.device("cpu")

def get_google_drive_service():
    """Authenticate with Google Drive"""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(f"{CREDENTIALS_FILE} not found. Add it to Hugging Face Secrets.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def download_model_from_drive():
    """Download model from Google Drive"""
    if os.path.exists(MODEL_PATH):
        return True

    try:
        service = get_google_drive_service()
        request = service.files().get_media(fileId=MODEL_FILE_ID)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request, chunksize=204800)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download: {int(status.progress() * 100)}%")

        with open(MODEL_PATH, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())

        print("✅ Model downloaded")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def load_model():
    """Load the trained model"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            if not download_model_from_drive():
                raise Exception("Failed to download model")

        model = torch.load(MODEL_PATH, map_location=device)
        model.eval()
        print("✅ Model loaded")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "name": "Uopo Food Recognition API",
        "version": "1.0",
        "model_loaded": model is not None,
        "endpoints": {"/predict": "POST with image file", "/docs": "API documentation"}
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict food from image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model loading...")

    try:
        image = Image.open(io.BytesIO(await file.read()))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        return {
            "success": True,
            "predicted_class": int(predicted_class.item()),
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))