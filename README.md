# ⚡ FaceFetch (EventAI)

FaceFetch is a modern, high-performance web application designed to automatically scan massive event albums and find every photo containing your face. Simply upload a few selfies to create your **Face DNA**, point the app to a Google Drive folder or upload local photos, and let the AI find you instantly!

![FaceFetch Preview](https://github.com/Nagendraas612/FaceFetch/assets/preview-placeholder.png) <!-- Update with an actual screenshot -->

## ✨ Features

- 🧬 **Biometric Face DNA:** Upload 3-5 selfies to create an encrypted, highly accurate reference matrix of your face.
- ⚡ **Hybrid Scan Architecture:** Leverages client-side AI (`face-api.js`) for lightning-fast pre-filtering (skips photos with no faces instantly) and server-side matching (`dlib/face_recognition`) for maximum accuracy.
- 📂 **Google Drive Integration:** Seamlessly scan public Google Drive event folders without downloading the entire album to your device.
- 🚀 **Built for Scale:** Optimized to run efficiently on low-memory environments (like Render's Free Tier) by keeping image batches small and using the optimized HOG detection model.
- 🔒 **Privacy First:** Biometric data is stored securely in MongoDB and photos from Google Drive are proxied safely.
- 📦 **One-Click ZIP Download:** Easily download a curated ZIP file containing only the photos where you appear!

---

## 🛠 Tech Stack

- **Frontend:** Vanilla JS, HTML5, CSS3, `face-api.js` (Client-side AI), JSZip
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **AI/ML:** `dlib`, `face_recognition`, NumPy, OpenCV, Pillow
- **Database:** MongoDB (Motor Async)
- **Auth:** Google OAuth 2.0

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.11+
- CMake & C++ Build Tools (Required to compile `dlib`)
- MongoDB cluster (Atlas or local)
- Google Cloud Console Project (for OAuth Client ID & Secret)

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/Nagendraas612/FaceFetch.git
cd FaceFetch
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (do not commit it!) and add your credentials:
```env
MONGO_URI=your_mongodb_connection_string
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_secret
SESSION_SECRET=a_long_random_string_for_security
ENVIRONMENT=development
```

### 4. Run the Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
The app will be available at: `http://localhost:8000`

---

## ⚙️ How It Works (The Hybrid Pipeline)

1. **Pre-Filtering:** The client browser fetches image data and runs an SSD MobileNet model (`face-api.js`) to ensure at least one human face exists in the photo.
2. **Batching:** Valid photos are grouped into micro-batches and sent to the server.
3. **Enhancement:** The server applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection in poor lighting.
4. **Encoding & Matching:** The `dlib` HOG model detects bounding boxes, extracts the 128-d face encodings, and compares them against the user's saved Face DNA using a strict distance threshold.
5. **Results:** Matched photos are immediately pushed back to the client interface for live preview and downloading.

---

## ⚠️ Notes on Free Tier Hosting (e.g., Render)

FaceFetch is heavily optimized to run on 512MB RAM environments. The **CNN model** option has been intentionally disabled in the UI for free-tier deployments as deep-learning facial recognition requires significantly more RAM and a dedicated GPU to prevent Out-Of-Memory (OOM) crashes. The optimized **HOG model** provides an excellent balance of speed and accuracy!

---

## 📄 License
This project is for personal use and portfolio demonstration.
