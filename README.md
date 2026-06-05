---
title: FaceFetch
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ⚡ EventAI | Deep Face Search Engine


![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css)

**EventAI** is an intelligent, high-performance facial recognition search engine. Instead of manually scrolling through hundreds of event photos, users upload a few selfies to generate a "Master Face DNA." The AI then scans massive public Google Drive folders or local albums, perfectly isolating every photo the user appears in, and delivers them as a convenient ZIP file.

---

## ✨ Key Features

* 🧬 **Master Face DNA:** Upload 3-5 reference selfies. The engine utilizes professional ML reference-averaging to merge them into a single, highly accurate 128-point facial encoding.
* 🚀 **Dual-Engine Scanning:** Toggle between **Fast Mode** (HOG-based, ~0.1s/photo) and **Accurate Mode** (CNN-based with upsampling, ~1s/photo) depending on the crowd size.
* 🎛️ **Dynamic AI Strictness:** Real-time Match Tolerance slider (0.35 to 0.65) allowing users to fine-tune the AI's Euclidean distance thresholds to eliminate false positives in group photos.
* 📂 **Multi-Source Input:** Scan Google Drive folders directly via link, or perform bulk uploads from your local device.
* 🔄 **Resilient OAuth 2.0:** Secure Google login with automated background Refresh Token swapping—ensuring the engine never drops connection during massive 2,000+ photo scans.
* 📡 **Live SSE Streaming:** Real-time Server-Sent Events stream progress, estimated time of arrival (ETA), hit rates, and live match thumbnails directly to the UI.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript, `face-api.js` (for client-side pre-validation).
* **Backend:** Python, FastAPI, `face_recognition` (dlib core), Pillow, OpenCV.
* **Database:** MongoDB Atlas (Motor async driver) for securely storing user Face DNA encodings and offline tokens.
* **Authentication:** Google OAuth 2.0 via Authlib & Starlette Sessions.

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Nagendraas612/EventAI.git](https://github.com/Nagendraas612/EventAI.git)
cd EventAI
