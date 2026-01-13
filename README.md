# 2025 LINE Bot Monorepo — Frontend, Backend, Pose Estimation

This repository hosts a complete LINE-based coaching experience for racket skills (e.g., serve, smash, clear):

- Frontend: a LIFF app built with Next.js that users open inside LINE.
- Backend: a Go server that handles LINE webhooks, user state, storage, and analytics APIs.
- Pose Estimation: a Python FastAPI service that analyzes uploaded videos, grades technique, and returns an annotated clip.

The three services are developed and run independently, but work together in local and production environments.

---

## Repository Layout

- `liff/` — Next.js 15 LIFF app, TailwindCSS, client-side auth with `@line/liff`.
- `linebot/` — Go 1.23 backend, LINE webhook + APIs, Firestore, Cloud Storage, GPT summarization, ffmpeg processing.
- `angle_analysis_ai/` — FastAPI server, Ultralytics YOLO pose detection, returns graded/annotated video.

---

## Prerequisites

- OS: macOS, Linux, or Windows (WSL recommended)
- Node.js: 18+ (Next.js 15 recommends Node 18+)
- Go: 1.23+
- Python: 3.10–3.11 recommended
- ffmpeg: CLI available on PATH (used by the backend for resize/thumbnail/stitching)
- Disk: Sufficient space for model weights and temporary videos

Install ffmpeg examples:
- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`

---

## Quick Start (Local, 3 terminals)

1) Pose Estimation API (Python/FastAPI)
- Path: `angle_analysis_ai/`
- Create venv and install dependencies:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Ensure the pose model file exists: `angle_analysis_ai/yolo11m-pose.pt` (already included in the repo)
- Start API: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Health check: `curl http://127.0.0.1:8000/health`

2) Backend (Go)
- Path: `linebot/`
- Ensure `ffmpeg` is installed and available on PATH
- Copy and edit environment variables:
  - `cp .env .env.local` (or create a secure `.env` with your own values)
- Start server: `go run main.go`
- Health check: `curl http://127.0.0.1:8080/test`

3) Frontend (Next.js/LIFF)
- Path: `liff/`
- Configure `liff/.env`:
  - `NEXT_PUBLIC_LIFF_ID=<your-liff-id>`
  - `NEXT_PUBLIC_BACKEND_BASE_URL=http://127.0.0.1:8080`
- Install deps: `npm install`
- Start dev server: `npm run dev`
- Open `http://localhost:3000` in a browser (LIFF flows are intended to run inside LINE, but most pages render locally for development).

---

## How It Works (End-to-End)

- User opens the LIFF app in LINE and navigates to the skill workflow.
- The backend receives LINE webhook events at `/callback` and guides the user through selecting skill/handedness and uploading a video.
- When a video is received, the backend:
  - Saves a temporary copy and resizes it with `ffmpeg`.
  - Sends it to the Pose Estimation API at `POST /upload` with HTTP Basic Auth and form fields: `skill`, `handedness`, and the `video` file.
  - Receives JSON containing a base64-encoded annotated clip and a grading payload.
  - Stitches the annotated clip with a corresponding expert video using `ffmpeg` (see `linebot/pro_videos/`).
  - Uploads the annotated and comparison videos (and a generated thumbnail) to Cloud Storage and persists metadata to Firestore.
  - Optionally summarizes analysis chats via GPT and caches per-day summaries in Firestore.
- LIFF fetches data from the backend APIs to render portfolio items, stats, and summaries.

---

## Frontend (LIFF) — `liff/`

- Framework: Next.js 15 + React 19 + TailwindCSS
- LIFF integration: `@line/liff` with a provider at `src/app/LiffProvider.tsx`
- Required env (`liff/.env`):
  - `NEXT_PUBLIC_LIFF_ID` — your LIFF app ID
  - `NEXT_PUBLIC_BACKEND_BASE_URL` — e.g. `http://127.0.0.1:8080`
- Useful scripts:
  - `npm run dev` — Dev server on `0.0.0.0:3000`
  - `npm run build` — Build production bundle
  - `npm start` — Start production server
- Entry page: `liff/src/app/page.tsx` redirects to `/personal` after LIFF boot/login
- Backend URL resolution: `liff/src/utils/env.ts`

---

## Backend (Go) — `linebot/`

- Language/Version: Go 1.23
- Web framework: `gin`
- External services: LINE Messaging API, Google Cloud Firestore/Storage, OpenAI GPT
- ffmpeg usage: resize incoming videos, extract thumbnail, hstack with expert video
- Core routes:
  - `POST /callback` — LINE webhook endpoint
  - `GET /test` — Simple health ping
  - `GET /api/chat/history?user_id=...&skill=...` — Chat history (with optional skill filter)
  - `POST /api/chat/summarize` — Summarize chat content and cache by user/day/skill
  - `GET /api/db/user?user_id=...` — Fetch user profile
  - `GET /api/db/users` — List users
  - `GET /api/db/stats/users/:id?skill=...` — Per-user skill stats
  - `GET /api/db/stats/class?skill=...` — Class-level aggregates by skill

Environment variables (see `linebot/config/config.go` and `.env`):

- General
  - `PORT` — backend port (default used in this repo: `8080`)
- LINE
  - `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_TOKEN`
- GCP
  - `GCP_PROJECT_ID`
  - `GCP_CREDENTIALS` — path or alias to service account credentials (see your setup)
  - Storage: `GCS_BUCKET_NAME`
  - Secret Manager: `GCP_SECRET_VERSION`
  - Firestore: `FIREBASE_DATA_DB`, `FIREBASE_SESSION_DB`
- GPT
  - `OPENAI_API_KEY`, `OPENAI_PROMPT_ID`
- Pose Estimation server
  - `POSE_ESTIMATION_SERVER_HOST` — e.g. `http://127.0.0.1:8000`
  - `POSE_ESTIMATION_SERVER_USER` — default FastAPI Basic Auth username: `admin`
  - `POSE_ESTIMATION_SERVER_PASSWORD` — default password: `thisisacomplicatedpassword`
- Debugging
  - `SAVE_PROCESSED_VIDEOS=1` — persist intermediate videos locally
  - `SKIP_EXTERNAL_CLIENTS=1` — start without Firestore/Storage/GPT clients (limited functionality)

Run locally:
- `go run main.go`
- Requires `ffmpeg` installed and on PATH

Docker (example):
- Build: `docker build -t linebot:local ./linebot`
- Run: `docker run --rm -p 8080:8080 --env-file ./linebot/.env linebot:local`

---

## Pose Estimation API (Python) — `angle_analysis_ai/`

- Framework: FastAPI
- Model: Ultralytics YOLO (weights: `yolo11m-pose.pt` in the same folder)
- Key modules: `PoseModule.py`, `VideoProcessor.py`, `Grader.py`, `Types.py`
- Endpoints:
  - `GET /health` — health probe
  - `POST /upload` — Basic Auth protected, accepts multipart form data:
    - file field `video` — user video (mp4)
    - form `skill` — one of the supported skills (e.g., `serve`, `smash`, `clear`)
    - form `handedness` — `left` or `right`
    - Response: JSON `{ processed_video: <base64 mp4>, grade: {...} }`

Start locally:
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `uvicorn main:app --host 0.0.0.0 --port 8000`

Example request:
```
curl -u admin:thisisacomplicatedpassword \
  -F "video=@/path/to/video.mp4" \
  -F "skill=serve" \
  -F "handedness=right" \
  http://127.0.0.1:8000/upload
```

Notes:
- The service auto-detects compute device: CUDA > Apple MPS > CPU.
- Outputs an annotated segment and angle metrics for grading.

---

## Data & Storage

- Expert videos for comparison: `linebot/pro_videos/pro_<handedness>_<skill>.mp4`
- Temporary workspace: `linebot/tmp/` (resized/stiched/intermediate videos)
- Backend uploads to Google Cloud Storage and stores metadata in Firestore.
- LIFF consumes direct GCS object URLs for playback.

---

## Troubleshooting

- ffmpeg not found: Install and ensure it is on PATH (`ffmpeg -version`).
- Pose API 401: Verify Basic Auth credentials match backend env.
- Large Python installs: consider Python 3.10/3.11 and a clean venv; on Linux, preinstall `build-essential` and `python3-dev` if needed.
- GPU not used: Verify CUDA drivers or Apple Metal support; service falls back to CPU.
- GCP permission errors: Confirm service account credentials and bucket name; ensure Firestore/Storage APIs are enabled.
- CORS issues in LIFF: Backend allows `http://localhost:3000` and the deployed LIFF origin; update CORS config if needed.

---

## Useful Paths

- Frontend entry: `liff/src/app/page.tsx`
- Backend entry: `linebot/main.go`
- Pose API entry: `angle_analysis_ai/main.py`
- Backend video pipeline: `linebot/app/video_utils.go`
- Pose detection core: `angle_analysis_ai/PoseModule.py`, `angle_analysis_ai/VideoProcessor.py`

