# 🏸 2025 LINE Bot – Badminton Angle Analysis Assistant

## 📘 Overview

This project integrates **computer vision**, **pose estimation**, and **LLM-based dialogue systems** into a LINE Bot platform to assist badminton players in understanding and improving their movements.
The bot analyzes user-submitted videos, estimates key body angles, and provides personalized feedback and visual overlays through an interactive LINE chat interface.

---

## 🚀 Key Features

* **Pose Estimation Engine**
  Uses MediaPipe/OpenCV-based pose detection to extract skeleton keypoints and compute joint angles.
* **Angle Analysis System**
  Evaluates posture correctness based on pre-defined biomechanical rules for badminton strokes.
* **ChatGPT Integration (RAG System)**
  Retrieves relevant motion data and prior results to generate context-aware responses.
* **Visualization Module**
  Draws annotated skeletons and angle overlays directly on user-submitted videos or frames.
* **LINE Bot Interface**
  Provides a conversational interface for video upload, guidance requests, and performance feedback.

---

## 🧠 System Architecture

```
User (LINE)
   ↓
LINE Messaging API
   ↓
Flask Server / FastAPI
   ↓
│ Pose Detector (OpenCV + MediaPipe)
│ Angle Analyzer (custom algorithm)
│ LLM Response Generator (OpenAI API)
│
↳ Firebase / Local DB (for storage)
```

---

## 🧩 Core Modules

| Module               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `app.py`             | Main Flask application handling LINE webhook events          |
| `pose_estimation.py` | Extracts pose landmarks from images or frames                |
| `angle_analysis.py`  | Computes and compares body angles                            |
| `retrieval.py`       | Retrieves contextual feedback from knowledge base            |
| `utils/`             | Contains helper functions for drawing, data conversion, etc. |

---

## 📱 User Flow

1. **User uploads** a badminton video or photo to the LINE chat.
2. The **server analyzes** the frame and computes key body angles.
3. The **bot replies** with annotated images and an explanatory message.
4. Users can **ask follow-up questions**, such as:

   * “Is my smash motion correct?”
   * “How can I improve my backhand posture?”
   * “Show me my previous performance.”

---

## 🧩 Example Interaction

**User:** “Analyze this video of my serve.”
**Bot:** “Your elbow angle during the swing is 152°, which is slightly lower than the recommended 165°. Try to extend your arm more at the peak of your motion.”
**Bot (Image):** Displays annotated frame with detected skeleton and angles.

---

## ⚙️ Installation

### Prerequisites

* Python 3.10+
* LINE Developers Account
* OpenAI API key (for GPT integration)
* ngrok (for local webhook testing)

### Setup

```bash
git clone https://github.com/HeavenAQ/2025-linebot.git
cd 2025-linebot
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
LINE_CHANNEL_SECRET=your_secret
LINE_CHANNEL_ACCESS_TOKEN=your_token
OPENAI_API_KEY=your_api_key
```

### Run

```bash
python app.py
```

---

## 🧪 Future Development

* Expansion to **real-time video feedback**
* **Personalized learning paths** using user history
* **Multilingual support** for broader accessibility

---

## 📚 Reference

This project is currently under development.

---

## 👨‍💻 Author

**Heaven Chen**

* Master’s Student, Waseda University
* GitHub: [HeavenAQ](https://github.com/HeavenAQ)
* Contact: [LinkedIn](https://www.linkedin.com/in/heavenchen)
