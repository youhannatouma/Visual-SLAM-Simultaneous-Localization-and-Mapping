# 🧠 AI Visual Navigation System

### Visual Odometry + Object Detection + AI Reasoning

---

## 📌 Overview

This project implements a **real-time AI-driven visual navigation system** that combines:

* Computer Vision (feature tracking)
* Deep Learning (object detection)
* AI Logic (rule-based reasoning)

The system uses a camera to **perceive its environment**, estimate motion, and **make decisions based on detected objects**.

---

## 🎯 Objectives

* Detect objects in real time using AI
* Estimate camera motion using visual features
* Build a simple understanding of the environment
* Apply AI reasoning to make decisions

---

## 🧠 System Architecture

```
Camera Input
    ↓
Object Detection (YOLOv8)
    ↓
Feature Tracking (ORB - Visual Odometry)
    ↓
State Interpretation
    ↓
AI Reasoning Engine
    ↓
Decision Output
    ↓
Visualization
```

---

## ⚙️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy

---

## 🔍 Features

* ✅ Real-time object detection
* ✅ ORB feature tracking (motion awareness)
* ✅ Rule-based AI reasoning system
* ✅ Live decision overlay
* ✅ Lightweight and real-time performance

---

## 🧠 AI Reasoning System

The system includes a **rule-based decision engine**:

* IF a *chair* is detected → **Target the chair**
* IF a *person* is detected → **Avoid**
* ELSE → **Explore the environment**

This demonstrates **symbolic AI logic and decision-making** based on perception.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the project

```bash
python main.py
```

---

## 📷 Demo

The system will:

* Open your webcam
* Detect objects in real time
* Track visual features
* Display decisions such as:

  * TARGET CHAIR
  * AVOID PERSON
  * EXPLORE

---

## 📁 Project Structure

```
project/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👥 Team Roles

* **Vision & Detection** → Object detection (YOLO)
* **Motion Estimation** → Feature tracking (ORB)
* **AI Logic** → Reasoning system and decision-making

---

## ⏱️ Development Timeline

* Week 1 → Setup + detection + motion tracking
* Week 2 → Reasoning system + integration + demo

---

## 🎓 Academic Context

This project was developed for a **3rd-year AI course**, focusing on:

* Perception → reasoning → action pipeline
* Integration of deep learning and symbolic AI
* Real-time intelligent systems

---

## ⚠️ Limitations

* No full 3D mapping (uses simplified visual odometry)
* Basic rule-based reasoning (not learning-based)
* Approximate motion estimation

---

## 🚀 Future Improvements

* Add object tracking across frames
* Improve motion estimation accuracy
* Implement path planning (A*)
* Upgrade reasoning with probabilistic models
* Integrate with robotics (ROS)

---

## � Training the Reasoning Model

This project now includes a learnable reasoning engine. The new workflow is:

1. Run the app with `python main.py`.
2. Use the keyboard to label examples while the camera is running:
   * `a` → save `AVOID_PERSON`
   * `c` → save `MOVE_TO_CHAIR`
   * `t` → save `CHECK_TABLE`
   * `e` → save `EXPLORE`
3. Train the model from the logged data:

```bash
python train_reasoning.py --data data/reasoning_data.csv --model models/reasoning_model.pt
```

4. Restart `main.py`.

The application will automatically load `models/reasoning_model.pt` if it exists and use the learned policy to choose decisions.

---

## �📄 License

This project is for academic and educational use.

---

## 💬 Acknowledgments

* Open-source computer vision and AI libraries
* YOLOv8 by Ultralytics
* OpenCV community

---
