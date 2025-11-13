# 🎥 Video–Text Captioning using VLLM with Streamlit Interface

This project demonstrates **Vision–Language understanding** for video data using **BLIP (Bootstrapped Language-Image Pretraining)** — a **VLLM (Vision–Language Large Model)**.  
The system automatically extracts frames from videos and generates **descriptive text captions**, showing how **deep learning models align heterogeneous modalities** such as vision and language.

---

## 🎯 Objective
To explore **cross-modal embeddings** and **attention-based multimodal architectures** that align image and text representations for video understanding.  
This work directly aligns with the **Multimodal Analysis and Retrieval** research focus.

---

## 🧩 Features
- 🧠 Uses pre-trained **BLIP model** from Salesforce (Hugging Face)
- 🎥 Samples frames automatically from input video
- 📝 Generates frame-wise and aggregated captions
- ⚙️ Works fully on **CPU or GPU**
- 💡 Demonstrates **VLLM, attention mechanisms, and cross-modal feature alignment**

---

**🧠 Model Details**

**Model:** Salesforce/blip-image-captioning-base

**Architecture:** Vision Transformer (ViT) + GPT-like text decoder

**Concepts Demonstrated:**

Cross-modal feature alignment

Attention mechanisms in multimodal transformers

Vision–Language Large Models (VLLM) for caption generation

---

**🧠 Learning Outcomes**

Implemented a Vision–Language model for real-world video understanding

Learned multimodal data processing and attention-based captioning

Explored temporal reasoning using sampled frame aggregation

---

## 🗂️ Folder Structure

Video-Text-Captioning-VLLM/
│
├── app_video_caption.py # Main CLI captioning script
├── requirements.txt
├── sample_data/ # Input videos
│ └── factory.mp4
├── results/ # Output frames + captions
│ ├── frame_01.jpg
│ ├── frame_01.txt
│ └── ...
└── README.md

---


---

## ⚙️ Installation and Usage

### 1️⃣ Create Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

3️⃣ Run Captioning
```bash
python app_video_caption.py --video sample_data/factory.mp4 --sample_fps 1 --max_frames 12

---

📊 Example Output

Terminal Output

Per-frame captions:
 [1] a man working on a piece of wood
 [2] a man working on a piece of furniture
 [3] a man working on a machine in a room
 ...
===== Final aggregated caption =====
a man working on a table in a kitchen

Result Folder

results/
├── frame_01.jpg
├── frame_01.txt
├── frame_02.jpg
├── frame_02.txt
...


