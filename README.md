# 🎥 Video–Text Captioning using VLLM with Streamlit Interface

This project demonstrates **Vision–Language understanding** for video data using **BLIP (Bootstrapped Language-Image Pretraining)** a **VLLM (Vision–Language Large Model)**.  
The system automatically extracts frames from videos and generates **descriptive text captions**, showing how **deep learning models align heterogeneous modalities** such as vision and language.

---

## 🎯 Objective
To explore **cross-modal embeddings** and **attention-based multimodal architectures** that align image and text representations for video understanding.  
This work directly aligns with the **Multimodal Analysis and Retrieval** research focus.

---

**📘 How Attention Mechanisms Enable Multimodal Understanding**

Modern multimodal AI models such as CLIP and BLIP use attention mechanisms — the core idea behind Transformers to learn relationships between words and visual elements.

**🧩 1. Self-Attention**

In both the vision and text encoders, self-attention allows each token (word or image patch) to focus on the most relevant parts of its own sequence.
For example:

In text, “cat sitting on the mat,” attention helps relate cat ↔ mat.

In images, attention lets the model highlight regions (e.g., ears, tail) most related to the concept cat.

**🔗 2. Cross-Attention (for Vision–Language Alignment)**

BLIP and other Vision–Language Large Models add cross-attention layers, which connect features across modalities:

The text decoder attends to visual embeddings to generate a caption or retrieve the correct image.

This enables “cross-modal grounding,” meaning the model learns which words correspond to which parts of an image or video frame.

**⚙️ 3. Contrastive or Generative Learning**

CLIP uses contrastive attention: it compares all image–text pairs and increases similarity for matching pairs while reducing it for mismatched ones.

BLIP uses generative attention: it focuses on relevant visual regions while predicting descriptive text tokens sequentially.

**🧠 4. Attention Maps for Explainability**

Attention weights act as interpretable “heatmaps.”
Higher attention scores show which image areas influenced certain words, providing explainability in multimodal retrieval or captioning tasks.

---

**📚 Key Takeaways**

Attention is the mechanism that aligns heterogeneous modalities.

It allows CLIP/BLIP to focus on semantically related regions across text, image, and video.

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

**🖼️ Conceptual Illustration**

```[Text Tokens] → Self-Attention → Text Embeddings
       ↓
 Cross-Attention ↔
       ↑
[Image Patches] → Self-Attention → Visual Embeddings
        ↓
     Shared Embedding Space → Retrieval / Caption Generation

```
Multimodal Pipeline Diagram showing how text, image, audio, and video flow into a shared embedding space

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c101814c-c8e2-485b-9ef4-6c1dd9538651" />

---

```
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

```
---

**🎯 Project Context: Multimodal Analysis and Retrieval**

This research theme investigates advanced deep learning techniques for modeling relationships between heterogeneous modalities text, image, audio, and video through cross-modal embeddings, attention mechanisms, and Vision–Language Large Models (VLLMs).
The objective is to align different feature spaces for efficient information retrieval and semantic understanding, with applications in smart agriculture, smart manufacturing, and digital twin systems.

```
| NCCU Research Focus                 | Demonstrated in This Project                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Cross-Modal Embeddings**          | CLIP and BLIP map image and text features into a shared semantic space.                             |
| **Attention Mechanisms**            | Transformer layers capture relationships among image patches and text tokens for precise alignment. |
| **VLLM Integration**                | BLIP implements a Vision–Language Large Model that decodes captions using cross-attention.          |
| **Heterogeneous Modality Modeling** | Combines visual and linguistic data (video → image frames → text).                                  |
| **Applications in Smart Systems**   | Demonstrated through examples related to manufacturing and digital environments.                    |
```

---

🧩 Research Significance

This project showcases practical application of multimodal representation learning, where attention-based transformers bridge the semantic gap between visual and textual domains.
It reflects the core goals by enabling semantic retrieval, cross-domain understanding, and explainable AI integration essential components of next-generation smart and context-aware systems.

---





