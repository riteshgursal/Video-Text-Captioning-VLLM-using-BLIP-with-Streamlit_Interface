import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from collections import Counter
from tqdm import tqdm

def sample_frames(video_path, sample_fps=1, max_frames=20):
    """
    Sample frames from video_path at sample_fps frames per second (fps).
    Returns list of PIL.Image frames.
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = vid.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps if video_fps else 0
    frames = []
    # sample every `sample_fps` seconds -> frame index step = video_fps/sample_fps
    if video_fps == 0:
        # fallback: sample first N frames evenly
        step = max(1, total_frames // max_frames)
        idxs = list(range(0, total_frames, step))[:max_frames]
    else:
        step = int(round(video_fps / sample_fps))
        if step <= 0:
            step = 1
        idxs = list(range(0, total_frames, step))[:max_frames]

    for idx in idxs:
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vid.read()
        if not ret:
            continue
        # convert BGR(OpenCV) -> RGB(PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    vid.release()
    return frames, duration

def generate_captions_for_frames(frames, processor, model, device):
    captions = []
    for img in tqdm(frames, desc="Captioning frames"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def aggregate_captions(captions):
    # Option A: majority exact caption (good if many frames identical)
    if not captions:
        return ""
    cnt = Counter(captions)
    most_common_caption, freq = cnt.most_common(1)[0]
    # Option B: if captions vary, join by semicolon / choose longest or best-scored frame
    if freq >= max(2, len(captions)//3):
        return most_common_caption
    # else join short unique captions into a summary
    unique = list(dict.fromkeys(captions))  # preserve order
    summary = " / ".join(unique[:5])  # limit length
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video (mp4, mkv, etc.)")
    parser.add_argument("--sample_fps", type=float, default=1.0, help="Frames to sample per second")
    parser.add_argument("--max_frames", type=int, default=12, help="Max number of sampled frames")
    parser.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-base", help="HuggingFace BLIP model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)

    print("Sampling frames...")
    frames, duration = sample_frames(args.video, sample_fps=args.sample_fps, max_frames=args.max_frames)
    print(f"Sampled {len(frames)} frames from video (duration ~{duration:.2f}s).")

    if not frames:
        print("No frames sampled. Exiting.")
        return

    captions = generate_captions_for_frames(frames, processor, model, device)
    print("\nPer-frame captions:")
    for i, c in enumerate(captions):
        print(f" [{i+1}] {c}")

    final_caption = aggregate_captions(captions)
    print("\n===== Final aggregated caption =====")
    print(final_caption)

    # optional: save per-frame images + captions to results/
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    for i, (img, cap) in enumerate(zip(frames, captions)):
        fn = os.path.join(results_dir, f"frame_{i+1:02d}.jpg")
        img.save(fn)
        with open(os.path.join(results_dir, f"frame_{i+1:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(cap)
    print(f"Saved sampled frames and captions to {results_dir}/")

if __name__ == "__main__":
    main()
