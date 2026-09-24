"""DEPRECATED (2026-05-10): One-shot debug script. Hardcodes paths for old /home/jxy/MemoryGuard workspace.
Kept for reference only. Use `ai2thor_grounded_sam2_verifier.py` for production GSAM verification."""
import json, sys, math, os
from pathlib import Path

# --- Config ---
FRAME = Path("results/ai2thor_rearrangement_6scene_seed29_capture/images/FloorPlan301/after/01_frame.jpg")
GROUNDING_CONFIG = Path("/home/jxy/.cache/memoryguard-grounded-sam2-src-unpacked2/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_CKPT = Path("/home/jxy/MemoryGuard/checkpoints/groundingdino_swint_ogc.pth")
SAM2_CONFIG = Path("/home/jxy/.cache/memoryguard-grounded-sam2-src-unpacked2/sam2-main/sam2/configs/sam2/sam2.1_hiera_t.yaml")
SAM2_CKPT = Path("/home/jxy/MemoryGuard/checkpoints/sam2.1_hiera_tiny.pt")
DEVICE = "cuda"
LABEL = "BaseballBat"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# --- Load models ---
print("Loading GroundingDINO...")
from groundingdino.util.inference import load_model, load_image, predict
model = load_model(str(GROUNDING_CONFIG), str(GROUNDING_CKPT), device=DEVICE)

print("Loading SAM2...")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(str(SAM2_CONFIG), str(SAM2_CKPT), device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# --- Run detection ---
print(f"Detecting '{LABEL}' in {FRAME.name}...")
import numpy as np
from PIL import Image

image_source, image_tensor = load_image(str(FRAME))
boxes, logits, phrases = predict(
    model=model, image=image_tensor, caption=f"{LABEL}.",
    box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE,
)

width, height = Image.open(FRAME).size
print(f"Frame size: {width}x{height}, diagonal={math.sqrt(width*width + height*height):.1f}")

# --- Known before bbox ---
# From manifest: [262, 259, 287, 300]
before_bbox = [262, 259, 287, 300]
before_center = ((before_bbox[0] + before_bbox[2]) / 2, (before_bbox[1] + before_bbox[3]) / 2)
print(f"Before remembered center (from manifest): ({before_center[0]:.1f}, {before_center[1]:.1f})")

print(f"\nDINO raw detections: {len(boxes)}")
for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
    # box = [cx, cy, bw, bh] normalized [0,1]
    cx, cy, bw, bh = [float(v) for v in box.tolist()]
    cx_px = cx * width
    cy_px = cy * height
    bw_px = bw * width
    bh_px = bh * height
    x1 = (cx - bw/2) * width
    y1 = (cy - bh/2) * height
    x2 = (cx + bw/2) * width
    y2 = (cy + bh/2) * height
    print(f"\n  Detection {i}: score={float(logit):.4f}, label='{phrase}'")
    print(f"    Norm box [cx,cy,bw,bh]: [{cx:.4f}, {cy:.4f}, {bw:.4f}, {bh:.4f}]")
    print(f"    Pixel box [x1,y1,x2,y2]: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"    Pixel center (DINO only): ({cx_px:.1f}, {cy_px:.1f})")
    pixel_dist_dino = math.sqrt((cx_px - before_center[0])**2 + (cy_px - before_center[1])**2)
    norm_dist_dino = pixel_dist_dino / math.sqrt(width*width + height*height)
    print(f"    Distance from remembered: {pixel_dist_dino:.1f}px = {norm_dist_dino:.4f} (norm)")

    # --- SAM2 refinement ---
    predictor.set_image(np.asarray(image_source))
    masks, _, _ = predictor.predict(
        box=np.asarray([x1, y1, x2, y2]),
        multimask_output=False,
    )
    mask = masks[0]
    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        sam2_cx = float(xs.mean())
        sam2_cy = float(ys.mean())
        print(f"    SAM2 mask centroid: ({sam2_cx:.1f}, {sam2_cy:.1f})")
        pixel_dist_sam2 = math.sqrt((sam2_cx - before_center[0])**2 + (sam2_cy - before_center[1])**2)
        norm_dist_sam2 = pixel_dist_sam2 / math.sqrt(width*width + height*height)
        print(f"    Distance (SAM2 refined) from remembered: {pixel_dist_sam2:.1f}px = {norm_dist_sam2:.4f} (norm)")
        # expected: 0.4591
        diff = abs(norm_dist_sam2 - 0.4591)
        if diff < 0.01:
            print(f"    ✓ MATCHES reported distance 0.4591!")

print(f"\nDistance to beat (threshold 0.02): {0.02 * math.sqrt(width*width + height*height):.1f}px")
print("Done.")
