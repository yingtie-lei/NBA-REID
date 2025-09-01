# pipeline_enhanced_modified.py
# ---------------------------------------------------------
# Enhanced Video Segmentation Pipeline with:
# 1. Left-right layout with larger visualization
# 2. Video processing (extract frames from video)
# 3. Custom categories: player_name and motion_class
# 4. Save to configurable output directory structure
# 5. Centralized annotations.json
# 6. Binary mask export with light pink visualization
# ---------------------------------------------------------

import os
import cv2
import json
import shutil
import tempfile
import traceback
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import colorsys

import gradio as gr
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Supervision for professional annotations
import supervision as sv

# ====== SAM2 modules from Grounded-SAM-2 repository ======
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================
# Global Configuration
# =========================
DEFAULT_VIDEO_PATH = ""
DEFAULT_PROMPT = "basketball player."
DEFAULT_PLAYER_NAME = ""
DEFAULT_MOTION_CLASS = ""
IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}

# Grounding DINO (HuggingFace)
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# SAM2 weights and configuration
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEFAULT_ALPHA = 0.6
DEFAULT_FOURCC = "mp4v"
DEFAULT_RESULTS_DIR = "./Results"
ANNOTATIONS_JSON_PATH = os.path.join(DEFAULT_RESULTS_DIR, "annotations.json")

# Color palette - only light pink for masks
LIGHT_PINK_COLOR = (255, 182, 193)  # Light pink color for masks

def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


# =========================
# Video Processing Functions
# =========================
def extract_frames_from_video(video_path: str, output_dir: str) -> Tuple[List[str], float]:
    """Extract all frames from video at original fps"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_names = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_name = f"{frame_count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_names.append(frame_name)
        frame_count += 1
    
    cap.release()
    
    if not frame_names:
        raise ValueError("No frames were extracted from the video")
    
    return frame_names, original_fps

def get_video_info(video_path: str) -> Dict:
    """Get video information"""
    if not os.path.exists(video_path):
        return {}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# =========================
# Supervision Annotators Initialization
# =========================
def get_annotators():
    """Create annotators based on supervision version"""
    try:
        bbox_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=3)
        selected_bbox_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=5)
    except AttributeError:
        try:
            bbox_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED, thickness=3)
            selected_bbox_annotator = sv.BoundingBoxAnnotator(color=sv.Color.GREEN, thickness=5)
        except AttributeError:
            bbox_annotator = sv.RoundBoxAnnotator(color=sv.Color.RED, thickness=3)
            selected_bbox_annotator = sv.RoundBoxAnnotator(color=sv.Color.GREEN, thickness=5)

    try:
        label_annotator = sv.LabelAnnotator(
            color=sv.Color.RED, text_color=sv.Color.WHITE,
            text_scale=0.7, text_thickness=2, text_padding=8, border_radius=5
        )
        selected_label_annotator = sv.LabelAnnotator(
            color=sv.Color.GREEN, text_color=sv.Color.WHITE,
            text_scale=0.7, text_thickness=2, text_padding=8, border_radius=5
        )
    except (AttributeError, TypeError):
        label_annotator = sv.LabelAnnotator()
        selected_label_annotator = sv.LabelAnnotator()

    try:
        mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN, opacity=DEFAULT_ALPHA)
    except (AttributeError, TypeError):
        mask_annotator = sv.MaskAnnotator()

    try:
        positive_dot_annotator = sv.DotAnnotator(
            color=sv.Color.GREEN, radius=10, outline_color=sv.Color.BLACK, outline_thickness=3
        )
        negative_dot_annotator = sv.DotAnnotator(
            color=sv.Color.RED, radius=10, outline_color=sv.Color.BLACK, outline_thickness=3
        )
    except (AttributeError, TypeError):
        try:
            positive_dot_annotator = sv.CircleAnnotator(color=sv.Color.GREEN)
            negative_dot_annotator = sv.CircleAnnotator(color=sv.Color.RED)
        except AttributeError:
            positive_dot_annotator = None
            negative_dot_annotator = None

    return (bbox_annotator, selected_bbox_annotator, 
            label_annotator, selected_label_annotator,
            mask_annotator, positive_dot_annotator, negative_dot_annotator)

# Get annotators
(bbox_annotator, selected_bbox_annotator, 
 label_annotator, selected_label_annotator,
 mask_annotator, positive_dot_annotator, negative_dot_annotator) = get_annotators()


# =========================
# Utility Functions
# =========================
def is_cuda_ok() -> bool:
    return torch.cuda.is_available()

def set_torch_perf_flags():
    if not is_cuda_ok():
        return
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def scan_frame_names(frames_dir: str) -> List[str]:
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Directory does not exist: {frames_dir}")
    names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] in IMAGE_EXTS]
    if not names:
        raise FileNotFoundError(f"No frame images found in directory")
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except Exception:
        names.sort()
    return names

def read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def create_detections(boxes_xyxy: np.ndarray, labels: List[str], scores: np.ndarray = None) -> sv.Detections:
    """Create supervision Detections object"""
    if scores is None:
        scores = np.ones(len(boxes_xyxy))
    
    detections = sv.Detections(
        xyxy=boxes_xyxy,
        confidence=scores,
        class_id=np.arange(len(boxes_xyxy))
    )
    return detections

def annotate_detections_with_labels(image: np.ndarray, 
                                  detections: sv.Detections, 
                                  labels: List[str],
                                  selected_idx: int = None) -> np.ndarray:
    """Annotate detection results using supervision"""
    annotated_image = image.copy()
    
    if selected_idx is not None:
        unselected_mask = np.ones(len(detections), dtype=bool)
        unselected_mask[selected_idx] = False
        
        if np.any(unselected_mask):
            unselected_detections = detections[unselected_mask]
            unselected_labels = [f"{i}: {labels[i]}" for i, keep in enumerate(unselected_mask) if keep]
            
            annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=unselected_detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=unselected_detections, labels=unselected_labels)
        
        selected_detection = detections[selected_idx:selected_idx+1]
        selected_label = [f"{selected_idx}: {labels[selected_idx]} ✓"]
        
        annotated_image = selected_bbox_annotator.annotate(scene=annotated_image, detections=selected_detection)
        annotated_image = selected_label_annotator.annotate(scene=annotated_image, detections=selected_detection, labels=selected_label)
    else:
        formatted_labels = [f"{i}: {label}" for i, label in enumerate(labels)]
        annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=formatted_labels)
    
    return annotated_image

def annotate_colorful_mask_on_image(rgb_img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Apply colorful mask overlay using supervision's MaskAnnotator"""
    H, W = rgb_img.shape[:2]
    m = np.array(mask)
    m = np.squeeze(m)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got shape {m.shape}")
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        m = m.astype(bool)
    
    if not np.any(m):
        return rgb_img.copy()
    
    mask_indices = np.where(m)
    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
    bbox = np.array([[x_min, y_min, x_max, y_max]])
    mask_3d = m[np.newaxis, ...]
    
    if color is None:
        color = LIGHT_PINK_COLOR
    
    try:
        custom_mask_annotator = sv.MaskAnnotator(
            color=sv.Color(r=color[0], g=color[1], b=color[2]),
            opacity=alpha
        )
    except (AttributeError, TypeError):
        custom_mask_annotator = mask_annotator
    
    detections = sv.Detections(
        xyxy=bbox,
        mask=mask_3d,
        confidence=np.array([1.0]),
        class_id=np.array([0])
    )
    
    annotated_image = custom_mask_annotator.annotate(scene=rgb_img.copy(), detections=detections)
    return annotated_image

def annotate_points(rgb_img: np.ndarray, points: List[Tuple[int, int]], labels: List[int]) -> np.ndarray:
    """Annotate points using supervision point annotators, fallback to OpenCV if unavailable"""
    if not points or not labels:
        return rgb_img.copy()
    
    annotated_image = rgb_img.copy()
    
    if positive_dot_annotator is not None and negative_dot_annotator is not None:
        positive_points = []
        negative_points = []
        
        for (x, y), lab in zip(points, labels):
            if lab == 1:
                positive_points.append([x, y])
            else:
                negative_points.append([x, y])
        
        if positive_points:
            pos_points_array = np.array(positive_points)
            pos_detections = sv.Detections(
                xyxy=np.array([[x-1, y-1, x+1, y+1] for x, y in pos_points_array]),
                confidence=np.ones(len(pos_points_array)),
                class_id=np.zeros(len(pos_points_array), dtype=int)
            )
            try:
                annotated_image = positive_dot_annotator.annotate(scene=annotated_image, detections=pos_detections)
            except (AttributeError, TypeError):
                for x, y in pos_points_array:
                    cv2.circle(annotated_image, (int(x), int(y)), 10, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(annotated_image, (int(x), int(y)), 12, (0, 0, 0), 3, cv2.LINE_AA)
        
        if negative_points:
            neg_points_array = np.array(negative_points)
            neg_detections = sv.Detections(
                xyxy=np.array([[x-1, y-1, x+1, y+1] for x, y in neg_points_array]),
                confidence=np.ones(len(neg_points_array)),
                class_id=np.zeros(len(neg_points_array), dtype=int)
            )
            try:
                annotated_image = negative_dot_annotator.annotate(scene=annotated_image, detections=neg_detections)
            except (AttributeError, TypeError):
                for x, y in neg_points_array:
                    cv2.circle(annotated_image, (int(x), int(y)), 10, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(annotated_image, (int(x), int(y)), 12, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        for (x, y), lab in zip(points, labels):
            color = (0, 255, 0) if lab == 1 else (255, 0, 0)
            cv2.circle(annotated_image, (int(x), int(y)), 10, color, -1, cv2.LINE_AA)
            cv2.circle(annotated_image, (int(x), int(y)), 12, (0, 0, 0), 3, cv2.LINE_AA)
    
    return annotated_image

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

def single_mask_to_rle(mask: np.ndarray) -> Dict:
    """Convert single mask to RLE format"""
    mask = np.array(mask, dtype=np.uint8, order='F')
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    try:
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    except ImportError:
        pixels = mask.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {'size': [mask.shape[0], mask.shape[1]], 'counts': runs.tolist()}

def save_annotations_to_json(video_segments: Dict[int, Dict[int, np.ndarray]],
                            frames_dir: str,
                            frame_names: List[str],
                            video_name: str,
                            player_name: str,
                            motion_class: str,
                            fps: float,
                            output_base_dir: str,
                            selected_box: np.ndarray = None,
                            frame_paths_for_json: List[str] = None) -> str:
    """Save annotations in the required JSON format and append to central annotations.json
    For the same video, overwrite previous annotations; for different videos, keep them separate"""
    
    print("Start dumping the annotation...")
    
    # Use the provided output base directory
    annotations_json_path = os.path.join(output_base_dir, "annotations.json")
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load existing annotations or create new
    existing_annotations = []
    if os.path.exists(annotations_json_path):
        try:
            with open(annotations_json_path, 'r') as f:
                loaded_annotations = json.load(f)
                if isinstance(loaded_annotations, list):
                    existing_annotations = loaded_annotations
        except Exception as e:
            print(f"Warning: Could not load existing annotations: {e}")
            existing_annotations = []
    
    # Filter out annotations from the current video (to overwrite them)
    filtered_annotations = []
    if existing_annotations:
        for annotation in existing_annotations:
            if annotation.get("video_name") != video_name:
                filtered_annotations.append(annotation)
        print(f"Removed {len(existing_annotations) - len(filtered_annotations)} existing annotations for video '{video_name}'")
    
    # Process each frame for the current video
    new_annotations = []
    for frame_idx in sorted(video_segments.keys()):
        if frame_idx >= len(frame_names):
            continue
            
        # Use the proper saved frame path if provided
        if frame_paths_for_json and frame_idx < len(frame_paths_for_json):
            frame_path = frame_paths_for_json[frame_idx]
        else:
            frame_name = frame_names[frame_idx]
            frame_path = os.path.join(frames_dir, frame_name)
        
        # Read image to get dimensions
        try:
            with Image.open(frame_path) as img:
                img_width, img_height = img.size
        except Exception:
            try:
                # Try reading the original frame from temp directory
                frame_name = frame_names[frame_idx]
                temp_frame_path = os.path.join(frames_dir, frame_name)
                img = cv2.imread(temp_frame_path)
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
            except:
                continue
        
        # Process masks for this frame
        segs = video_segments[frame_idx]
        if not segs:
            continue
            
        # Collect all masks and boxes for this frame
        masks = []
        boxes = []
        scores = []
        
        for obj_id, mask_logits in segs.items():
            mask = np.squeeze(mask_logits).astype(np.uint8)
            
            if not np.any(mask):
                continue
                
            masks.append(mask)
            
            # Calculate bounding box from mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
                
            x_min, x_max = float(x_indices.min()), float(x_indices.max())
            y_min, y_max = float(y_indices.min()), float(y_indices.max())
            
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(1.0)
        
        if not masks:
            continue
            
        # Convert masks to RLE format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        
        # Create annotation for this frame
        frame_annotation = {
            "image_path": frame_path,
            "video_name": video_name,
            "fps": fps,
            "annotations": [
                {
                    "class_name": motion_class.strip() if motion_class else "object",
                    "player_name": player_name.strip() if player_name else "unknown",
                    "bbox": box,
                    "segmentation": mask_rle
                }
                for box, mask_rle, score in zip(boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": img_width,
            "img_height": img_height,
        }
        
        new_annotations.append(frame_annotation)
    
    # Combine filtered existing annotations with new annotations
    final_annotations = filtered_annotations + new_annotations
    
    # Save updated annotations
    with open(annotations_json_path, "w") as f:
        json.dump(final_annotations, f, indent=4)
    
    print(f'Added {len(new_annotations)} new annotations for video "{video_name}"')
    print(f'Total annotations in file: {len(final_annotations)} (from {len(set([ann.get("video_name", "unknown") for ann in final_annotations]))} videos)')
    print(f'Annotations have been saved to "{annotations_json_path}"')
    
    return annotations_json_path

def export_results(video_segments: Dict[int, Dict[int, np.ndarray]],
                  frames_dir: str,
                  frame_names: List[str],
                  output_base_dir: str,
                  video_name: str,
                  player_name: str,
                  motion_class: str,
                  fps: float,
                  selected_box: np.ndarray = None):
    """Export frames and binary masks to organized folder structure"""
    
    # Create main output directory for this video
    video_output_dir = os.path.join(output_base_dir, video_name)
    frames_output_dir = os.path.join(video_output_dir, "frames")
    masks_output_dir = os.path.join(video_output_dir, "masks")
    
    os.makedirs(frames_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    # Export frame by frame
    exported_count = 0
    frame_paths_for_json = []
    
    for frame_idx in sorted(video_segments.keys()):
        if frame_idx >= len(frame_names):
            continue
            
        frame_name = frame_names[frame_idx]
        frame_path = os.path.join(frames_dir, frame_name)
        original_rgb = read_image_rgb(frame_path)
        
        segs = video_segments[frame_idx]
        obj_ids = sorted(segs.keys())
        if not obj_ids:
            continue
            
        mask = np.squeeze(segs[obj_ids[0]]).astype(bool)
        
        # Save original frame (without mask overlay)
        output_frame_name = f"frame_{frame_idx:05d}.jpg"
        output_frame_path = os.path.join(frames_output_dir, output_frame_name)
        frame_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_frame_path, frame_bgr)
        
        # Save binary mask (255 for mask, 0 for background) as PNG
        binary_mask = mask.astype(np.uint8) * 255  # Convert True->255, False->0
        output_mask_name = f"mask_{frame_idx:05d}.jpg"
        output_mask_path = os.path.join(masks_output_dir, output_mask_name)
        cv2.imwrite(output_mask_path, binary_mask)
        
        frame_paths_for_json.append(output_frame_path)
        exported_count += 1
    
    # Save annotations to JSON using user-specified output directory
    json_path = save_annotations_to_json(
        video_segments, frames_output_dir, frame_names, video_name,
        player_name, motion_class, fps, output_base_dir, selected_box, frame_paths_for_json
    )
    
    return exported_count, json_path


# =========================
# Model Loading
# =========================
_GDINO = None
_GDINO_PROC = None
_SAM2_VIDEO = None
_SAM2_IMG = None

def get_gdino():
    global _GDINO, _GDINO_PROC
    if _GDINO is None or _GDINO_PROC is None:
        device = "cuda" if is_cuda_ok() else "cpu"
        _GDINO_PROC = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        _GDINO = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
    return _GDINO, _GDINO_PROC

def get_sam2_models():
    global _SAM2_VIDEO, _SAM2_IMG
    if _SAM2_VIDEO is None or _SAM2_IMG is None:
        if not os.path.isfile(SAM2_CHECKPOINT):
            raise FileNotFoundError(f"SAM2 weights not found: {SAM2_CHECKPOINT}")
        try:
            vp = build_sam2_video_predictor(SAM2_CFG, SAM2_CHECKPOINT)
            im = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CHECKPOINT))
            _SAM2_VIDEO, _SAM2_IMG = vp, im
        except Exception as e:
            raise RuntimeError(f"SAM2 config loading failed: {SAM2_CFG}\n{e}")
    return _SAM2_VIDEO, _SAM2_IMG

def reset_sam2_models():
    global _SAM2_VIDEO, _SAM2_IMG
    _SAM2_VIDEO = None
    _SAM2_IMG = None


# =========================
# Business Logic
# =========================
def process_video_and_detect(video_path: str, text_prompt: str, box_thr: float, text_thr: float):
    """Process video: extract frames and run detection on first frame"""
    set_torch_perf_flags()
    
    # Get video info
    video_info = get_video_info(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create temporary directory for frames
    temp_frames_dir = tempfile.mkdtemp(prefix=f"frames_{video_name}_")
    
    # Extract all frames at original fps
    frame_names, original_fps = extract_frames_from_video(video_path, temp_frames_dir)
    
    # Update video info with fps
    video_info['original_fps'] = original_fps
    
    # Run detection on first frame
    first_path = os.path.join(temp_frames_dir, frame_names[0])
    image_pil = Image.open(first_path).convert("RGB")
    w, h = image_pil.size

    model, proc = get_gdino()
    device = next(model.parameters()).device.type

    text_prompt = (text_prompt or "").strip()
    if not text_prompt.endswith("."):
        text_prompt += "."
    text_prompt = text_prompt.lower()

    inputs = proc(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[(h, w)]
    )
    boxes = results[0]["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = results[0]["scores"].detach().cpu().numpy().astype(np.float32)
    labels = results[0]["labels"]

    rgb = read_image_rgb(first_path)
    detections = create_detections(boxes, labels, scores)
    vis = annotate_detections_with_labels(rgb, detections, labels, selected_idx=None)
    
    return vis, boxes, labels, frame_names, temp_frames_dir, video_name, video_info

def initial_propagate_with_selected_box(frames_dir: str, frame_names: List[str], 
                                       boxes_xyxy: np.ndarray, labels: List[str], selected_idx: int):
    set_torch_perf_flags()
    sam2_video, sam2_img = get_sam2_models()

    inference_state = sam2_video.init_state(video_path=frames_dir)

    first_path = os.path.join(frames_dir, frame_names[0])
    first_rgb = read_image_rgb(first_path)

    sel_box = boxes_xyxy[selected_idx].astype(np.float32)
    sam2_img.set_image(first_rgb)
    masks, scores, logits = sam2_img.predict(
        point_coords=None, point_labels=None, box=sel_box[None, :], multimask_output=False,
    )
    mask0 = np.squeeze(masks[0]).astype(bool)

    obj_id = 1
    _ret, _obj_ids, _mask_logits = sam2_video.add_new_mask(
        inference_state=inference_state, frame_idx=0, obj_id=obj_id, mask=mask0
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return inference_state, video_segments

def repropagate_with_points(frames_dir: str, frame_idx: int, points_xy: np.ndarray, 
                           points_labels: np.ndarray, inference_state, obj_id: int = 1):
    set_torch_perf_flags()
    sam2_video, _ = get_sam2_models()

    _ret, _obj_ids, _mask_logits = sam2_video.add_new_points_or_box(
        inference_state=inference_state, frame_idx=int(frame_idx), obj_id=int(obj_id),
        points=points_xy.astype(np.float32), labels=points_labels.astype(np.int32),
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


# =========================
# Gradio Interface
# =========================
def create_custom_css():
    return """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1600px !important;
        margin: 0 auto;
    }
    
    .main-container {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 20px;
        padding: 20px;
    }
    
    .left-column {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        width: 45%;
        margin: 0 auto;
    }
    
    .right-column {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        padding: 20px;
        width: 55%;
        margin: 0 auto;
    }
    
    .gr-button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        margin: 4px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    .secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    .gr-textbox, .gr-file {
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        transition: border-color 0.3s ease;
        text-align: center;
    }
    
    .gr-textbox:focus {
        border-color: #667eea;
    }
    
    .gr-image {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 0 auto;
    }
    
    .section-header {
        color: #2d3748;
        font-size: 1.3em;
        font-weight: 700;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .info-card {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    """

with gr.Blocks(title="NBA Re-id Annotation Pipeline", theme="soft", css=create_custom_css()) as demo:
    gr.Markdown(
        """
        # NBA Re-id Annotation Pipeline with Custom Categories
        """,
        elem_classes=["section-header"]
    )

    with gr.Row():
        # Left Column - Configuration and Controls
        with gr.Column(scale=1, elem_classes=["left-column"]):
            gr.Markdown("## Configuration & Controls", elem_classes=["section-header"])
            
            # Video Input
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Video Input")
                video_path_inp = gr.Textbox(
                    label="Video File Path",
                    placeholder="Enter path to video file (e.g., /path/to/video.mp4)",
                    value=""
                )
                load_video_btn = gr.Button("Load Video Info", variant="secondary")
                video_info_display = gr.Textbox(
                    label="Video Information",
                    interactive=False,
                    lines=4
                )

            # Output Path Configuration
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Output Settings")
                output_base_dir_inp = gr.Textbox(
                    label="Output Base Directory",
                    value=DEFAULT_RESULTS_DIR,
                    placeholder="Enter base directory for results"
                )

            # Detection Settings
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Detection Settings")
                text_inp = gr.Textbox(
                    label="Detection Prompt",
                    value=DEFAULT_PROMPT,
                    placeholder="basketball player."
                )
                with gr.Row():
                    box_thr = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Box Threshold")
                    text_thr = gr.Slider(0.05, 0.9, value=0.30, step=0.05, label="Text Threshold")

            # Custom Categories
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Custom Categories")
                player_name_inp = gr.Textbox(
                    label="Player Name",
                    value=DEFAULT_PLAYER_NAME,
                    placeholder="Enter player name"
                )
                motion_class_inp = gr.Textbox(
                    label="Motion Class",
                    value=DEFAULT_MOTION_CLASS,
                    placeholder="2points shooting, 3points shooting, freethrow, etc."
                )

            # Processing Settings
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Processing Settings")
                alpha_inp = gr.Slider(0.1, 1.0, value=DEFAULT_ALPHA, step=0.05, label="Overlay Opacity")

            # Main Control Buttons
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Main Controls")
                process_video_btn = gr.Button("Process Video & Detect", variant="primary", size="lg")
                run_segmentation_btn = gr.Button("Start Segmentation & Tracking", variant="primary", size="lg")
                with gr.Row():
                    reset_selection_btn = gr.Button("Clear Selection", variant="secondary")
                    reset_models_btn = gr.Button("Reset Models", variant="secondary")

            # Frame Correction Controls
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Frame Correction")
                frame_slider = gr.Slider(value=0, minimum=0, maximum=0, step=1, label="Frame Index")
                point_label_radio = gr.Radio(
                    choices=[("Positive (Include)", 1), ("Negative (Exclude)", 0)], 
                    value=1, 
                    label="Point Type"
                )
                with gr.Row():
                    clear_points_btn = gr.Button("Clear Points", variant="secondary")
                    apply_points_btn = gr.Button("Apply Points", variant="primary")

        # Right Column - Display and Results (Made Larger)
        with gr.Column(scale=3, elem_classes=["right-column"]):
            gr.Markdown("## Detection & Results", elem_classes=["section-header"])
            
            # Detection Results
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Detection Results")
                det_img = gr.Image(
                    label="First Frame Detection (Click to Select Target)",
                    interactive=True,
                    height=500
                )
                selection_info = gr.Textbox(
                    label="Selection Status",
                    interactive=False,
                    value="No selection yet"
                )

            # Preview and Correction
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Preview & Interactive Correction")
                
                with gr.Tab("Current Preview"):
                    preview_img = gr.Image(
                        label="Current Frame Preview",
                        interactive=False,
                        height=500
                    )
                
                with gr.Tab("Interactive Correction"):
                    interact_img = gr.Image(
                        label="Click to Add Correction Points",
                        interactive=True,
                        height=500
                    )
                    pending_info = gr.Textbox(
                        label="Pending Points",
                        interactive=False,
                        placeholder="No pending points"
                    )

            # Results
            with gr.Group(elem_classes=["info-card"]):
                gr.Markdown("### Export Results")
                with gr.Row():
                    results_info = gr.Textbox(
                        label="Processing Results",
                        interactive=False,
                        lines=3
                    )
                with gr.Row():
                    download_json = gr.File(label="Download Annotations JSON")

    # State Management
    st_boxes = gr.State(None)
    st_labels = gr.State(None)
    st_frame_names = gr.State(None)
    st_frames_dir = gr.State(None)
    st_video_name = gr.State(None)
    st_video_info = gr.State(None)
    st_selected_idx = gr.State(None)
    st_selected_label = gr.State(None)
    st_selected_box = gr.State(None)
    st_infer_state = gr.State(None)
    st_segments = gr.State(None)
    st_points = gr.State(dict())
    st_obj_id = gr.State(1)

    # Event Handlers

    # Load video info button click
    def on_load_video_info(video_path):
        if not video_path or not video_path.strip():
            return "No video path provided"
        
        try:
            info = get_video_info(video_path)
            if not info:
                return "Could not read video information. Please check the file path."
            
            info_text = f"""
Video: {os.path.basename(video_path)}
Duration: {info['duration']:.2f} seconds
FPS: {info['fps']:.2f}
Resolution: {info['width']} x {info['height']}
Total Frames: {info['frame_count']}
            """.strip()
            return info_text
        except Exception as e:
            return f"Error reading video: {str(e)}"

    load_video_btn.click(
        fn=on_load_video_info,
        inputs=[video_path_inp],
        outputs=[video_info_display]
    )

    # Step 1: Process video and detect
    def on_process_video(video_path, text_prompt, bthr, tthr):
        try:
            if not video_path or not video_path.strip():
                raise gr.Error("Please enter a video file path first.")
            
            vis, boxes, labels, frame_names, frames_dir, video_name, video_info = process_video_and_detect(
                video_path, text_prompt, float(bthr), float(tthr)
            )
            
            max_idx = len(frame_names) - 1
            info_msg = f"Detection completed for {len(frame_names)} frames. Click on a bounding box to select your target object."
            
            return (
                vis, boxes, labels, frame_names, frames_dir, video_name, video_info,
                info_msg, gr.update(maximum=max_idx, value=0),
                None, None, {}, None, None, None, "Video processed successfully!"
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise gr.Error(f"Video processing failed: {e}\n{tb}")

    process_video_btn.click(
        fn=on_process_video,
        inputs=[video_path_inp, text_inp, box_thr, text_thr],
        outputs=[
            det_img, st_boxes, st_labels, st_frame_names, st_frames_dir, 
            st_video_name, st_video_info, selection_info, frame_slider,
            preview_img, interact_img, st_points, st_infer_state, st_segments, 
            st_selected_idx, results_info
        ]
    )

    # Clear selection
    def on_reset_selection():
        return None, None, None, "Selection cleared. Please click on a bounding box to select target.", None

    reset_selection_btn.click(
        fn=on_reset_selection,
        outputs=[st_selected_idx, st_selected_label, st_selected_box, selection_info, preview_img]
    )

    # Reset models
    def on_reset_models():
        reset_sam2_models()
        return None, {}, "Models and inference state have been reset."

    reset_models_btn.click(
        fn=on_reset_models,
        outputs=[st_infer_state, st_points, results_info]
    )

    # Click to select object
    def on_click_detection(image, boxes, labels, selected_idx, evt: gr.SelectData):
        try:
            if image is None or boxes is None or labels is None:
                raise gr.Error("Please process video first.")
            
            x, y = int(evt.index[0]), int(evt.index[1])
            boxes_np = np.array(boxes, dtype=np.float32)
            hit = None
            
            for i, box in enumerate(boxes_np):
                if point_in_box(x, y, box):
                    hit = i
                    break
            
            if hit is not None:
                detections = create_detections(boxes_np, labels)
                vis = annotate_detections_with_labels(
                    np.array(image, dtype=np.uint8), detections, labels, selected_idx=hit
                )
                
                selected_label = labels[hit]
                selected_box = boxes_np[hit]
                info = f"Selected: {hit}: {selected_label}"
                return hit, selected_label, selected_box, info, vis
            else:
                info = "No bounding box clicked. Please click inside a detection box."
                return selected_idx, None, None, info, image
        except Exception as e:
            raise gr.Error(f"Click processing failed: {e}")

    det_img.select(
        fn=on_click_detection,
        inputs=[det_img, st_boxes, st_labels, st_selected_idx],
        outputs=[st_selected_idx, st_selected_label, st_selected_box, selection_info, det_img]
    )

    # Step 2: Run segmentation
    def on_run_segmentation(frames_dir, frame_names, boxes, labels, selected_idx, selected_label, 
                           selected_box, video_name, player_name, motion_class, alpha, video_info, output_base_dir):
        try:
            # Fix the validation check for numpy arrays
            if not (boxes is not None and labels is not None and frame_names is not None and selected_idx is not None):
                raise gr.Error("Please complete video processing and object selection first.")

            gr.Info("Starting segmentation and tracking...")
            
            infer_state, segments = initial_propagate_with_selected_box(
                frames_dir, list(frame_names), np.array(boxes, dtype=np.float32), 
                list(labels), int(selected_idx)
            )

            # Get fps from video info
            fps = video_info.get('original_fps', 30.0) if video_info else 30.0
            
            # Export results
            exported_count, json_path = export_results(
                segments, frames_dir, list(frame_names), output_base_dir or DEFAULT_RESULTS_DIR, 
                video_name, player_name, motion_class, fps, selected_box
            )

            # Preview current frame with light pink mask
            cur_idx = 0
            rgb = read_image_rgb(os.path.join(frames_dir, frame_names[cur_idx]))
            seg = segments.get(cur_idx, {})
            if seg:
                mask = np.squeeze(seg[min(seg.keys())]).astype(bool)
                preview = annotate_colorful_mask_on_image(rgb, mask, color=LIGHT_PINK_COLOR, alpha=float(alpha))
            else:
                preview = rgb

            result_msg = f"Segmentation completed! Exported {exported_count} frames and binary masks to {output_base_dir}/{video_name}"
            gr.Info("Segmentation and tracking completed successfully!")
            
            return preview, infer_state, segments, {}, json_path, result_msg
        except Exception as e:
            tb = traceback.format_exc()
            raise gr.Error(f"Segmentation failed: {e}\n{tb}")

    run_segmentation_btn.click(
        fn=on_run_segmentation,
        inputs=[
            st_frames_dir, st_frame_names, st_boxes, st_labels, st_selected_idx, 
            st_selected_label, st_selected_box, st_video_name, player_name_inp, 
            motion_class_inp, alpha_inp, st_video_info, output_base_dir_inp
        ],
        outputs=[preview_img, st_infer_state, st_segments, st_points, download_json, results_info]
    )

    # Render interactive image for frame correction
    def render_correction_image(frames_dir, frame_names, segments, cur_idx, alpha, pending_points_dict):
        if not frame_names:
            return None, "Please complete previous steps first."
        
        cur_idx = int(cur_idx)
        rgb = read_image_rgb(os.path.join(frames_dir, frame_names[cur_idx]))

        # Overlay existing mask with light pink color
        if segments and (cur_idx in segments) and segments[cur_idx]:
            seg = segments[cur_idx]
            mask = np.squeeze(seg[min(seg.keys())]).astype(bool)
            canvas = annotate_colorful_mask_on_image(rgb, mask, color=LIGHT_PINK_COLOR, alpha=float(alpha))
        else:
            canvas = rgb

        # Draw pending points
        pts, labs = [], []
        if pending_points_dict and (cur_idx in pending_points_dict):
            pts = pending_points_dict[cur_idx].get("pts", [])
            labs = pending_points_dict[cur_idx].get("labs", [])
            canvas = annotate_points(canvas, pts, labs)

        info_text = f"Frame {cur_idx}: {len(pts)} pending points"
        return canvas, info_text

    # Update display when slider changes
    frame_slider.change(
        fn=render_correction_image,
        inputs=[st_frames_dir, st_frame_names, st_segments, frame_slider, alpha_inp, st_points],
        outputs=[interact_img, pending_info]
    )

    # Add correction point
    def on_add_correction_point(image, cur_idx, radio_label, pending_points_dict, evt: gr.SelectData):
        try:
            x, y = int(evt.index[0]), int(evt.index[1])
            cur_idx = int(cur_idx)
            lab = int(radio_label)

            if pending_points_dict is None:
                pending_points_dict = {}
            if cur_idx not in pending_points_dict:
                pending_points_dict[cur_idx] = {"pts": [], "labs": []}
            
            pending_points_dict[cur_idx]["pts"].append((x, y))
            pending_points_dict[cur_idx]["labs"].append(lab)

            # Visual feedback
            img_np = np.array(image, dtype=np.uint8)
            img_np = annotate_points(img_np, [(x, y)], [lab])

            point_type = "Positive" if lab == 1 else "Negative"
            total_points = len(pending_points_dict[cur_idx]["pts"])
            info = f"Frame {cur_idx}: Added {point_type} point at ({x}, {y}). Total: {total_points} points"
            
            return img_np, pending_points_dict, info
        except Exception as e:
            raise gr.Error(f"Failed to add point: {e}")

    interact_img.select(
        fn=on_add_correction_point,
        inputs=[interact_img, frame_slider, point_label_radio, st_points],
        outputs=[interact_img, st_points, pending_info]
    )

    # Clear points
    def on_clear_correction_points(cur_idx, pending_points_dict):
        cur_idx = int(cur_idx)
        if pending_points_dict and (cur_idx in pending_points_dict):
            pending_points_dict[cur_idx] = {"pts": [], "labs": []}
        return pending_points_dict, f"Cleared all pending points for frame {cur_idx}."

    clear_points_btn.click(
        fn=on_clear_correction_points,
        inputs=[frame_slider, st_points],
        outputs=[st_points, pending_info]
    )

    # Apply correction points
    def on_apply_correction_points(frames_dir, frame_names, cur_idx, pending_points_dict, 
                                 infer_state, segments, obj_id, alpha, video_name, 
                                 player_name, motion_class, selected_box, video_info, output_base_dir):
        try:
            if infer_state is None or not segments:
                raise gr.Error("Please complete initial segmentation first.")

            cur_idx = int(cur_idx)
            if not pending_points_dict or (cur_idx not in pending_points_dict) or \
               (len(pending_points_dict[cur_idx].get("pts", [])) == 0):
                raise gr.Error("No pending points for this frame.")

            pts = np.array(pending_points_dict[cur_idx]["pts"], dtype=np.float32)
            labs = np.array(pending_points_dict[cur_idx]["labs"], dtype=np.int32)

            gr.Info("Re-propagating with correction points...")
            
            # Re-propagate
            new_segments = repropagate_with_points(
                frames_dir, cur_idx, pts, labs, infer_state, int(obj_id)
            )

            # Get fps from video info
            fps = video_info.get('original_fps', 30.0) if video_info else 30.0

            # Re-export results
            exported_count, json_path = export_results(
                new_segments, frames_dir, list(frame_names), output_base_dir or DEFAULT_RESULTS_DIR, 
                video_name, player_name, motion_class, fps, selected_box
            )

            # Clear pending points
            pending_points_dict[cur_idx] = {"pts": [], "labs": []}

            # Update preview with light pink mask
            rgb = read_image_rgb(os.path.join(frames_dir, frame_names[cur_idx]))
            if new_segments and (cur_idx in new_segments) and new_segments[cur_idx]:
                seg = new_segments[cur_idx]
                mask = np.squeeze(seg[min(seg.keys())]).astype(bool)
                preview = annotate_colorful_mask_on_image(rgb, mask, color=LIGHT_PINK_COLOR, alpha=float(alpha))
            else:
                preview = rgb

            result_msg = f"Applied {len(pts)} correction points and re-propagated. Updated {exported_count} frames and binary masks."
            gr.Info("Re-propagation completed successfully!")
            
            return preview, new_segments, pending_points_dict, json_path, result_msg, f"Frame {cur_idx}: cleared pending points"
        except Exception as e:
            tb = traceback.format_exc()
            raise gr.Error(f"Re-propagation failed: {e}\n{tb}")

    apply_points_btn.click(
        fn=on_apply_correction_points,
        inputs=[
            st_frames_dir, st_frame_names, frame_slider, st_points, st_infer_state, 
            st_segments, st_obj_id, alpha_inp, st_video_name, player_name_inp, 
            motion_class_inp, st_selected_box, st_video_info, output_base_dir_inp
        ],
        outputs=[preview_img, st_segments, st_points, download_json, results_info, pending_info]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860, share=False)