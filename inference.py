"""
CIIBS Inference Engine
Real-Time Intelligent Cargo X-ray Risk Screening

Handles: object detection, anomaly heatmap, material density classification,
concealment pattern detection, declaration mismatch, and risk fusion.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger("ciibs.inference")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


# ──────────────────────────────────────────────
# Model Management
# ──────────────────────────────────────────────

_model_cache: dict = {}

def load_model(mode: str = "realtime") -> object:
    from ultralytics import YOLO
    config = load_config()
    mode_cfg = config["detection"]["modes"].get(mode, config["detection"]["modes"]["realtime"])
    model_name = mode_cfg["model"]
    if model_name not in _model_cache:
        logger.info(f"Loading model: {model_name} (mode={mode})")
        model = YOLO(model_name)
        _model_cache[model_name] = model
        logger.info(f"Model {model_name} loaded successfully")
    return _model_cache[model_name]


# ──────────────────────────────────────────────
# Object Detection
# ──────────────────────────────────────────────

PIDRAY_CLASSES = [
    "Baton", "Pliers", "Hammer", "Powerbank", "Scissors",
    "Wrench", "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter"
]

COCO_TO_PIDRAY = {
    43: "Knife", 76: "Scissors", 77: "Pliers", 42: "Gun",
    73: "Hammer", 74: "Wrench", 39: "Baton", 75: "Lighter",
    46: "Bullet", 64: "Powerbank", 84: "Sprayer", 80: "HandCuffs"
}

THREAT_LEVELS = {
    "Gun": 5, "Bullet": 5, "Knife": 4, "Baton": 3,
    "Scissors": 3, "Hammer": 3, "HandCuffs": 3,
    "Wrench": 2, "Pliers": 2, "Sprayer": 2,
    "Lighter": 2, "Powerbank": 1
}


def detect_objects(
    image: np.ndarray, model: object, mode: str = "realtime"
) -> Tuple[List[Dict], float, np.ndarray]:
    config = load_config()
    mode_cfg = config["detection"]["modes"].get(mode, config["detection"]["modes"]["realtime"])

    t0 = time.time()
    results = model(
        image,
        imgsz=mode_cfg["imgsz"],
        conf=mode_cfg["conf_threshold"],
        iou=mode_cfg["iou_threshold"],
        max_det=mode_cfg["max_det"],
        verbose=False
    )
    inference_time = time.time() - t0

    detections = []
    annotated = image.copy()

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                if cls_id in COCO_TO_PIDRAY:
                    pidray_class = COCO_TO_PIDRAY[cls_id]
                else:
                    pidray_class = PIDRAY_CLASSES[cls_id % len(PIDRAY_CLASSES)]
                    conf *= 0.7

                threat = THREAT_LEVELS.get(pidray_class, 1)

                # Material classification for this detection
                mat_type, mat_score = classify_material(image, xyxy)

                detections.append({
                    "bbox": xyxy.tolist(),
                    "class": pidray_class,
                    "confidence": round(conf, 3),
                    "threat_level": threat,
                    "material_type": mat_type,
                    "material_score": round(mat_score, 3),
                })

                color = _threat_color(threat)
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                label = f"{pidray_class} {conf:.0%} [{mat_type[:3]}]"
                _draw_label(annotated, label, (xyxy[0], xyxy[1] - 10), color)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, inference_time, annotated


def _threat_color(threat_level: int) -> Tuple[int, int, int]:
    colors = {
        5: (0, 0, 255), 4: (0, 69, 255), 3: (0, 165, 255),
        2: (0, 255, 255), 1: (0, 255, 0),
    }
    return colors.get(threat_level, (255, 255, 255))


def _draw_label(img, text, pos, color, scale=0.5, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 5), (x + w, y + 2), color, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)


# ──────────────────────────────────────────────
# 🔬 Material Density Classification (Novel)
# ──────────────────────────────────────────────

def classify_material(
    image: np.ndarray, bbox: np.ndarray
) -> Tuple[str, float]:
    """
    Classify material type within a bounding box using X-ray color channel analysis.
    In X-ray images:
      - Blue/dark channels → metallic/dense materials
      - Orange/warm channels → organic/light materials
      - Green → intermediate density

    Returns:
        material_type: "Metallic" | "Organic" | "Intermediate"
        metallic_score: float [0,1] — higher = more metallic
    """
    x1, y1, x2, y2 = bbox[:4]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return "Unknown", 0.5

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown", 0.5

    # Analyze color channels (BGR)
    b_mean = float(np.mean(roi[:, :, 0]))
    g_mean = float(np.mean(roi[:, :, 1]))
    r_mean = float(np.mean(roi[:, :, 2]))
    total = b_mean + g_mean + r_mean + 1e-6

    blue_ratio = b_mean / total
    warm_ratio = (r_mean + g_mean * 0.5) / total

    # Analyze intensity variance (metallic objects have sharper edges)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge_density = float(np.mean(cv2.Canny(gray_roi, 50, 150))) / 255.0

    # Metallic score: high blue ratio + high edge density
    metallic_score = (blue_ratio * 0.5 + edge_density * 0.3 +
                      (1 - warm_ratio) * 0.2)
    metallic_score = float(np.clip(metallic_score, 0, 1))

    if metallic_score > 0.45:
        return "Metallic", metallic_score
    elif metallic_score < 0.25:
        return "Organic", metallic_score
    else:
        return "Intermediate", metallic_score


def compute_material_risk(
    detections: List[Dict], declared_cargo: Optional[str]
) -> Tuple[float, Dict]:
    """
    Assess material-based risk.
    Metallic objects in organic-declared cargo = suspicious.
    """
    if not detections:
        return 0.0, {"metallic_count": 0, "message": "No objects to classify"}

    metallic_dets = [d for d in detections if d.get("material_type") == "Metallic"]
    metallic_count = len(metallic_dets)

    if metallic_count == 0:
        return 0.0, {"metallic_count": 0, "message": "No metallic objects detected"}

    # Higher risk if cargo declared as organic/soft
    organic_cargo = {"clothing", "food", "personal", "fragile", "documents"}
    is_organic_declared = False
    if declared_cargo:
        for cat in organic_cargo:
            if cat in declared_cargo.lower():
                is_organic_declared = True
                break

    base_risk = min(1.0, metallic_count * 0.3)
    if is_organic_declared:
        base_risk = min(1.0, base_risk * 1.5)

    avg_metallic = np.mean([d["material_score"] for d in metallic_dets])

    return base_risk, {
        "metallic_count": metallic_count,
        "avg_metallic_score": round(float(avg_metallic), 3),
        "organic_declared": is_organic_declared,
        "message": (
            f"{metallic_count} metallic object(s) detected"
            + (f" — suspicious in '{declared_cargo}' cargo" if is_organic_declared else "")
        )
    }


# ──────────────────────────────────────────────
# 🕵️ Concealment Pattern Detection (Novel)
# ──────────────────────────────────────────────

def compute_concealment(
    image: np.ndarray, detections: List[Dict]
) -> Tuple[float, Dict]:
    """
    Detect concealment patterns:
    1. Object overlap ratio — items stacked to hide shapes
    2. Edge discontinuity — broken edges suggesting hidden objects
    3. Density gradient anomaly — unusual density within bounding boxes
    """
    if len(detections) < 1:
        return 0.0, {
            "overlap_score": 0, "edge_score": 0, "density_score": 0,
            "message": "No detections to analyze for concealment"
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Overlap analysis
    overlap_score = _compute_overlap_score(detections)

    # 2. Edge discontinuity in ROIs
    edge_score = _compute_edge_discontinuity(gray, detections)

    # 3. Density gradient within detections
    density_score = _compute_density_anomaly(gray, detections)

    # Combine
    concealment_score = float(np.clip(
        overlap_score * 0.3 + edge_score * 0.35 + density_score * 0.35,
        0, 1
    ))

    details = {
        "overlap_score": round(overlap_score, 3),
        "edge_score": round(edge_score, 3),
        "density_score": round(density_score, 3),
        "message": _concealment_message(concealment_score)
    }

    return concealment_score, details


def _compute_overlap_score(detections: List[Dict]) -> float:
    """How much do detected bounding boxes overlap? High overlap = potential concealment."""
    if len(detections) < 2:
        return 0.0

    bboxes = [d["bbox"] for d in detections]
    max_iou = 0.0

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou = _bbox_iou(bboxes[i], bboxes[j])
            max_iou = max(max_iou, iou)

    return min(1.0, max_iou * 3)  # Scale up — even 0.33 IoU is suspicious


def _bbox_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def _compute_edge_discontinuity(gray: np.ndarray, detections: List[Dict]) -> float:
    """Detect broken/discontinuous edges within detected regions — sign of concealment."""
    if not detections:
        return 0.0

    scores = []
    h, w = gray.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = gray[y1:y2, x1:x2]
        edges = cv2.Canny(roi, 50, 150)

        # Compute edge continuity — ratio of edge pixels in connected components
        num_labels, labels = cv2.connectedComponents(edges)
        if num_labels <= 1:
            scores.append(0.0)
            continue

        # Many small components = broken edges = possible concealment
        component_sizes = [(labels == l).sum() for l in range(1, num_labels)]
        if not component_sizes:
            scores.append(0.0)
            continue

        avg_size = np.mean(component_sizes)
        fragmentation = 1.0 - min(1.0, avg_size / max(roi.shape[0], 1))
        scores.append(fragmentation)

    return float(np.mean(scores)) if scores else 0.0


def _compute_density_anomaly(gray: np.ndarray, detections: List[Dict]) -> float:
    """Check for unusual density gradients within detected objects."""
    if not detections:
        return 0.0

    scores = []
    h, w = gray.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = gray[y1:y2, x1:x2].astype(np.float32)

        # Split into quadrants and compare density
        mid_y, mid_x = roi.shape[0] // 2, roi.shape[1] // 2
        if mid_y < 2 or mid_x < 2:
            continue

        quads = [
            roi[:mid_y, :mid_x], roi[:mid_y, mid_x:],
            roi[mid_y:, :mid_x], roi[mid_y:, mid_x:]
        ]
        means = [float(np.mean(q)) for q in quads if q.size > 0]

        if len(means) >= 2:
            density_var = float(np.std(means) / (np.mean(means) + 1e-6))
            scores.append(min(1.0, density_var * 2))

    return float(np.mean(scores)) if scores else 0.0


def _concealment_message(score: float) -> str:
    if score > 0.6:
        return "⚠️ HIGH concealment indicators — items may be deliberately hidden"
    elif score > 0.3:
        return "🟡 Moderate concealment patterns — overlapping items detected"
    else:
        return "✅ Low concealment risk — objects clearly visible"


# ──────────────────────────────────────────────
# Anomaly Heatmap Generation
# ──────────────────────────────────────────────

def compute_anomaly(image: np.ndarray) -> Tuple[np.ndarray, float]:
    config = load_config()
    acfg = config["anomaly"]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]

    # 1. Edge density map
    edges = cv2.Canny(gray, acfg["canny_low"], acfg["canny_high"])
    edge_map = cv2.GaussianBlur(
        edges.astype(np.float32),
        (acfg["gaussian_ksize"], acfg["gaussian_ksize"]),
        acfg["blur_sigma"]
    )
    if edge_map.max() > 0:
        edge_map /= edge_map.max()

    # 2. Frequency domain energy
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log1p(np.abs(dft_shift))

    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    Y, X = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    mask = ((X - cx)**2 + (Y - cy)**2 > r**2).astype(np.float32)

    freq_map = magnitude * mask
    freq_map = cv2.GaussianBlur(
        freq_map.astype(np.float32),
        (acfg["gaussian_ksize"], acfg["gaussian_ksize"]),
        acfg["blur_sigma"]
    )
    freq_map = cv2.resize(freq_map, (w, h))
    if freq_map.max() > 0:
        freq_map /= freq_map.max()

    # 3. Local density variance
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 10)
    local_var = np.abs(gray.astype(np.float32) - blurred)
    local_var = cv2.GaussianBlur(
        local_var,
        (acfg["gaussian_ksize"], acfg["gaussian_ksize"]),
        acfg["blur_sigma"]
    )
    if local_var.max() > 0:
        local_var /= local_var.max()

    heatmap = (acfg["edge_weight"] * edge_map +
               acfg["frequency_weight"] * freq_map +
               acfg["density_weight"] * local_var)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = np.clip(heatmap, 0, 1).astype(np.float32)
    anomaly_score = float(np.percentile(heatmap, 95))

    return heatmap, anomaly_score


def render_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return overlay


# ──────────────────────────────────────────────
# Declaration Mismatch
# ──────────────────────────────────────────────

def compute_mismatch(
    detections: List[Dict], declared_cargo: Optional[str] = None
) -> Tuple[float, Dict]:
    if not declared_cargo or not declared_cargo.strip():
        return 0.0, {"declared": None, "conflicts": [], "message": "No declaration provided"}

    config = load_config()
    conflict_map = config["declaration"]["conflict_map"]
    declared_lower = declared_cargo.lower().strip()

    matched_type = None
    for cargo_type in conflict_map:
        if cargo_type in declared_lower:
            matched_type = cargo_type
            break

    if matched_type is None:
        return 0.2, {
            "declared": declared_cargo, "conflicts": [],
            "message": f"Cargo type '{declared_cargo}' not in known categories — flagged for review"
        }

    conflict_items = set(conflict_map[matched_type])
    detected_classes = {d["class"] for d in detections}
    conflicts = detected_classes & conflict_items

    if not conflicts:
        return 0.0, {
            "declared": declared_cargo, "matched_type": matched_type,
            "conflicts": [], "message": f"Declared '{declared_cargo}' — no conflicts detected"
        }

    max_threat = max(THREAT_LEVELS.get(c, 1) for c in conflicts)
    penalty = min(1.0, (len(conflicts) * 0.25) + (max_threat * 0.1))

    return penalty, {
        "declared": declared_cargo, "matched_type": matched_type,
        "conflicts": list(conflicts),
        "message": f"MISMATCH: Declared '{declared_cargo}' but detected {', '.join(conflicts)}"
    }


# ──────────────────────────────────────────────
# Risk Fusion (Updated with all signals)
# ──────────────────────────────────────────────

def compute_risk(
    detections: List[Dict],
    anomaly_score: float,
    mismatch_penalty: float,
    material_risk: float = 0.0,
    concealment_score: float = 0.0,
    vlm_threat: float = 0.0
) -> Tuple[int, str, Dict]:
    config = load_config()
    weights = config["risk"]["weights"]
    levels = config["risk"]["levels"]

    if detections:
        max_det_score = max(
            d["confidence"] * (d["threat_level"] / 5.0)
            for d in detections
        )
    else:
        max_det_score = 0.0

    raw_risk = (
        weights["detection"] * max_det_score +
        weights["anomaly"] * anomaly_score +
        weights["mismatch"] * mismatch_penalty +
        weights.get("material", 0.1) * material_risk +
        weights.get("concealment", 0.1) * concealment_score +
        weights.get("vlm_agreement", 0.1) * vlm_threat
    ) * 100

    risk_score = int(np.clip(raw_risk, 0, 100))

    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= levels["medium"]:
        risk_level = "HIGH"
    elif risk_score >= levels["low"]:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    breakdown = {
        "detection_component": round(weights["detection"] * max_det_score * 100, 1),
        "anomaly_component": round(weights["anomaly"] * anomaly_score * 100, 1),
        "mismatch_component": round(weights["mismatch"] * mismatch_penalty * 100, 1),
        "material_component": round(weights.get("material", 0.1) * material_risk * 100, 1),
        "concealment_component": round(weights.get("concealment", 0.1) * concealment_score * 100, 1),
        "vlm_component": round(weights.get("vlm_agreement", 0.1) * vlm_threat * 100, 1),
        "max_detection_conf": round(max_det_score, 3),
        "anomaly_score": round(anomaly_score, 3),
        "mismatch_penalty": round(mismatch_penalty, 3),
    }

    return risk_score, risk_level, breakdown


# ──────────────────────────────────────────────
# Template Explanation (fallback)
# ──────────────────────────────────────────────

def explain_risk(
    detections: List[Dict], risk_score: int, risk_level: str,
    anomaly_score: float, mismatch_info: Dict, breakdown: Dict,
    concealment_info: Optional[Dict] = None,
    material_info: Optional[Dict] = None
) -> str:
    lines = []

    if risk_level == "CRITICAL":
        lines.append("⚠️ CRITICAL RISK — Immediate manual inspection required.")
    elif risk_level == "HIGH":
        lines.append("🔴 HIGH RISK — Manual review strongly recommended.")
    elif risk_level == "MEDIUM":
        lines.append("🟡 MEDIUM RISK — Secondary screening advised.")
    else:
        lines.append("🟢 LOW RISK — No immediate threat indicators.")

    if detections:
        top = detections[:3]
        items = ", ".join(
            f"{d['class']} ({d['confidence']:.0%}, {d.get('material_type', '?')})"
            for d in top
        )
        lines.append(f"Detected: {items}.")
        high_threat = [d for d in detections if d["threat_level"] >= 4]
        if high_threat:
            lines.append(f"⚡ {len(high_threat)} high-threat item(s) identified.")
    else:
        lines.append("No prohibited items detected in scan.")

    if anomaly_score > 0.6:
        lines.append(f"Anomaly: {anomaly_score:.0%} — elevated density/concealment patterns.")
    elif anomaly_score > 0.3:
        lines.append(f"Anomaly: {anomaly_score:.0%} — minor irregularities noted.")

    if material_info and material_info.get("metallic_count", 0) > 0:
        lines.append(f"🔬 Material: {material_info['message']}")

    if concealment_info and concealment_info.get("overlap_score", 0) > 0.2:
        lines.append(f"🕵️ Concealment: {concealment_info['message']}")

    if mismatch_info.get("conflicts"):
        lines.append(f"📋 Declaration: {mismatch_info['message']}")

    if risk_score >= 70:
        lines.append("→ Action: Escalate for physical inspection.")
    elif risk_score >= 35:
        lines.append("→ Action: Flag for secondary review.")
    else:
        lines.append("→ Action: Clear for processing.")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────

def run_full_pipeline(
    image: np.ndarray,
    mode: str = "realtime",
    declared_cargo: Optional[str] = None,
    model: object = None
) -> Dict:
    if model is None:
        model = load_model(mode)

    t_start = time.time()

    # 1. Detection (with material classification per bbox)
    detections, det_time, annotated = detect_objects(image, model, mode)

    # 2. Anomaly
    heatmap, anomaly_score = compute_anomaly(image)
    heatmap_overlay = render_heatmap(image, heatmap)

    # 3. Declaration Mismatch
    mismatch_penalty, mismatch_info = compute_mismatch(detections, declared_cargo)

    # 4. Material Risk (novel)
    material_risk, material_info = compute_material_risk(detections, declared_cargo)

    # 5. Concealment Detection (novel)
    concealment_score, concealment_info = compute_concealment(image, detections)

    # 6. Risk Fusion (all signals)
    risk_score, risk_level, breakdown = compute_risk(
        detections, anomaly_score, mismatch_penalty,
        material_risk, concealment_score
    )

    # 7. Template Explanation
    explanation = explain_risk(
        detections, risk_score, risk_level, anomaly_score,
        mismatch_info, breakdown, concealment_info, material_info
    )

    total_time = time.time() - t_start

    return {
        "detections": detections,
        "detection_count": len(detections),
        "inference_time_ms": round(det_time * 1000, 1),
        "total_time_ms": round(total_time * 1000, 1),
        "fps": round(1.0 / max(total_time, 0.001), 1),
        "anomaly_score": round(anomaly_score, 3),
        "heatmap": heatmap,
        "heatmap_overlay": heatmap_overlay,
        "annotated_image": annotated,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_breakdown": breakdown,
        "mismatch_info": mismatch_info,
        "material_info": material_info,
        "concealment_score": round(concealment_score, 3),
        "concealment_info": concealment_info,
        "explanation": explanation,
        "mode": mode,
        "declared_cargo": declared_cargo,
    }
