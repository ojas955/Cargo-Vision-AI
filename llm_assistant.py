"""
CIIBS LLM Copilot Assistant — Gemini-Powered (google-genai SDK)
Dual-Brain Analysis: Text Explanation + Vision Analysis
Non-blocking with deterministic template fallback.
"""

import asyncio
import logging
import os
import time
from typing import Dict

import cv2
import numpy as np
import yaml

logger = logging.getLogger("ciibs.llm")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _get_genai_client():
    """Initialize and return the new google.genai Client."""
    from google import genai
    config = _load_config()
    api_key = os.environ.get("GEMINI_API_KEY") or config.get("llm", {}).get("api_key", "")
    if not api_key:
        raise ValueError(
            "API key not set in config.yaml or GEMINI_API_KEY env var."
        )
    client = genai.Client(api_key=api_key)
    return client


# ──────────────────────────────────────────────
# Gemini Text Explanation
# ──────────────────────────────────────────────

async def _call_gemini_text(prompt: str, config: dict) -> str:
    """Call Gemini text API for explanation generation."""
    try:
        client = _get_genai_client()
        model_name = config["llm"].get("model", "gemini-2.0-flash")
        system_prompt = config["llm"].get("system_prompt", "")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=full_prompt
            ),
            timeout=config["llm"]["timeout"]
        )

        return response.text.strip()

    except asyncio.TimeoutError:
        logger.warning("Gemini text API timeout — falling back to template")
        raise
    except Exception as e:
        logger.warning(f"Gemini text API error: {e}")
        raise


# ──────────────────────────────────────────────
# Gemini Vision — Dual-Brain Analysis ⭐
# ──────────────────────────────────────────────

async def _call_gemini_vision(image: np.ndarray, config: dict) -> str:
    """
    Send actual X-ray image to Gemini Vision for independent analysis.
    This is the "second brain" — a VLM that can SEE the image.
    """
    try:
        client = _get_genai_client()
        from PIL import Image as PILImage

        vision_cfg = config["llm"].get("vision", {})
        model_name = vision_cfg.get("model", "gemini-2.0-flash")
        vision_prompt = vision_cfg.get("prompt", "Analyze this X-ray scan image.")

        # Convert numpy BGR → RGB → PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        # Resize if too large (max 1024px for speed)
        max_dim = 1024
        if max(pil_image.size) > max_dim:
            ratio = max_dim / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size)

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=[vision_prompt, pil_image]
            ),
            timeout=config["llm"]["timeout"]
        )

        return response.text.strip()

    except asyncio.TimeoutError:
        logger.warning("Gemini Vision timeout")
        raise
    except Exception as e:
        logger.warning(f"Gemini Vision error: {e}")
        raise


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(analysis: Dict) -> str:
    """Build a structured prompt from analysis results."""
    det_summary = ""
    if analysis.get("detections"):
        items = []
        for d in analysis["detections"][:5]:
            mat = d.get("material_type", "unknown")
            items.append(
                f"- {d['class']}: {d['confidence']:.0%} confidence "
                f"(threat {d['threat_level']}/5, material: {mat})"
            )
        det_summary = "\n".join(items)
    else:
        det_summary = "- No prohibited items detected"

    mismatch = analysis.get("mismatch_info", {})
    mismatch_text = mismatch.get("message", "No declaration provided")

    concealment = analysis.get("concealment_score", 0)

    prompt = f"""X-ray scan result — Risk: {analysis['risk_score']}/100 ({analysis['risk_level']}), Anomaly: {analysis['anomaly_score']:.0%}, Concealment: {concealment:.0%}.
Items: {det_summary}. Declaration: {mismatch_text}. Declared: {analysis.get('declared_cargo', 'N/A')}.
Give a 3-4 sentence customs assessment: risk verdict, concerns, and recommended action."""

    return prompt


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def generate_explanation(analysis: Dict) -> Dict:
    """
    Generate Gemini-powered explanation.
    Falls back to template if API fails.
    """
    config = _load_config()

    if not config["llm"]["enabled"]:
        return {
            "explanation": analysis.get("explanation", "LLM disabled"),
            "source": "template",
            "latency_ms": 0
        }

    prompt = build_prompt(analysis)
    t0 = time.time()

    try:
        explanation = await _call_gemini_text(prompt, config)
        latency = (time.time() - t0) * 1000
        logger.info(f"Gemini explanation generated in {latency:.0f}ms")

        return {
            "explanation": explanation,
            "source": "gemini",
            "latency_ms": round(latency, 1)
        }

    except Exception:
        latency = (time.time() - t0) * 1000
        return {
            "explanation": analysis.get(
                "explanation",
                "Analysis complete — see risk score and detections."
            ),
            "source": "template",
            "latency_ms": round(latency, 1)
        }


async def analyze_with_vision(image: np.ndarray) -> Dict:
    """
    🧠 Dual-Brain: Send X-ray image to Gemini Vision for independent analysis.
    Returns structured VLM assessment.
    """
    config = _load_config()
    vision_cfg = config["llm"].get("vision", {})

    if not vision_cfg.get("enabled", False):
        return {
            "vlm_analysis": "Vision analysis disabled",
            "source": "disabled",
            "latency_ms": 0
        }

    t0 = time.time()

    try:
        analysis = await _call_gemini_vision(image, config)
        latency = (time.time() - t0) * 1000
        logger.info(f"Gemini Vision analysis in {latency:.0f}ms")

        return {
            "vlm_analysis": analysis,
            "source": "gemini-vision",
            "latency_ms": round(latency, 1)
        }

    except Exception as e:
        latency = (time.time() - t0) * 1000
        err_msg = str(e)
        return {
            "vlm_analysis": f"Vision analysis unavailable: {err_msg[:100]}",
            "source": "error",
            "latency_ms": round(latency, 1)
        }


def generate_explanation_sync(analysis: Dict) -> Dict:
    """Synchronous wrapper for generate_explanation."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, generate_explanation(analysis)
                )
                return future.result(timeout=15)
        else:
            return loop.run_until_complete(generate_explanation(analysis))
    except RuntimeError:
        return asyncio.run(generate_explanation(analysis))


def analyze_vision_sync(image: np.ndarray) -> Dict:
    """Synchronous wrapper for analyze_with_vision."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, analyze_with_vision(image)
                )
                return future.result(timeout=15)
        else:
            return loop.run_until_complete(analyze_with_vision(image))
    except RuntimeError:
        return asyncio.run(analyze_with_vision(image))
