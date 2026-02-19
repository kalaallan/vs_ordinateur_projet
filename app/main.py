from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from live_yolo12_mistral_commentary import (
    BALL_CLS,
    HOOP_CLS,
    PLAYER_CLS,
    CommentaryWorker,
    bucket_score,
    compute_metrics,
    predict_with_legacy_nms_if_needed,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "storage" / "uploads"
PREVIEWS_DIR = BASE_DIR / "storage" / "previews"
OUTPUTS_DIR = BASE_DIR / "storage" / "outputs"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
LLM_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.1.162:1234")
LLM_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen2.5-vl-7b")
LLM_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")
COMMENT_INTERVAL_SEC = 1.2
LLM_TIMEOUT_SEC = 4.0
TTS_BASE_URL = os.getenv("TTS_BASE_URL", LLM_BASE_URL)
TTS_MODEL = os.getenv("TTS_MODEL", "neutts-nano-french")
TTS_API_KEY = os.getenv("TTS_API_KEY", LLM_API_KEY)
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_TIMEOUT_SEC = 25.0
TTS_MACOS_FALLBACK = os.getenv("TTS_MACOS_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
TTS_MACOS_VOICE = os.getenv("TTS_MACOS_VOICE", "Thomas")
TTS_MACOS_RATE = int(os.getenv("TTS_MACOS_RATE", "230"))

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "yolo11": {
        "label": "YOLO11",
        "description": "Rapide, bonne précision",
        "primary_path": "runs/detect/train3/weights/best.pt",
        "fallback_path": "yolo11n.pt",
    },
    "yolo12": {
        "label": "YOLO12",
        "description": "Équilibré, recommandé",
        "primary_path": "runs/detect/yolo12/weights/best.pt",
        "fallback_path": "yolo12n.pt",
    },
    "yolo26": {
        "label": "YOLO26",
        "description": "Haute précision, plus lent",
        "primary_path": "runs/detect/yolo26/weights/best.pt",
        "fallback_path": "yolo26n.pt",
    },
}

uploads: dict[str, Path] = {}
jobs: dict[str, dict[str, Any]] = {}
model_cache: dict[str, YOLO] = {}
job_streams: dict[str, dict[str, Any]] = {}

uploads_lock = threading.Lock()
jobs_lock = threading.Lock()
model_lock = threading.Lock()
inference_lock = threading.Lock()
streams_lock = threading.Lock()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NBA Commentary Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    upload_id: str
    model_id: str


class TTSRequest(BaseModel):
    text: str
    hype: bool = True
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tts_text(text: str) -> str:
    clean = " ".join((text or "").strip().split())
    if len(clean) <= 420:
        return clean
    clipped = clean[:420]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()


def guess_audio_mime(audio_bytes: bytes, fallback: str = "audio/wav") -> str:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes[:3] == b"ID3" or (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio/ogg"
    if audio_bytes[:4] == b"fLaC":
        return "audio/flac"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return "audio/mp4"
    return fallback


def _decode_base64_candidate(value: str) -> bytes | None:
    if not value:
        return None
    candidate = value.strip()
    if candidate.startswith("data:audio/") and "," in candidate:
        candidate = candidate.split(",", 1)[1]
    candidate = candidate.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(candidate) < 80:
        return None

    for decoder in (
        lambda s: base64.b64decode(s, validate=True),
        base64.b64decode,
        base64.urlsafe_b64decode,
    ):
        try:
            padded = candidate + "=" * ((4 - (len(candidate) % 4)) % 4)
            decoded = decoder(padded)
        except Exception:
            continue
        if len(decoded) < 128:
            continue
        if guess_audio_mime(decoded, fallback="application/octet-stream") != "application/octet-stream":
            return decoded
    return None


def _extract_audio_from_json(obj: Any, timeout_sec: float) -> tuple[bytes | None, str | None]:
    audio_key_re = re.compile(r"(audio|speech|wav|mp3|ogg|flac|b64|base64|voice|tts|file|path|url)", re.IGNORECASE)
    url_candidates: list[str] = []
    path_candidates: list[str] = []
    b64_candidates: list[str] = []

    def walk(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                k_hint = f"{key_hint}.{k}" if key_hint else str(k)
                walk(v, k_hint)
            return
        if isinstance(node, list):
            for idx, item in enumerate(node):
                walk(item, f"{key_hint}[{idx}]")
            return
        if not isinstance(node, str):
            return

        value = node.strip()
        lowered_hint = key_hint.lower()
        is_audio_field = bool(audio_key_re.search(lowered_hint))
        if value.startswith(("http://", "https://")) and (is_audio_field or "/audio/" in value or value.endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a"))):
            url_candidates.append(value)
            return
        if value.startswith("data:audio/"):
            b64_candidates.append(value)
            return
        if is_audio_field and ("/" in value or value.endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a"))):
            path_candidates.append(value)
            return
        if is_audio_field and len(value) >= 80:
            b64_candidates.append(value)
            return
        if len(value) >= 200 and re.fullmatch(r"[A-Za-z0-9_\-+/=\s]+", value):
            b64_candidates.append(value)

    walk(obj)

    for candidate in b64_candidates:
        decoded = _decode_base64_candidate(candidate)
        if decoded:
            return decoded, guess_audio_mime(decoded)

    for url in url_candidates:
        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as audio_resp:
                audio_url_bytes = audio_resp.read()
                audio_url_type = (audio_resp.headers.get("Content-Type", "") or "").lower()
                if audio_url_bytes:
                    return audio_url_bytes, audio_url_type or guess_audio_mime(audio_url_bytes)
        except Exception:
            continue

    for path_str in path_candidates:
        candidate_path = Path(path_str)
        if candidate_path.exists() and candidate_path.is_file():
            data = candidate_path.read_bytes()
            if data:
                return data, guess_audio_mime(data)
        relative_in_app = BASE_DIR / candidate_path
        if relative_in_app.exists() and relative_in_app.is_file():
            data = relative_in_app.read_bytes()
            if data:
                return data, guess_audio_mime(data)

    return None, None


def hype_commentary_text(text: str) -> str:
    clean = normalize_tts_text(text)
    if not clean:
        return clean
    if clean[-1] not in ".!?":
        clean += "."
    starters = [
        "Le public pousse fort,",
        "Quelle ambiance NBA,",
        "Ça monte en intensité,",
        "On sent la salle qui s'enflamme,",
    ]
    closers = [
        "Ça fait lever les fans.",
        "Le rythme est énorme.",
        "On vit un vrai temps fort.",
        "La séquence est brûlante.",
    ]
    digest = hashlib.md5(clean.encode("utf-8")).hexdigest()
    start_idx = int(digest[:2], 16) % len(starters)
    end_idx = int(digest[2:4], 16) % len(closers)
    return f"{starters[start_idx]} {clean} {closers[end_idx]}"


def call_lmstudio_tts(
    base_url: str,
    model_id: str,
    text: str,
    response_format: str,
    speed: float,
    voice: str | None,
    timeout_sec: float,
    api_key: str | None = None,
) -> tuple[bytes, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payloads: list[dict[str, Any]] = []
    full_payload: dict[str, Any] = {
        "model": model_id,
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    if voice:
        full_payload["voice"] = voice
    payloads.append(full_payload)

    alt_payload: dict[str, Any] = {
        "model": model_id,
        "input": text,
        "format": response_format,
    }
    if voice:
        alt_payload["voice"] = voice
    payloads.append(alt_payload)
    payloads.append({"model": model_id, "input": text})

    errors: list[str] = []
    endpoints = ["/v1/audio/speech", "/v1/tts"]
    for endpoint in endpoints:
        for payload in payloads:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                base_url.rstrip("/") + endpoint,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    audio = resp.read()
                    content_type = (resp.headers.get("Content-Type", "") or "").lower()
                    if not audio:
                        raise RuntimeError("Réponse audio vide.")
                    if "application/json" in content_type or "text/" in content_type:
                        try:
                            obj = json.loads(audio.decode("utf-8", errors="ignore"))
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("Réponse TTS JSON invalide.") from exc

                        extracted_bytes, extracted_mime = _extract_audio_from_json(obj, timeout_sec=timeout_sec)
                        if extracted_bytes:
                            return extracted_bytes, (extracted_mime or guess_audio_mime(extracted_bytes))

                        if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                            raise RuntimeError(
                                "Le serveur a répondu en mode chat (texte) au lieu de TTS audio. "
                                "Vérifie que le modèle TTS est chargé et que l'endpoint TTS est activé."
                            )

                        preview = json.dumps(obj, ensure_ascii=True)[:300]
                        raise RuntimeError(f"Réponse TTS JSON sans audio exploitable. Aperçu: {preview}")
                    return audio, content_type
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                errors.append(f"{endpoint} HTTP {exc.code}: {body[:240]}")
            except urllib.error.URLError as exc:
                errors.append(f"{endpoint} URL error: {exc}")
            except TimeoutError:
                errors.append(f"{endpoint} timeout")

    raise RuntimeError("TTS indisponible. " + " | ".join(errors[-3:]))


def synthesize_with_macos_say(text: str, voice: str | None = None, rate_wpm: int = 230) -> bytes:
    # Fallback local robuste sur macOS quand LM Studio ne sert pas de endpoint TTS audio.
    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            output_path = tmp.name

        def run_say(selected_voice: str | None) -> subprocess.CompletedProcess[str]:
            cmd = ["say", "-r", str(rate_wpm), "-o", output_path]
            if selected_voice:
                cmd.extend(["-v", selected_voice])
            cmd.append(text)
            return subprocess.run(cmd, check=True, capture_output=True, text=True)

        try:
            run_say(voice)
        except subprocess.CalledProcessError:
            run_say(None)

        data = Path(output_path).read_bytes()
        if not data:
            raise RuntimeError("Audio macOS vide.")
        return data
    except FileNotFoundError as exc:
        raise RuntimeError("Commande macOS 'say' introuvable.") from exc
    finally:
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass


def resolve_model_weights(model_id: str) -> Path:
    model_cfg = MODEL_CONFIGS[model_id]
    candidates = [
        ROOT_DIR / model_cfg["primary_path"],
        ROOT_DIR / model_cfg["fallback_path"],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Poids introuvables pour {model_cfg['label']}")


def load_model(model_id: str) -> YOLO:
    with model_lock:
        if model_id in model_cache:
            return model_cache[model_id]
        weights = resolve_model_weights(model_id)
        model = YOLO(str(weights))
        model_cache[model_id] = model
        return model


def find_upload(upload_id: str) -> Path | None:
    with uploads_lock:
        existing = uploads.get(upload_id)
    if existing and existing.exists():
        return existing

    matches = list(UPLOADS_DIR.glob(f"{upload_id}.*"))
    if not matches:
        return None

    with uploads_lock:
        uploads[upload_id] = matches[0]
    return matches[0]


def update_job(job_id: str, **fields: Any) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(fields)


def summarize_commentary(attack_score: float, defense_score: float, players_avg: float) -> str:
    if attack_score >= 65 and defense_score < 55:
        return "L'attaque prend le dessus, le rythme est clairement offensif."
    if defense_score >= 65 and attack_score < 55:
        return "La défense verrouille bien les espaces et coupe les options adverses."
    if players_avg < 6:
        return "Peu de joueurs clairement visibles, la séquence reste difficile à lire."
    return "Le jeu est équilibré, avec des phases d'attaque et de défense stables."


def draw_overlay_boxes_only(frame: np.ndarray, detections: dict[str, Any], metrics: dict[str, Any], fps: float) -> np.ndarray:
    out = frame.copy()

    team0_color = (255, 0, 0)
    team1_color = (0, 0, 255)
    unknown_color = (0, 255, 255)

    players_xyxy = detections["players_xyxy"]
    player_confs = detections["player_confs"]
    balls_xyxy = detections["balls_xyxy"]
    hoops_xyxy = detections["hoops_xyxy"]
    team = detections["team"]
    attack_team = detections["attack_team"]

    for i, xyxy in enumerate(players_xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        t = team[i]
        color = team0_color if t == 0 else team1_color if t == 1 else unknown_color
        role = ""
        if attack_team != -1 and t != -1:
            role = "ATT" if t == attack_team else "DEF"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if role == "DEF" else 3)
        label = f"{role} T{t} {player_confs[i]:.2f}" if role else f"T{t} {player_confs[i]:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for xyxy in balls_xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "Ball", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for xyxy in hoops_xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(out, "Hoop", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    top_line = (
        f"ATQ {metrics['attack_score']:.0f} ({bucket_score(metrics['attack_score'])})  "
        f"DEF {metrics['defense_score']:.0f} ({bucket_score(metrics['defense_score'])})  "
        f"FPS {fps:.1f}"
    )
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 760), 44), (0, 0, 0), -1)
    cv2.putText(out, top_line, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)

    return out


def run_analysis(job_id: str, upload_id: str, model_id: str) -> None:
    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    worker: CommentaryWorker | None = None
    try:
        upload_path = find_upload(upload_id)
        if upload_path is None:
            raise FileNotFoundError("Vidéo introuvable.")

        model = load_model(model_id)
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la vidéo uploadée.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            source_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = OUTPUTS_DIR / f"{job_id}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Impossible de créer la vidéo annotée.")

        attack_scores: list[float] = []
        defense_scores: list[float] = []
        player_counts: list[int] = []
        confidences: list[float] = []
        processed = 0  # Frames réellement traitées par YOLO.
        read_frames = 0  # Frames lues de la vidéo source.
        preview_written = False
        start_ts = time.time()
        t_prev = start_ts
        fps_smooth = 0.0

        worker = CommentaryWorker(
            base_url=LLM_BASE_URL,
            model_id=LLM_MODEL,
            interval_sec=COMMENT_INTERVAL_SEC,
            timeout_sec=LLM_TIMEOUT_SEC,
            api_key=LLM_API_KEY or None,
        )
        worker.start()

        update_job(
            job_id,
            status="running",
            progress=0.01,
            detail=f"Analyse + annotation en cours ({MODEL_CONFIGS[model_id]['label']})…",
        )
        with streams_lock:
            if job_id in job_streams:
                job_streams[job_id]["done"] = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            read_frames += 1

            with inference_lock:
                result = predict_with_legacy_nms_if_needed(
                    model,
                    frame,
                    conf=0.45,
                    iou=0.5,
                    imgsz=640,
                    classes=[BALL_CLS, HOOP_CLS, PLAYER_CLS],
                    verbose=False,
                )[0]

            metrics, detections = compute_metrics(frame, result)

            now = time.time()
            dt = max(1e-6, now - t_prev)
            t_prev = now
            fps_now = 1.0 / dt
            fps_smooth = fps_now if fps_smooth <= 0 else (0.9 * fps_smooth + 0.1 * fps_now)

            payload = {
                "frame": read_frames,
                "equipe_en_attaque": metrics["attack_team"],
                "intensite_attaque": bucket_score(metrics["attack_score"]),
                "intensite_defense": bucket_score(metrics["defense_score"]),
                "pression_defensive": (
                    "forte"
                    if (metrics["defenders_close_to_carrier"] or 0) >= 2
                    else "moyenne"
                    if (metrics["defenders_close_to_carrier"] or 0) == 1
                    else "faible"
                ),
                "proximite_panier": (
                    "haute"
                    if (metrics["closest_att_to_hoop_px"] is not None and metrics["closest_att_to_hoop_px"] < 150)
                    else "moyenne"
                    if (metrics["closest_att_to_hoop_px"] is not None and metrics["closest_att_to_hoop_px"] < 260)
                    else "basse"
                ),
                "joueurs_detectes": metrics["n_players"],
                "ballon_detecte": metrics["n_ball"] > 0,
            }
            worker.maybe_submit(payload)

            frame_out = draw_overlay_boxes_only(frame, detections, metrics, fps_smooth)
            writer.write(frame_out)
            ok_jpg, jpg = cv2.imencode(".jpg", frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok_jpg:
                with streams_lock:
                    stream = job_streams.get(job_id)
                    if stream is not None:
                        stream["frame_jpeg"] = jpg.tobytes()
                        stream["seq"] += 1

            if not preview_written:
                preview_path = PREVIEWS_DIR / f"{job_id}.jpg"
                cv2.imwrite(str(preview_path), frame_out)
                preview_written = True

            attack_scores.append(metrics["attack_score"])
            defense_scores.append(metrics["defense_score"])
            player_counts.append(metrics["n_players"])
            if detections["player_confs"]:
                confidences.append(float(np.mean(detections["player_confs"])))

            processed += 1
            if processed % 8 == 0:
                progress_base = processed / max(1, total_frames) if total_frames > 0 else min(0.95, processed / 300.0)
                progress = min(0.97, progress_base)
                live_comment = worker.last_text if worker is not None else ""
                update_job(
                    job_id,
                    progress=round(progress, 3),
                    detail=f"Frames annotées: {processed}{f'/{total_frames}' if total_frames > 0 else ''}",
                    live_comment=live_comment,
                    live_attack=round(metrics["attack_score"], 1),
                    live_defense=round(metrics["defense_score"], 1),
                    live_players=metrics["n_players"],
                )

        elapsed = max(1e-6, time.time() - start_ts)
        fps = processed / elapsed

        avg_attack = float(np.mean(attack_scores)) if attack_scores else 0.0
        avg_defense = float(np.mean(defense_scores)) if defense_scores else 0.0
        avg_players = float(np.mean(player_counts)) if player_counts else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        live_comment = worker.last_text if worker is not None else ""
        if not live_comment or live_comment.lower().startswith("initialisation"):
            live_comment = summarize_commentary(avg_attack, avg_defense, avg_players)

        result_payload = {
            "model": MODEL_CONFIGS[model_id]["label"],
            "fps": round(fps, 1),
            "confidence": round(avg_conf * 100, 1),
            "attack_score": round(avg_attack, 1),
            "defense_score": round(avg_defense, 1),
            "players_average": round(avg_players, 1),
            "commentary": live_comment,
            "summary": summarize_commentary(avg_attack, avg_defense, avg_players),
            "video_url": f"/api/outputs/{job_id}",
            "preview_url": f"/api/previews/{job_id}",
            "llm_model": LLM_MODEL,
            "llm_base_url": LLM_BASE_URL,
        }

        update_job(
            job_id,
            status="completed",
            progress=1.0,
            detail="Analyse terminée.",
            result=result_payload,
            finished_at=utc_now(),
        )
        with streams_lock:
            stream = job_streams.get(job_id)
            if stream is not None:
                stream["done"] = True
    except Exception as exc:  # noqa: BLE001
        update_job(
            job_id,
            status="failed",
            detail=f"Erreur: {exc}",
            finished_at=utc_now(),
        )
        with streams_lock:
            stream = job_streams.get(job_id)
            if stream is not None:
                stream["done"] = True
    finally:
        if worker is not None:
            worker.stop()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tts/health")
def tts_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "base_url": TTS_BASE_URL,
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_id, cfg in MODEL_CONFIGS.items():
        available = any((ROOT_DIR / cfg[path_key]).exists() for path_key in ("primary_path", "fallback_path"))
        models.append(
            {
                "id": model_id,
                "name": cfg["label"],
                "description": cfg["description"],
                "available": available,
            }
        )
    return {"items": models, "default": "yolo11"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Format vidéo non supporté.")

    upload_id = uuid.uuid4().hex
    out_path = UPLOADS_DIR / f"{upload_id}{ext}"

    with out_path.open("wb") as stream:
        shutil.copyfileobj(file.file, stream)

    with uploads_lock:
        uploads[upload_id] = out_path

    return {
        "upload_id": upload_id,
        "file_name": file.filename or out_path.name,
        "video_url": f"/api/uploads/{upload_id}",
    }


@app.get("/api/uploads/{upload_id}")
def serve_uploaded_video(upload_id: str) -> FileResponse:
    upload_path = find_upload(upload_id)
    if upload_path is None:
        raise HTTPException(status_code=404, detail="Vidéo non trouvée.")
    return FileResponse(upload_path)


@app.get("/api/outputs/{job_id}")
def serve_annotated_output(job_id: str) -> FileResponse:
    output_path = OUTPUTS_DIR / f"{job_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Vidéo annotée non trouvée.")
    return FileResponse(output_path)


@app.get("/api/stream/{job_id}")
def stream_job(job_id: str) -> StreamingResponse:
    with streams_lock:
        if job_id not in job_streams:
            raise HTTPException(status_code=404, detail="Stream non trouvé.")

    def generate():
        last_seq = -1
        while True:
            with streams_lock:
                stream = job_streams.get(job_id)
                if stream is None:
                    return
                seq = int(stream.get("seq", 0))
                frame_jpeg = stream.get("frame_jpeg")
                done = bool(stream.get("done", False))

            if frame_jpeg is not None and seq != last_seq:
                last_seq = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_jpeg + b"\r\n"
                continue

            if done:
                return

            time.sleep(0.04)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/previews/{job_id}")
def serve_preview(job_id: str) -> FileResponse:
    preview_path = PREVIEWS_DIR / f"{job_id}.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview indisponible.")
    return FileResponse(preview_path)


@app.post("/api/tts")
def synthesize_commentary(request: TTSRequest) -> Response:
    base_text = normalize_tts_text(request.text)
    if len(base_text) < 3:
        raise HTTPException(status_code=400, detail="Texte trop court pour la synthèse.")

    styled_text = hype_commentary_text(base_text) if request.hype else base_text
    model_id = (request.model or TTS_MODEL).strip()
    voice = (request.voice or TTS_VOICE).strip()
    response_format = (request.response_format or "mp3").strip().lower()
    speed = max(0.75, min(1.25, float(request.speed)))

    try:
        audio_bytes, media_type = call_lmstudio_tts(
            base_url=TTS_BASE_URL,
            model_id=model_id,
            text=styled_text,
            response_format=response_format,
            speed=speed,
            voice=voice or None,
            timeout_sec=TTS_TIMEOUT_SEC,
            api_key=TTS_API_KEY or None,
        )
    except RuntimeError as exc:
        err = str(exc)
        if TTS_MACOS_FALLBACK and ("Unexpected endpoint or method" in err or "TTS indisponible" in err):
            try:
                audio_bytes = synthesize_with_macos_say(
                    text=styled_text,
                    voice=TTS_MACOS_VOICE or None,
                    rate_wpm=TTS_MACOS_RATE,
                )
                return Response(
                    content=audio_bytes,
                    media_type="audio/aiff",
                    headers={
                        "Cache-Control": "no-store",
                        "X-TTS-Provider": "macos-say-fallback",
                    },
                )
            except RuntimeError as fallback_exc:
                raise HTTPException(status_code=502, detail=f"{err} | fallback macOS: {fallback_exc}") from fallback_exc
        raise HTTPException(status_code=502, detail=err) from exc

    if not media_type.startswith("audio/"):
        if response_format == "mp3":
            media_type = "audio/mpeg"
        elif response_format == "wav":
            media_type = "audio/wav"
        else:
            media_type = "application/octet-stream"

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    if request.model_id not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Modèle inconnu.")

    upload_path = find_upload(request.upload_id)
    if upload_path is None:
        raise HTTPException(status_code=404, detail="Upload introuvable.")

    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "upload_id": request.upload_id,
        "model_id": request.model_id,
        "status": "queued",
        "progress": 0.0,
        "detail": "Job en attente…",
        "result": None,
        "created_at": utc_now(),
        "finished_at": None,
    }

    with jobs_lock:
        jobs[job_id] = job
    with streams_lock:
        job_streams[job_id] = {
            "frame_jpeg": None,
            "seq": 0,
            "done": False,
        }

    worker = threading.Thread(
        target=run_analysis,
        args=(job_id, request.upload_id, request.model_id),
        daemon=True,
    )
    worker.start()

    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trouvé.")
    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
