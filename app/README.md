# NBA Commentary Studio (app)

## Lancer le site

Depuis la racine du projet:

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload
```

Puis ouvrir [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Endpoints API inclus

- `GET /api/health`
- `GET /api/models`
- `POST /api/upload` (multipart `file`)
- `POST /api/analyze` (JSON: `upload_id`, `model_id`)
- `GET /api/jobs/{job_id}`
- `GET /api/uploads/{upload_id}`
- `GET /api/previews/{job_id}`
- `GET /api/outputs/{job_id}` (vidéo annotée: boîtes + commentaire overlay)
- `GET /api/stream/{job_id}` (stream live MJPEG pendant l'analyse)
- `GET /api/tts/health`
- `POST /api/tts` (JSON: `text`, optionnels: `hype`, `model`, `voice`, `response_format`, `speed`)

## Modèles supportés

- `yolo11`
- `yolo12`
- `yolo26`

Le backend utilise en priorité les poids entraînés du repo, sinon bascule sur les poids de base (`yolo11n.pt`, `yolo12n.pt`, `yolo26n.pt`).

## LLM commentaire

Par défaut:
- base URL: `http://192.168.1.162:1234`
- modèle: `qwen/qwen2.5-vl-7b`

Variables optionnelles:
- `LMSTUDIO_BASE_URL`
- `LMSTUDIO_MODEL`
- `LMSTUDIO_API_KEY`

## TTS local (commentaire audio)

Par défaut:
- base URL: `http://192.168.1.162:1234`
- modèle: `neutts-nano-french`

Variables optionnelles:
- `TTS_BASE_URL`
- `TTS_MODEL`
- `TTS_API_KEY`
- `TTS_VOICE`
- `TTS_MACOS_FALLBACK` (`1` par défaut)
- `TTS_MACOS_VOICE` (`Thomas` par défaut)
- `TTS_MACOS_RATE` (`230` par défaut)
