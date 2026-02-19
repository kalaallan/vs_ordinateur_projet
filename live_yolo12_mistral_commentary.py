import argparse
import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


BALL_CLS = 0
HOOP_CLS = 1
PLAYER_CLS = 3


def clamp01(x):
    return max(0.0, min(1.0, x))


def bucket_score(value):
    if value is None:
        return "inconnu"
    if value >= 70:
        return "fort"
    if value >= 45:
        return "moyen"
    return "faible"


def center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_pairwise_distance(points):
    if len(points) < 2:
        return None
    dsum, cnt = 0.0, 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dsum += dist(points[i], points[j])
            cnt += 1
    return dsum / cnt


def crop_jersey(img, xyxy):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 5 or bh < 5:
        return None

    cx1 = x1 + int(0.25 * bw)
    cx2 = x1 + int(0.75 * bw)
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.55 * bh)

    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(w - 1, cx2)
    cy2 = min(h - 1, cy2)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def jersey_feat_hsv(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(-1, 3).astype(np.float32)
    _, s, v = pix[:, 0], pix[:, 1], pix[:, 2]
    mask = (s > 40) & (v > 40) & (v < 240)
    pix = pix[mask]
    if len(pix) < 20:
        return hsv.reshape(-1, 3).mean(axis=0)
    return pix.mean(axis=0)


def pick_best_ball(ball_centers, player_centers):
    best_ball = None
    best_dist = 1e18
    for bc in ball_centers:
        for pc in player_centers:
            d = dist(bc, pc)
            if d < best_dist:
                best_dist = d
                best_ball = bc
    return best_ball


def predict_with_legacy_nms_if_needed(model, source, **kwargs):
    uses_end2end_head = bool(getattr(getattr(model, "model", None), "end2end", False))
    if not uses_end2end_head:
        return model.predict(source, **kwargs)
    try:
        return model.predict(source, end2end=False, **kwargs)
    except Exception as exc:
        if "end2end" in str(exc).lower():
            return model.predict(source, **kwargs)
        raise


def compute_metrics(frame, result):
    players_xyxy = []
    player_confs = []
    balls_xyxy = []
    hoops_xyxy = []

    for b in result.boxes:
        cls = int(b.cls[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0].item())
        if cls == PLAYER_CLS:
            players_xyxy.append(xyxy)
            player_confs.append(conf)
        elif cls == BALL_CLS:
            balls_xyxy.append(xyxy)
        elif cls == HOOP_CLS:
            hoops_xyxy.append(xyxy)

    player_centers = [center_xyxy(p) for p in players_xyxy]
    ball_centers = [center_xyxy(b) for b in balls_xyxy]
    hoop_center = center_xyxy(hoops_xyxy[0]) if len(hoops_xyxy) > 0 else None

    jersey_feats = []
    valid_idxs = []
    for i, p in enumerate(players_xyxy):
        crop = crop_jersey(frame, p)
        if crop is None:
            continue
        jersey_feats.append(jersey_feat_hsv(crop))
        valid_idxs.append(i)

    team = [-1] * len(players_xyxy)
    if len(jersey_feats) >= 2:
        x = np.array(jersey_feats, dtype=np.float32)
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(x)
        for j, idx in enumerate(valid_idxs):
            team[idx] = int(labels[j])
    else:
        team = [0] * len(players_xyxy)

    carrier_idx = None
    best_ball = None
    if len(ball_centers) > 0 and len(player_centers) > 0:
        best_ball = pick_best_ball(ball_centers, player_centers)
        best_d = 1e18
        for i, pc in enumerate(player_centers):
            d = dist(pc, best_ball)
            if d < best_d:
                best_d = d
                carrier_idx = i

    attack_team = -1
    if carrier_idx is not None and team[carrier_idx] != -1:
        attack_team = team[carrier_idx]
    else:
        if hoop_center is not None and len(player_centers) > 0:
            hoop_radius = 260
            score0 = 0
            score1 = 0
            for i, pc in enumerate(player_centers):
                if team[i] == -1:
                    continue
                if dist(pc, hoop_center) < hoop_radius:
                    if team[i] == 0:
                        score0 += 1
                    elif team[i] == 1:
                        score1 += 1
            if score0 != score1:
                attack_team = 0 if score0 > score1 else 1
            else:
                xs0 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 0]
                xs1 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 1]
                if len(xs0) > 0 and len(xs1) > 0:
                    attack_team = 0 if (sum(xs0) / len(xs0)) > (sum(xs1) / len(xs1)) else 1
                else:
                    attack_team = 0
        else:
            xs0 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 0]
            xs1 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 1]
            if len(xs0) > 0 and len(xs1) > 0:
                attack_team = 0 if (sum(xs0) / len(xs0)) > (sum(xs1) / len(xs1)) else 1
            else:
                attack_team = 0

    attackers = [player_centers[i] for i in range(len(player_centers)) if team[i] == attack_team]
    defenders = [player_centers[i] for i in range(len(player_centers)) if team[i] != -1 and team[i] != attack_team]

    spacing_px = mean_pairwise_distance(attackers)
    carrier_nearest_def_px = None
    defenders_close_to_carrier = None
    closest_att_to_hoop_px = None

    if carrier_idx is not None and len(defenders) > 0 and attack_team != -1:
        carrier = player_centers[carrier_idx]
        carrier_nearest_def_px = min(dist(carrier, d) for d in defenders)
        radius = 120
        defenders_close_to_carrier = sum(1 for d in defenders if dist(d, carrier) < radius)

    if hoop_center is not None and len(attackers) > 0:
        closest_att_to_hoop_px = min(dist(hoop_center, a) for a in attackers)

    spacing_score = 0.5 if spacing_px is None else clamp01((spacing_px - 80) / (220 - 80))
    pressure_score = (
        0.5
        if carrier_nearest_def_px is None
        else clamp01((carrier_nearest_def_px - 60) / (200 - 60))
    )
    threat_score = (
        0.5
        if closest_att_to_hoop_px is None
        else 1.0 - clamp01((closest_att_to_hoop_px - 80) / (350 - 80))
    )
    pressure_def_score = 1.0 - pressure_score

    hoop_def_score = 0.5
    if hoop_center is not None and len(defenders) > 0:
        hoop_radius2 = 180
        n_close = sum(1 for d in defenders if dist(d, hoop_center) < hoop_radius2)
        hoop_def_score = clamp01(n_close / 4.0)

    attack_score = round(100 * (0.40 * spacing_score + 0.35 * pressure_score + 0.25 * threat_score), 1)
    defense_score = round(100 * (0.60 * pressure_def_score + 0.40 * hoop_def_score), 1)

    metrics = {
        "n_players": len(players_xyxy),
        "n_team0": sum(1 for t in team if t == 0),
        "n_team1": sum(1 for t in team if t == 1),
        "n_ball": len(balls_xyxy),
        "n_hoop": len(hoops_xyxy),
        "attack_team": int(attack_team),
        "attack_score": attack_score,
        "defense_score": defense_score,
        "spacing_px": None if spacing_px is None else round(float(spacing_px), 1),
        "carrier_nearest_def_px": None if carrier_nearest_def_px is None else round(float(carrier_nearest_def_px), 1),
        "closest_att_to_hoop_px": None if closest_att_to_hoop_px is None else round(float(closest_att_to_hoop_px), 1),
        "defenders_close_to_carrier": defenders_close_to_carrier,
    }

    detections = {
        "players_xyxy": players_xyxy,
        "player_confs": player_confs,
        "balls_xyxy": balls_xyxy,
        "hoops_xyxy": hoops_xyxy,
        "team": team,
        "attack_team": attack_team,
    }
    return metrics, detections


def call_lmstudio_chat(base_url, model_id, system_prompt, user_prompt, timeout, api_key=None):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 64,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return obj["choices"][0]["message"]["content"].strip()


class CommentaryWorker:
    def __init__(self, base_url, model_id, interval_sec, timeout_sec, api_key=None):
        self.base_url = base_url
        self.model_id = model_id
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec
        self.api_key = api_key
        self.queue = queue.Queue(maxsize=1)
        self.last_text = ""
        self.last_error = None
        self.last_sent_ts = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.recent_lines = []
        self.system_prompt = (
            "Tu es un commentateur NBA en direct, style TV francophone. "
            "Objectif: produire une phrase vivante, naturelle, imagee et facile a entendre. "
            "Contraintes: 12 a 24 mots, en une seule phrase, sans JSON, sans puces, sans prefixe "
            "('commentaire', 'live', etc.). "
            "Ne repete pas la meme tournure que les dernieres phrases. "
            "Reste strictement base sur les infos recues; n'invente ni score ni joueur."
        )

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def maybe_submit(self, payload):
        now = time.time()
        if now - self.last_sent_ts < self.interval_sec:
            return
        self.last_sent_ts = now
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    @staticmethod
    def _clean_commentary(text):
        clean = (text or "").replace("\n", " ").strip().strip('"').strip("'")
        clean = re.sub(r"\s+", " ", clean)
        clean = re.sub(
            r"^(commentaire|commentaire live|live|analyse|observation)\s*[:\-]\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        words = clean.split()
        if len(words) > 24:
            clean = " ".join(words[:24]).rstrip(",;:")
        if clean and clean[-1] not in ".!?":
            clean += "."
        if clean:
            clean = clean[0].upper() + clean[1:]
        return clean

    @staticmethod
    def _fallback_commentary(payload):
        att = payload.get("intensite_attaque", "moyen")
        deff = payload.get("intensite_defense", "moyen")
        pressure = payload.get("pression_defensive", "moyenne")
        hoop = payload.get("proximite_panier", "moyenne")
        players = int(payload.get("joueurs_detectes") or 0)
        has_ball = bool(payload.get("ballon_detecte"))
        atk_team = payload.get("equipe_en_attaque")
        frame = int(payload.get("frame") or 0)

        team_hint = (
            "l'equipe en attaque"
            if atk_team in (-1, None)
            else f"l'equipe {atk_team}"
        )

        options = []
        if att == "fort" and pressure in ("faible", "moyenne"):
            options.append(f"{team_hint.capitalize()} accelere, les espaces s'ouvrent et la defense recule.")
        if deff == "fort" or pressure == "forte":
            options.append("La defense monte fort sur le porteur, chaque dribble est conteste.")
        if hoop == "haute" and has_ball:
            options.append("Ca se rapproche du cercle, l'action peut basculer tres vite.")
        if players < 6:
            options.append("Sequence brouillonne, peu de joueurs nets, mais l'intensite reste presente.")
        if not has_ball:
            options.append("Le ballon disparait un instant, mais le mouvement collectif continue.")
        if not options:
            options.append("Le rythme monte, attaque et defense se repondent possession apres possession.")

        return options[frame % len(options)]

    def _remember_line(self, line):
        if not line:
            return
        self.recent_lines.append(line)
        if len(self.recent_lines) > 6:
            self.recent_lines = self.recent_lines[-6:]

    def _run(self):
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            recent = self.recent_lines[-3:]
            user_prompt = (
                "Situation de jeu:\n"
                f"{json.dumps(payload, ensure_ascii=True)}\n"
                f"Dernieres phrases a eviter:\n{json.dumps(recent, ensure_ascii=True)}\n"
                "Donne un commentaire de diffusion TV, naturel et energique."
            )
            try:
                txt = call_lmstudio_chat(
                    base_url=self.base_url,
                    model_id=self.model_id,
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    timeout=self.timeout_sec,
                    api_key=self.api_key,
                )
                clean = self._clean_commentary(txt)
                if not clean or len(clean.split()) < 8:
                    clean = self._fallback_commentary(payload)
                if clean == self.last_text:
                    clean = self._fallback_commentary(payload)
                self.last_text = clean
                self._remember_line(clean)
                self.last_error = None
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                self.last_error = str(exc)
                fallback = self._fallback_commentary(payload)
                self.last_text = fallback
                self._remember_line(fallback)


def wrap_text(text, width=70):
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add_len = len(w) + (1 if cur else 0)
        if cur_len + add_len > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add_len
    if cur:
        lines.append(" ".join(cur))
    return lines if lines else [""]


def draw_overlay(frame, detections, metrics, comment_text, fps):
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

    att_level = bucket_score(metrics["attack_score"])
    def_level = bucket_score(metrics["defense_score"])
    top_line = (
        f"ATQ {metrics['attack_score']:.0f} ({att_level})  "
        f"DEF {metrics['defense_score']:.0f} ({def_level})  "
        f"FPS {fps:.1f}"
    )
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 760), 44), (0, 0, 0), -1)
    cv2.putText(out, top_line, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)

    # Commentaire en haut-droite pour laisser libre la zone du bas (scoreboard video).
    comment_lines = wrap_text(comment_text, width=46)[:2]
    panel_w = 560
    panel_h = 86
    x2 = out.shape[1] - 10
    x1 = max(10, x2 - panel_w)
    y1 = 52
    y2 = min(out.shape[0] - 10, y1 + panel_h)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.52, out, 0.48, 0)
    cv2.putText(out, "Commentaire live", (x1 + 10, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    y = y1 + 49
    for ln in comment_lines:
        cv2.putText(out, ln, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (170, 255, 170), 2)
        y += 24

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO12 + attack/defense + commentaire Mistral (LM Studio) en temps reel")
    parser.add_argument("--video", default="Ja Morant Block.mp4", help="Chemin video")
    parser.add_argument("--model", default="runs/detect/yolo12/weights/best.pt", help="Poids YOLO")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance YOLO")
    parser.add_argument("--iou", type=float, default=0.5, help="Seuil IOU NMS YOLO")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille image YOLO")
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234"), help="Base URL LM Studio")
    parser.add_argument("--llm-model", default=os.getenv("LMSTUDIO_MODEL", "mistralai/ministral-3-3b"), help="Identifiant modele LM Studio")
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", ""), help="API key LM Studio (optionnel)")
    parser.add_argument("--comment-interval", type=float, default=1.6, help="Intervalle entre 2 appels LLM (s)")
    parser.add_argument("--llm-timeout", type=float, default=4.0, help="Timeout requete LLM (s)")
    parser.add_argument("--save", default="", help="Chemin video de sortie (optionnel)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video: {args.video}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    worker = CommentaryWorker(
        base_url=args.base_url,
        model_id=args.llm_model,
        interval_sec=args.comment_interval,
        timeout_sec=args.llm_timeout,
        api_key=args.api_key or None,
    )
    worker.start()

    t_prev = time.time()
    fps_smooth = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = predict_with_legacy_nms_if_needed(
                model,
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
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
                "frame": frame_idx,
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

            frame_out = draw_overlay(frame, detections, metrics, worker.last_text, fps_smooth)
            cv2.imshow("YOLO12 + Attack/Defense + Mistral Live Commentary", frame_out)
            if writer is not None:
                writer.write(frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            frame_idx += 1
    finally:
        worker.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
