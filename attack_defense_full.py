from ultralytics import YOLO
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv
import math

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"runs/detect/train3/weights/best.pt"  # <-- adapte si nécessaire
SOURCE_DIR = r"valid/images"  # ou test/images
OUT_DIR = r"attack_defense_results"
CONF = 0.6
IOU = 0.5

# classes (selon ton data.yaml)
BALL_CLS = 0
HOOP_CLS = 1
PLAYER_CLS = 3

# affichage: garde les cadres + texte (comme tu veux)
SHOW_LABELS = True
SHOW_CONF = True
LINE_WIDTH = 2

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

model = YOLO(MODEL_PATH)


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


def center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp01(x):
    return max(0.0, min(1.0, x))


def crop_jersey(img, xyxy):
    """Crop torse pour récupérer couleur maillot."""
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

    # torse: zone centrale, moitié haute
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
    """Couleur maillot (HSV moyenne, filtrée)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(-1, 3).astype(np.float32)
    _, S, V = pix[:, 0], pix[:, 1], pix[:, 2]
    mask = (S > 40) & (V > 40) & (V < 240)
    pix = pix[mask]
    if len(pix) < 20:
        return hsv.reshape(-1, 3).mean(axis=0)
    return pix.mean(axis=0)


def pick_best_ball(ball_centers, player_centers):
    """Choisit la balle la plus plausible (la plus proche d'un joueur)."""
    best_ball = None
    best_dist = 1e18
    for bc in ball_centers:
        for pc in player_centers:
            d = dist(bc, pc)
            if d < best_dist:
                best_dist = d
                best_ball = bc
    return best_ball


def mean_pairwise_distance(points):
    if len(points) < 2:
        return None
    dsum, cnt = 0.0, 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dsum += dist(points[i], points[j])
            cnt += 1
    return dsum / cnt


# -------------------------
# CSV OUTPUT
# -------------------------
csv_path = os.path.join(OUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "image",
            "n_players",
            "n_team0",
            "n_team1",
            "attack_team",
            "attack_score",
            "defense_score",
            "spacing_px",
            "carrier_nearest_def_px",
            "closest_att_to_hoop_px",
            "defenders_close_to_carrier",
        ]
    )

    # -------------------------
    # PROCESS IMAGES
    # -------------------------
    for name in sorted(os.listdir(SOURCE_DIR)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(SOURCE_DIR, name)
        img = cv2.imread(path)
        if img is None:
            continue

        # 1) Predict (only needed classes)
        r = predict_with_legacy_nms_if_needed(
            model,
            path,
            conf=CONF,
            iou=IOU,
            classes=[BALL_CLS, HOOP_CLS, PLAYER_CLS],
            verbose=False,
        )[0]

        # 2) Extract objects
        players_xyxy = []
        player_confs = []
        balls_xyxy = []
        hoops_xyxy = []

        for b in r.boxes:
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

        # 3) Team assignment by jersey color (KMeans k=2)
        jersey_feats = []
        valid_idxs = []
        for i, p in enumerate(players_xyxy):
            crop = crop_jersey(img, p)
            if crop is None:
                continue
            feat = jersey_feat_hsv(crop)  # (H,S,V)
            jersey_feats.append(feat)
            valid_idxs.append(i)

        team = [-1] * len(players_xyxy)
        if len(jersey_feats) >= 2:
            X = np.array(jersey_feats, dtype=np.float32)
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            labs = km.fit_predict(X)
            for j, idx in enumerate(valid_idxs):
                team[idx] = int(labs[j])
        else:
            team = [0] * len(players_xyxy)

        # 4) Find ball carrier (closest player to best ball)
        carrier_idx = None
        best_ball = None
        if len(ball_centers) > 0 and len(player_centers) > 0:
            best_ball = pick_best_ball(ball_centers, player_centers)
            # closest player to best_ball
            best_d = 1e18
            for i, pc in enumerate(player_centers):
                d = dist(pc, best_ball)
                if d < best_d:
                    best_d = d
                    carrier_idx = i

        # 5) Determine attack team
        attack_team = -1

        # Case 1: ball detected -> carrier -> attack team
        if carrier_idx is not None and team[carrier_idx] != -1:
            attack_team = team[carrier_idx]
        else:
            # Case 2: fallback using HOOP (if detected)
            if hoop_center is not None and len(player_centers) > 0:
                hoop_radius = 260  # px
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
                    # Case 3: fallback using "advanced side" (x-axis)
                    xs0 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 0]
                    xs1 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 1]
                    if len(xs0) > 0 and len(xs1) > 0:
                        attack_team = 0 if (sum(xs0) / len(xs0)) > (sum(xs1) / len(xs1)) else 1
                    else:
                        attack_team = 0
            else:
                # No hoop -> fallback x-axis
                xs0 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 0]
                xs1 = [player_centers[i][0] for i in range(len(player_centers)) if team[i] == 1]
                if len(xs0) > 0 and len(xs1) > 0:
                    attack_team = 0 if (sum(xs0) / len(xs0)) > (sum(xs1) / len(xs1)) else 1
                else:
                    attack_team = 0

        attackers = [player_centers[i] for i in range(len(player_centers)) if team[i] == attack_team]
        defenders = [player_centers[i] for i in range(len(player_centers)) if team[i] != -1 and team[i] != attack_team]

        # -------------------------
        # METRICS
        # -------------------------
        spacing_px = mean_pairwise_distance(attackers)

        carrier_nearest_def_px = None
        defenders_close_to_carrier = None
        closest_att_to_hoop_px = None

        if carrier_idx is not None and len(defenders) > 0 and attack_team != -1:
            carrier = player_centers[carrier_idx]
            carrier_nearest_def_px = min(dist(carrier, d) for d in defenders)

            radius = 120  # px
            defenders_close_to_carrier = sum(1 for d in defenders if dist(d, carrier) < radius)

        if hoop_center is not None and len(attackers) > 0:
            closest_att_to_hoop_px = min(dist(hoop_center, a) for a in attackers)

        # -------------------------
        # SCORES
        # -------------------------
        spacing_score = 0.5
        if spacing_px is not None:
            spacing_score = clamp01((spacing_px - 80) / (220 - 80))

        pressure_score = 0.5
        if carrier_nearest_def_px is not None:
            pressure_score = clamp01((carrier_nearest_def_px - 60) / (200 - 60))

        threat_score = 0.5
        if closest_att_to_hoop_px is not None:
            threat_score = 1.0 - clamp01((closest_att_to_hoop_px - 80) / (350 - 80))

        pressure_def_score = 1.0 - pressure_score

        hoop_def_score = 0.5
        if hoop_center is not None and len(defenders) > 0:
            hoop_radius2 = 180
            n_close = sum(1 for d in defenders if dist(d, hoop_center) < hoop_radius2)
            hoop_def_score = clamp01(n_close / 4.0)

        attack_score = round(100 * (0.40 * spacing_score + 0.35 * pressure_score + 0.25 * threat_score), 1)
        defense_score = round(100 * (0.60 * pressure_def_score + 0.40 * hoop_def_score), 1)

        # -------------------------
        # DRAWING
        # -------------------------
        out = img.copy()

        team0_color = (255, 0, 0)    # bleu
        team1_color = (0, 0, 255)    # rouge
        unknown_color = (0, 255, 255)

        # Si moins de 2 joueurs détectés, on désactive ATT/DEF
        enable_roles = len(players_xyxy) >= 2
        for i, xyxy in enumerate(players_xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            t = team[i]

            # couleur équipe (bleu/rouge)
            base_color = team0_color if t == 0 else team1_color if t == 1 else unknown_color

            # rôle ATT/DEF selon attack_team
            role = ""
            thickness = LINE_WIDTH
            if enable_roles and attack_team != -1 and t != -1:
                if t == attack_team:
                    role = "ATT"
                    thickness = LINE_WIDTH + 1   # un peu plus épais pour l'attaque
                else:
                    role = "DEF"
                    thickness = LINE_WIDTH       # normal

            cv2.rectangle(out, (x1, y1), (x2, y2), base_color, thickness)

            if SHOW_LABELS:
                lab = f"{role} T{t}" if (t != -1 and role != "") else f"T{t}"
                if SHOW_CONF:
                    lab += f" {player_confs[i]:.2f}"
                cv2.putText(out, lab, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, base_color, 2)

        for bxyxy in balls_xyxy:
            x1, y1, x2, y2 = map(int, bxyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), LINE_WIDTH)
            if SHOW_LABELS:
                cv2.putText(out, "Ball", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for hxyxy in hoops_xyxy:
            x1, y1, x2, y2 = map(int, hxyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), LINE_WIDTH)
            if SHOW_LABELS:
                cv2.putText(out, "Hoop", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        header = f"attack_team={attack_team}  attack={attack_score}  defense={defense_score}"
        cv2.rectangle(out, (5, 5), (min(900, out.shape[1] - 5), 40), (0, 0, 0), -1)
        cv2.putText(out, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(OUT_DIR, "images", name), out)

        n0 = sum(1 for t in team if t == 0)
        n1 = sum(1 for t in team if t == 1)

        w.writerow([
            name,
            len(players_xyxy),
            n0,
            n1,
            attack_team,
            attack_score,
            defense_score,
            spacing_px,
            carrier_nearest_def_px,
            closest_att_to_hoop_px,
            defenders_close_to_carrier,
        ])


print("✅ Terminé.")
print("-> Images:", os.path.join(OUT_DIR, "images"))
print("-> CSV:", os.path.join(OUT_DIR, "metrics.csv"))
