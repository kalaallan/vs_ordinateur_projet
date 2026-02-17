from ultralytics import YOLO
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv
import math

# -------------------------
# Config
# -------------------------
MODEL_PATH = r"runs/detect/train3/weights/best.pt"  # adapte si besoin
SOURCE_DIR = r"valid/images"  # ou test/images
OUT_DIR = r"team_results"
CONF = 0.5
IOU = 0.5

# classes d'après ton data.yaml
BALL_CLS = 0
PLAYER_CLS = 3

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

model = YOLO(MODEL_PATH)


def box_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def crop_jersey(img, xyxy):
    """
    Prend une zone du 'torse' pour capter la couleur du maillot.
    On évite les jambes + parquet.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 2 or bh <= 2:
        return None

    # zone torse : au centre, plutôt dans la moitié haute
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


def dominant_color_hsv(crop_bgr):
    """
    Retourne une couleur représentative en HSV (moyenne robuste).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # flatten
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    # on enlève pixels très sombres ou très clairs (souvent bruit / parquet /
    # pub)
    H, S, V = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    mask = (S > 40) & (V > 40) & (V < 240)
    pixels = pixels[mask]
    if len(pixels) < 20:
        # fallback simple
        mean = hsv.reshape(-1, 3).mean(axis=0)
        return mean
    return pixels.mean(axis=0)


def euclidean(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


# CSV output
csv_path = os.path.join(OUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "image",
            "n_players",
            "n_team0",
            "n_team1",
            "attack_team",
            "def_pressure",
            "off_spacing",
        ]
    )

    for name in sorted(os.listdir(SOURCE_DIR)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(SOURCE_DIR, name)
        img = cv2.imread(path)
        if img is None:
            continue

        pred = model.predict(
            path, conf=CONF, iou=IOU, classes=[BALL_CLS, PLAYER_CLS],
            verbose=False
        )[0]

        players = []
        ball_centers = []

        for b in pred.boxes:
            cls = int(b.cls[0].item())
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            if cls == PLAYER_CLS:
                players.append(xyxy)
            elif cls == BALL_CLS:
                ball_centers.append(box_center_xyxy(xyxy))

        # ---- Team assignment by jersey color clustering
        jersey_feats = []
        valid_player_idxs = []

        for i, xyxy in enumerate(players):
            crop = crop_jersey(img, xyxy)
            if crop is None:
                continue
            feat = dominant_color_hsv(crop)  # (H,S,V)
            jersey_feats.append(feat)
            valid_player_idxs.append(i)

        team_labels = [-1] * len(players)

        if len(jersey_feats) >= 2:
            X = np.array(jersey_feats, dtype=np.float32)
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            labs = km.fit_predict(X)
            for j, pi in enumerate(valid_player_idxs):
                team_labels[pi] = int(labs[j])
        else:
            # pas assez d'info couleur → tout team0 (fallback)
            team_labels = [0] * len(players)

        # ---- Determine attack team (by ball proximity)
        attack_team = -1
        def_pressure = None
        off_spacing = None

        player_centers = [box_center_xyxy(p) for p in players]

        if len(ball_centers) > 0 and len(players) > 0:
            # prend la balle la plus "probable" : celle la plus proche d'un
            # joueur
            best_ball = None
            best_dist = 1e18
            best_player_idx = None
            for bc in ball_centers:
                for i, pc in enumerate(player_centers):
                    d = euclidean(bc, pc)
                    if d < best_dist:
                        best_dist = d
                        best_ball = bc
                        best_player_idx = i

            if best_player_idx is not None and team_labels[best_player_idx] != -1:
                attack_team = team_labels[best_player_idx]

                # pression défensive = nombre d'adversaires proches du porteur
                carrier_center = player_centers[best_player_idx]
                radius = 120  # pixels (à ajuster selon résolution)
                defenders_close = 0
                for i, pc in enumerate(player_centers):
                    if i == best_player_idx:
                        continue
                    if team_labels[i] == -1:
                        continue
                    if (
                        team_labels[i] != attack_team
                        and euclidean(pc, carrier_center) < radius
                    ):
                        defenders_close += 1
                def_pressure = defenders_close

                # spacing offensif = distance moyenne entre attaquants
                attackers = [
                    player_centers[i]
                    for i in range(len(players))
                    if team_labels[i] == attack_team
                ]
                if len(attackers) >= 2:
                    dsum = 0.0
                    cnt = 0
                    for i in range(len(attackers)):
                        for j in range(i + 1, len(attackers)):
                            dsum += euclidean(attackers[i], attackers[j])
                            cnt += 1
                    off_spacing = dsum / cnt
        else:
            # si pas de balle → on garde attack_team=-1
            pass

        # ---- Draw clean visualization
        out = img.copy()

        # couleurs BGR
        team0_color = (255, 0, 0)  # bleu
        team1_color = (0, 0, 255)  # rouge
        unknown_color = (0, 255, 255)

        for i, xyxy in enumerate(players):
            x1, y1, x2, y2 = map(int, xyxy)
            t = team_labels[i]
            if t == 0:
                c = team0_color
            elif t == 1:
                c = team1_color
            else:
                c = unknown_color
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

        # dessiner la balle si présente
        for bc in ball_centers:
            cv2.circle(out, (int(bc[0]), int(bc[1])), 6, (0, 255, 0), -1)

        # texte en haut
        txt = f"attack_team={attack_team}  def_pressure={def_pressure}  off_spacing={None if off_spacing is None else round(off_spacing,1)}"
        cv2.rectangle(
            out, (5, 5), (5 + min(1200, out.shape[1] - 10), 35), (0, 0, 0), -1
        )
        cv2.putText(
            out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
            2
        )

        cv2.imwrite(os.path.join(OUT_DIR, "images", name), out)

        n0 = sum(1 for t in team_labels if t == 0)
        n1 = sum(1 for t in team_labels if t == 1)

        writer.writerow(
            [name, len(players), n0, n1, attack_team, def_pressure,
             off_spacing]
        )

print("✅ Terminé. Images:", os.path.join(OUT_DIR, "images"))
print("✅ CSV:", csv_path)
