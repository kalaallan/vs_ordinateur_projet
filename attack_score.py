from ultralytics import YOLO
import os
import cv2
import math
import numpy as np

MODEL_PATH = r"runs/detect/train3/weights/best.pt"
SOURCE_DIR = r"valid/images"
OUT_DIR = r"attack_scored"
os.makedirs(OUT_DIR, exist_ok=True)

# classes
BALL, HOOP, PLAYER = 0, 1, 3

model = YOLO(MODEL_PATH)


def center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp01(x):
    return max(0.0, min(1.0, x))


for name in sorted(os.listdir(SOURCE_DIR)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(SOURCE_DIR, name)
    img = cv2.imread(path)
    if img is None:
        continue

    r = model.predict(
        path, conf=0.5, iou=0.5, classes=[BALL, HOOP, PLAYER], verbose=False
    )[0]

    players = []
    balls = []
    hoops = []
    for b in r.boxes:
        cls = int(b.cls[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        if cls == PLAYER:
            players.append(xyxy)
        elif cls == BALL:
            balls.append(xyxy)
        elif cls == HOOP:
            hoops.append(xyxy)

    # --- Si pas assez de joueurs : skip
    if len(players) < 2:
        continue

    # --- Estimer porteur: joueur le plus proche du ballon (si balle détectée)
    carrier_idx = None
    if len(balls) > 0:
        bc = center(balls[0])  # si plusieurs balles, on prend la 1ère (simple)
        best = 1e18
        for i, p in enumerate(players):
            d = dist(center(p), bc)
            if d < best:
                best = d
                carrier_idx = i

    # --- Approx teams (TRÈS simple fallback): split par x (gauche/droite)
    # Si tu as déjà team assignment couleur, on remplacera ça par tes teams.
    pcs = [center(p) for p in players]
    xs = [p[0] for p in pcs]
    median_x = float(np.median(xs))
    team = [0 if p[0] < median_x else 1 for p in pcs]  # 2 "camps" approx

    # --- Si porteur connu => équipe attaque = équipe du porteur
    attack_team = None
    if carrier_idx is not None:
        attack_team = team[carrier_idx]
    else:
        # fallback : équipe avec le plus de joueurs "avancés" vers le hoop
        # si hoop détecté
        # sinon: on ne peut pas conclure => on met team 0 par défaut
        attack_team = 0

    attackers = [pcs[i] for i in range(len(players)) if team[i] == attack_team]
    defenders = [pcs[i] for i in range(len(players)) if team[i] != attack_team]

    # 1) Spacing: distance moyenne entre attaquants (normalisée)
    spacing = 0.0
    if len(attackers) >= 2:
        dsum, cnt = 0.0, 0
        for i in range(len(attackers)):
            for j in range(i + 1, len(attackers)):
                dsum += dist(attackers[i], attackers[j])
                cnt += 1
        spacing = dsum / cnt

    # normalisation grossière (à l’échelle image)
    # bonne valeur typique ~ 120-250px selon résolution
    spacing_score = clamp01((spacing - 80) / (220 - 80))  # 0..1

    # 2) Pression sur porteur : distance du défenseur le plus proche
    pressure_score = 0.5
    if carrier_idx is not None and len(defenders) > 0:
        carrier = pcs[carrier_idx]
        nearest_def = min(dist(carrier, d) for d in defenders)
        # plus c'est loin, mieux c'est
        pressure_score = clamp01((nearest_def - 60) / (200 - 60))

    # 3) Menace panier : attaquant le plus proche du hoop
    threat_score = 0.5
    if len(hoops) > 0:
        hc = center(hoops[0])
        nearest_att = min(dist(hc, a) for a in attackers) if attackers else 1e9
        # plus c'est proche, mieux c'est (inversé)
        threat_score = 1.0 - clamp01((nearest_att - 80) / (350 - 80))

    # Score final attaque (0-100)
    attack_score = round(
        100 * (0.40 * spacing_score + 0.35 * pressure_score + 0.25 * threat_score), 1
    )

    out = img.copy()
    cv2.rectangle(out, (5, 5), (520, 40), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"Attack score: {attack_score}/100",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(os.path.join(OUT_DIR, name), out)

print("✅ Terminé. Images avec score dans:", OUT_DIR)
