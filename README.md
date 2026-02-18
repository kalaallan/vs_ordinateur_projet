# Basketball Detection - Training Guide (YOLO11, YOLO12, YOLO26)

Ce README explique comment entrainer et executer les 3 modeles dans ce projet:
- `YOLO11` (base: `yolo11n.pt`)
- `YOLO12` (base: `yolo12n.pt`)
- `YOLO26` (base: `yolo26n.pt`)

## 1. Prerequis

Depuis la racine du projet:

```bash
python3 -m venv vision
source vision/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python scikit-learn torch
```

## 2. Dataset utilise

Configuration dataset: `data.yaml`

- train: `train/images`
- val: `valid/images`
- test: `test/images`
- classes (9): `Ball, Hoop, Period, Player, Ref, Shot Clock, Team Name, Team Points, Time Remaining`

## 3. Entrainement des 3 modeles

### A. YOLO11

Script: `train_yolo.py`

```bash
python3 train_yolo.py
```

Parametres du script:
- model: `yolo11n.pt`
- epochs: `50`
- imgsz: `640`
- batch: `16`
- device: `0`

Sortie principale:
- `runs/detect/train/weights/best.pt`


### B. YOLO12

Script: `train_yolo12.py`

```bash
python3 train_yolo12.py
```

Parametres du script:
- model: `yolo12n.pt`
- epochs: `50`
- imgsz: `640`
- batch: `16`
- device: `0`
- name: `yolo12`

Sortie principale:
- `runs/detect/yolo12/weights/best.pt`

### C. YOLO26

Script: `train_yolo26.py`

```bash
python3 train_yolo26.py
```

Parametres du script:
- model: `yolo26n.pt`
- epochs: `10`
- imgsz: `640`
- batch: `5`
- device: auto (`0` si CUDA, sinon `mps`, sinon `cpu`)
- name: `yolo26`

Sortie principale:
- `runs/detect/yolo26/weights/best.pt`

## 4. Comparer les 3 modeles

comparaisanr sur le meme split avec les memes parametres:

```bash
yolo val model="runs/detect/train3/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo11
yolo val model="runs/detect/yolo12/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo12
yolo val model="runs/detect/yolo26/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo26
```

Metriques a comparer:
- `mAP50-95` (metrique principale)
- `mAP50`
- `precision`
- `recall`
- performance par classe (surtout `Ball`, `Player`, `Hoop`)

## 6. Scripts d'analyse

### `attack_score.py`
- Entree: images de `valid/images`
- Detection: `Ball (0)`, `Hoop (1)`, `Player (3)`
- Sortie: score d'attaque par image (`Attack score: x/100`) en overlay
- Dossier de sortie: `attack_scored`

### `team_attack_defense.py`
- Entree: images de `valid/images` (ou `test/images` si change dans le script)
- Detection: `Ball (0)`, `Player (3)`
- Fonction:
  - separation des 2 equipes par couleur de maillot (KMeans sur HSV)
  - estimation de l'equipe en attaque via proximite balle-joueur
  - calcul de `def_pressure` et `off_spacing`
- Sorties:
  - images annotees: `team_results/images`
  - CSV metriques: `team_results/metrics.csv`

### `attack_defense_full.py`
- Entree: images de `valid/images` (ou `test/images` si change dans le script)
- Detection: `Ball (0)`, `Hoop (1)`, `Player (3)`
- Fonction:
  - separation des equipes par couleur maillot
  - attribution attaque/defense
  - calcul de scores complets `attack_score` et `defense_score`
  - calcul de metriques intermediaires (spacing, nearest defender, menace panier, etc.)
- Sorties:
  - images annotees: `attack_defense_results/images`
  - CSV metriques: `attack_defense_results/metrics.csv`

### `clean_view.py`
- Entree: images de `valid/images`
- Detection: `Ball (0)`, `Player (3)`
- Fonction: visualisation "propre" des detections (sans labels ni conf)
- Dossier de sortie: `clean_preds`
