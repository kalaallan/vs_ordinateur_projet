# Projet Vision Ordinateur - Analyse Basket (YOLO11, YOLO12, YOLO26)

## 1. Objet du projet
Ce projet entraine et compare plusieurs modeles YOLO sur une dataset basket, puis produit des analyses terrain (attaque/defense) a partir des detections:
- detection des objets du match (joueurs, ballon, panier, elements scoreboard)
- comparaison de modeles (precision, robustesse, vitesse)
- extraction de metriques de jeu (spacing, pression defensive, score attaque/defense)

## 2. Dataset
Fichier de configuration: `data.yaml`

- train: `train/images`
- val: `valid/images`
- test: `test/images`
- classes (`nc=9`):
  - `Ball`
  - `Hoop`
  - `Period`
  - `Player`
  - `Ref`
  - `Shot Clock`
  - `Team Name`
  - `Team Points`
  - `Time Remaining`

Volumes actuels:
- train: 1140 images
- valid: 32 images
- test: 24 images

## 3. Arborescence utile

- scripts entrainement:
  - `train_yolo.py`
  - `train_yolo12.py`
  - `train_yolo26.py`
- scripts inference/visualisation:
  - `clean_view.py`
  - `comparaisonyolo.py`
  - `sam2_boxes.py`
- scripts evaluation/benchmark:
  - `benchmark_models.py`
  - `robust_eval.py`
- scripts analyses metier basket:
  - `attack_score.py`
  - `team_attack_defense.py`
  - `attack_defense_full.py`
- sorties:
  - `runs/detect/...` (entrainement, validation)
  - `clean_preds/`
  - `visual_compare/`
  - `attack_scored/`
  - `team_results/`
  - `attack_defense_results/`

## 4. Environnement et installation

Depuis la racine du projet:

```bash
python3 -m venv vision
source vision/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy scikit-learn torch
```

Remarque: un `requirements.txt` existe, mais il correspond a un environnement large (non limite au projet YOLO).

## 5. Entrainement des modeles

### 5.1 YOLO11
Script: `train_yolo.py`

```bash
python3 train_yolo.py
```

Parametres principaux:
- base model: `yolo11n.pt`
- epochs: 50
- imgsz: 640
- batch: 16
- device: 0

Sortie:
- `runs/detect/train/weights/best.pt`

### 5.2 YOLO12
Script: `train_yolo12.py`

```bash
python3 train_yolo12.py
```

Parametres principaux:
- base model: `yolo12n.pt`
- epochs: 50
- imgsz: 640
- batch: 16
- device: 0
- run name: `yolo12`

Sortie:
- `runs/detect/yolo12/weights/best.pt`

### 5.3 YOLO26
Script: `train_yolo26.py`

```bash
python3 train_yolo26.py
```

Parametres principaux:
- base model: `yolo26n.pt`
- epochs: 10
- imgsz: 640
- batch: 5
- device: auto (`cuda` -> `mps` -> `cpu`)
- run name: `yolo26`

Sortie:
- `runs/detect/yolo26/weights/best.pt`

## 6. Evaluation des modeles

### 6.1 Validation standard
Executer sur le meme split et les memes hyperparametres:

```bash
yolo val model="runs/detect/train3/weights/best.pt" data="data.yaml" split=val imgsz=640
yolo val model="runs/detect/yolo12/weights/best.pt" data="data.yaml" split=val imgsz=640
yolo val model="runs/detect/yolo26/weights/best.pt" data="data.yaml" split=val imgsz=640
```

### 6.2 Comparaison stricte sur test

```bash
yolo val model="runs/detect/train3/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo11
yolo val model="runs/detect/yolo12/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo12
yolo val model="runs/detect/yolo26/weights/best.pt" data="data.yaml" split=test imgsz=640 batch=16 conf=0.001 iou=0.7 name=cmp_yolo26
```

Metriques de decision:
- `mAP50-95` (metrique principale)
- `mAP50`
- `precision`
- `recall`
- detail par classe (`Ball`, `Player`, `Hoop` en priorite)

### 6.3 Resultats deja observes sur les runs d'entrainement

- YOLO11 (`runs/detect/train3`):
  - best epoch: 33
  - precision: 0.83323
  - recall: 0.81859
  - mAP50: 0.82592
  - mAP50-95: 0.55321
- YOLO12 (`runs/detect/yolo12`):
  - best epoch: 36
  - precision: 0.84572
  - recall: 0.82390
  - mAP50: 0.84159
  - mAP50-95: 0.56565
- YOLO26 (`runs/detect/yolo26`):
  - best epoch: 8
  - precision: 0.67480
  - recall: 0.65431
  - mAP50: 0.69236
  - mAP50-95: 0.43048

## 7. Scripts d'inference et d'analyse

### 7.1 `clean_view.py`
But:
- produire des rendus visuels epures des detections (`Ball`, `Player`) sans labels ni score.

Entree:
- `valid/images`

Sortie:
- `clean_preds/`

Execution:
```bash
python3 clean_view.py
```

### 7.2 `comparaisonyolo.py`
But:
- comparaison visuelle YOLO11 vs YOLO12 sur 10 images aleatoires.

Entree:
- `valid/images`

Sortie:
- `visual_compare/` avec images prefixees `y11_` et `y12_`

Execution:
```bash
python3 comparaisonyolo.py
```

### 7.3 `benchmark_models.py`
But:
- benchmark vitesse inference (FPS, ms/img, memoire GPU si CUDA).

Modeles compares:
- YOLO11 (`runs/detect/train3/weights/best.pt`)
- YOLO12 (`runs/detect/yolo12/weights/best.pt`)

Execution:
```bash
python3 benchmark_models.py
```

### 7.4 `robust_eval.py`
But:
- test de robustesse sur images degradees (`blur`, `dark`, `noise`, `zoom`).
- compte les detections `Ball` et `Player` sur 20 images.

Sortie intermediaire:
- `robust_tmp/tmp.jpg` (fichier temporaire)

Execution:
```bash
python3 robust_eval.py
```

### 7.5 `attack_score.py`
But:
- calcul d'un score d'attaque par image (`Attack score / 100`) a partir de:
  - spacing offensif
  - pression defensive sur porteur
  - proximite panier

Entree:
- `valid/images`

Sortie:
- `attack_scored/`

Execution:
```bash
python3 attack_score.py
```

### 7.6 `team_attack_defense.py`
But:
- separation des equipes par couleur de maillot (KMeans HSV)
- estimation equipe en attaque via ballon
- calcul de:
  - `def_pressure`
  - `off_spacing`

Entree:
- `valid/images` (modifiable en `test/images` dans le script)

Sorties:
- images: `team_results/images/`
- metriques tabulaires: `team_results/metrics.csv`

Execution:
```bash
python3 team_attack_defense.py
```

### 7.7 `attack_defense_full.py`
But:
- version complete de l'analyse attaque/defense:
  - attribution des equipes
  - score attaque
  - score defense
  - metriques detaillees (spacing, nearest defender, menace panier, etc.)

Entree:
- `valid/images` (modifiable en `test/images`)

Sorties:
- images: `attack_defense_results/images/`
- metriques tabulaires: `attack_defense_results/metrics.csv`

Execution:
```bash
python3 attack_defense_full.py
```

### 7.8 `sam2_boxes.py`
But:
- pipeline detection + segmentation:
  - detection des joueurs via YOLO
  - segmentation guidee par boites avec SAM2

Entree:
- `valid/images`

Sortie:
- `yolo_sam2_results/`

Execution:
```bash
python3 sam2_boxes.py
```

Pre-requis specifique:
- poids SAM2 disponible (`sam2.1_b.pt`)

### 7.9 `analyze.py`
- fichier present mais non implemente (vide).

## 8. Procedure d'execution

1. activer l'environnement
2. verifier la presence des poids de base (`yolo11n.pt`, `yolo12n.pt`, `yolo26n.pt`)
3. entrainer les modeles (`train_yolo.py`, `train_yolo12.py`, `train_yolo26.py`)
4. evaluer avec `yolo val` sur `val` puis `test`
5. lancer les scripts d'analyse metier:
   - `attack_score.py`
   - `team_attack_defense.py`
   - `attack_defense_full.py`
6. exporter les dossiers resultats (`runs/detect`, `team_results`, `attack_defense_results`, `visual_compare`)

## 9. Detection en temps reel sur video

Exemple avec la video `Ja Morant Block.mp4`:

```bash
cd "/Users/wajdi/Computer vision/vs_ordinateur_projet"
source "/Users/wajdi/Computer vision/vs_ordinateur_projet/vision/bin/activate"

yolo predict model="/Users/wajdi/Computer vision/vs_ordinateur_projet/runs/detect/yolo12/weights/best.pt" source="/Users/wajdi/Computer vision/vs_ordinateur_projet/Ja Morant Block.mp4" show=True conf=0.4 iou=0.5 imgsz=640
```

Modeles disponibles:
- YOLO11: `runs/detect/train3/weights/best.pt`
- YOLO12: `runs/detect/yolo12/weights/best.pt`
- YOLO26: `runs/detect/yolo26/weights/best.pt`

Detection live webcam:

```bash
yolo predict model="/Users/wajdi/Computer vision/vs_ordinateur_projet/runs/detect/yolo12/weights/best.pt" source=0 show=True conf=0.4 iou=0.5 imgsz=640
```

Si la commande `yolo` n'est pas reconnue:

```bash
python -m ultralytics yolo predict model="/Users/wajdi/Computer vision/vs_ordinateur_projet/runs/detect/yolo12/weights/best.pt" source="/Users/wajdi/Computer vision/vs_ordinateur_projet/Ja Morant Block.mp4" show=True conf=0.4 iou=0.5 imgsz=640
```
