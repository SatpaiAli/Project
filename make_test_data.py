"""
make_test_data.py — генератор синтетических тестовых данных с ТОЧНОЙ разметкой.

Зачем: чтобы проверять «линейку» количественно, нужно знать истинные размеры.
Мы моделируем настоящую камеру (pinhole + дисторсия), стол с листом бумаги,
шахматную доску и предметы с известными размерами, а затем «фотографируем» сцену
с разных ракурсов и в разных условиях (свет, шум, размытие, блики, тени, перекрытие).

Генерируются:
  data/calib/calib_XX.jpg      — снимки доски для калибровки
  data/scenes/<сценарий>_K.jpg — сцены для измерения + <...>.json с разметкой
  data/camera_true.json        — истинные параметры камеры (для оценки калибровки)

Запуск:  python make_test_data.py
"""
import json
import os

import cv2
import numpy as np

from ruler_core import UNDIST_CRITERIA, find_board_corners

# ---------------- «истинная» камера ----------------
IMG_W, IMG_H = 1280, 960
TRUE_K = np.array([[1005.0, 0, 645.0], [0, 998.0, 478.0], [0, 0, 1]])
TRUE_DIST = np.array([-0.28, 0.12, 0.0012, -0.0006, 0.0])   # k1 k2 p1 p2 k3 (заметная бочкообразная дисторсия)

PATTERN = (9, 6)       # внутренних углов (cols, rows)
SQUARE = 25.0          # мм
TEX_PPM = 3.0          # разрешение текстуры плоскости, px/мм
TEX_X0, TEX_Y0, TEX_X1, TEX_Y1 = -300, -300, 900, 600     # размеры стола, мм
PAPER = (-40, -40, 580, 290)                               # лист бумаги, мм

# Предметы: (вид, центр, размеры, угол, цвет BGR)
OBJECTS = [
    ("rect",   (310, 50),  (80, 50),  20,  (110, 50, 30)),
    ("rect",   (440, 60),  (60, 60),  35,  (30, 30, 160)),
    ("rect",   (330, 140), (120, 30), -15, (40, 110, 40)),
    ("rect",   (500, 150), (45, 95),  10,  (100, 60, 100)),
    ("circle", (300, 225), (40, 40),  0,   (30, 30, 160)),
    ("circle", (420, 220), (70, 70),  0,   (110, 50, 30)),
]

SCENARIOS = {
    "frontal":   dict(tilt=(0, 5)),
    "tilt30":    dict(tilt=(25, 35)),
    "tilt50":    dict(tilt=(45, 55)),
    "dark":      dict(tilt=(10, 30), gain=0.3, noise=6),
    "glare":     dict(tilt=(10, 30), gain=1.25, glare=True),
    "noise":     dict(tilt=(10, 30), noise=20),
    "blur":      dict(tilt=(10, 30), blur=2.5),
    "clutter":   dict(tilt=(10, 30), clutter=True, shadow=True),
    "occlusion": dict(tilt=(10, 30), occlusion=True),
}
N_PER_SCENARIO = 6


# ---------------- текстура плоскости ----------------
def _tex_xy(x, y):
    """мм -> координаты рисования в текстуре (центр пикселя = целое число)."""
    return ((x - TEX_X0) * TEX_PPM - 0.5, (y - TEX_Y0) * TEX_PPM - 0.5)


def _fill_poly(tex, pts_mm, color):
    pts = np.array([_tex_xy(x, y) for x, y in pts_mm]) * 16
    cv2.fillPoly(tex, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA, shift=4)


def _fill_circle(tex, c_mm, r_mm, color):
    cx, cy = _tex_xy(*c_mm)
    cv2.circle(tex, (int(round(cx * 16)), int(round(cy * 16))), int(round(r_mm * TEX_PPM * 16)),
               color, -1, cv2.LINE_AA, shift=4)


def _rot_rect(center, size, ang):
    return cv2.boxPoints((center, size, ang))


def make_texture(rng, objects=True, clutter=False, occlusion=False):
    """Стол (дерево) + лист бумаги + шахматная доска + предметы."""
    H, W = int((TEX_Y1 - TEX_Y0) * TEX_PPM), int((TEX_X1 - TEX_X0) * TEX_PPM)
    low = cv2.resize(rng.normal(0, 1, (12, 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (0, 0), 1.5)
    grain = cv2.resize(grain, (W, H), interpolation=cv2.INTER_LINEAR)
    base = np.array([60, 85, 120], np.float32)
    tex = base[None, None, :] + (low * 10 + grain * 25)[..., None]
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    _fill_poly(tex, [(PAPER[0], PAPER[1]), (PAPER[2], PAPER[1]), (PAPER[2], PAPER[3]), (PAPER[0], PAPER[3])],
               (238, 238, 235))

    # шахматная доска: 10x7 клеток, внутренний угол (0,0) совпадает с началом координат
    cols, rows = PATTERN
    for i in range(cols + 1):
        for j in range(rows + 1):
            if (i + j) % 2 == 0:
                x, y = (i - 1) * SQUARE, (j - 1) * SQUARE
                _fill_poly(tex, [(x, y), (x + SQUARE, y), (x + SQUARE, y + SQUARE), (x, y + SQUARE)], (25, 25, 25))
    gt = []
    if objects:
        for kind, c, size, ang, col in OBJECTS:
            if kind == "rect":
                _fill_poly(tex, _rot_rect(c, size, ang), col)
            else:
                _fill_circle(tex, c, size[0] / 2, col)
            gt.append(dict(kind=kind, center_mm=list(c), size_mm=list(size), angle_deg=ang))
    if clutter:   # мелкие винтики (отфильтруются) и один «ластик» (ложная цель)
        for c in [(270, 110), (380, 100), (470, 250)]:
            _fill_circle(tex, c, 4, (60, 60, 60))
        _fill_poly(tex, _rot_rect((540, 245), (25, 15), 30), (90, 90, 160))
    if occlusion:  # «рука» закрывает часть предметов + карандаш лежит вплотную к предмету
        _fill_poly(tex, cv2.ellipse2Poly((515, 190), (38, 28), 40, 0, 360, 10).astype(float), (150, 170, 215))
        _fill_poly(tex, cv2.ellipse2Poly((395, 220), (22, 30), 10, 0, 360, 10).astype(float), (150, 170, 215))
        _fill_poly(tex, _rot_rect((330, 168), (140, 4), -15), (50, 50, 50))
    return tex, gt


# ---------------- рендер вида ----------------
_NORM_CACHE = {}


def _normalized_grid():
    """Для каждого пикселя изображения — луч в нормированных координатах (с обратной дисторсией). Считается один раз."""
    if "n" not in _NORM_CACHE:
        uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
        pts = np.stack([uu, vv], -1).reshape(-1, 1, 2).astype(np.float64)
        _NORM_CACHE["n"] = cv2.undistortPointsIter(pts, TRUE_K, TRUE_DIST, None, None, UNDIST_CRITERIA).reshape(-1, 2)
    return _NORM_CACHE["n"]


def look_at(p0, dist_mm, theta_deg, phi_deg):
    """Поза камеры (R, t), смотрящей на точку p0 плоскости Z=0.
    theta — наклон оптической оси от нормали, phi — азимут положения камеры.
    Камера «стоит вертикально»: низ кадра направлен в сторону камеры."""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    p0 = np.asarray(p0, float)
    C = p0 + dist_mm * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)])
    z = (p0 - C) / np.linalg.norm(p0 - C)
    hint = np.array([np.cos(ph), np.sin(ph), 0.0])
    y = hint - hint.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    return R, -R @ C


def render_view(tex, R, t):
    n = _normalized_grid()
    C = -R.T @ t
    d = np.c_[n, np.ones(len(n))] @ R
    s = -C[2] / d[:, 2]
    P = C + s[:, None] * d
    mx = ((P[:, 0] - TEX_X0) * TEX_PPM - 0.5).reshape(IMG_H, IMG_W).astype(np.float32)
    my = ((P[:, 1] - TEX_Y0) * TEX_PPM - 0.5).reshape(IMG_H, IMG_W).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def project(R, t, pts_mm):
    rvec, _ = cv2.Rodrigues(R)
    pts = np.c_[np.asarray(pts_mm, float).reshape(-1, 2), np.zeros(len(pts_mm))]
    px, _ = cv2.projectPoints(pts, rvec, t, TRUE_K, TRUE_DIST)
    return px.reshape(-1, 2)


def degrade(img, rng, R, t, blur=0.7, gain=1.0, noise=3.0, glare=False, shadow=False):
    """Оптика, экспозиция, блик, тень, шум датчика."""
    out = cv2.GaussianBlur(img, (0, 0), blur).astype(np.float32)
    out *= gain
    if glare:   # засвет поверх предметов
        c = project(R, t, [(400, 100)])[0]
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        g = 230 * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * 55.0 ** 2))
        out += g[..., None]
    if shadow:  # диагональный градиент освещённости
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        out *= (1 - 0.45 * (xx + yy) / (IMG_W + IMG_H))[..., None]
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_jpg(path, img):
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ---------------- наборы ----------------
def make_calibration_set(out_dir, n, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols, rows = PATTERN
    center = np.array([(cols - 1) * SQUARE / 2, (rows - 1) * SQUARE / 2])
    k, tries = 0, 0
    while k < n and tries < 400:
        tries += 1
        tex, _ = make_texture(rng, objects=False)
        p0 = np.r_[center + rng.uniform([-130, -100], [130, 100]), 0]
        R, t = look_at(p0, rng.uniform(380, 700), rng.uniform(0, 55), rng.uniform(0, 360))
        img = degrade(render_view(tex, R, t), rng, R, t, noise=3.0)
        if find_board_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), PATTERN) is not None:
            save_jpg(os.path.join(out_dir, f"calib_{k:02d}.jpg"), img)
            k += 1
    print(f"calib: {k} снимков -> {out_dir}")


def make_scene_set(out_dir, seed=1):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name, p in SCENARIOS.items():
        for k in range(N_PER_SCENARIO):
            tex, objs = make_texture(rng, True, p.get("clutter", False), p.get("occlusion", False))
            theta = rng.uniform(*p["tilt"])
            phi = 90 + rng.uniform(-40, 40)
            R, t = look_at(np.r_[(250, 120) + rng.uniform(-30, 30, 2), 0], rng.uniform(650, 850), theta, phi)
            img = degrade(render_view(tex, R, t), rng, R, t, blur=p.get("blur", 0.7), gain=p.get("gain", 1.0),
                          noise=p.get("noise", 3.0), glare=p.get("glare", False), shadow=p.get("shadow", False))
            # пары точек для «линейки»: случайные точки на бумаге, обе видны в кадре
            pairs = []
            while len(pairs) < 8:
                a = rng.uniform([-20, -20], [560, 270]); b = rng.uniform([-20, -20], [560, 270])
                L = float(np.linalg.norm(a - b))
                pa, pb = project(R, t, [a, b])
                ok = all(30 < q[0] < IMG_W - 30 and 30 < q[1] < IMG_H - 30 for q in (pa, pb))
                if 40 <= L <= 300 and ok:
                    pairs.append(dict(p1_px=pa.tolist(), p2_px=pb.tolist(), dist_mm=L))
            meta = dict(scenario=name, theta=float(theta), phi=float(phi), objects=objs, pairs=pairs,
                        roi_mm=[235, -20, 575, 285], pattern=list(PATTERN), square=SQUARE,
                        center_px=[project(R, t, [o["center_mm"]])[0].tolist() for o in objs])
            base = os.path.join(out_dir, f"{name}_{k}")
            save_jpg(base + ".jpg", img)
            json.dump(meta, open(base + ".json", "w"), indent=1)
        print(f"scene: {name} x{N_PER_SCENARIO}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    json.dump(dict(K=TRUE_K.tolist(), dist=TRUE_DIST.tolist(), size=[IMG_W, IMG_H], pattern=list(PATTERN),
                   square=SQUARE), open("data/camera_true.json", "w"), indent=1)
    make_calibration_set("data/calib", 20)
    make_scene_set("data/scenes")
