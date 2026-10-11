"""
visualize.py — рисунки для отчёта и README (results/fig_*.png, results/demo_*.jpg).

  fig_pipeline.png      — конвейер на одном кадре: фото -> углы доски -> вид сверху -> маска -> результат
  fig_undistort.png     — калибровка: «до/после» снятия дисторсии (прямизна ряда углов доски)
  fig_scenarios.png     — мозаика всех сценариев (свет, шум, блики, тени, перекрытие) с измерениями
  fig_error_scenarios.png — ошибка линейки по сценариям для трёх методов
  fig_error_tilt.png    — ошибка линейки в зависимости от наклона камеры
  fig_calibration.png   — точность калибровки vs число снимков

Запуск (после evaluate.py):  python visualize.py
"""
import json
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ruler_core import Calibration, PlaneRuler, draw_measurement, draw_objects

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"naive": "#d1495b", "homography": "#edae49", "calibrated": "#00798c"}
LABELS = {"naive": "масштаб по клетке", "homography": "гомография (без дисторсии)", "calibrated": "калибровка + поза (наш метод)"}


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load(name, calib):
    img = cv2.imread(f"data/scenes/{name}.jpg")
    meta = json.load(open(f"data/scenes/{name}.json"))
    r = PlaneRuler(calib)
    assert r.fit(img), name
    return img, meta, r


def fig_pipeline(calib):
    img, meta, r = load("tilt30_0", calib)
    objs, dbg = r.measure_objects(img, meta["roi_mm"], return_debug=True)
    corners_img = img.copy()
    cv2.drawChessboardCorners(corners_img, calib.pattern, r.corners.astype(np.float32), True)
    final = img.copy()
    draw_objects(final, r, objs)
    for p in meta["pairs"][:2]:
        draw_measurement(final, r, p["p1_px"], p["p2_px"])
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("1. Исходное фото (наклон %.0f°, дисторсия)" % meta["theta"], rgb(img)),
              ("2. Углы шахматной доски (SB + subpixel)", rgb(corners_img)),
              ("3. Вид сверху: перспектива и дисторсия сняты (remap)", rgb(dbg["rect"])),
              ("4. Выравнивание освещения + Otsu", dbg["flat"]),
              ("5. Маска после морфологии (open + close)", dbg["mask"]),
              ("6. Результат: размеры в мм + линейка", rgb(final))]
    for a, (t, im) in zip(ax.ravel(), panels):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(t, fontsize=10); a.axis("off")
    fig.tight_layout(); fig.savefig("results/fig_pipeline.png", dpi=110); plt.close(fig)
    cv2.imwrite("results/demo_after.jpg", final)
    cv2.imwrite("results/demo_before.jpg", img)


def row_straightness(corners, pattern):
    """Макс. отклонение углов первой строки доски от прямой (px)."""
    row = corners.reshape(pattern[1], pattern[0], 2)[0]
    vx, vy, x0, y0 = cv2.fitLine(row.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return float(np.max(np.abs((row[:, 0] - x0) * vy - (row[:, 1] - y0) * vx)))


def fig_undistort(calib):
    from ruler_core import find_board_corners
    # снимок, где доска ближе всего к краю кадра (дисторсия там максимальна)
    best, best_dev = None, -1
    for k in range(20):
        im = cv2.imread(f"data/calib/calib_{k:02d}.jpg")
        c = find_board_corners(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), calib.pattern)
        if c is None:
            continue
        dev = row_straightness(c, calib.pattern)
        if dev > best_dev:
            best, best_dev, best_c = im, dev, c
    und = cv2.undistort(best, calib.K, calib.dist)
    c_und = cv2.undistortPoints(best_c.astype(np.float32), calib.K, calib.dist, P=calib.K).reshape(-1, 1, 2)
    d_after = row_straightness(c_und, calib.pattern)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for a, im, c, t in [(ax[0], best, best_c, f"До: ряд углов изогнут, макс. отклонение от прямой {best_dev:.1f} px"),
                        (ax[1], und, c_und, f"После cv2.undistort: макс. отклонение {d_after:.2f} px")]:
        a.imshow(rgb(im))
        row = c.reshape(calib.pattern[1], calib.pattern[0], 2)[0]
        a.plot(row[:, 0], row[:, 1], "r.-", lw=1)
        a.plot([row[0, 0], row[-1, 0]], [row[0, 1], row[-1, 1]], "c--", lw=1)
        a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig("results/fig_undistort.png", dpi=110); plt.close(fig)
    return best_dev, d_after


def fig_scenarios(calib):
    names = ["frontal", "tilt30", "tilt50", "dark", "glare", "noise", "blur", "clutter", "occlusion"]
    fig, ax = plt.subplots(3, 3, figsize=(15, 11.5))
    for a, n in zip(ax.ravel(), names):
        img, meta, r = load(f"{n}_1", calib)
        objs = r.measure_objects(img, meta["roi_mm"])
        canvas = img.copy()
        draw_objects(canvas, r, objs)
        err = np.mean([abs(r.distance(p["p1_px"], p["p2_px"]) - p["dist_mm"]) for p in meta["pairs"]])
        a.imshow(rgb(canvas)); a.axis("off")
        a.set_title(f"{n}: наклон {meta['theta']:.0f}°, найдено {len(objs)} предм., MAE линейки {err:.2f} мм", fontsize=9)
        if n in ("frontal", "occlusion", "glare"):
            cv2.imwrite(f"results/demo_{n}.jpg", canvas)
    fig.tight_layout(); fig.savefig("results/fig_scenarios.png", dpi=100); plt.close(fig)


def fig_errors():
    summ = pd.read_csv("results/summary.csv")
    x = np.arange(len(summ)); w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, m in enumerate(("naive", "homography", "calibrated")):
        ax.bar(x + (i - 1) * w, summ[f"{m}_MAE_mm"], w, label=LABELS[m], color=COLORS[m])
        for xi, v in zip(x, summ[f"{m}_MAE_mm"]):
            ax.text(xi + (i - 1) * w, v * 1.08, f"{v:.1f}" if v >= 1 else f"{v:.2f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(summ.scenario)
    ax.set_ylabel("средняя абсолютная ошибка, мм (лог. шкала)"); ax.legend()
    ax.set_title("Ошибка измерения расстояния точка-точка по сценариям (6 кадров x 8 пар на сценарий)")
    fig.tight_layout(); fig.savefig("results/fig_error_scenarios.png", dpi=110); plt.close(fig)


def fig_tilt():
    pairs, imgs = pd.read_csv("results/metrics_pairs.csv"), pd.read_csv("results/metrics_images.csv")
    d = pairs.merge(imgs[["image", "theta"]], on="image")
    g = d.groupby(["method", "image"]).agg(theta=("theta", "first"), err=("rel_err_pct", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in ("naive", "homography", "calibrated"):
        s = g[g.method == m]
        ax.scatter(s.theta, s.err, s=22, alpha=0.7, color=COLORS[m], label=LABELS[m])
    ax.set_yscale("log"); ax.set_xlabel("наклон оптической оси от нормали к плоскости, °")
    ax.set_ylabel("средняя относительная ошибка кадра, %"); ax.legend()
    ax.set_title("Ошибка линейки растёт с наклоном — кроме калибровочного метода")
    fig.tight_layout(); fig.savefig("results/fig_error_tilt.png", dpi=110); plt.close(fig)


def fig_calibration():
    sw = pd.read_csv("results/calibration_sweep.csv")
    g = sw.groupby("n_images")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, col, t in [(ax[0], "fx_err_pct", "Ошибка фокусного расстояния fx, %"), (ax[1], "c_err_px", "Ошибка главной точки, px")]:
        m, s = g[col].mean(), g[col].std().fillna(0)
        a.errorbar(m.index, m.values, yerr=s.values, fmt="o-", color=COLORS["calibrated"], capsize=3)
        a.set_xlabel("число снимков доски"); a.set_title(t); a.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig("results/fig_calibration.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    calib = Calibration.load("results/calibration.npz")
    fig_errors(); fig_tilt(); fig_calibration()
    before, after = fig_undistort(calib)
    print(f"прямизна ряда углов: до {before:.2f} px, после {after:.2f} px")
    fig_pipeline(calib)
    fig_scenarios(calib)
    print("готово: results/fig_*.png")
