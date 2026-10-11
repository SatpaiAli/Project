"""
evaluate_real.py — проверка линейки на ВАШИХ реальных фото с известными размерами.

Подготовка:
  1. Откалибруйте камеру:  python calibrate.py --images data/real_calib --square 25 --out results/calibration_real.npz
  2. Сфотографируйте сцены (доска + предметы, разные ракурсы/свет) ТОЙ ЖЕ камерой, с теми же настройками.
  3. Измерьте настоящей линейкой несколько расстояний на предметах (например, длины сторон книги, карточки).
  4. Запишите в data/real/pairs.csv (заголовок обязателен):
         image,x1,y1,x2,y2,true_mm
         IMG_001.jpg,812,455,1103,470,148.0
     где (x1,y1) и (x2,y2) — пиксели концов отрезка (кликните мышью через measure.py — он печатает только мм,
     поэтому координаты удобно взять из любого просмотрщика изображений).
Запуск:
  python evaluate_real.py --calib results/calibration_real.npz --csv data/real/pairs.csv --dir data/real
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from ruler_core import METHODS, Calibration, PlaneRuler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="results/calibration_real.npz")
    ap.add_argument("--csv", default="data/real/pairs.csv")
    ap.add_argument("--dir", default="data/real")
    ap.add_argument("--out", default="results/metrics_real.csv")
    a = ap.parse_args()

    calib = Calibration.load(a.calib)
    df = pd.read_csv(a.csv)
    rows, rulers = [], {}
    for _, r in df.iterrows():
        if r.image not in rulers:
            img = cv2.imread(os.path.join(a.dir, r.image))
            ruler = PlaneRuler(calib)
            rulers[r.image] = ruler if (img is not None and ruler.fit(img)) else None
            if rulers[r.image] is None:
                print(f"[!] {r.image}: доска не найдена — кадр пропущен")
        ruler = rulers[r.image]
        if ruler is None:
            continue
        for m in METHODS:
            est = ruler.distance((r.x1, r.y1), (r.x2, r.y2), m)
            rows.append(dict(image=r.image, method=m, true_mm=r.true_mm, est_mm=est,
                             abs_err=abs(est - r.true_mm), rel_err_pct=100 * abs(est - r.true_mm) / r.true_mm))
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    print(res.groupby("method")[["abs_err", "rel_err_pct"]].agg(["mean", "max"]).round(2))
    print(f"кадров с найденной доской: {sum(v is not None for v in rulers.values())}/{len(rulers)}; сохранено: {a.out}")


if __name__ == "__main__":
    main()
