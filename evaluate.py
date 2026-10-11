"""
evaluate.py — количественная оценка «Виртуальной линейки» на синтетических сценах.

Что считаем:
  1. Калибровка: ошибка найденных параметров камеры в зависимости от числа снимков доски.
  2. «Линейка» (точка-точка): абсолютная/относительная ошибка расстояний в мм для трёх методов
       naive      — один масштаб мм/пиксель по клетке доски,
       homography — гомография по доске, без калибровки и без учёта дисторсии,
       calibrated — калибровка + поза плоскости + пересечение луча с плоскостью (наш метод).
     Клик пользователя моделируется шумом 0.5 px.
  3. Автоизмерение предметов (сегментация + контуры): ошибка размеров, полнота и точность обнаружения.
  4. Время обработки одного кадра.

Результаты: results/*.csv и results/summary.md
Запуск:  python make_test_data.py && python calibrate.py && python evaluate.py
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from ruler_core import METHODS, Calibration, PlaneRuler, calibrate_from_corners, detect_corners_batch

CLICK_NOISE_PX = 0.5
MATCH_RADIUS_PX = 30.0


def calibration_sweep(rng, true, n_trials=8, sizes=(3, 5, 8, 12, 20)):
    """Как качество калибровки зависит от числа снимков доски."""
    paths = sorted(glob.glob("data/calib/*.jpg"))
    imgs = [cv2.imread(p) for p in paths]
    pattern = tuple(true["pattern"])
    corners_all, size = detect_corners_batch(imgs, pattern)    # доску ищем один раз
    rows = []
    for n in sizes:
        for trial in range(n_trials if n < len(imgs) else 1):
            idx = rng.choice(len(imgs), n, replace=False) if n < len(imgs) else np.arange(len(imgs))
            c, _, _ = calibrate_from_corners([corners_all[i] for i in idx], size, pattern, true["square"])
            rows.append(dict(n_images=n, trial=trial, rms_px=c.rms,
                             fx_err_pct=100 * abs(c.K[0, 0] - true["K"][0][0]) / true["K"][0][0],
                             fy_err_pct=100 * abs(c.K[1, 1] - true["K"][1][1]) / true["K"][1][1],
                             c_err_px=float(np.hypot(c.K[0, 2] - true["K"][0][2], c.K[1, 2] - true["K"][1][2])),
                             k1_err=abs(c.dist[0] - true["dist"][0])))
    return pd.DataFrame(rows)


def evaluate_scenes(calib, rng):
    pair_rows, obj_rows, img_rows = [], [], []
    for jpg in sorted(glob.glob("data/scenes/*.jpg")):
        meta = json.load(open(jpg.replace(".jpg", ".json")))
        scen, name = meta["scenario"], os.path.basename(jpg)[:-4]
        img = cv2.imread(jpg)
        ruler = PlaneRuler(calib)
        t0 = time.perf_counter()
        ok = ruler.fit(img)
        t_fit = time.perf_counter() - t0
        img_rows.append(dict(scenario=scen, image=name, board_found=ok, theta=meta["theta"], fit_s=t_fit))
        if not ok:
            continue
        # --- линейка точка-точка ---
        for k, p in enumerate(meta["pairs"]):
            p1 = np.array(p["p1_px"]) + rng.normal(0, CLICK_NOISE_PX, 2)
            p2 = np.array(p["p2_px"]) + rng.normal(0, CLICK_NOISE_PX, 2)
            for m in METHODS:
                est = ruler.distance(p1, p2, m)
                pair_rows.append(dict(scenario=scen, image=name, method=m, true_mm=p["dist_mm"], est_mm=est,
                                      abs_err=abs(est - p["dist_mm"]), rel_err_pct=100 * abs(est - p["dist_mm"]) / p["dist_mm"]))
        # --- автоизмерение предметов ---
        t0 = time.perf_counter()
        objs = ruler.measure_objects(img, meta["roi_mm"])
        img_rows[-1]["objects_s"] = time.perf_counter() - t0
        gt_px = np.array(meta["center_px"])
        used = set()
        for o in objs:
            c_px = ruler.to_pixels([o["center_mm"]])[0]
            d = np.linalg.norm(gt_px - c_px, axis=1)
            j = int(np.argmin(d))
            if d[j] < MATCH_RADIUS_PX and j not in used:
                used.add(j)
                g = meta["objects"][j]
                gs, es = sorted(g["size_mm"], reverse=True), sorted(o["size_mm"], reverse=True)
                if g["kind"] != o["kind"]:
                    err = np.nan
                else:
                    err = float(np.mean([abs(es[0] - gs[0]), abs(es[1] - gs[1])]))
                obj_rows.append(dict(scenario=scen, image=name, gt_idx=j, kind=g["kind"], matched=True,
                                     kind_ok=g["kind"] == o["kind"], size_err_mm=err, partial=o["partial"]))
            else:
                obj_rows.append(dict(scenario=scen, image=name, gt_idx=-1, kind=o["kind"], matched=False,
                                     kind_ok=False, size_err_mm=np.nan, partial=o["partial"]))
        for j in range(len(meta["objects"])):
            if j not in used:
                obj_rows.append(dict(scenario=scen, image=name, gt_idx=j, kind=meta["objects"][j]["kind"],
                                     matched=False, kind_ok=False, size_err_mm=np.nan, partial=False, missed=True))
    return pd.DataFrame(pair_rows), pd.DataFrame(obj_rows), pd.DataFrame(img_rows)


def summarize(pairs, objs, imgs):
    order = ["frontal", "tilt30", "tilt50", "dark", "glare", "noise", "blur", "clutter", "occlusion"]
    rows = []
    for scen in order:
        im = imgs[imgs.scenario == scen]
        row = dict(scenario=scen, board_found=f"{int(im.board_found.sum())}/{len(im)}")
        for m in METHODS:
            e = pairs[(pairs.scenario == scen) & (pairs.method == m)]
            row[f"{m}_MAE_mm"] = e.abs_err.mean()
            row[f"{m}_MRE_pct"] = e.rel_err_pct.mean()
        o = objs[objs.scenario == scen]
        tp = ((o.matched) & (o.kind_ok)).sum()
        fp = (~o.matched & o.gt_idx.eq(-1)).sum()
        fn = o.get("missed", pd.Series(False, index=o.index)).fillna(False).sum()
        n_gt = 6 * len(im)
        row["obj_recall_pct"] = 100 * tp / max(n_gt, 1)
        row["obj_false_pos"] = int(fp)
        row["obj_size_err_mm"] = o[o.matched & o.kind_ok].size_err_mm.mean()
        row["fit_ms"] = 1000 * im.fit_s.mean()
        row["objects_ms"] = 1000 * im.objects_s.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df, floatfmt=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(123)
    true = json.load(open("data/camera_true.json"))
    calib = Calibration.load("results/calibration.npz")

    sweep = calibration_sweep(rng, true)
    sweep.to_csv("results/calibration_sweep.csv", index=False)
    pairs, objs, imgs = evaluate_scenes(calib, rng)
    pairs.to_csv("results/metrics_pairs.csv", index=False)
    objs.to_csv("results/metrics_objects.csv", index=False)
    imgs.to_csv("results/metrics_images.csv", index=False)
    summ = summarize(pairs, objs, imgs)
    summ.to_csv("results/summary.csv", index=False)

    overall = pairs.groupby("method")[["abs_err", "rel_err_pct"]].agg(["mean", "median", lambda x: np.percentile(x, 95)])
    overall.columns = ["MAE_mm", "median_mm", "p95_mm", "MRE_pct", "median_pct", "p95_pct"]
    sweep_mean = sweep.groupby("n_images").mean(numeric_only=True).drop(columns="trial").reset_index()
    with open("results/summary.md", "w") as f:
        f.write("## Сводка по сценариям\n\n" + to_markdown(summ) + "\n\n")
        f.write("## Общая ошибка линейки по методам (все сцены)\n\n" + to_markdown(overall.reset_index()) + "\n\n")
        f.write("## Калибровка vs число снимков\n\n" + to_markdown(sweep_mean, 3) + "\n")
    print(open("results/summary.md").read())


if __name__ == "__main__":
    main()
