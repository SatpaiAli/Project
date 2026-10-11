"""
calibrate.py — калибровка камеры по снимкам шахматной доски.

Пример:
    python calibrate.py --images data/calib --cols 9 --rows 6 --square 25 --out results/calibration.npz

--cols/--rows — число ВНУТРЕННИХ углов (у доски 10x7 клеток это 9x6),
--square — сторона клетки в мм (измерьте линейкой на распечатке!).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from ruler_core import calibrate_camera


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/calib", help="папка со снимками доски")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--out", default="results/calibration.npz")
    args = ap.parse_args()

    paths = sorted(p for ext in ("jpg", "jpeg", "png", "bmp") for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    imgs = [cv2.imread(p) for p in paths]
    calib, errs, used = calibrate_camera(imgs, (args.cols, args.rows), args.square)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    calib.save(args.out)
    K, d = calib.K, calib.dist
    print(f"Снимков: {len(paths)}, доска найдена на {len(used)}")
    print(f"RMS ошибка репроекции: {calib.rms:.3f} px")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"dist (k1,k2,p1,p2,k3) = {np.round(d, 4).tolist()}")
    worst = np.argsort(errs)[::-1][:3]
    print("Худшие снимки:", [(os.path.basename(paths[used[i]]), round(errs[i], 3)) for i in worst])
    json.dump(dict(K=K.tolist(), dist=d.tolist(), rms=calib.rms, size=list(calib.size),
                   per_image_error=dict(zip([os.path.basename(paths[i]) for i in used], errs))),
              open(args.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
