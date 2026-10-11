"""
measure.py — «Виртуальная линейка»: измерение расстояний на фото в миллиметрах.

Нужно: калибровка камеры (calibrate.py) и фото, на котором лежит шахматная доска
на той же плоскости, что и измеряемые предметы (бумага, стол).

Режимы:
  интерактивный (по умолчанию)  — кликайте по двум точкам, линейка покажет мм
      ЛКМ — поставить точку | u — отменить | s — сохранить картинку | q/Esc — выход
  --points x1 y1 x2 y2 [...]    — измерить заданные пары точек без окна
  --auto                        — найти предметы автоматически и измерить их

Примеры:
  python measure.py data/scenes/tilt30_0.jpg
  python measure.py data/scenes/tilt30_0.jpg --points 800 500 1000 620 --out results/demo_points.jpg
  python measure.py data/scenes/tilt30_0.jpg --auto --roi 235 -20 575 285 --out results/demo_auto.jpg
"""
import argparse
import sys

import cv2
import numpy as np

from ruler_core import Calibration, PlaneRuler, draw_measurement, draw_objects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--calib", default="results/calibration.npz")
    ap.add_argument("--points", type=float, nargs="+", help="x1 y1 x2 y2 [x1 y1 x2 y2 ...] в пикселях")
    ap.add_argument("--auto", action="store_true", help="автоматически измерить предметы")
    ap.add_argument("--roi", type=float, nargs=4, default=[235, -20, 575, 285],
                    metavar=("X0", "Y0", "X1", "Y1"), help="область поиска предметов в мм (система координат доски)")
    ap.add_argument("--out", default=None, help="куда сохранить результат")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не удалось открыть {args.image}")
    calib = Calibration.load(args.calib)
    if (img.shape[1], img.shape[0]) != calib.size:
        print(f"ВНИМАНИЕ: размер кадра {img.shape[1]}x{img.shape[0]} не совпадает с калибровочным {calib.size}. "
              "Калибровка верна только для того же разрешения и фокусировки.")
    ruler = PlaneRuler(calib)
    if not ruler.fit(img):
        sys.exit("Шахматная доска не найдена. Она должна целиком помещаться в кадр, быть в фокусе и без бликов.")

    canvas = img.copy()
    if args.auto:
        objs = ruler.measure_objects(img, args.roi)
        for i, o in enumerate(sorted(objs, key=lambda o: o["center_mm"]), 1):
            w, h = o["size_mm"]
            print(f"#{i} {o['kind']:6s} {'d = %.1f мм' % w if o['kind'] == 'circle' else '%.1f x %.1f мм' % (max(w, h), min(w, h))}"
                  f"{'  (обрезан краем области!)' if o['partial'] else ''}")
        draw_objects(canvas, ruler, objs)
    elif args.points:
        if len(args.points) % 4:
            sys.exit("--points: нужно кратно 4 числам")
        for x1, y1, x2, y2 in np.array(args.points).reshape(-1, 4):
            print(f"({x1:.0f},{y1:.0f}) - ({x2:.0f},{y2:.0f}): {draw_measurement(canvas, ruler, (x1, y1), (x2, y2)):.1f} мм")
    else:
        interactive(img, ruler, args.out)
        return
    if args.out:
        cv2.imwrite(args.out, canvas)
        print("сохранено:", args.out)


def interactive(img, ruler, out):
    pts, lines = [], []                     # pts — текущая незавершённая пара; lines — готовые пары
    win = "Virtual ruler (LMB: points, u: undo, s: save, q: quit)"

    def redraw():
        canvas = img.copy()
        for a, b in lines:
            draw_measurement(canvas, ruler, a, b)
        for p in pts:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 2:
                lines.append(tuple(pts))
                print(f"{ruler.distance(*pts):.1f} мм")
                pts.clear()
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    canvas = redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            (pts.pop() if pts else (lines.pop() if lines else None))
            canvas = redraw()
        if key == ord("s"):
            path = out or "results/measure_result.jpg"
            cv2.imwrite(path, redraw())
            print("сохранено:", path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
