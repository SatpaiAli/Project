"""
ruler_core.py — ядро проекта «Виртуальная линейка».

Идея: если камера откалибрована (знаем матрицу K и коэффициенты дисторсии),
а в кадре есть плоскость с известным эталоном (шахматная доска с квадратом
известного размера), то для ЛЮБОГО пикселя на этой плоскости можно найти
реальные координаты в мм: луч из камеры через пиксель пересекаем с плоскостью.
Расстояние между двумя точками = расстояние между точками пересечения.

Содержимое:
  1. Шахматная доска: поиск углов, объектные точки.
  2. Калибровка камеры (метод Чжана, cv2.calibrateCamera).
  3. Класс PlaneRuler: поза плоскости, пиксель -> мм, три метода измерения
     (naive / homography / calibrated), ректификация, автоизмерение объектов.
  4. Функции отрисовки.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Критерий для итеративного снятия дисторсии (по умолчанию в OpenCV всего 5 итераций)
UNDIST_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-12)


# --------------------------------------------------------------------------
# 1. Шахматная доска
# --------------------------------------------------------------------------
def object_points(pattern, square):
    """3D-координаты внутренних углов доски в её системе (Z=0), мм.
    pattern = (cols, rows) — число ВНУТРЕННИХ углов."""
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def find_board_corners(gray, pattern, origin_dark=True):
    """Ищет внутренние углы шахматной доски с субпиксельной точностью.
    Сначала устойчивый детектор (SB, на основе радон-преобразования),
    при неудаче — классический + cornerSubPix. После поиска порядок углов
    приводится к единому виду (см. fix_orientation)."""
    cols, rows = pattern
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    if not ok:
        return None
    corners = corners.reshape(-1, 1, 2).astype(np.float64)
    return fix_orientation(gray, corners, pattern, origin_dark)


def fix_orientation(gray, corners, pattern, origin_dark=True):
    """OpenCV может вернуть углы в порядке, повёрнутом на 180 градусов (начало координат
    оказывается в противоположном углу доски). Тогда абсолютные координаты в мм «переезжают».
    Лечим по цвету клеток: у первого угла клетки по диагонали (-,-) и (+,+) должны быть
    тёмными (origin_dark=True, как у доски из make_chessboard.py). Если нет — разворачиваем порядок."""
    cols, rows = pattern
    c = corners.reshape(rows, cols, 2)
    ex, ey = c[0, 1] - c[0, 0], c[1, 0] - c[0, 0]          # шаги сетки в пикселях

    def patch(p):
        x, y = float(p[0]), float(p[1])
        if not (2 < x < gray.shape[1] - 3 and 2 < y < gray.shape[0] - 3):
            return None
        return float(cv2.getRectSubPix(gray, (5, 5), (x, y)).mean())

    p0 = c[0, 0]
    diag_same = [patch(p0 - (ex + ey) / 2), patch(p0 + (ex + ey) / 2)]
    diag_other = [patch(p0 + (ex - ey) / 2), patch(p0 + (ey - ex) / 2)]
    a = [v for v in diag_same if v is not None]
    b = [v for v in diag_other if v is not None]
    if not a or not b:
        return corners
    first_is_dark = np.mean(a) < np.mean(b)
    if first_is_dark != origin_dark:
        corners = corners[::-1].copy()
    return corners


# --------------------------------------------------------------------------
# 2. Калибровка камеры
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    K: np.ndarray          # матрица внутренних параметров 3x3
    dist: np.ndarray       # k1, k2, p1, p2, k3
    size: tuple            # (ширина, высота) изображения
    rms: float             # общая RMS-ошибка репроекции, px
    pattern: tuple         # (cols, rows) внутренних углов
    square: float          # сторона клетки, мм

    def save(self, path):
        np.savez(path, K=self.K, dist=self.dist, size=np.array(self.size), rms=self.rms,
                 pattern=np.array(self.pattern), square=self.square)

    @staticmethod
    def load(path):
        d = np.load(path)
        return Calibration(d["K"], d["dist"], tuple(int(x) for x in d["size"]), float(d["rms"]),
                           tuple(int(x) for x in d["pattern"]), float(d["square"]))


def detect_corners_batch(images_bgr, pattern):
    """Ищет доску на каждом снимке. Возвращает (список углов или None, размер изображения)."""
    out, size = [], None
    for img in images_bgr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        size = (gray.shape[1], gray.shape[0])
        out.append(find_board_corners(gray, pattern))
    return out, size


def calibrate_from_corners(corner_list, size, pattern, square):
    """Калибровка по уже найденным углам (позволяет не искать доску повторно)."""
    objp = object_points(pattern, square)
    used = [i for i, c in enumerate(corner_list) if c is not None]
    if len(used) < 3:
        raise RuntimeError(f"Доска найдена только на {len(used)} снимках, нужно минимум 3 (лучше 10-20)")
    obj_pts = [objp for _ in used]
    img_pts = [corner_list[i] for i in used]
    # calibrateCamera принимает только float32
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [o.astype(np.float32) for o in obj_pts], [c.astype(np.float32) for c in img_pts], size, None, None)
    errs = []
    for o, c, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=-1)))))
    return Calibration(K, dist.ravel(), size, float(rms), tuple(pattern), float(square)), errs, used


def calibrate_camera(images_bgr, pattern, square):
    """Калибровка по набору снимков шахматной доски.
    Возвращает (Calibration, per_image_errors, used_indices)."""
    corners, size = detect_corners_batch(images_bgr, pattern)
    return calibrate_from_corners(corners, size, pattern, square)


# --------------------------------------------------------------------------
# 3. Измерения на плоскости
# --------------------------------------------------------------------------
METHODS = ("naive", "homography", "calibrated")


class PlaneRuler:
    """Линейка для плоскости, на которой лежит шахматная доска.
    Система координат плоскости: начало — первый внутренний угол доски, X вдоль
    строки углов, Y вдоль столбца, единицы — мм."""

    def __init__(self, calib: Calibration):
        self.calib = calib
        self.K, self.dist = calib.K, calib.dist
        self.pattern, self.square = calib.pattern, calib.square
        self.rvec = self.tvec = self.H = self.mm_per_px = None
        self.corners = None

    # ---- привязка к кадру ------------------------------------------------
    def fit(self, bgr) -> bool:
        """Находит доску в кадре и вычисляет всё нужное для трёх методов измерения."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners = find_board_corners(gray, self.pattern)
        if corners is None:
            return False
        self.corners = corners
        objp = object_points(self.pattern, self.square)

        # метод 3: поза плоскости (solvePnP) с учётом K и дисторсии
        ok, rvec, tvec = cv2.solvePnP(objp, corners, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return False
        self.rvec, self.tvec = cv2.solvePnPRefineLM(objp, corners, self.K, self.dist, rvec, tvec)

        # метод 2: гомография «пиксели -> плоскость» (без K и без учёта дисторсии)
        self.H, _ = cv2.findHomography(corners.reshape(-1, 2), objp[:, :2], 0)

        # метод 1: один масштаб мм/пиксель по среднему шагу сетки углов
        c = corners.reshape(self.pattern[1], self.pattern[0], 2)
        step_px = np.mean(np.r_[np.linalg.norm(np.diff(c, axis=1), axis=-1).ravel(),
                                np.linalg.norm(np.diff(c, axis=0), axis=-1).ravel()])
        self.mm_per_px = self.square / step_px
        return True

    # ---- преобразования пиксель <-> плоскость ---------------------------
    def to_plane(self, px):
        """Пиксели (N,2) -> координаты на плоскости (N,2), мм.
        Шаги: снять дисторсию -> нормированные координаты -> луч -> пересечение с Z=0."""
        px = np.asarray(px, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPointsIter(px, self.K, self.dist, None, None, UNDIST_CRITERIA).reshape(-1, 2)
        R, _ = cv2.Rodrigues(self.rvec)
        C = -R.T @ self.tvec.ravel()                       # центр камеры в системе доски
        rays = np.c_[n, np.ones(len(n))] @ R               # направления лучей в системе доски
        s = -C[2] / rays[:, 2]
        return (C + s[:, None] * rays)[:, :2]

    def to_pixels(self, xy):
        """Координаты на плоскости (N,2), мм -> пиксели (N,2) (с дисторсией)."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        pts = np.c_[xy, np.zeros(len(xy))]
        px, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    # ---- расстояние -------------------------------------------------------
    def distance(self, p1, p2, method="calibrated") -> float:
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        if method == "naive":
            return float(np.linalg.norm(p1 - p2) * self.mm_per_px)
        if method == "homography":
            q = cv2.perspectiveTransform(np.array([[p1, p2]], np.float64), self.H)[0]
            return float(np.linalg.norm(q[0] - q[1]))
        if method == "calibrated":
            q = self.to_plane([p1, p2])
            return float(np.linalg.norm(q[0] - q[1]))
        raise ValueError(method)

    # ---- вид сверху ("ректификация") -------------------------------------
    def rectify(self, bgr, roi, ppm):
        """Строит вид сверху прямоугольной области плоскости roi=(x0,y0,x1,y1) в мм
        с разрешением ppm пикселей на мм. За один remap исправляются и перспектива, и дисторсия."""
        x0, y0, x1, y1 = roi
        W, H = int(round((x1 - x0) * ppm)), int(round((y1 - y0) * ppm))
        xs = x0 + (np.arange(W) + 0.5) / ppm
        ys = y0 + (np.arange(H) + 0.5) / ppm
        gx, gy = np.meshgrid(xs, ys)
        px = self.to_pixels(np.c_[gx.ravel(), gy.ravel()]).reshape(H, W, 2).astype(np.float32)
        return cv2.remap(bgr, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def board_rect_mm(self, margin=5.0):
        """Прямоугольник, занятый доской (с белой каймой в одну клетку), в мм."""
        cols, rows = self.pattern
        s = self.square
        return (-s - margin, -s - margin, cols * s + margin, rows * s + margin)

    # ---- автоматическое измерение предметов ------------------------------
    def measure_objects(self, bgr, roi, ppm=3.0, min_area_mm2=150.0, return_debug=False):
        """Находит тёмные предметы на светлом фоне в области roi и измеряет их в мм.
        Конвейер: вид сверху -> GaussianBlur -> заливка доски -> Otsu ->
        выравнивание освещения -> Otsu -> морфология -> контуры -> minAreaRect."""
        rect = self.rectify(bgr, roi, ppm)
        gray = cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)

        # 1) закрашиваем доску медианным цветом фона, чтобы она не мешала сегментации
        bx0, by0, bx1, by1 = self.board_rect_mm()
        x0, y0 = roi[0], roi[1]
        i0, i1 = int((bx0 - x0) * ppm), int((bx1 - x0) * ppm)
        j0, j1 = int((by0 - y0) * ppm), int((by1 - y0) * ppm)
        board = np.zeros(gray.shape, bool)
        board[max(j0, 0):max(j1, 0), max(i0, 0):max(i1, 0)] = True
        bg_level = np.median(gray[~board]) if (~board).any() else 128
        gray = gray.copy()
        gray[board] = bg_level

        # 2) первый проход Otsu (предметы темнее фона), затем выравнивание освещения
        thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if bw.mean() / 255 > 0.5:                      # предметы светлее фона — меняем полярность
            bw = 255 - bw
            polarity = -1
        else:
            polarity = 1
        flat = _flatten_illumination(gray, bw)
        flat_u8 = np.clip(flat, 0, 255).astype(np.uint8)
        thr, bw = cv2.threshold(flat_u8, 0, 255,
                                (cv2.THRESH_BINARY_INV if polarity == 1 else cv2.THRESH_BINARY) | cv2.THRESH_OTSU)

        # защита: если контраст предметов с фоном слишком мал — предметов нет
        fg, bgd = flat_u8[bw > 0], flat_u8[bw == 0]
        if len(fg) == 0 or len(bgd) == 0 or abs(float(bgd.mean()) - float(fg.mean())) < 25:
            bw[:] = 0

        # 3) морфология: убрать мелкий шум, заполнить дыры
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # 4) контуры и измерение
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        objs = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            area_mm2 = area_px / ppm ** 2
            if area_mm2 < min_area_mm2:
                continue
            per = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area_px / (per ** 2 + 1e-9)
            (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
            bx, by, bw_, bh = cv2.boundingRect(cnt)
            partial = bx <= 1 or by <= 1 or bx + bw_ >= bw.shape[1] - 1 or by + bh >= bw.shape[0] - 1
            center_mm = (x0 + (cx + 0.5) / ppm, y0 + (cy + 0.5) / ppm)
            if circ > 0.88:                              # круг
                d = np.sqrt(4 * area_mm2 / np.pi)
                t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
                poly = np.c_[center_mm[0] + d / 2 * np.cos(t), center_mm[1] + d / 2 * np.sin(t)]
                objs.append(dict(kind="circle", center_mm=center_mm, size_mm=(d, d), area_mm2=area_mm2,
                                 circularity=circ, partial=partial, polygon_mm=poly))
            else:                                        # прямоугольник (повёрнутый)
                box = cv2.boxPoints(((cx, cy), (w, h), ang))
                poly = np.c_[x0 + (box[:, 0] + 0.5) / ppm, y0 + (box[:, 1] + 0.5) / ppm]
                objs.append(dict(kind="rect", center_mm=center_mm, size_mm=(w / ppm, h / ppm),
                                 area_mm2=area_mm2, circularity=circ, partial=partial, polygon_mm=poly))
        if return_debug:
            return objs, dict(rect=rect, mask=bw, flat=flat_u8, board=board)
        return objs


def _flatten_illumination(gray, fg_mask):
    """Делит изображение на гладкую (квадратичную) модель освещения фона,
    чтобы тени и градиенты не ломали пороговую сегментацию."""
    h, w = gray.shape
    k = 4
    small = cv2.resize(gray.astype(np.float32), (w // k, h // k), interpolation=cv2.INTER_AREA)
    fg = cv2.dilate(fg_mask, np.ones((15, 15), np.uint8))
    bgmask = cv2.resize(fg, (w // k, h // k), interpolation=cv2.INTER_NEAREST) == 0

    def basis(x, y):
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)

    sy, sx = np.mgrid[0:h // k, 0:w // k]
    A = basis(sx / (w / k), sy / (h / k))
    if bgmask.sum() < 200:
        return gray.astype(np.float32)
    coef, *_ = np.linalg.lstsq(A[bgmask], small[bgmask], rcond=None)
    fy, fx = np.mgrid[0:h, 0:w]
    model = basis(fx / w, fy / h) @ coef
    model = np.maximum(model, 1.0)
    return gray.astype(np.float32) / model * float(np.mean(model))


# --------------------------------------------------------------------------
# 4. Отрисовка
# --------------------------------------------------------------------------
def draw_measurement(img, ruler: PlaneRuler, p1, p2, color=(0, 200, 255), label=None):
    """Рисует линейку между p1 и p2. Линия строится как отрезок НА ПЛОСКОСТИ и
    проецируется обратно в кадр, поэтому при дисторсии она слегка изгибается, как и в реальности."""
    q = ruler.to_plane([p1, p2])
    t = np.linspace(0, 1, 40)[:, None]
    line_px = ruler.to_pixels(q[0] * (1 - t) + q[1] * t).astype(np.int32)
    cv2.polylines(img, [line_px.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 5, color, -1, cv2.LINE_AA)
    d = float(np.linalg.norm(q[0] - q[1]))
    text = label or f"{d:.1f} mm"
    mid = (np.asarray(p1) + np.asarray(p2)) / 2
    _put_label(img, text, (int(mid[0]) + 8, int(mid[1]) - 8), color)
    return d


def draw_objects(img, ruler: PlaneRuler, objs, color=(0, 220, 0)):
    for o in objs:
        poly = o["polygon_mm"]
        closed = np.vstack([poly, poly[:1]])
        dense = np.vstack([np.linspace(closed[i], closed[i + 1], 20, endpoint=False) for i in range(len(closed) - 1)])
        px = ruler.to_pixels(dense).astype(np.int32)
        col = (0, 165, 255) if o["partial"] else color
        cv2.polylines(img, [px.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        c = ruler.to_pixels([o["center_mm"]])[0]
        w, h = o["size_mm"]
        text = f"d={w:.1f}" if o["kind"] == "circle" else f"{max(w, h):.1f}x{min(w, h):.1f}"
        _put_label(img, text, (int(c[0]) - 40, int(c[1]) + 5), col)
    return img


def _put_label(img, text, org, color):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, thh), _ = cv2.getTextSize(text, font, scale, th)
    x, y = org
    cv2.rectangle(img, (x - 3, y - thh - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
