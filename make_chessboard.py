"""
make_chessboard.py — печатная шахматная доска точного размера (PDF, лист A4 альбомный).

Доска 10x7 клеток (9x6 внутренних углов), клетка 25 мм. Левая верхняя клетка ЧЁРНАЯ —
именно от неё программа определяет начало координат.
ВАЖНО: печатать в масштабе 100% («Actual size»), затем измерить клетку линейкой
и, если вышло не 25 мм, передать реальное значение в --square.

Запуск:  python make_chessboard.py            -> chessboard_A4.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares-x", type=int, default=10)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square", type=float, default=25.0, help="мм")
    ap.add_argument("--out", default="chessboard_A4.pdf")
    a = ap.parse_args()
    page_w, page_h = 297.0, 210.0
    bw, bh = a.squares_x * a.square, a.squares_y * a.square
    ox, oy = (page_w - bw) / 2, (page_h - bh) / 2
    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w); ax.set_ylim(page_h, 0); ax.axis("off")
    for i in range(a.squares_x):
        for j in range(a.squares_y):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((ox + i * a.square, oy + j * a.square), a.square, a.square, color="black", lw=0))
    ax.text(page_w / 2, page_h - 8, f"клетка {a.square:g} мм — печатать в масштабе 100%", ha="center", fontsize=8)
    fig.savefig(a.out)
    print("сохранено:", a.out)


if __name__ == "__main__":
    main()
