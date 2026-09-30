
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib.colors import rgb_to_hsv as mpl_rgb_to_hsv
from matplotlib.widgets import Slider, Button

# Каталог, куда складываются результаты работы программы.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

WEIGHTS_BT601 = (0.299, 0.587, 0.114)

WEIGHTS_BT709 = (0.2126, 0.7152, 0.0722)



# Вспомогательные функции


def load_image(path: str | None) -> np.ndarray:
    
    if path and os.path.isfile(path):
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    if path:
        print(f"Файл '{path}' не найден — используется тестовое изображение.")
    else:
        print("Изображение не задано — используется тестовое изображение.")
    return make_test_image()


def make_test_image(height: int = 360, width: int = 540) -> np.ndarray:

    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)          # горизонтальная координата в [0, 1]
    v = y / (height - 1)         # вертикальная координата в [0, 1]

    image = np.zeros((height, width, 3), dtype=np.float64)

    # Верхняя половина -вертикальные полосы чистых и составных цветов.
    bars = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    half = height // 2
    bar_width = width / len(bars)
    for i, color in enumerate(bars):
        left = int(round(i * bar_width))
        right = int(round((i + 1) * bar_width))
        image[:half, left:right] = color

    # Нижняя половина плавные градиенты по каждому каналу.
    image[half:, :, 0] = 255 * u[half:]
    image[half:, :, 1] = 255 * (1 - u[half:])
    image[half:, :, 2] = 255 * v[half:]

    return image.astype(np.uint8)


def ensure_output_dir() -> str:
    """Создать каталог для результатов (если его ещё нет) и вернуть путь."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def save_gray(image: np.ndarray, filename: str) -> None:
    """Сохранить полутоновое изображение в каталог результатов."""
    path = os.path.join(ensure_output_dir(), filename)
    Image.fromarray(image).save(path)  # 2D-массив uint8 -> режим "L"
    print(f"Сохранено: {path}")



#Задание 1

def rgb_to_gray(image: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Перевести RGB в оттенки серого: Y = wR * R + wG * G + wB * B.

    Вычисления ведутся в float, результат округляется и приводится к uint8.
    """
    wr, wg, wb = weights
    rgb = image.astype(np.float64)
    gray = wr * rgb[:, :, 0] + wg * rgb[:, :, 1] + wb * rgb[:, :, 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def gray_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Модуль разности двух полутоновых изображений (тип uint8).

    Приведение к int16 нужно, чтобы вычитание uint8 не «заворачивалось».
    """
    diff = first.astype(np.int16) - second.astype(np.int16)
    return np.abs(diff).astype(np.uint8)


def stretch_contrast(image: np.ndarray) -> np.ndarray:
    """Растянуть диапазон значений на [0, 255] — разность обычно очень тёмная."""
    lo, hi = int(image.min()), int(image.max())
    if hi == lo:
        return np.zeros_like(image)
    scaled = (image.astype(np.float64) - lo) * 255.0 / (hi - lo)
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def intensity_histogram(gray: np.ndarray) -> np.ndarray:
    """Гистограмма интенсивности: количество пикселей для каждого уровня 0-255."""
    return np.bincount(gray.ravel(), minlength=256)


def task_1(image: np.ndarray) -> None:
    gray_601 = rgb_to_gray(image, WEIGHTS_BT601)
    gray_709 = rgb_to_gray(image, WEIGHTS_BT709)
    diff = gray_difference(gray_601, gray_709)

    label_601 = "BT.601: 0.299R + 0.587G + 0.114B"
    label_709 = "BT.709: 0.2126R + 0.7152G + 0.0722B"

    print("\n--- Задание 1: RGB -> оттенки серого ---")
    print(f"Формула 1 ({label_601}): средняя яркость {gray_601.mean():.2f}")
    print(f"Формула 2 ({label_709}): средняя яркость {gray_709.mean():.2f}")
    print(f"Разность: min={diff.min()}, max={diff.max()}, "
          f"среднее={diff.mean():.2f}, "
          f"различающихся пикселей={np.count_nonzero(diff)} из {diff.size}")

    save_gray(gray_601, "task1_gray_bt601.png")
    save_gray(gray_709, "task1_gray_bt709.png")
    save_gray(diff, "task1_difference.png")
    save_gray(stretch_contrast(diff), "task1_difference_stretched.png")

    # Изображения: оригинал, два полутоновых варианта и их разность 
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Задание 1. Преобразование RGB в оттенки серого", fontsize=14)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Исходное изображение (RGB)")

    axes[0, 1].imshow(gray_601, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Вариант 1\n{label_601}")

    axes[0, 2].imshow(gray_709, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(f"Вариант 2\n{label_709}")

    axes[1, 0].imshow(diff, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Разность |вариант 1 - вариант 2|")

    axes[1, 1].imshow(stretch_contrast(diff), cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(f"Та же разность с растянутым контрастом\n"
                         f"(исходный диапазон 0..{diff.max()})")

    # Гистограмма самой разности — показывает, насколько сильно расходятся формулы.
    axes[1, 2].bar(np.arange(256), intensity_histogram(diff),
                   width=1.0, color="tab:purple")
    axes[1, 2].set_title("Гистограмма разности")
    axes[1, 2].set_xlabel("Модуль разности")
    axes[1, 2].set_ylabel("Количество пикселей")
    axes[1, 2].set_xlim(-1, 256)

    for ax in axes.ravel()[:5]:
        ax.axis("off")
    fig.tight_layout()

    # --- Гистограммы интенсивности для обоих преобразований ---
    hist_601 = intensity_histogram(gray_601)
    hist_709 = intensity_histogram(gray_709)
    levels = np.arange(256)

    fig2, hist_axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig2.suptitle("Задание 1. Гистограммы интенсивности", fontsize=14)

    hist_axes[0].bar(levels, hist_601, width=1.0, color="tab:blue")
    hist_axes[0].set_title(f"Вариант 1\n{label_601}")

    hist_axes[1].bar(levels, hist_709, width=1.0, color="tab:orange")
    hist_axes[1].set_title(f"Вариант 2\n{label_709}")

    # Третий график — наложение, чтобы разница между формулами была наглядной.
    hist_axes[2].plot(levels, hist_601, color="tab:blue", linewidth=1.0,
                      label="Вариант 1 (BT.601)")
    hist_axes[2].plot(levels, hist_709, color="tab:orange", linewidth=1.0,
                      label="Вариант 2 (BT.709)")
    hist_axes[2].set_title("Сравнение гистограмм")
    hist_axes[2].legend()

    top = max(hist_601.max(), hist_709.max()) * 1.05
    for ax in hist_axes:
        ax.set_xlabel("Уровень яркости (0..255)")
        ax.set_ylabel("Количество пикселей")
        ax.set_xlim(-1, 256)
        ax.set_ylim(0, top)
    fig2.tight_layout()

    out_dir = ensure_output_dir()
    fig.savefig(os.path.join(out_dir, "task1_images.png"), dpi=120)
    fig2.savefig(os.path.join(out_dir, "task1_histograms.png"), dpi=120)
    print(f"Сохранено: {os.path.join(out_dir, 'task1_images.png')}")
    print(f"Сохранено: {os.path.join(out_dir, 'task1_histograms.png')}")

    plt.show()



# Задание 2


def task2(image_path):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    
    red_image = np.zeros_like(img)
    red_image[:, :, 0] = R

    green_image = np.zeros_like(img)
    green_image[:, :, 1] = G

    blue_image = np.zeros_like(img)
    blue_image[:, :, 2] = B

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Исходное изображение")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(red_image)
    plt.title("Канал R")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(green_image)
    plt.title("Канал G")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(blue_image)
    plt.title("Канал B")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.hist(R.ravel(), bins=256, range=(0, 255))
    plt.title("Гистограмма канала R")
    plt.xlim(0, 255)

    plt.subplot(3, 1, 2)
    plt.hist(G.ravel(), bins=256, range=(0, 255))
    plt.title("Гистограмма канала G")
    plt.xlim(0, 255)

    plt.subplot(3, 1, 3)
    plt.hist(B.ravel(), bins=256, range=(0, 255))
    plt.title("Гистограмма канала B")
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.show()


#Задание 3

def rgb_to_hsv(RGB_matrix: np.ndarray) -> np.ndarray:
    rgb = RGB_matrix.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    MAX = np.max(rgb, axis=2)
    MIN = np.min(rgb, axis=2)
    delta = MAX - MIN

    V = MAX

    S = np.zeros_like(MAX)
    mask_max_nonzero = MAX > 0
    S[mask_max_nonzero] = 1 - MIN[mask_max_nonzero] / MAX[mask_max_nonzero]

    H = np.zeros_like(MAX)

    condition_1 = (delta > 0) & (MAX == r) & (g >= b)
    H[condition_1] = 60 * (g[condition_1] - b[condition_1]) / delta[condition_1] + 0

    condition_2 = (delta > 0) & (MAX == r) & (g < b)
    H[condition_2] = 60 * (g[condition_2] - b[condition_2]) / delta[condition_2] + 360

    condition_3 = (delta > 0) & (MAX == g)
    H[condition_3] = 60 * (b[condition_3] - r[condition_3]) / delta[condition_3] + 120

    condition_4 = (delta > 0) & (MAX == b)
    H[condition_4] = 60 * (r[condition_4] - g[condition_4]) / delta[condition_4] + 240

    return np.stack([H, S, V], axis=2)


def hsv_to_rgb(HSV_matrix: np.ndarray) -> np.ndarray:
    H, S, V = HSV_matrix[:, :, 0], HSV_matrix[:, :, 1], HSV_matrix[:, :, 2]

    Hi = (np.floor(H / 60).astype(int)) % 6
    Vmin = (1 - S) * V
    a = (V - Vmin) * ((H % 60) / 60)
    Vinc = Vmin + a
    Vdec = V - a

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    case_0 = Hi == 0
    R[case_0], G[case_0], B[case_0] = V[case_0], Vinc[case_0], Vmin[case_0]

    case_1 = Hi == 1
    R[case_1], G[case_1], B[case_1] = Vdec[case_1], V[case_1], Vmin[case_1]

    case_2 = Hi == 2
    R[case_2], G[case_2], B[case_2] = Vmin[case_2], V[case_2], Vinc[case_2]

    case_3 = Hi == 3
    R[case_3], G[case_3], B[case_3] = Vmin[case_3], Vdec[case_3], V[case_3]

    case_4 = Hi == 4
    R[case_4], G[case_4], B[case_4] = Vinc[case_4], Vmin[case_4], V[case_4]

    case_5 = Hi == 5
    R[case_5], G[case_5], B[case_5] = V[case_5], Vmin[case_5], Vdec[case_5]

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def compare_with_matplotlib(image: np.ndarray) -> None:
    own = rgb_to_hsv(image)
    reference = mpl_rgb_to_hsv(image.astype(np.float64) / 255.0)

    own_h = own[:, :, 0] / 360.0
    diff_h = np.abs(own_h - reference[:, :, 0])
    diff_h = np.minimum(diff_h, 1 - diff_h)
    diff_s = np.abs(own[:, :, 1] - reference[:, :, 1])
    diff_v = np.abs(own[:, :, 2] - reference[:, :, 2])

    print("Сравнение со встроенным matplotlib.colors.rgb_to_hsv:")
    print(f"  H: максимальное расхождение {diff_h.max():.6f}")
    print(f"  S: максимальное расхождение {diff_s.max():.6f}")
    print(f"  V: максимальное расхождение {diff_v.max():.6f}")


def run_interactive(image: np.ndarray) -> None:
    base_hsv = rgb_to_hsv(image)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    display = ax.imshow(image)
    ax.axis("off")
    ax.set_title("Задание 3. RGB <-> HSV")

    ax_h = plt.axes([0.25, 0.20, 0.5, 0.03])
    ax_s = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_v = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_save = plt.axes([0.4, 0.02, 0.2, 0.05])

    slider_h = Slider(ax_h, "Hue, сдвиг", -180, 180, valinit=0)
    slider_s = Slider(ax_s, "Saturation, x", 0.0, 2.0, valinit=1.0)
    slider_v = Slider(ax_v, "Value, x", 0.0, 2.0, valinit=1.0)
    button_save = Button(ax_save, "Сохранить")

    def current_rgb() -> np.ndarray:
        hsv = base_hsv.copy()
        hsv[:, :, 0] = (hsv[:, :, 0] + slider_h.val) % 360
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * slider_s.val, 0, 1)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * slider_v.val, 0, 1)
        return hsv_to_rgb(hsv)

    def update(_) -> None:
        display.set_data(current_rgb())
        fig.canvas.draw_idle()

    def save(_) -> None:
        path = os.path.join(ensure_output_dir(), "task3_result.png")
        Image.fromarray(current_rgb()).save(path)
        print(f"Сохранено: {path}")

    slider_h.on_changed(update)
    slider_s.on_changed(update)
    slider_v.on_changed(update)
    button_save.on_clicked(save)

    plt.show()


def task_3(image: np.ndarray) -> None:
    compare_with_matplotlib(image)
    run_interactive(image)

#начало


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа #2: преобразование цветовых пространств.")
    parser.add_argument("-i", "--image", default=None,
                        help="путь к изображению (по умолчанию — тестовое)")
    parser.add_argument("-t", "--task", type=int, choices=(1, 2, 3), default=1,
                        help="номер задания (по умолчанию 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    print(f"Размер изображения: {image.shape[1]}x{image.shape[0]} пикселей")

    tasks = {1: task_1, 3: task_3}  # задания 2 и 3 добавить сюда после реализации
    if args.task not in tasks:
        print(f"Задание {args.task} пока не реализовано.")
        return
    tasks[args.task](image)
    task_1(load_image("cat.avif"))
    task2("task2.png")

    task_3(load_image("task3.png"))

if __name__ == "__main__":
    main()

#чтобы запустить первое python graphic_1.py -i "ваше фото без скобок" в терминале