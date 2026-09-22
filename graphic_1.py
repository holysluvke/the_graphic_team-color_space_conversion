
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    # args = parse_args()
    # image = load_image(args.image)
    # print(f"Размер изображения: {image.shape[1]}x{image.shape[0]} пикселей")

    # tasks = {1: task_1}  # задания 2 и 3 добавить сюда после реализации
    # if args.task not in tasks:
    #     print(f"Задание {args.task} пока не реализовано.")
    #     return
    # tasks[args.task](image)
    task2("task2.png")

if __name__ == "__main__":
    main()

#чтобы запустить первое python graphic_1.py -i "ваше фото без скобок" в терминале