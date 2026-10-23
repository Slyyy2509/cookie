"""Авторазметка скриншотов обученной YOLO-моделью.

Прогоняет модель по сырым кадрам из папки --input и раскладывает результат
в images/<split> + labels/<split> в формате YOLO. Кадры, где модель не нашла
ничего, копируются в needs_review/ — их нужно проверить глазами руками.

Запуск:
    py autolabel.py --input raw --split train
    py autolabel.py --input raw --val-ratio 0.1
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Для авторазметки важна полнота, а не точность: лишний бокс вы снимете
# руками за секунду, а пропущенный печеньку даже не заметите на превью.
DEFAULT_CONF = 0.15

BASE_DIR = Path(__file__).resolve().parent


def format_label(box) -> str:
    """YOLO-строка: class cx cy w h в нормализованных координатах [0, 1]."""
    cls = int(box.cls[0])
    cx, cy, w, h = box.xywhn[0].tolist()
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def autolabel(
    model: YOLO,
    image_path: Path,
    split: str,
    conf: float,
    imgsz: int,
    review_dir: Path,
    overwrite: bool,
) -> int:
    """Размечает один кадр. Возвращает число боксов, -1 если кадр пропущен."""
    out_dir = BASE_DIR / "labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / f"{image_path.stem}.txt"

    if label_path.exists() and not overwrite:
        print(f"  пропуск (уже размечен): {image_path.name}")
        return -1

    boxes = model(str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0].boxes
    lines = [format_label(b) for b in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")

    target = BASE_DIR / "images" / split / image_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(image_path, target)

    if not lines:
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, review_dir / image_path.name)
        print(f"  ! ничего не найдено -> {review_dir.name}/{image_path.name}")

    print(f"  {image_path.name}: боксов {len(lines)}")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Авторазметка скриншотов через YOLO")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "raw",
        help="папка с сырыми кадрами (default: ./raw)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=BASE_DIR / "best.pt",
        help="путь к весам (default: ./best.pt)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="доля кадров в val; 0 = всё в train (default: 0.0)",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="перезаписывать уже существующие .txt",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Папка с кадрами не найдена: {args.input}")

    images = sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"В {args.input} нет изображений. Снимите кадры через screener.py")

    print(f"Модель: {args.weights}")
    print(f"Кадров: {len(images)}  |  conf={args.conf}  imgsz={args.imgsz}\n")

    model = YOLO(str(args.weights))
    review_dir = BASE_DIR / "needs_review"

    # Детерминированный шаг вместо random.shuffle: результат воспроизводим.
    val_count = int(len(images) * args.val_ratio)
    splits = ["val"] * val_count + ["train"] * (len(images) - val_count)

    empty = 0
    for i, (image_path, split) in enumerate(zip(images, splits), start=1):
        print(f"[{i}/{len(images)}] {split}")
        found = autolabel(
            model,
            image_path,
            split,
            args.conf,
            args.imgsz,
            review_dir,
            args.overwrite,
        )
        if found == 0:
            empty += 1

    print("\n" + "=" * 50)
    print(f"Готово. Обработано: {len(images)}")
    print(f"Без детекций: {empty} -> проверьте их вручную в {review_dir.name}/")
    if empty:
        print("Эти кадры модель не поняла. Разметьте их руками и повторите обучение.")
    print("Не забудьте удалить labels/*.cache перед новым обучением.")
    print("=" * 50)


if __name__ == "__main__":
    main()