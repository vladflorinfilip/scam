"""Build a four-column visual index of all lab-book figure PNGs."""

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "data/experiments"
OUTPUT = EXPERIMENTS / "lab_book/contact_sheet.png"
COLUMNS = 4
CELL_WIDTH = 720
CELL_HEIGHT = 440
IMAGE_HEIGHT = 390


def main():
    sources = sorted((EXPERIMENTS / "lab_book").glob("*.png"))
    sources += sorted((EXPERIMENTS / "close_gaps").glob("fig_*.png"))
    sources = [path for path in sources if path != OUTPUT]
    rows = ceil(len(sources) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)

    for index, path in enumerate(sources):
        column, row = index % COLUMNS, index // COLUMNS
        x, y = column * CELL_WIDTH, row * CELL_HEIGHT
        label = str(path.relative_to(EXPERIMENTS))
        draw.text((x + 12, y + 8), label, fill="#222222", font=font)
        with Image.open(path) as original:
            image = original.convert("RGB")
            image.thumbnail((CELL_WIDTH - 24, IMAGE_HEIGHT - 8), Image.Resampling.LANCZOS)
        image_x = x + (CELL_WIDTH - image.width) // 2
        image_y = y + 38 + (IMAGE_HEIGHT - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.rectangle((x + 4, y + 4, x + CELL_WIDTH - 5, y + CELL_HEIGHT - 5), outline="#CCCCCC")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)


if __name__ == "__main__":
    main()
