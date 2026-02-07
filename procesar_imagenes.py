#!/usr/bin/env python3
"""Procesa un conjunto de imágenes para entrenamiento.

1) Convierte cada imagen en cuadrada rellenando con contenido reflejado de los bordes
   para que el resultado se vea natural.
2) Si la imagen resultante supera el tamaño objetivo, la reduce a 1024x1024 con
   remuestreo de alta calidad (LANCZOS).

Uso rápido:
    python procesar_imagenes.py ./mis_imagenes ./salida

También puedes usar banderas explícitas:
    python procesar_imagenes.py --input-dir ./mis_imagenes --output-dir ./salida

Modo interactivo:
    python procesar_imagenes.py --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _mirror_tile(strip: Image.Image, target_width: int, horizontal: bool) -> Image.Image:
    """Genera una franja del tamaño objetivo alternando espejo para evitar cortes bruscos."""
    if horizontal:
        tile_size = strip.width
    else:
        tile_size = strip.height

    if tile_size <= 0:
        raise ValueError("La franja de borde tiene tamaño inválido.")

    pieces: list[Image.Image] = []
    consumed = 0
    flipped = False
    while consumed < target_width:
        piece = strip.transpose(Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM) if flipped else strip
        remaining = target_width - consumed
        if horizontal:
            if piece.width > remaining:
                piece = piece.crop((0, 0, remaining, piece.height))
            consumed += piece.width
        else:
            if piece.height > remaining:
                piece = piece.crop((0, 0, piece.width, remaining))
            consumed += piece.height
        pieces.append(piece)
        flipped = not flipped

    if len(pieces) == 1:
        return pieces[0]

    if horizontal:
        out = Image.new(strip.mode, (target_width, strip.height))
        x = 0
        for p in pieces:
            out.paste(p, (x, 0))
            x += p.width
        return out

    out = Image.new(strip.mode, (strip.width, target_width))
    y = 0
    for p in pieces:
        out.paste(p, (0, y))
        y += p.height
    return out


def square_with_mirrored_edges(img: Image.Image) -> Image.Image:
    """Convierte una imagen rectangular en cuadrada extendiendo bordes con espejo."""
    w, h = img.size
    if w == h:
        return img

    side = max(w, h)
    canvas = Image.new(img.mode, (side, side))
    x_off = (side - w) // 2
    y_off = (side - h) // 2

    canvas.paste(img, (x_off, y_off))

    if w < side:
        # Falta ancho: rellenar izquierda y/o derecha con espejo de los bordes verticales.
        left_pad = x_off
        right_pad = side - (x_off + w)

        strip_w = min(max(1, w // 8), w)
        left_strip = img.crop((0, 0, strip_w, h))
        right_strip = img.crop((w - strip_w, 0, w, h))

        if left_pad > 0:
            fill = _mirror_tile(left_strip.transpose(Image.FLIP_LEFT_RIGHT), left_pad, horizontal=True)
            canvas.paste(fill, (0, y_off))
        if right_pad > 0:
            fill = _mirror_tile(right_strip.transpose(Image.FLIP_LEFT_RIGHT), right_pad, horizontal=True)
            canvas.paste(fill, (x_off + w, y_off))

    if h < side:
        # Falta alto: rellenar arriba y/o abajo con espejo de los bordes horizontales.
        top_pad = y_off
        bottom_pad = side - (y_off + h)

        strip_h = min(max(1, h // 8), h)
        top_strip = canvas.crop((0, y_off, side, y_off + strip_h))
        bottom_strip = canvas.crop((0, y_off + h - strip_h, side, y_off + h))

        if top_pad > 0:
            fill = _mirror_tile(top_strip.transpose(Image.FLIP_TOP_BOTTOM), top_pad, horizontal=False)
            canvas.paste(fill, (0, 0))
        if bottom_pad > 0:
            fill = _mirror_tile(bottom_strip.transpose(Image.FLIP_TOP_BOTTOM), bottom_pad, horizontal=False)
            canvas.paste(fill, (0, y_off + h))

    return canvas


def resize_if_needed(img: Image.Image, target_size: int, upscale: bool) -> Image.Image:
    if img.width == target_size and img.height == target_size:
        return img

    if img.width > target_size or img.height > target_size or upscale:
        return img.resize((target_size, target_size), Image.Resampling.LANCZOS, reducing_gap=3.0)

    return img


def process_image(src: Path, dst: Path, target_size: int, upscale: bool, quality: int) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = square_with_mirrored_edges(img)
        img = resize_if_needed(img, target_size=target_size, upscale=upscale)

        dst.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": quality, "optimize": True, "subsampling": 0})
        elif dst.suffix.lower() == ".png":
            save_kwargs.update({"optimize": True})

        img.save(dst, **save_kwargs)


def build_output_path(src: Path, input_dir: Path, output_dir: Path, out_format: str | None) -> Path:
    rel = src.relative_to(input_dir)
    if out_format:
        return output_dir / rel.with_suffix(f".{out_format.lower()}")
    return output_dir / rel



def prompt_for_path(message: str, default: Path) -> Path:
    raw = input(f"{message} [{default}]: ").strip()
    return Path(raw).expanduser() if raw else default


def interactive_paths() -> tuple[Path, Path]:
    cwd = Path.cwd()
    print("Modo interactivo de rutas")
    print("1) Usar rutas personalizadas")
    print("2) Usar carpeta actual como entrada y ./salida_procesada como salida")

    choice = input("Selecciona una opción [1/2] (default 2): ").strip() or "2"

    if choice == "1":
        input_dir = prompt_for_path("Ruta de entrada", cwd)
        output_dir = prompt_for_path("Ruta de salida", cwd / "salida_procesada")
        return input_dir, output_dir

    if choice == "2":
        return cwd, cwd / "salida_procesada"

    raise SystemExit("Opción inválida en modo interactivo. Usa 1 o 2.")

def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.interactive:
        return interactive_paths()

    input_dir = args.input_dir or args.input_dir_flag
    output_dir = args.output_dir or args.output_dir_flag

    if input_dir is None or output_dir is None:
        raise SystemExit(
            "Debes indicar rutas de entrada y salida o usar --interactive.\n"
            "Ejemplo: python procesar_imagenes.py ./imagenes ./salida\n"
            "O:       python procesar_imagenes.py --input-dir ./imagenes --output-dir ./salida\n"
            "O:       python procesar_imagenes.py --interactive"
        )

    return input_dir, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte imágenes a formato cuadrado y limita tamaño para datasets de entrenamiento. "
            "Toma automáticamente archivos compatibles dentro de la carpeta de entrada (recursivo)."
        )
    )
    parser.add_argument("input_dir", nargs="?", type=Path, help="Carpeta con imágenes de entrada")
    parser.add_argument("output_dir", nargs="?", type=Path, help="Carpeta donde guardar imágenes procesadas")
    parser.add_argument("--input-dir", dest="input_dir_flag", type=Path, help="Alternativa explícita a input_dir")
    parser.add_argument("--output-dir", dest="output_dir_flag", type=Path, help="Alternativa explícita a output_dir")
    parser.add_argument("--interactive", action="store_true", help="Pregunta en consola las rutas o usa la ruta actual")
    parser.add_argument("--size", type=int, default=1024, help="Tamaño final objetivo (default: 1024)")
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Si se activa, también escala imágenes pequeñas hasta el tamaño objetivo.",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "jpeg", "png", "webp"],
        default=None,
        help="Formato de salida opcional. Si no se define, mantiene extensión original.",
    )
    parser.add_argument("--quality", type=int, default=95, help="Calidad para JPG (default: 95)")

    args = parser.parse_args()
    input_dir, output_dir = resolve_paths(args)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"La carpeta de entrada no existe o no es válida: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        extensiones = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(
            "No se encontraron imágenes compatibles en la carpeta de entrada. "
            f"Extensiones soportadas: {extensiones}"
        )

    print(f"Entrada: {input_dir}")
    print(f"Salida:  {output_dir}")
    print(f"Detectadas {len(images)} imágenes compatibles")

    for src in images:
        dst = build_output_path(src, input_dir, output_dir, args.format)
        process_image(src, dst, target_size=args.size, upscale=args.upscale, quality=args.quality)

    print(f"Procesadas {len(images)} imágenes en: {output_dir}")


if __name__ == "__main__":
    main()
