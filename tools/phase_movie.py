from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_EXTS = {".svg", ".png", ".jpg", ".jpeg"}


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    out = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part)
    return tuple(out)


def find_frames(input_dir: Path, pattern: str | None = None, limit: int | None = None) -> list[Path]:
    frames = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue
        frames.append(p)

    frames = sorted(frames, key=natural_key)

    if limit is not None:
        frames = frames[:limit]

    return frames


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def rasterize_svg_to_png(svg_path: Path, png_path: Path) -> None:
    try:
        import cairosvg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SVG input detected but cairosvg is not installed.\n"
            "Install it with: pip install cairosvg"
        ) from exc

    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))


def jpeg_to_png(src: Path, dst: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "JPEG input detected but Pillow is not installed.\n"
            "Install it with: pip install pillow"
        ) from exc

    img = Image.open(src)
    img.save(dst, format="PNG")


def normalize_to_png_sequence(frames: list[Path], temp_dir: Path, hold: int) -> list[Path]:
    png_frames: list[Path] = []
    frame_idx = 0

    for frame in frames:
        suffix = frame.suffix.lower()

        if suffix == ".svg":
            base_png = temp_dir / f"base_{frame_idx:06d}.png"
            rasterize_svg_to_png(frame, base_png)
        elif suffix == ".png":
            base_png = temp_dir / f"base_{frame_idx:06d}.png"
            shutil.copy2(frame, base_png)
        elif suffix in {".jpg", ".jpeg"}:
            base_png = temp_dir / f"base_{frame_idx:06d}.png"
            jpeg_to_png(frame, base_png)
        else:
            continue

        for _ in range(max(1, hold)):
            out_png = temp_dir / f"frame_{frame_idx:06d}.png"
            shutil.copy2(base_png, out_png)
            png_frames.append(out_png)
            frame_idx += 1

    return png_frames


def build_mp4_with_ffmpeg(temp_dir: Path, output_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def build_gif_with_ffmpeg(temp_dir: Path, output_path: Path, fps: int) -> None:
    palette = temp_dir / "palette.png"

    cmd_palette = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-vf",
        "palettegen",
        str(palette),
    ]
    subprocess.run(cmd_palette, check=True)

    cmd_gif = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-i",
        str(palette),
        "-lavfi",
        "paletteuse",
        str(output_path),
    ]
    subprocess.run(cmd_gif, check=True)


def build_gif_with_pillow(png_frames: list[Path], output_path: Path, fps: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable and Pillow is not installed.\n"
            "Install Pillow with: pip install pillow"
        ) from exc

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in png_frames]
    if not images:
        raise RuntimeError("No PNG frames available for GIF creation.")

    duration_ms = max(20, int(1000 / max(1, fps)))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GIF or MP4 from PAM Observatory screenshots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("tui/screenshots"),
        help="Directory containing screenshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tui/screenshots/phase_movie.gif"),
        help="Output file path (.gif or .mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second.",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=1,
        help="Duplicate each source frame this many times.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='Optional filename glob filter, e.g. "obs_r*.svg".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of source frames.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("tui/screenshots/.phase_movie_tmp"),
        help="Temporary working directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    frames = find_frames(args.input_dir, pattern=args.pattern, limit=args.limit)
    if not frames:
        print(f"No supported frames found in {args.input_dir}", file=sys.stderr)
        return 1

    ensure_clean_dir(args.temp_dir)
    png_frames = normalize_to_png_sequence(frames, args.temp_dir, hold=args.hold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()

    if suffix == ".mp4":
        if not ffmpeg_available():
            print("ffmpeg is required for MP4 output.", file=sys.stderr)
            return 1
        build_mp4_with_ffmpeg(args.temp_dir, args.output, args.fps)

    elif suffix == ".gif":
        if ffmpeg_available():
            build_gif_with_ffmpeg(args.temp_dir, args.output, args.fps)
        else:
            build_gif_with_pillow(png_frames, args.output, args.fps)

    else:
        print("Output file must end with .gif or .mp4", file=sys.stderr)
        return 1

    print(f"Wrote movie: {args.output}")
    print(f"Source frames: {len(frames)}")
    print(f"Expanded frames: {len(png_frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
