import argparse
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".webm", ".mov"}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr[:2000]}")
    return p

def extract_frames_ffmpeg(video_path: Path, out_dir: Path, size: int, fps: int = None, max_frames: int = None):
    """
    Extract frames as JPEG, resized to size x size, 8-digit naming: 00000001.jpg, ...
    Optional fps to downsample; optional max_frames to truncate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "%08d.jpg")

    vf_parts = []
    # resize to square size x size
    vf_parts.append(f"scale={size}:{size}")
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vf", vf, "-q:v", "5", out_pattern]
    run(cmd)

    if max_frames is not None:
        # Keep only first max_frames
        frames = sorted(out_dir.glob("*.jpg"))
        for f in frames[max_frames:]:
            f.unlink()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmdb_root", type=str, required=True,
                    help="Path to HMDB51 extracted videos root: class_name/*.avi")
    ap.add_argument("--out_zip", type=str, required=True,
                    help="Output zip path, e.g. ~/trx_data/video_datasets/data/hmdb51_256q5.zip")
    ap.add_argument("--size", type=int, default=256, help="Resize frames to size x size (default 256)")
    ap.add_argument("--fps", type=int, default=None, help="Optional FPS sampling (e.g., 10). Default: keep native fps.")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap frames per video.")
    args = ap.parse_args()

    hmdb_root = Path(args.hmdb_root).expanduser().resolve()
    out_zip = Path(args.out_zip).expanduser().resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # collect class/video files
    class_dirs = [p for p in hmdb_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {hmdb_root}")

    video_list = []
    for cdir in sorted(class_dirs):
        for v in sorted(cdir.iterdir()):
            if v.is_file() and v.suffix.lower() in VIDEO_EXTS:
                video_list.append((cdir.name, v))

    if not video_list:
        raise RuntimeError(f"No video files found under {hmdb_root} with extensions {VIDEO_EXTS}")

    # Build zip by extracting each video to a temp folder, then writing frames into zip under:
    # <class>/<video_id>/<frame>.jpg
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for cls, vpath in tqdm(video_list, desc="Videos"):
                video_id = vpath.stem  # filename without extension
                out_dir = tmp_path / cls / video_id
                # clean any residue
                if out_dir.exists():
                    for f in out_dir.glob("*.jpg"):
                        f.unlink()
                extract_frames_ffmpeg(vpath, out_dir, size=args.size, fps=args.fps, max_frames=args.max_frames)

                frames = sorted(out_dir.glob("*.jpg"))
                if len(frames) == 0:
                    print(f"[WARN] no frames extracted: {vpath}")
                    continue

                for f in frames:
                    arcname = f"{cls}/{video_id}/{f.name}"
                    zf.write(f, arcname)

    print(f"Done. Wrote: {out_zip}")

if __name__ == "__main__":
    main()
