"""Run the person detection -> tracking -> Re-ID -> gallery pipeline.

    python main.py                                      # every camera in config.yaml
    python main.py --source ../videos/test1.mp4         # one camera, offline test on a file
    python main.py --source ../videos/a.mp4 ../videos/b.mp4   # two cameras (also accepts a.mp4,b.mp4)
    python main.py --no-show --save out.mp4             # headless, record out_cam1.mp4, out_cam2.mp4
    python main.py --source a.mp4 b.mp4 --gallery shared   # one gallery for all cameras: same person, same ID
"""

from __future__ import annotations

import argparse
import os
import signal
from dataclasses import replace
from pathlib import Path

from person_pipeline import CameraConfig, PersonPipeline, load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="config file")
    p.add_argument(
        "--source",
        nargs="+",
        help="RTSP URLs or video files, one per camera, separated by spaces or commas; replaces the cameras' urls "
        "in order and runs only that many cameras",
    )
    p.add_argument("--backend", choices=["opencv", "gstreamer"], help="override source.backend")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--save", help="record the annotated video to this path (one file per camera)")
    p.add_argument("--gallery", choices=["per_camera", "shared"], help="override gallery.mode")
    p.add_argument("--no-cupola", action="store_true", help="do not push boxes to Cupola")
    return p.parse_args()


def override_sources(cameras: list[CameraConfig], sources: list[str]) -> list[CameraConfig]:
    """Camera i gets sources[i]; keeps its name and Cupola target. Extra sources become cams without Cupola push."""
    urls = [u.strip() for arg in sources for u in arg.split(",") if u.strip()]
    out = []
    for i, url in enumerate(urls):
        url = os.path.abspath(url) if os.path.isfile(url) else url
        out.append(replace(cameras[i], url=url) if i < len(cameras) else CameraConfig(name=f"cam{i + 1}", url=url))
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.cameras = override_sources(cfg.cameras, args.source)
    if args.backend:
        cfg.source.backend = args.backend
    if args.no_show:
        cfg.display.show = False
    if args.gallery:
        cfg.gallery.mode = args.gallery
    if args.no_cupola:
        cfg.cupola.enabled = False
    if args.save:
        cfg.display.save_path = os.path.abspath(args.save)

    pipeline = PersonPipeline(cfg)

    def on_signal(signum, _frame):
        print(f"[main] {signal.Signals(signum).name}, stopping")
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
