"""Run the person detection -> tracking -> Re-ID -> gallery pipeline.

    python main.py                                  # uses config.yaml next to this file
    python main.py --source ../videos/test1.mp4     # offline test on a file
    python main.py --no-show --save out.mp4         # headless, record the result
"""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

from person_pipeline import PersonPipeline, load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="config file")
    p.add_argument("--source", help="override source.url (RTSP URL or video file)")
    p.add_argument("--backend", choices=["opencv", "gstreamer"], help="override source.backend")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--save", help="record the annotated video to this path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.source.url = os.path.abspath(args.source) if os.path.isfile(args.source) else args.source
    if args.backend:
        cfg.source.backend = args.backend
    if args.no_show:
        cfg.display.show = False
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
