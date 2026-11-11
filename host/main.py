#!/usr/bin/env python3
"""python main.py [--config config.yaml]

Starts the Active Camera and the official XR Teleoperation as independent processes, starts
the read-only recorder, and gives you r / s / q. See README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1rec import config as config_mod          # noqa: E402
from g1rec.orchestrator import Orchestrator      # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = HERE / "config.yaml"
    parser.add_argument("--config", default=str(default) if default.exists() else None)
    parser.add_argument("--no-act-check", action="store_true",
                        help="skip the ACT forward-pass check after each saved episode")
    args = parser.parse_args(argv)
    cfg = config_mod.load(args.config)
    orchestrator = Orchestrator(cfg, args.config, act_check=not args.no_act_check)
    try:
        if not orchestrator.start_all():
            return 1
        orchestrator.loop()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
