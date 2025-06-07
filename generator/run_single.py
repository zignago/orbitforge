import uuid
from pathlib import Path
from .mission import MissionSpec
from .basic_frame import build_basic_frame


def main(mission_json: str):
    with open(mission_json) as fp:
        spec = MissionSpec.model_validate_json(fp.read())
    out = Path("outputs") / f"design_{uuid.uuid4().hex[:4]}"
    out.mkdir(parents=True, exist_ok=True)
    build_basic_frame(spec, out)
    print("✓ generated frame at", out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mission_json")
    main(**vars(ap.parse_args()))
