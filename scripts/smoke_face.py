"""Face and Liveness for real: buffalo_l on real photographs, through the HTTP API.

Speaks only HTTP to `app.main:app`, so it runs unchanged against any revision -
and its `cases` output, compared between two revisions, is the Face regression
check: same photographs, same model, same scores, or something moved.

The photographs are the samples that ship inside the insightface package: one
portrait (the enrolled person) and a group photo (the impostor comes from it).
Variants stand in for five enrolment shots and a capture.

    .venv/bin/python scripts/smoke_face.py --output face.json [--preimport av]

Also measures the Face runtime's memory: idle, buffalo_l loaded, peak while
verifying, and after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


def rss_mb() -> float:
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
    import subprocess

    command = ["ps", "-o", "rss=", "-p", str(os.getpid())]
    return int(subprocess.run(command, capture_output=True, text=True).stdout.strip()) / 1024


class PeakRss:
    def __init__(self) -> None:
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> PeakRss:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, rss_mb())
        self.peak = max(self.peak, rss_mb())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preimport", help="Import this module first (e.g. av).")
    args = parser.parse_args()

    if args.preimport:
        __import__(args.preimport)

    os.environ.setdefault("FSA_API_KEYS", "smoke:smoke-key")
    os.environ.setdefault("FSA_CHALLENGE_SECRET", "smoke-secret")
    os.environ.setdefault("FSA_RUNTIME_ROLE", "face")  # ignored by revisions without roles
    os.environ["FSA_WARM_UP_ON_STARTUP"] = "true"

    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from insightface.data import get_image

    from app.api.deps import get_face_engine
    from app.main import app

    memory = {"rss_idle_mb": round(rss_mb(), 1)}

    def png(image: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()

    def portrait(image: np.ndarray, size: int = 320) -> np.ndarray:
        """A tight aligned crop, given the margin a detector expects."""
        padded = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_REPLICATE)
        return cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)

    def variant(image: np.ndarray, *, angle=0.0, brightness=0, flip=False, zoom=1.0):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, zoom)
        out = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=brightness)
        return cv2.flip(out, 1) if flip else out

    def clip(image: np.ndarray, frames: int = 40) -> bytes:
        path = Path(tempfile.mkdtemp()) / "clip.mp4"
        height, width = image.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        for index in range(frames):
            shift = np.float32([[1, 0, 3 * np.sin(index / 4)], [0, 1, 2 * np.cos(index / 5)]])
            frame = cv2.warpAffine(image, shift, (width, height), borderMode=cv2.BORDER_REPLICATE)
            writer.write(frame)
        writer.release()
        return path.read_bytes()

    person = portrait(get_image("Tom_Hanks_54745"))
    enrolment = [
        variant(person),
        variant(person, flip=True),
        variant(person, brightness=25),
        variant(person, angle=6),
        variant(person, zoom=0.9, brightness=-15),
    ]
    capture = variant(person, angle=-5, brightness=10)

    headers = {"X-API-Key": os.environ["FSA_API_KEYS"].split(",")[0].split(":", 1)[-1]}
    results: dict = {}

    with TestClient(app, headers=headers) as client:  # lifespan loads buffalo_l
        memory["rss_model_loaded_mb"] = round(rss_mb(), 1)

        group = get_image("t1")
        faces = get_face_engine().detect(group)
        x1, y1, x2, y2 = faces[0].bbox
        margin = int(0.4 * (x2 - x1))
        stranger = cv2.resize(
            group[max(0, y1 - margin) : y2 + margin, max(0, x1 - margin) : x2 + margin],
            (320, 320),
            interpolation=cv2.INTER_CUBIC,
        )
        refs = [
            ("images", (f"ref{index}.png", png(image), "image/png"))
            for index, image in enumerate(enrolment)
        ]

        def still(name: str, image: np.ndarray) -> None:
            response = client.post(
                "/api/v1/face/verify-image",
                files=[*refs, ("image", ("capture.png", png(image), "image/png"))],
            )
            assert response.status_code == 200, response.text
            body = response.json()
            results[name] = {
                "decision": body["decision"],
                "passed": body["passed"],
                "score": round(body["score"], 6),
                "similarities": [
                    None if s is None else round(s, 6)
                    for s in (i["capture_similarity"] for i in body["reference"]["images"])
                ],
            }

        def video(name: str, image: np.ndarray, token: str | None = None) -> None:
            data = {"challenge_token": token} if token else {}
            response = client.post(
                "/api/v1/face/verify",
                files=[*refs, ("video", ("clip.mp4", clip(image), "video/mp4"))],
                data=data,
            )
            assert response.status_code == 200, response.text
            body = response.json()
            results[name] = {
                "decision": body["decision"],
                "passed": body["passed"],
                "score": round(body["score"], 6),
                "frames_with_face": body["video"]["frames_with_face"],
                "match_ratio": round(body["video"]["match_ratio"], 6),
                # Action names are drawn at random; whether they were performed is not.
                "live": (body.get("liveness") or {}).get("live"),
            }

        with PeakRss() as peak:
            started = time.perf_counter()
            still("image_same_person", capture)
            still("image_impostor", stranger)
            video("video_same_person", person)
            video("video_impostor", stranger)

            challenge = client.post("/api/v1/liveness/challenge", json={}).json()
            response = client.post(
                "/api/v1/liveness/verify",
                data={"token": challenge["token"]},
                files={"video": ("clip.mp4", clip(person), "video/mp4")},
            )
            assert response.status_code == 200, response.text
            results["liveness_static_clip"] = {
                "live": response.json()["live"],
                "frames_with_face": response.json()["frames_with_face"],
            }
            token = client.post("/api/v1/liveness/challenge", json={}).json()["token"]
            video("video_with_challenge", person, token=token)
            memory["verify_seconds_total"] = round(time.perf_counter() - started, 1)

        memory["rss_peak_verifying_mb"] = round(peak.peak, 1)
        memory["rss_after_mb"] = round(rss_mb(), 1)

    expectations = {
        "image_same_person": results["image_same_person"]["decision"] == "match",
        "image_impostor": results["image_impostor"]["decision"] == "no_match",
        "video_same_person": results["video_same_person"]["decision"] == "match",
        "video_impostor": results["video_impostor"]["decision"] == "no_match",
        # A still photograph moved around performs no action.
        "liveness_static_clip": results["liveness_static_clip"]["live"] is False,
        "video_with_challenge": results["video_with_challenge"]["passed"] is False,
    }
    report = {
        "cases": results,
        "expectations": expectations,
        "memory": memory,
        "preimported": args.preimport,
        "modules": sorted(m for m in ("av", "cv2", "insightface") if m in sys.modules),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(expectations.values()):
        raise SystemExit(f"unexpected outcome: {expectations}")


if __name__ == "__main__":
    main()
