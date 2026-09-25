"""Exercise the same local HTTP path as an analyst using the viewer."""

import hashlib
from io import BytesIO

import httpx
from PIL import Image

from galaxeye.contracts import CLASSES
from galaxeye.settings import API_URL


def main() -> None:
    tile = Image.new("RGB", (64, 64))
    pixels = tile.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = ((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
    buffer = BytesIO()
    tile.save(buffer, format="PNG")
    raw = buffer.getvalue()
    expected_hash = hashlib.sha256(raw).hexdigest()
    with httpx.Client(base_url=API_URL, timeout=130) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["inference"]["ready"] is True
        results = {}
        for version in ("v1", "v2", "v3"):
            response = client.post(
                "/classify",
                params={"version": version},
                files={"file": ("canary.png", raw, "image/png")},
            )
            response.raise_for_status()
            result = response.json()
            assert result["image_sha256"] == expected_hash
            assert result["version"] == version
            assert result["predicted_class"] in CLASSES
            assert tuple(result["scores"]) == CLASSES
            assert abs(sum(result["scores"].values()) - 1) < 0.05
            repeat = client.post(
                "/classify",
                params={"version": version},
                files={"file": ("canary.png", raw, "image/png")},
            )
            repeat.raise_for_status()
            assert repeat.json()["id"] == result["id"]
            stored = client.get(f"/predictions/{result['id']}")
            stored.raise_for_status()
            assert stored.json()["image_sha256"] == expected_hash
            results[version] = result
        image = client.get(f"/images/{expected_hash}")
        image.raise_for_status()
        assert image.content == raw
        assert len({result["id"] for result in results.values()}) == len(results)
        broken = client.post(
            "/classify",
            files={"file": ("broken.png", b"not a PNG", "image/png")},
        )
        assert broken.status_code == 422
        history = client.get("/predictions", params={"limit": 100})
        history.raise_for_status()
        assert {result["id"] for result in results.values()} <= {
            item["id"] for item in history.json()
        }
    print(
        "Local upload, all CPU models, persistence, retry, query, image retrieval, and invalid input: PASS"
    )


if __name__ == "__main__":
    main()
