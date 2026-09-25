"""Package runnable code, model reports, and selected local checkpoints."""

import hashlib
from zipfile import ZIP_DEFLATED, ZipFile

from galaxeye.settings import ROOT

FILES = [
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    ROOT / ".python-version",
    ROOT / ".streamlit" / "config.toml",
    *sorted((ROOT / "src").rglob("*.py")),
    *sorted((ROOT / "scripts").glob("*.py")),
    *sorted((ROOT / "artifacts" / "v1").glob("*.json")),
    ROOT / "artifacts" / "v1" / "model.pt",
    *sorted((ROOT / "artifacts" / "v2").glob("*.json")),
    ROOT / "artifacts" / "v2" / "model.pt",
    *sorted((ROOT / "artifacts" / "v3").glob("*.json")),
    ROOT / "artifacts" / "v3" / "model.pt",
]


def main() -> None:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package missing files: {missing}")
    output = ROOT / "dist" / "GalaxEye-Backend-ML-Systems-Local.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in FILES:
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(output) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt archive member: {bad_member}")
        names = set(archive.namelist())
        for required in (
            "artifacts/v1/model.pt",
            "artifacts/v2/model.pt",
            "artifacts/v3/model.pt",
            "README.md",
        ):
            if required not in names:
                raise RuntimeError(f"Archive missing {required}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Created {output} ({output.stat().st_size:,} bytes, SHA-256 {digest})")


if __name__ == "__main__":
    main()
