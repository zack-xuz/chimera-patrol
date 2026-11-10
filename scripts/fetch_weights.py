"""下载 Chimera Patrol 所需的烟火与安全帽检测权重到 weights/ 目录。

权重不进入 git（.gitignore 已排除 weights/ 与 *.pt）。运行方式：

    python scripts/fetch_weights.py

每个文件下载后校验 SHA-256，校验失败则删除并报错退出。
person 权重 yolov8n.pt 由 Ultralytics 首次使用时自动下载，本脚本不处理。

来源与许可证见 README「权重获取」一节。
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"


@dataclass(frozen=True)
class WeightSource:
    name: str
    url: str
    sha256: str
    note: str


SOURCES: tuple[WeightSource, ...] = (
    WeightSource(
        name="fire_smoke.pt",
        url="https://github.com/andreasN78/fire-smoke-detection-yolov8/raw/main/weights/best.pt",
        sha256="22091935e9dbd7bd02a6a923f77875db599bafc0c784fe624de3c387925dba15",
        note="YOLOv8s 烟火检测，类别 Fire / smoke（另有一个 default 类别，规则中忽略）。AGPL-3.0。",
    ),
    WeightSource(
        name="helmet.pt",
        url="https://huggingface.co/iaac-maai/day4-example-ppe-detector/resolve/main/model.pt",
        sha256="8c03c378eeaa80f16e688142730717eca6963e7455bf264b281e1a101a50ebf7",
        note="YOLO26s 工地 PPE 检测，类别含 Hardhat / NO-Hardhat。AGPL-3.0，数据集 CC BY 4.0。",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: WeightSource, dest_dir: Path = WEIGHTS_DIR) -> Path:
    dest = dest_dir / source.name
    if dest.exists() and _sha256(dest) == source.sha256:
        print(f"[skip] {source.name} 已存在且校验通过")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[get ] {source.name} <- {source.url}")
    with urllib.request.urlopen(source.url, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    actual = _sha256(tmp)
    if actual != source.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{source.name} SHA-256 不匹配：期望 {source.sha256}，实际 {actual}")
    tmp.replace(dest)
    print(f"[ok  ] {source.name} {dest.stat().st_size} 字节")
    return dest


def main() -> int:
    for source in SOURCES:
        fetch(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
