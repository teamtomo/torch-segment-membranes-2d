from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("torch-segment-membranes-2d"),
    base_url="doi:10.5281/zenodo.15700479",
    registry={"epoch=12-step=5000.ckpt": "md5:ddcebb73df1270ce432a72779df672df"},
)


def get_latest_checkpoint() -> Path:
    """Retrieve the latest checkpoint from cache if available or download."""
    checkpoint_file = Path(GOODBOY.fetch("epoch=12-step=5000.ckpt"))
    return checkpoint_file
