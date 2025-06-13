from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("torch-segment-membranes-2d"),
    base_url="doi:10.5281/zenodo.15660525",
    registry={"epoch=4-step=2000.ckpt": "md5:b4455a96b86c07ef5cf9dbc0b08e48c2"},
)


def get_latest_checkpoint() -> Path:
    """Retrieve the latest checkpoint from cache if available or download."""
    checkpoint_file = Path(GOODBOY.fetch("epoch=4-step=2000.ckpt"))
    return checkpoint_file
