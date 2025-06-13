import numpy as np

import torch_segment_membranes_2d
from torch_segment_membranes_2d.model._pretrained_weights import get_latest_checkpoint
from torch_segment_membranes_2d.model import ResidualUNet18
from torch_segment_membranes_2d import predict_membrane_mask


def test_imports_with_version():
    assert isinstance(torch_segment_membranes_2d.__version__, str)


def test_model_from_latest_checkpoint():
    checkpoint_path = torch_segment_membranes_2d.model.get_latest_checkpoint()
    model = ResidualUNet18.load_from_checkpoint(checkpoint_path)
    assert isinstance(model, ResidualUNet18)


def test_predict():
    # smoke test for prediction API
    image = np.random.random((128, 128))
    result = predict_membrane_mask(
        image=image,
        pixel_spacing=8,
        probability_threshold=0.5
    )
    assert image.shape == result.shape
