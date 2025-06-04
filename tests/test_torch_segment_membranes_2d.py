import torch_segment_membranes_2d


def test_imports_with_version():
    assert isinstance(torch_segment_membranes_2d.__version__, str)
