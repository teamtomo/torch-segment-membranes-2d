"""Segment membranes in cryo-EM images"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-segment-membranes-2d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
