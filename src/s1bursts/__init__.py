"""Sentinel-1 burst download, SAR and InSAR processing.

A burst is the smallest independently processable unit of a Sentinel-1
acquisition. Working at that level keeps downloads small (~100 MB instead of
~4 GB per SLC scene) and makes dense time-series processing of a fixed
footprint practical.

This namespace holds what a pipeline is built from: the file manager, the
search, processing, and compositing services, and the configs they take. What
those services return lives alongside them, in ``s1bursts.burst``,
``s1bursts.product``, and ``s1bursts.summary``.
"""

from importlib.metadata import version as _version

from s1bursts.burstmap import download_burst_map
from s1bursts.database import BurstDatabase
from s1bursts.dem import compensate_dem_for_snap
from s1bursts.filemanager import FileManager
from s1bursts.processor import BurstProcessor
from s1bursts.query import SearchQuery
from s1bursts.resolution_weighter import ResolutionWeighter
from s1bursts.snap_client import SNAPConfig
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    GeocodingConfig,
    InterferogramConfig,
)

__version__ = _version("s1bursts")

__all__ = [
    "BackscatterConfig",
    "BurstDatabase",
    "BurstProcessor",
    "CoherenceConfig",
    "FileManager",
    "GeocodingConfig",
    "InterferogramConfig",
    "ResolutionWeighter",
    "SNAPConfig",
    "SearchQuery",
    "compensate_dem_for_snap",
    "download_burst_map",
]
