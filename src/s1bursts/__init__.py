"""Sentinel-1 burst download, SAR and InSAR processing.

A burst is the smallest independently processable unit of a Sentinel-1
acquisition. Working at that level keeps downloads small and makes dense
time-series processing of a fixed footprint practical.

This namespace holds what a pipeline is built from: the file manager, the
search, processing, and compositing services, and the configs they take.
"""

from importlib.metadata import version as _version

from s1bursts.burstmap import download_burst_map
from s1bursts.database import BurstDatabase
from s1bursts.dem import prepare_dem_for_snap
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
from s1bursts.time_range import TimeRange

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
    "TimeRange",
    "download_burst_map",
    "prepare_dem_for_snap",
]
