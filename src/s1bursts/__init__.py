"""Submodule for Sentinel-1 burst processing."""

from s1bursts.burst import Burst, BurstCollection, BurstPair, BurstPairCollection
from s1bursts.burstmap import download_burst_map
from s1bursts.database import BurstDatabase
from s1bursts.dimap import BeamDimapDataset
from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstNotFoundError,
    ConfigurationError,
    S1BurstsError,
    SnapProcessingError,
)
from s1bursts.filemanager import FileManager
from s1bursts.processor import BurstProcessor
from s1bursts.query import Mode, OrbitDirection, Polarization, SearchQuery, SubSwath
from s1bursts.resolution_weighter import ResolutionWeighter
from s1bursts.snap_client import SNAPConfig
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    GeocodingConfig,
    InterferogramConfig,
)
from s1bursts.summary import OperationSummary

__all__ = [
    "BackscatterConfig",
    "BeamDimapDataset",
    "Burst",
    "BurstCollection",
    "BurstDatabase",
    "BurstDownloadError",
    "BurstError",
    "BurstNotFoundError",
    "BurstPair",
    "BurstPairCollection",
    "BurstProcessor",
    "CoherenceConfig",
    "ConfigurationError",
    "FileManager",
    "GeocodingConfig",
    "InterferogramConfig",
    "Mode",
    "OperationSummary",
    "OrbitDirection",
    "Polarization",
    "ResolutionWeighter",
    "S1BurstsError",
    "SNAPConfig",
    "SearchQuery",
    "SnapProcessingError",
    "SubSwath",
    "download_burst_map",
]
