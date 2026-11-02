"""Search and download Sentinel-1 bursts from the Copernicus Data Space Ecosystem (CDSE)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from freezebase.download import HTTPDownloader, retry_request
from freezebase.pandas_utils import clean_names
from freezebase.utils import EPSG_WGS84
import geopandas as gpd
import pandas as pd
from pydantic import SecretStr
import requests
from requests import PreparedRequest
from requests.auth import AuthBase

from s1bursts.burst_extractor import extract_burst_from_s3
from s1bursts.config import get_settings
from s1bursts.errors import CatalogError, ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from upath import UPath


TIMEOUT_DEFAULT = 30

MAX_PAGE_SIZE = 1000
"""Largest ``$top`` the CDSE catalog honours per request, and so the most one search returns."""

_RESPONSE_COLS_KEEP = [
    "Id",
    "Name",
    "S3Path",
    "ParentProductId",
    "ParentProductName",
    "ByteOffset",
    "BurstId",
    "AbsoluteBurstId",
    "AzimuthTime",
    "AzimuthAnxTime",
    "ParentProductType",
    "SwathIdentifier",
    "RelativeOrbitNumber",
    "OrbitDirection",
    "PlatformSerialIdentifier",
    "PolarisationChannels",
    "OperationalMode",
    "DatatakeID",
    "Lines",
    "LinesPerBurst",
    "SamplesPerBurst",
    "BeginningDateTime",
    "EndingDateTime",
]
_FOOTPRINT_COL = "Footprint"
_DATE_COLS = ["BeginningDateTime", "EndingDateTime", "AzimuthTime"]

logger = logging.getLogger(__name__)


_REFRESH_REJECTED_STATUS = frozenset({400, 401})
"""Identity-server responses that mean the refresh token is no longer accepted."""

_LOGIN_REJECTED_STATUS = frozenset({400, 401, 403})
"""Identity-server responses that mean the username and password are not accepted."""

# Longest error description of the identity server quoted in an error message.
_MAX_DESCRIPTION_CHARS = 200


def _error_description(response: requests.Response) -> str | None:
    """Return the ``error_description`` an OpenID Connect error response carries."""
    try:
        body = response.json()
    except ValueError:
        return None
    description = body.get("error_description") if isinstance(body, dict) else None
    if not isinstance(description, str) or not description.strip():
        return None
    return description.strip()[:_MAX_DESCRIPTION_CHARS]


def _without_request(error: requests.RequestException) -> requests.RequestException:
    """Copy a token-request error without its request, whose body holds a credential."""
    try:
        return type(error)(str(error))
    except TypeError:
        # A subclass whose constructor takes more than a message.
        return requests.RequestException(str(error))


class CDSEAuth(AuthBase):
    """CDSE authentication handler using OpenIDConnect bearer tokens.

    Automatically refreshes access tokens when expired.

    Parameters
    ----------
    username
        CDSE username used to request the initial token. If not provided, falls
        back to the ``CDSE_USERNAME`` environment variable.
    password
        CDSE password used to request the initial token. If not provided, falls
        back to the ``CDSE_PASSWORD`` environment variable.

    Raises
    ------
    ConfigurationError
        If credentials are not provided as arguments and not found in the
        environment, or if CDSE rejects them (HTTP 400, 401 or 403).
    requests.RequestException
        If the token request fails otherwise. The error keeps neither the
        request nor the response, so that the password does not travel with it.
    """

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        if username is None or password is None:
            cdse = get_settings().cdse
            if username is None:
                username = cdse.username
            if password is None and cdse.password is not None:
                password = cdse.password.get_secret_value()

        if username is None or password is None:
            msg = (
                "CDSE credentials not configured. Pass username/password explicitly, "
                "or set CDSE_USERNAME and CDSE_PASSWORD in your environment."
            )
            raise ConfigurationError(msg)

        self._username = username
        self._password = SecretStr(password)
        self.id_endpoint = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        # Timeout for request.post() in seconds
        self.request_timeout = TIMEOUT_DEFAULT
        # Buffer in seconds before token is refreshed
        self.expire_buffer = 60
        self._lock = threading.Lock()
        self._get_initial_tokens()

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        """Attach the current access token to the given HTTP request.

        Parameters
        ----------
        r
            The request object to modify.

        Returns
        -------
        requests.PreparedRequest
            The modified request with the `Authorization` header set.
        """
        if self._token_expired():
            with self._lock:
                # Double-check: token might have been refreshed while we waited for the lock
                if self._token_expired():
                    self._refresh_access_token()
        r.headers["Authorization"] = f"Bearer {self.access_token}"
        return r

    def _token_expired(self) -> bool:
        """Check whether the current access token has expired."""
        return time.time() >= self.token_expires_at

    def _refresh_token_expired(self) -> bool:
        """Check whether the current refresh token has expired."""
        return time.time() >= self.refresh_expires_at

    def _get_initial_tokens(self) -> None:
        """Request the initial access and refresh tokens, keeping the password out of errors."""
        failure: requests.RequestException | None = None
        try:
            self._request_initial_tokens()
        except requests.RequestException as e:
            failure = _without_request(e)
        # Raised outside the except block, so the original is not kept as its context.
        if failure is not None:
            raise failure

    @retry_request(logger=logger)
    def _request_initial_tokens(self) -> None:
        """Request the initial access and refresh tokens with the password grant."""
        data = {
            "client_id": "cdse-public",
            "username": self._username,
            "password": self._password.get_secret_value(),
            "grant_type": "password",
        }
        response = requests.post(
            self.id_endpoint,
            data=data,
            timeout=self.request_timeout,
        )
        if response.status_code in _LOGIN_REJECTED_STATUS:
            # Raised outside an except block, so no chained HTTPError carries the
            # request, whose body holds the password.
            description = _error_description(response)
            reason = f": {description}" if description else ""
            msg = (
                f"CDSE rejected the username and password (HTTP {response.status_code}"
                f"{reason}). Check CDSE_USERNAME and CDSE_PASSWORD, or the credentials "
                "passed to CDSEAuth."
            )
            raise ConfigurationError(msg)
        response.raise_for_status()
        self._update_tokens(response.json())

    def _refresh_access_token(self) -> None:
        """Refresh the access token, keeping the refresh token out of errors."""
        failure: requests.RequestException | None = None
        try:
            self._request_refreshed_tokens()
        except requests.RequestException as e:
            failure = _without_request(e)
        # Raised outside the except block, so the original is not kept as its context.
        if failure is not None:
            raise failure

    @retry_request(logger=logger)
    def _request_refreshed_tokens(self) -> None:
        """Refresh the access token using the refresh token."""
        if self._refresh_token_expired():
            logger.info("Refresh token expired, re-authenticating with stored credentials.")
            self._get_initial_tokens()
            return
        data = {
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        response = requests.post(
            self.id_endpoint,
            data=data,
            timeout=self.request_timeout,
        )
        if response.status_code in _REFRESH_REJECTED_STATUS:
            # The identity server invalidated the refresh token before its nominal
            # expiry (session limits, token rotation). Retrying the refresh would
            # fail forever, so fall back to the password grant.
            logger.info(
                "Refresh token rejected (HTTP %d), re-authenticating with stored credentials.",
                response.status_code,
            )
            self._get_initial_tokens()
            return
        response.raise_for_status()
        self._update_tokens(response.json())

    def _update_tokens(self, token_data: dict[str, Any]) -> None:
        """Update tokens properties from token response."""
        self.access_token = str(token_data["access_token"])
        self.refresh_token = str(token_data["refresh_token"])
        # Calculate token expiration
        now = time.time()
        token_expires_in = float(token_data["expires_in"])
        refresh_expires_in = float(token_data["refresh_expires_in"])
        self.token_expires_at = now + (token_expires_in) - self.expire_buffer
        self.refresh_expires_at = now + refresh_expires_in - self.expire_buffer


@dataclass
class CDSEResponse:
    """Response object for Sentinel-1 burst search operations.

    Parameters
    ----------
    context
        OData context metadata string (usually includes entity type).
    value
        List of result items returned by the search.
    count
        Total number of matches on the server (if requested).
    """

    context: str
    value: list[dict[str, Any]]
    count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CDSEResponse:
        """Create a CDSEResponse from a raw API response dictionary.

        Parameters
        ----------
        data
            JSON response dictionary from the search API.

        Returns
        -------
        CDSEResponse
            Parsed CDSEResponse instance.
        """
        return cls(
            context=data["@odata.context"],
            value=data["value"],
            count=data.get("@odata.count"),
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the search results to a GeoPandas GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial representation of the search results.

        Raises
        ------
        CatalogError
            If an unknown EPSG code is found in the geometry definitions.
        """
        df_results = pd.DataFrame(self.value)
        geometry = gpd.GeoSeries([], crs=EPSG_WGS84, dtype="geometry")

        if not df_results.empty:
            # Bursts crossing the antimeridian come back as MULTIPOLYGON.
            extracted = df_results[_FOOTPRINT_COL].str.extract(
                r"SRID=(\d+);((?:MULTI)?POLYGON \(.*\))",
            )
            epsg_codes = extracted[0].astype(int)
            if not (epsg_codes == EPSG_WGS84).all():
                msg = f"Found unknown EPSG codes:\n{epsg_codes}"
                raise CatalogError(msg)
            geometry = gpd.GeoSeries.from_wkt(extracted[1], crs=EPSG_WGS84)

        df_results = df_results.reindex(columns=_RESPONSE_COLS_KEEP)
        df_results[_DATE_COLS] = df_results[_DATE_COLS].apply(pd.to_datetime)
        gdf = gpd.GeoDataFrame(df_results, geometry=geometry, crs=EPSG_WGS84)
        return clean_names(gdf, case_type="snake")

    def __len__(self) -> int:
        """Return the number of items in the search result.

        Returns
        -------
        int
            Number of bursts returned.
        """
        return len(self.value)

    def __repr__(self) -> str:
        """Return a human-readable summary of the CDSEResponse.

        Returns
        -------
        str
            A string representation showing count and entity type.
        """
        # Extract entity type from context (e.g., "Bursts" from "$metadata#Bursts")
        entity_type = self.context.split("#")[-1] if "#" in self.context else "Unknown"

        return (
            f"CDSEResponse({len(self.value)}/{self.count} {entity_type})"
            if self.count is not None
            else f"CDSEResponse({len(self.value)} {entity_type})"
        )

    def is_complete(self) -> bool:
        """Check whether the response contains all available bursts that match the query.

        Returns
        -------
        bool
            True if response length is equal to count.

        Raises
        ------
        CatalogError
            If count is None.
        """
        if self.count is None:
            msg = (
                "Cannot determine completeness because count is None. "
                "Set `count=True` during burst search."
            )
            raise CatalogError(msg)
        return len(self) == self.count


class CDSEBurstClient:
    """Client for searching and downloading Sentinel-1 bursts from CDSE using the OData API.

    Implemented according to the CDSE API documentation [1].

    Parameters
    ----------
    auth
        Authentication instance. Required for downloading bursts.

    References
    ----------
    .. [1] Sentinel-1 SLC Burst API. Copernicus Data Space Ecosystem documentation.
       https://documentation.dataspace.copernicus.eu/APIs/Sentinel-1%20SLC%20Burst.html
    """

    def __init__(self, auth: CDSEAuth | None = None) -> None:
        self.auth = auth
        self.catalog_endpoint = "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
        self.download_endpoint = "https://bursts.dataspace.copernicus.eu/v1/product"
        self.request_timeout = TIMEOUT_DEFAULT

    def _build_query_params(
        self,
        filters: Iterable[str] | None = None,
        *,
        ascending: bool = True,
        count: bool | None = None,
    ) -> dict[str, Any]:
        """Construct OData query parameters from filter expressions."""
        params: dict[str, Any] = {}
        if filters is not None:
            params["$filter"] = " and ".join(filters)
        params["$top"] = MAX_PAGE_SIZE
        params["$orderby"] = "ContentDate/Start" if ascending else "ContentDate/Start desc"
        if count is not None:
            params["$count"] = count
        return params

    # Search has no fallback endpoint, so retry its intermittent HTTP 500 responses.
    @retry_request(logger=logger, extra_status_codes=frozenset({500}))
    def search_bursts(
        self,
        filters: list[str] | None = None,
        *,
        ascending: bool = True,
        count: bool | None = None,
    ) -> CDSEResponse:
        """Search for Sentinel-1 bursts matching specified criteria.

        Returns a single page of at most ``MAX_PAGE_SIZE`` items, the most the
        catalog serves per request. Request ``count`` to tell whether more
        bursts match than the page holds.

        Parameters
        ----------
        filters
            List of OData filter expressions to apply.
        ascending
            Whether to sort results by start time in ascending
            or descending order.
        count
            Whether to include a total result count in the response.

        Returns
        -------
        CDSEResponse
            Structured response with result metadata.

        Raises
        ------
        CatalogError
            If the API response is malformed or invalid.
        requests.HTTPError
            If the search request fails.
        """
        query = self._build_query_params(filters=filters, ascending=ascending, count=count)
        # The catalog endpoint requires no authentication
        response = requests.get(
            url=self.catalog_endpoint,
            params=query,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            msg = f"Failed to query bursts: \n{response.text}"
            raise CatalogError(msg)
        json_ = response.json()
        if not isinstance(json_, dict) or json_.get("@odata.context") != "$metadata#Bursts":
            msg = f"Failed to query bursts: \n{json_}"
            raise CatalogError(msg)
        return CDSEResponse.from_dict(json_)

    def download_burst(
        self,
        cdse_id: str,
        save_path: Path | UPath,
        *,
        progress: bool = True,
    ) -> Path | UPath:
        """Download a single Sentinel-1 burst by CDSE ID.

        Supports both local and remote (e.g. S3 via UPath) destinations.

        Parameters
        ----------
        cdse_id
            The burst's CDSE product ID.
        save_path
            Full destination path (including filename) for the downloaded file.
            Can be local or a remote UPath (e.g. S3). A remote file is first
            downloaded whole to the system temp directory and uploaded only
            once complete, so it needs local space for the burst (about
            100 to 200 MB).
        progress
            Whether to display download progress.

        Returns
        -------
        Path or UPath
            Path of the downloaded file.

        Raises
        ------
        TypeError
            If the client has no valid CDSEAuth instance.
        requests.HTTPError
            If the download request fails.
        freezebase.errors.DownloadError
            If the response is not a file, or redirects without a host or too often.
        """
        url = f"{self.download_endpoint}/{cdse_id}"
        if not (isinstance(self.auth, CDSEAuth)):
            msg = "`auth` not valid. Must be an instance of 'CDSEAuth' to download bursts."
            raise TypeError(msg)

        # Closed on exit, releasing the pooled connection immediately
        with HTTPDownloader(
            auth=self.auth,
            progress=progress,
            method="POST",
            timeout=self.request_timeout,
        ) as downloader:
            return downloader(url=url, save_dir=save_path.parent, filename=save_path.name)

    def download_burst_from_s3(
        self,
        save_path: Path | UPath,
        *,
        parent_product_name: str,
        polarization: str,
        sub_swath: str,
        burst_id: int,
    ) -> Path | UPath:
        """Download a single Sentinel-1 burst from CDSE S3.

        Extracts the burst from the parent product with s3fs and rasterio,
        writing the result as a ``.zip`` to ``save_path`` (local or remote).

        Parameters
        ----------
        save_path
            Full destination path (including filename) for the ``.zip``.
            Can be local or a remote UPath (e.g. S3).
        parent_product_name
            Name of the parent Sentinel-1 product.
        polarization
            Polarization channels (e.g., 'VV', 'VH', 'HH', 'HV').
        sub_swath
            Sub-swath identifier (e.g., 'IW1', 'IW2', 'IW3').
        burst_id
            Relative burst ID within the swath.

        Returns
        -------
        Path or UPath
            Path of the produced ``.zip`` file.

        Raises
        ------
        ValueError
            If ``sub_swath`` is not a recognised sub-swath identifier.
        BurstNotFoundError
            If the parent product or the requested burst cannot be found.
        BurstDownloadError
            If the product's annotation, calibration, noise, RFI or manifest
            file is malformed.
        ConfigurationError
            If the CDSE S3 credentials are not configured.
        """
        return extract_burst_from_s3(
            save_path,
            parent_product_name=parent_product_name,
            polarization=polarization,
            sub_swath=sub_swath,
            burst_id=burst_id,
        )
