"""Tests for s1bursts.cdse (auth handler, response parsing, client)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from s1bursts.cdse import CDSEAuth, CDSEBurstClient, CDSEResponse

POLYGON_WKT = "POLYGON ((7 46, 8 46, 8 47, 7 47, 7 46))"


def make_token_response(expires_in: float = 600, refresh_expires_in: float = 3600) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "expires_in": expires_in,
        "refresh_expires_in": refresh_expires_in,
    }
    return response


class TestCDSEAuth:
    def test_missing_credentials_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)
        # Ensure no .env file in the working directory provides credentials
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="CDSE credentials not configured"):
            CDSEAuth()

    def test_initial_token_fetch_and_header(self) -> None:
        with patch(
            "s1bursts.cdse.requests.post",
            return_value=make_token_response(),
        ) as mock_post:
            auth = CDSEAuth(username="user", password="pass")

        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["data"]["grant_type"] == "password"
        assert auth.access_token == "access-1"

        request = MagicMock()
        request.headers = {}
        auth(request)
        assert request.headers["Authorization"] == "Bearer access-1"

    def test_expired_token_is_refreshed(self) -> None:
        with patch(
            "s1bursts.cdse.requests.post",
            return_value=make_token_response(),
        ):
            auth = CDSEAuth(username="user", password="pass")

        # Force expiry of the access token but not the refresh token
        auth.token_expires_at = time.time() - 1

        refreshed = make_token_response()
        refreshed.json.return_value["access_token"] = "access-2"
        with patch(
            "s1bursts.cdse.requests.post",
            return_value=refreshed,
        ) as mock_post:
            request = MagicMock()
            request.headers = {}
            auth(request)

        assert mock_post.call_args.kwargs["data"]["grant_type"] == "refresh_token"
        assert request.headers["Authorization"] == "Bearer access-2"

    def test_expired_refresh_token_reauthenticates(self) -> None:
        with patch(
            "s1bursts.cdse.requests.post",
            return_value=make_token_response(),
        ):
            auth = CDSEAuth(username="user", password="pass")

        auth.token_expires_at = time.time() - 1
        auth.refresh_expires_at = time.time() - 1

        with patch(
            "s1bursts.cdse.requests.post",
            return_value=make_token_response(),
        ) as mock_post:
            request = MagicMock()
            request.headers = {}
            auth(request)

        assert mock_post.call_args.kwargs["data"]["grant_type"] == "password"


def make_response_value(footprint: str = f"SRID=4326;{POLYGON_WKT}") -> dict[str, Any]:
    return {
        "Id": "some-id",
        "Name": "some-name",
        "Footprint": footprint,
        "BurstId": 30346,
        "AbsoluteBurstId": 123456,
        "SwathIdentifier": "IW1",
        "PolarisationChannels": "VV",
        "ParentProductName": "S1A_IW_SLC__1SDV",
        "BeginningDateTime": "2022-01-01T05:00:00.000Z",
        "EndingDateTime": "2022-01-01T05:00:03.000Z",
        "AzimuthTime": "2022-01-01T05:00:01.000Z",
    }


class TestCDSEResponse:
    def test_from_dict(self) -> None:
        response = CDSEResponse.from_dict(
            {
                "@odata.context": "$metadata#Bursts",
                "value": [make_response_value()],
                "@odata.count": 10,
            },
        )

        assert len(response) == 1
        assert response.count == 10
        assert repr(response) == "CDSEResponse(1/10 Bursts)"

    def test_repr_without_count(self) -> None:
        response = CDSEResponse(context="$metadata#Bursts", value=[make_response_value()])

        assert repr(response) == "CDSEResponse(1 Bursts)"

    def test_is_complete(self) -> None:
        response = CDSEResponse(
            context="$metadata#Bursts",
            value=[make_response_value()],
            count=1,
        )

        assert response.is_complete()

    def test_is_complete_without_count_raises(self) -> None:
        response = CDSEResponse(context="$metadata#Bursts", value=[])

        with pytest.raises(AttributeError, match="count is None"):
            response.is_complete()

    def test_to_geodataframe(self) -> None:
        response = CDSEResponse(
            context="$metadata#Bursts",
            value=[make_response_value()],
            count=1,
        )

        gdf = response.to_geodataframe()

        assert len(gdf) == 1
        # Column names are converted to snake_case
        assert "burst_id" in gdf.columns
        assert "parent_product_name" in gdf.columns
        assert gdf.crs.to_epsg() == 4326
        assert gdf.geometry.iloc[0].bounds == (7.0, 46.0, 8.0, 47.0)

    def test_to_geodataframe_parses_multipolygon(self) -> None:
        """Bursts crossing the antimeridian come back as MULTIPOLYGON."""
        multipolygon = (
            "MULTIPOLYGON (((180 52, 179 52, 179 53, 180 53, 180 52)), "
            "((-180 52, -179 52, -179 53, -180 53, -180 52)))"
        )
        response = CDSEResponse(
            context="$metadata#Bursts",
            value=[make_response_value(footprint=f"SRID=4326;{multipolygon}")],
            count=1,
        )

        gdf = response.to_geodataframe()

        assert gdf.geometry.iloc[0].geom_type == "MultiPolygon"

    def test_to_geodataframe_empty(self) -> None:
        response = CDSEResponse(context="$metadata#Bursts", value=[], count=0)

        gdf = response.to_geodataframe()

        assert len(gdf) == 0
        assert "burst_id" in gdf.columns

    def test_to_geodataframe_unknown_epsg_raises(self) -> None:
        response = CDSEResponse(
            context="$metadata#Bursts",
            value=[make_response_value(footprint=f"SRID=3857;{POLYGON_WKT}")],
        )

        with pytest.raises(RuntimeError, match="unknown EPSG"):
            response.to_geodataframe()


class TestCDSEBurstClient:
    def test_build_query_params(self) -> None:
        client = CDSEBurstClient()

        params = client._build_query_params(
            filters=["BurstId eq 1", "PolarisationChannels eq 'VV'"],
            top=50,
            ascending=False,
            count=True,
        )

        assert params["$filter"] == "BurstId eq 1 and PolarisationChannels eq 'VV'"
        assert params["$top"] == 50
        assert params["$orderby"] == "ContentDate/Start desc"
        assert params["$count"] is True

    def test_build_query_params_defaults(self) -> None:
        client = CDSEBurstClient()

        params = client._build_query_params()

        assert "$filter" not in params
        assert "$top" not in params
        assert "$count" not in params
        assert params["$orderby"] == "ContentDate/Start"

    def test_search_bursts_parses_response(self) -> None:
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json; charset=utf-8"}
        http_response.json.return_value = {
            "@odata.context": "$metadata#Bursts",
            "value": [make_response_value()],
        }

        with patch(
            "s1bursts.cdse.requests.get",
            return_value=http_response,
        ) as mock_get:
            response = client.search_bursts(["BurstId eq 30346"], top=10)

        assert isinstance(response, CDSEResponse)
        assert len(response) == 1
        assert mock_get.call_args.kwargs["params"]["$top"] == 10

    def test_search_bursts_rejects_non_json(self) -> None:
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "text/html"}
        http_response.text = "<html>gateway error</html>"

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(RuntimeError, match="Failed to query bursts"),
        ):
            client.search_bursts()

    def test_search_bursts_rejects_wrong_context(self) -> None:
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = {"@odata.context": "$metadata#Products", "value": []}

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(RuntimeError, match="Failed to query bursts"),
        ):
            client.search_bursts()

    def test_download_burst_requires_auth(self, tmp_path: Any) -> None:
        client = CDSEBurstClient(auth=None)

        with pytest.raises(TypeError, match="Must be an instance of 'CDSEAuth'"):
            client.download_burst("some-id", save_path=tmp_path / "burst.zip")
