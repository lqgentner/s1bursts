"""Tests for s1bursts.cdse (auth handler, response parsing, client)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from s1bursts.cdse import MAX_PAGE_SIZE, CDSEAuth, CDSEBurstClient, CDSEResponse
from s1bursts.errors import CatalogError, ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterator

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

        with pytest.raises(ConfigurationError, match="CDSE credentials not configured"):
            CDSEAuth()

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_rejected_credentials_raise_a_configuration_error(self, status: int) -> None:
        """Previously an HTTPError, which a download recorded per burst and retried."""
        rejected = MagicMock()
        rejected.status_code = status
        rejected.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Invalid user credentials",
        }
        with (
            patch("s1bursts.cdse.requests.post", return_value=rejected) as mock_post,
            pytest.raises(ConfigurationError, match=r"rejected the username and password") as e,
        ):
            CDSEAuth(username="someone", password="hunter2")

        mock_post.assert_called_once()
        assert f"(HTTP {status}: Invalid user credentials)" in str(e.value)
        assert "hunter2" not in str(e.value)
        assert "someone" not in str(e.value)
        # A chained HTTPError would keep the request, whose body holds the password.
        assert e.value.__cause__ is None
        assert e.value.__context__ is None

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(ValueError("not JSON"), id="not-json"),
            pytest.param(["a list"], id="not-an-object"),
            pytest.param({"error_description": "  "}, id="blank"),
        ],
    )
    def test_a_rejection_without_a_description_names_the_status(self, body: object) -> None:
        rejected = MagicMock()
        rejected.status_code = 401
        if isinstance(body, Exception):
            rejected.json.side_effect = body
        else:
            rejected.json.return_value = body

        with (
            patch("s1bursts.cdse.requests.post", return_value=rejected),
            pytest.raises(ConfigurationError, match=r"\(HTTP 401\)\."),
        ):
            CDSEAuth(username="someone", password="hunter2")

    @staticmethod
    def _failing_response(status: int, secret: str) -> MagicMock:
        """Return a token response that fails with an HTTPError holding ``secret``."""
        request = requests.Request(
            "POST",
            "https://identity.example/token",
            data={"password": secret},
        ).prepare()
        response = MagicMock()
        response.status_code = status
        response.request = request
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status} Server Error for url: https://identity.example/token",
            response=response,
        )
        return response

    def test_another_login_failure_keeps_no_request(self) -> None:
        """The request holds the password in its body, so the error must not keep it."""
        failing = self._failing_response(500, "hunter2")

        with (
            patch("s1bursts.cdse.requests.post", return_value=failing),
            pytest.raises(requests.HTTPError, match="500 Server Error") as e,
        ):
            CDSEAuth(username="someone", password="hunter2")

        assert e.value.request is None
        assert e.value.response is None
        assert e.value.__cause__ is None
        assert e.value.__context__ is None

    def test_a_failed_refresh_keeps_no_request(self) -> None:
        """The refresh request holds the refresh token in its body."""
        with patch("s1bursts.cdse.requests.post", return_value=make_token_response()):
            auth = CDSEAuth(username="user", password="pass")
        auth.token_expires_at = time.time() - 1
        request = MagicMock()
        headers: dict[str, str] = {}
        request.headers = headers

        with (
            patch(
                "s1bursts.cdse.requests.post",
                return_value=self._failing_response(500, "refresh-1"),
            ),
            pytest.raises(requests.HTTPError, match="500 Server Error") as e,
        ):
            auth(request)

        assert e.value.request is None
        assert e.value.__context__ is None

    def test_an_error_with_another_constructor_is_copied_as_a_request_exception(self) -> None:
        """Requests' JSONDecodeError takes a document and a position besides the message."""
        broken = MagicMock()
        broken.status_code = 200
        broken.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)

        with (
            patch("s1bursts.cdse.requests.post", return_value=broken),
            pytest.raises(requests.RequestException, match="Expecting value") as e,
        ):
            CDSEAuth(username="someone", password="hunter2")

        assert type(e.value) is requests.RequestException

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
        headers: dict[str, str] = {}
        request.headers = headers
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
            headers: dict[str, str] = {}
            request.headers = headers
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
            headers: dict[str, str] = {}
            request.headers = headers
            auth(request)

        assert mock_post.call_args.kwargs["data"]["grant_type"] == "password"

    def test_rejected_refresh_token_reauthenticates(self) -> None:
        with patch(
            "s1bursts.cdse.requests.post",
            return_value=make_token_response(),
        ):
            auth = CDSEAuth(username="user", password="pass")

        # Access token expired, refresh token nominally valid but rejected by the server
        auth.token_expires_at = time.time() - 1

        rejected = MagicMock()
        rejected.status_code = 400
        rejected.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request", response=rejected
        )
        fresh = make_token_response()
        fresh.json.return_value["access_token"] = "access-3"
        with patch(
            "s1bursts.cdse.requests.post",
            side_effect=[rejected, fresh],
        ) as mock_post:
            request = MagicMock()
            headers: dict[str, str] = {}
            request.headers = headers
            auth(request)

        grants = [c.kwargs["data"]["grant_type"] for c in mock_post.call_args_list]
        assert grants == ["refresh_token", "password"]
        assert request.headers["Authorization"] == "Bearer access-3"


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

        with pytest.raises(CatalogError, match="count is None"):
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
        assert gdf.crs is not None
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

        with pytest.raises(CatalogError, match="unknown EPSG"):
            response.to_geodataframe()


class TestCDSEBurstClient:
    def test_build_query_params(self) -> None:
        client = CDSEBurstClient()

        params = client._build_query_params(
            filters=["BurstId eq 1", "PolarisationChannels eq 'VV'"],
            ascending=False,
            count=True,
        )

        assert params["$filter"] == "BurstId eq 1 and PolarisationChannels eq 'VV'"
        assert params["$orderby"] == "ContentDate/Start desc"
        assert params["$count"] is True

    def test_build_query_params_defaults(self) -> None:
        client = CDSEBurstClient()

        params = client._build_query_params()

        assert "$filter" not in params
        # A search is one request, so it asks for the largest page CDSE serves.
        assert params["$top"] == MAX_PAGE_SIZE
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
            response = client.search_bursts(["BurstId eq 30346"], count=True)

        assert isinstance(response, CDSEResponse)
        assert len(response) == 1
        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["$top"] == MAX_PAGE_SIZE
        assert mock_get.call_args.kwargs["params"]["$count"] is True

    def test_search_bursts_ignores_next_link(self) -> None:
        """One page is the whole result; a longer one is refused by the caller."""
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = {
            "@odata.context": "$metadata#Bursts",
            "@odata.count": 3,
            "@odata.nextLink": "https://catalogue.example/next",
            "value": [make_response_value(), make_response_value()],
        }

        with patch("s1bursts.cdse.requests.get", return_value=http_response) as mock_get:
            response = client.search_bursts(count=True)

        assert mock_get.call_count == 1
        assert len(response) == 2
        assert response.count == 3
        assert not response.is_complete()

    def test_search_bursts_rejects_non_json(self) -> None:
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "text/html"}
        http_response.text = "<html>gateway error</html>"

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(CatalogError, match="Failed to query bursts"),
        ):
            client.search_bursts()

    def test_search_bursts_rejects_wrong_context(self) -> None:
        client = CDSEBurstClient()
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {"@odata.context": "$metadata#Products", "value": []}
        http_response.json.return_value = payload

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(CatalogError, match="Failed to query bursts"),
        ):
            client.search_bursts()

    def test_search_bursts_rejects_a_response_without_value(self) -> None:
        # Previously a bare KeyError from CDSEResponse.from_dict.
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.return_value = {"@odata.context": "$metadata#Bursts"}

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(CatalogError, match="unexpected response") as info,
        ):
            CDSEBurstClient().search_bursts()
        assert info.value.status_code is None

    def test_search_bursts_rejects_invalid_json(self) -> None:
        # Previously requests' JSONDecodeError.
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "application/json"}
        http_response.json.side_effect = requests.JSONDecodeError("Expecting value", "<", 0)

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(CatalogError, match="not valid JSON") as info,
        ):
            CDSEBurstClient().search_bursts()
        assert isinstance(info.value.__cause__, requests.JSONDecodeError)

    def test_search_bursts_quotes_a_long_non_json_body_in_part(self) -> None:
        http_response = MagicMock()
        http_response.headers = {"Content-Type": "text/html"}
        http_response.text = "x" * 5000

        with (
            patch("s1bursts.cdse.requests.get", return_value=http_response),
            pytest.raises(CatalogError, match=r"'text/html' instead of JSON") as info,
        ):
            CDSEBurstClient().search_bursts()
        assert "(5000 characters)" in str(info.value)
        assert len(str(info.value)) < 1000

    def test_download_burst_requires_auth(self, tmp_path: Any) -> None:
        client = CDSEBurstClient(auth=None)

        with pytest.raises(TypeError, match="Must be an instance of 'CDSEAuth'"):
            client.download_burst("some-id", save_path=tmp_path / "burst.zip")


def _error_response(
    status: int,
    reason: str,
    *,
    body: bytes = b"",
    url: str = "https://catalogue.example/odata/v1/Bursts",
) -> requests.Response:
    """Return a real catalog error response, so ``raise_for_status`` raises as it would."""
    response = requests.Response()
    response.status_code = status
    response.reason = reason
    response.url = url
    response._content = body
    response.headers["Content-Type"] = "application/json"
    return response


class TestSearchRequestFailures:
    """A failed catalog request raises CatalogError with the HTTPError as its cause."""

    @pytest.fixture(autouse=True)
    def _no_retry_wait(self) -> Iterator[None]:
        with patch("tenacity.nap.time.sleep"):
            yield

    def test_uri_too_long_explains_the_geometry(self) -> None:
        # Previously a raw requests.HTTPError, e.g. for the outline of an MGRS cell.
        url = "https://catalogue.example/odata/v1/Bursts?$filter=" + "x" * 15_000
        response = _error_response(414, "Request-URI Too Large", url=url)

        with (
            patch("s1bursts.cdse.requests.get", return_value=response) as get,
            pytest.raises(CatalogError, match="HTTP 414 Request-URI Too Large") as info,
        ):
            CDSEBurstClient().search_bursts(["BurstId eq 1"])

        assert get.call_count == 1
        assert info.value.status_code == 414
        assert isinstance(info.value.__cause__, requests.HTTPError)
        message = str(info.value)
        assert f"{len(url)} characters long" in message
        assert "shapely.box(*geometry.bounds)" in message
        assert "BurstCollection.intersecting(geometry)" in message

    def test_a_rejected_filter_names_the_catalogs_reason(self) -> None:
        body = (
            b'{"detail":{"message":"Invalid value: Geometry is invalid. '
            b'Self-intersection[0.5 0.5]","request_id":"abc"}}'
        )
        response = _error_response(400, "Bad Request", body=body)

        with (
            patch("s1bursts.cdse.requests.get", return_value=response),
            pytest.raises(CatalogError, match=r"Self-intersection\[0.5 0.5\]") as info,
        ):
            CDSEBurstClient().search_bursts()

        assert info.value.status_code == 400
        assert "intersecting" not in str(info.value)

    @pytest.mark.parametrize(
        "body",
        [
            b"<html>Bad Request</html>",
            b'{"detail": "Invalid field: Name"}',
            b"[]",
            b'{"detail": 1}',
        ],
    )
    def test_any_error_body_is_reported(self, body: bytes) -> None:
        response = _error_response(400, "Bad Request", body=body)

        with (
            patch("s1bursts.cdse.requests.get", return_value=response),
            pytest.raises(CatalogError, match="HTTP 400 Bad Request") as info,
        ):
            CDSEBurstClient().search_bursts()

        detail = "Invalid field: Name" in str(info.value)
        assert detail == (body == b'{"detail": "Invalid field: Name"}')

    def test_a_persistent_server_error_is_retried_then_raised(self) -> None:
        with (
            patch(
                "s1bursts.cdse.requests.get",
                side_effect=lambda **_kwargs: _error_response(500, "Internal Server Error"),
            ) as get,
            pytest.raises(CatalogError, match="HTTP 500") as info,
        ):
            CDSEBurstClient().search_bursts()

        assert get.call_count == 3
        assert info.value.status_code == 500

    def test_a_transient_server_error_is_retried(self) -> None:
        ok = MagicMock()
        ok.headers = {"Content-Type": "application/json"}
        empty: list[dict[str, Any]] = []
        ok.json.return_value = {"@odata.context": "$metadata#Bursts", "value": empty}

        with patch(
            "s1bursts.cdse.requests.get",
            side_effect=[_error_response(500, "Internal Server Error"), ok],
        ) as get:
            response = CDSEBurstClient().search_bursts()

        assert get.call_count == 2
        assert len(response) == 0

    def test_an_unreachable_catalog_raises_catalog_error(self) -> None:
        # Previously requests.ConnectionError.
        with (
            patch(
                "s1bursts.cdse.requests.get",
                side_effect=requests.ConnectionError("Name or service not known"),
            ) as get,
            pytest.raises(CatalogError, match="could not be reached") as info,
        ):
            CDSEBurstClient().search_bursts()

        assert get.call_count == 3
        assert info.value.status_code is None
        assert isinstance(info.value.__cause__, requests.ConnectionError)

    def test_an_http_error_without_response_is_reported(self) -> None:
        with (
            patch("s1bursts.cdse.requests.get", side_effect=requests.HTTPError("boom")),
            pytest.raises(CatalogError, match="Failed to query bursts: boom") as info,
        ):
            CDSEBurstClient().search_bursts()

        assert info.value.status_code is None


class TestFootprintParsing:
    def test_a_burst_without_footprint_raises_catalog_error(self) -> None:
        value = make_response_value()
        del value["Footprint"]
        response = CDSEResponse(context="$metadata#Bursts", value=[value], count=1)

        with pytest.raises(CatalogError, match="without a 'Footprint'"):
            response.to_geodataframe()

    @pytest.mark.parametrize("footprint", ["POINT (7 46)", None])
    def test_an_unrecognised_footprint_raises_catalog_error(self, footprint: str | None) -> None:
        # Previously a ValueError from casting NaN to int.
        value = make_response_value()
        value["Footprint"] = footprint
        response = CDSEResponse(context="$metadata#Bursts", value=[value], count=1)

        with pytest.raises(CatalogError, match="unrecognised footprint"):
            response.to_geodataframe()
