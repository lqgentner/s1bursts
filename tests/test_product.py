"""Tests for s1bursts.product metadata containers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING
from unittest.mock import patch

from pandas import Timestamp
from pyproj import CRS
import pytest

from s1bursts.burst import Burst, BurstPair
from s1bursts.product import (
    RTCLRWQA,
    CoherenceComplexProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    CohPassQA,
    GTCProduct,
    InterferogramProduct,
    MGRSTileProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

BURST_ID = 30346
SUB_SWATH = "IW1"
CRS_UTM32 = CRS("EPSG:32632")


@pytest.fixture(autouse=True)
def _patched_orbit_lut() -> Iterator[None]:
    with patch(
        "s1bursts.burst.create_s1_orbit_number_lut",
        return_value={(BURST_ID, SUB_SWATH): 15},
    ):
        yield


def make_burst(date: str) -> Burst:
    return Burst(
        burst_id=BURST_ID,
        sub_swath=SUB_SWATH,
        polarization="VV",
        date=Timestamp(date),
    )


class TestBackscatterProducts:
    def test_from_burst(self) -> None:
        product = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)

        assert product.burst_id == BURST_ID
        assert product.sub_swath == SUB_SWATH
        assert product.polarization == "VV"
        assert product.date == Timestamp("2022-01-01")
        assert isinstance(product.crs, CRS)

    def test_product_types_and_flattening(self) -> None:
        rtc = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)
        gtc = GTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)

        assert rtc.product_type == "RTC"
        assert rtc.is_terrain_flattened is True
        assert gtc.product_type == "GTC"
        assert gtc.is_terrain_flattened is False

    def test_rel_orbit_number_resolved_via_lut(self) -> None:
        product = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)

        assert product.rel_orbit_number == 15
        assert asdict(product)["rel_orbit_number"] == 15

    def test_repr_does_not_touch_lut(self) -> None:
        product = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)

        with patch(
            "s1bursts.burst.create_s1_orbit_number_lut",
            side_effect=AssertionError("repr must not resolve the burst map"),
        ):
            text = repr(product)

        assert text.startswith("RTCProduct(")
        assert "2022-01-01" in text

    def test_from_dict_ignores_non_init_fields(self) -> None:
        product = RTCProduct.from_dict(
            {
                "burst_id": BURST_ID,
                "sub_swath": SUB_SWATH,
                "polarization": "VV",
                "date": Timestamp("2022-01-01"),
                "crs": CRS_UTM32,
                "rel_orbit_number": 999,  # init=False -> dropped
            },
        )

        assert product.rel_orbit_number == 15


class TestCoregisteredProducts:
    def make_pair(self) -> BurstPair:
        return BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13"))

    def test_from_burst_pair(self) -> None:
        product = CoherenceProduct.from_burst_pair(self.make_pair(), crs=CRS_UTM32)

        assert product.date_primary == Timestamp("2022-01-01")
        assert product.date_secondary == Timestamp("2022-01-13")
        assert product.baseline_days == 12

    def test_product_types(self) -> None:
        pair = self.make_pair()

        assert CoherenceProduct.from_burst_pair(pair, crs=CRS_UTM32).product_type == "COH"
        assert (
            CoherenceComplexProduct.from_burst_pair(pair, crs=CRS_UTM32).product_type
            == "COH_CMPLX"
        )
        assert InterferogramProduct.from_burst_pair(pair, crs=CRS_UTM32).product_type == "IFG"

    def test_is_complex(self) -> None:
        pair = self.make_pair()

        assert CoherenceProduct.from_burst_pair(pair, crs=CRS_UTM32).is_complex is False
        assert CoherenceComplexProduct.from_burst_pair(pair, crs=CRS_UTM32).is_complex is True


class TestMGRSTileProducts:
    KWARGS: dict = {  # noqa: RUF012
        "mgrs_code": "32TMS",
        "start_date": Timestamp("2022-01-01"),
        "end_date": Timestamp("2022-12-31"),
        "polarization": "VV",
    }

    def test_crs_derived_from_mgrs_code(self) -> None:
        product = RTCLRWComposite(**self.KWARGS)

        assert product.crs == CRS("EPSG:32632")

    def test_product_types(self) -> None:
        assert RTCLRWComposite(**self.KWARGS).product_type == "RTC_LRW"
        assert CohLRWComposite(**self.KWARGS, baseline_days=12).product_type == "COH_LRW"
        assert (
            RTCPassComposite(**self.KWARGS, orbit_direction="ASCENDING").product_type
            == "RTC_COMP_A"
        )
        assert (
            CohPassComposite(
                **self.KWARGS,
                orbit_direction="DESCENDING",
                baseline_days=12,
            ).product_type
            == "COH_COMP_D"
        )
        assert RTCLRWQA(**self.KWARGS, qa_metric=QaMetric.NUM).product_type == "RTC_LRW_QA-NUM"
        assert (
            RTCPassQA(**self.KWARGS, orbit_direction="ASCENDING").product_type
            == "RTC_COMP_A_QA-NUM"
        )
        assert (
            CohPassQA(
                **self.KWARGS,
                orbit_direction="ASCENDING",
                baseline_days=12,
            ).product_type
            == "COH_COMP_A_QA-NUM"
        )

    def test_is_frozen(self) -> None:
        product = RTCLRWComposite(**self.KWARGS)

        with pytest.raises(AttributeError):
            product.mgrs_code = "33TUN"  # type: ignore[misc]

    def test_repr_formats_timestamps(self) -> None:
        text = repr(RTCLRWComposite(**self.KWARGS))

        assert "start_date='2022-01-01'" in text
        assert isinstance(RTCLRWComposite(**self.KWARGS), MGRSTileProduct)
