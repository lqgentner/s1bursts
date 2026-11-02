"""Tests for s1bursts.product metadata containers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio.crs

from s1bursts.burst import Burst, BurstPair
from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    CohPassQA,
    GTCPassComposite,
    GTCPassQA,
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
def _patched_luts() -> Iterator[None]:
    with (
        patch(
            "s1bursts.burst.create_s1_orbit_number_lut",
            return_value={(BURST_ID, SUB_SWATH): 15},
        ),
        patch(
            "s1bursts.burst.create_s1_pass_direction_lut",
            return_value={(BURST_ID, SUB_SWATH): "Ascending"},
        ),
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

    @pytest.mark.parametrize("crs", ["EPSG:32632", 32632, CRS_UTM32])
    def test_from_burst_accepts_any_crs_input(self, crs: str | int | CRS) -> None:
        product = RTCProduct.from_burst(make_burst("2022-01-01"), crs=crs)

        assert product.crs == CRS_UTM32

    @pytest.mark.parametrize("product_cls", [RTCProduct, GTCProduct])
    @pytest.mark.parametrize("crs", ["EPSG:25832", "EPSG:3035", 4326])
    def test_from_burst_rejects_a_crs_that_is_not_a_wgs84_utm_zone(
        self,
        product_cls: type[RTCProduct | GTCProduct],
        crs: str | int,
    ) -> None:
        with pytest.raises(ValueError, match="must be a WGS 84 / UTM zone"):
            product_cls.from_burst(make_burst("2022-01-01"), crs=crs)

    def test_direct_construction_normalizes_crs_like(self) -> None:
        product = RTCProduct(
            BURST_ID,
            SUB_SWATH,
            rasterio.crs.CRS.from_epsg(32632),
            Timestamp("2022-01-01"),
            "VV",
            pass_direction="Ascending",
        )

        assert isinstance(product.crs, CRS)
        assert product.crs == CRS_UTM32

    def test_from_burst_carries_pass_direction(self) -> None:
        burst = Burst(
            burst_id=BURST_ID,
            sub_swath=SUB_SWATH,
            polarization="VV",
            date=Timestamp("2022-01-01"),
            _pass_direction="Descending",
        )
        with patch(
            "s1bursts.burst.create_s1_pass_direction_lut",
            side_effect=AssertionError("must not resolve the burst map"),
        ):
            product = RTCProduct.from_burst(burst, crs=CRS_UTM32)

        assert product.pass_direction == "Descending"

    def test_from_burst_looks_up_a_pass_direction_the_burst_lacks(self) -> None:
        product = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)

        assert product.pass_direction == "Ascending"

    def test_pass_direction_is_part_of_the_identity(self) -> None:
        # The file name records it, so two products that differ in it are two files.
        ascending = RTCProduct.from_burst(make_burst("2022-01-01"), crs=CRS_UTM32)
        descending = RTCProduct(
            BURST_ID,
            SUB_SWATH,
            CRS_UTM32,
            Timestamp("2022-01-01"),
            "VV",
            pass_direction="Descending",
        )

        assert ascending != descending
        assert "pass_direction='Ascending'" in repr(ascending)

    @pytest.mark.parametrize("direction", ["ascending", "A", "", None])
    def test_an_unknown_pass_direction_is_refused(self, direction: str | None) -> None:
        # Previously "ascending" wrote "066a", which discovery never found again.
        with pytest.raises(ValueError, match="must be 'Ascending' or 'Descending'"):
            RTCProduct(
                BURST_ID,
                SUB_SWATH,
                CRS_UTM32,
                Timestamp("2022-01-01"),
                "VV",
                # pyrefly: ignore [bad-argument-type]
                pass_direction=direction,
            )

    def test_pass_direction_is_required(self) -> None:
        with pytest.raises(TypeError, match="pass_direction"):
            # pyrefly: ignore [missing-argument]
            RTCProduct(BURST_ID, SUB_SWATH, CRS_UTM32, Timestamp("2022-01-01"), "VV")

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

    def test_from_dict_keeps_parsed_footprint_values(self) -> None:
        # Previously dropped, so products found on disk fell back to the burst map.
        product = RTCProduct.from_dict(
            {
                "burst_id": BURST_ID,
                "sub_swath": SUB_SWATH,
                "polarization": "VV",
                "date": Timestamp("2022-01-01"),
                "crs": CRS_UTM32,
                "rel_orbit_number": 66,
                "pass_direction": "Ascending",
                "unknown": "ignored",
            },
        )

        with patch(
            "s1bursts.burst.create_s1_orbit_number_lut",
            side_effect=AssertionError("must not resolve the burst map"),
        ):
            assert product.rel_orbit_number == 66
        assert product.pass_direction == "Ascending"

    def test_from_dict_prefers_explicit_override(self) -> None:
        product = RTCProduct.from_dict(
            {
                "burst_id": BURST_ID,
                "sub_swath": SUB_SWATH,
                "polarization": "VV",
                "date": Timestamp("2022-01-01"),
                "crs": CRS_UTM32,
                "pass_direction": "Ascending",
                "rel_orbit_number": 66,
                "_rel_orbit_number": 15,
            },
        )

        assert product.rel_orbit_number == 15

    def test_propagated_orbit_does_not_affect_equality(self) -> None:
        # Previously a product found on disk, which carries the orbit from its file
        # name, differed from the same product built without one.
        found = RTCProduct(
            BURST_ID,
            SUB_SWATH,
            CRS_UTM32,
            Timestamp("2022-01-01"),
            "VV",
            pass_direction="Ascending",
            _rel_orbit_number=15,
        )
        built = RTCProduct(
            BURST_ID,
            SUB_SWATH,
            CRS_UTM32,
            Timestamp("2022-01-01"),
            "VV",
            pass_direction="Ascending",
        )

        assert found == built
        assert hash(found) == hash(built)

    def test_equality_does_not_need_the_burst_map(self) -> None:
        # Previously ``==`` resolved the orbit and raised without the burst map.
        first, second = (
            RTCProduct(
                BURST_ID,
                SUB_SWATH,
                CRS_UTM32,
                Timestamp("2022-01-01"),
                "VV",
                pass_direction="Ascending",
            )
            for _ in range(2)
        )

        with patch(
            "s1bursts.burst.create_s1_orbit_number_lut",
            side_effect=FileNotFoundError("burst map not downloaded"),
        ):
            assert first == second
            assert hash(first) == hash(second)


class TestCoregisteredProducts:
    def make_pair(self) -> BurstPair:
        return BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13"))

    def test_from_burst_pair(self) -> None:
        product = CoherenceProduct.from_burst_pair(self.make_pair(), crs=CRS_UTM32)

        assert product.date_primary == Timestamp("2022-01-01")
        assert product.date_secondary == Timestamp("2022-01-13")
        assert product.baseline_days == 12

    @pytest.mark.parametrize("product_cls", [CoherenceProduct, InterferogramProduct])
    def test_from_burst_pair_rejects_a_utm_zone_on_another_datum(
        self,
        product_cls: type[CoherenceProduct | InterferogramProduct],
    ) -> None:
        with pytest.raises(ValueError, match="received 'ETRS89 / UTM zone 32N'"):
            product_cls.from_burst_pair(self.make_pair(), crs="EPSG:25832")

    def test_from_burst_pair_carries_pass_direction_and_crs_string(self) -> None:
        primary, secondary = (
            Burst(
                burst_id=BURST_ID,
                sub_swath=SUB_SWATH,
                polarization="VV",
                date=Timestamp(date),
                _pass_direction="Ascending",
            )
            for date in ("2022-01-01", "2022-01-13")
        )
        product = CoherenceProduct.from_burst_pair(BurstPair(primary, secondary), crs="EPSG:32632")

        assert product.pass_direction == "Ascending"
        assert product.crs == CRS_UTM32

    def test_product_types(self) -> None:
        pair = self.make_pair()

        assert CoherenceProduct.from_burst_pair(pair, crs=CRS_UTM32).product_type == "COH"
        assert InterferogramProduct.from_burst_pair(pair, crs=CRS_UTM32).product_type == "IFG"


class TestMGRSTileProducts:
    KWARGS: dict[str, Any] = {  # noqa: RUF012
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
            RTCPassComposite(**self.KWARGS, pass_direction="Ascending").product_type
            == "RTC_COMP_A"
        )
        assert (
            CohPassComposite(
                **self.KWARGS,
                pass_direction="Descending",
                baseline_days=12,
            ).product_type
            == "COH_COMP_D"
        )
        assert RTCLRWQA(**self.KWARGS, qa_metric=QaMetric.NUM).product_type == "RTC_LRW_QA-NUM"
        assert (
            RTCPassQA(**self.KWARGS, pass_direction="Ascending").product_type
            == "RTC_COMP_A_QA-NUM"
        )
        assert (
            CohPassQA(
                **self.KWARGS,
                pass_direction="Ascending",
                baseline_days=12,
            ).product_type
            == "COH_COMP_A_QA-NUM"
        )

    def test_is_frozen(self) -> None:
        product = RTCLRWComposite(**self.KWARGS)

        with pytest.raises(AttributeError):
            # pyrefly: ignore [read-only]
            product.mgrs_code = "33TUN"

    def test_repr_formats_timestamps(self) -> None:
        text = repr(RTCLRWComposite(**self.KWARGS))

        assert "start_date='2022-01-01'" in text
        assert isinstance(RTCLRWComposite(**self.KWARGS), MGRSTileProduct)


def test_gtc_pass_composite_product_type() -> None:
    product = GTCPassComposite(
        mgrs_code="32TMS",
        start_date=Timestamp("2019-01-01"),
        end_date=Timestamp("2019-01-31"),
        polarization="VV",
        pass_direction="Descending",
    )
    assert product.product_type == "GTC_COMP_D"
    assert product.crs.to_epsg() == 32632
    qa = GTCPassQA(
        mgrs_code="32TMS",
        start_date=Timestamp("2019-01-01"),
        end_date=Timestamp("2019-01-31"),
        polarization="VV",
        pass_direction="Ascending",
    )
    assert qa.product_type == "GTC_COMP_A_QA-NUM"
