import numpy as np
from compliance_checker.cf import cf_1_6
from netCDF4 import Dataset

# Importing the plugin base module installs the check_geographic_region
# monkeypatch as a side effect (see plugins/wcrp_base.py).
from plugins import wcrp_base  # noqa: F401


def _write_region_var(path, *, region_names, strlen=20):
    """Write a netCDF file with a single labeled-region variable.

    A 1D char array (single name) when region_names has one entry, a 2D
    char array (one row per name) when it has more than one -- matching
    the shape ocean-basin-decomposed diagnostics (hfbasin, htovgyre, ...)
    actually use on disk.
    """
    with Dataset(path, "w") as ds:
        ds.createDimension("strlen", strlen)
        if len(region_names) == 1:
            var = ds.createVariable("region_var", "S1", ("strlen",))
            var[:] = np.array(list(region_names[0].ljust(strlen)), dtype="S1")
        else:
            ds.createDimension("basin", len(region_names))
            var = ds.createVariable("region_var", "S1", ("basin", "strlen"))
            data = [list(name.ljust(strlen)) for name in region_names]
            var[:] = np.array(data, dtype="S1")
        var.standard_name = "region"


def test_multi_region_2d_array_does_not_crash(tmp_path):
    path = tmp_path / "multi_region.nc"
    _write_region_var(
        path, region_names=["global_ocean", "atlantic_ocean", "not_a_real_region"]
    )
    with Dataset(path) as ds:
        results = cf_1_6.CF1_6Check().check_geographic_region(ds)
    assert len(results) == 1
    score, out_of = results[0].value
    assert out_of == 3
    assert score == 2
    assert len(results[0].msgs) == 1
    assert "not_a_real_region" in results[0].msgs[0]


def test_single_region_1d_array_still_works(tmp_path):
    path = tmp_path / "single_region.nc"
    _write_region_var(path, region_names=["global_ocean"])
    with Dataset(path) as ds:
        results = cf_1_6.CF1_6Check().check_geographic_region(ds)
    assert len(results) == 1
    score, out_of = results[0].value
    assert (score, out_of) == (1, 1)
    assert results[0].msgs == []


def test_no_region_variable_returns_empty(tmp_path):
    path = tmp_path / "no_region.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("x", 2)
        ds.createVariable("plain_var", "f4", ("x",))
    with Dataset(path) as ds:
        results = cf_1_6.CF1_6Check().check_geographic_region(ds)
    assert results == []
