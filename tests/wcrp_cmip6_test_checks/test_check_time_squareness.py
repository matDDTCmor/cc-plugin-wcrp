"""
Regression tests for cc-plugin-wcrp#80: TIME001 double-shifted the
theoretical axis by half a step for frequencies whose filename token
already carries the true cell midpoint (1hr/3hr/6hr, dec), because it
assumed every filename token was a period start and re-derived a
midpoint from it.

Both fixtures below use real-world numbers, not arbitrary values:
- The 6hr case matches a real BSC EC-Earth3-ESM-1-1 hurs file exactly
  (time_first=1850-01-01 03:00:00, bnds_first=00:00->06:00,
  time_step_raw=0.25, from generate_mapfiles.py/nc_metadata_extract.py
  output) and reproduces the exact ccreport failure this fix addresses:
  "Mismatch at index 0: expected 0.250000, got 0.125000."
- The dec case matches cc-plugin-wcrp#80's own reported numbers
  exactly: "Mismatch at index 0: expected 3652.500000, got 1826.000000."

A regression fixture (monthly tavg) is included because monthly/yearly
filename tokens are deliberately excluded from the fix (TOKEN_IS_MIDPOINT_FREQ
in time_constants.py) -- their tokens are coarser than the period (no
day-of-month), so _parse_filename_start's own defaulting reconstructs
a genuine period start, and the existing "start + half increment"
derivation is correct for them. This must keep passing.
"""
import os
import tempfile

import cftime
import numpy as np
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck

from checks.time_checks.check_time_squareness import check_time_squareness


def _diskless_dataset(filename):
    return Dataset(filename, mode="w", diskless=True, persist=False)


def test_6hr_tavg_midpoint_token_passes():
    # Real numbers: BSC EC-Earth3-ESM-1-1 esm-hist hurs, 1850, 6hr.
    cal = "proleptic_gregorian"
    units = "days since 1850-01-01 00:00:00"
    n = 1460  # 365 days * 4

    fname = (
        "hurs_tavg-h2m-hxy-u_6hr_glb_g114_EC-Earth3-ESM-1-1_esm-hist_"
        "r1i1p1f1_185001010300-185012312100.nc"
    )
    ds = _diskless_dataset(fname)
    ds.createDimension("time", n)
    ds.createDimension("bnds", 2)

    time_var = ds.createVariable("time", "f8", ("time",))
    time_var.units = units
    time_var.calendar = cal
    time_var.bounds = "time_bnds"
    bnds_var = ds.createVariable("time_bnds", "f8", ("time", "bnds"))

    edge0 = cftime.date2num(
        cftime.datetime(1850, 1, 1, 0, 0, 0, calendar=cal), units=units, calendar=cal
    )
    step = 0.25
    edges = edge0 + np.arange(n + 1) * step
    time_var[:] = 0.5 * (edges[:-1] + edges[1:])
    bnds_var[:, 0] = edges[:-1]
    bnds_var[:, 1] = edges[1:]

    hurs = ds.createVariable("hurs", "f4", ("time",))
    hurs.standard_name = "relative_humidity"
    hurs.units = "%"
    hurs.cell_methods = "area: time: mean"

    ds.table_id = "None"
    ds.frequency = "6hr"
    ds.variable_id = "hurs"

    results = check_time_squareness(ds, severity=BaseCheck.HIGH)
    assert len(results) == 1
    assert results[0].value[0] == results[0].value[1], results[0].msgs
    ds.close()


def test_dec_tavg_midpoint_token_passes():
    # Real numbers: cc-plugin-wcrp#80's own reported case.
    cal = "proleptic_gregorian"
    units = "days since 1850-01-01"

    fname = (
        "volo_tavg-u-hm-sea_dec_glb_g190_AWI-ESM3-4-2-veg-HR_piControl_"
        "r1i1p1f1_1855-1855.nc"
    )
    ds = _diskless_dataset(fname)
    ds.createDimension("time", 1)
    ds.createDimension("bnds", 2)

    time_var = ds.createVariable("time", "f8", ("time",))
    time_var.units = units
    time_var.calendar = cal
    time_var.bounds = "time_bnds"
    bnds_var = ds.createVariable("time_bnds", "f8", ("time", "bnds"))

    b0 = cftime.date2num(cftime.datetime(1850, 1, 1, calendar=cal), units=units, calendar=cal)
    b1 = cftime.date2num(cftime.datetime(1860, 1, 1, calendar=cal), units=units, calendar=cal)
    time_var[:] = [0.5 * (b0 + b1)]
    bnds_var[:, 0] = [b0]
    bnds_var[:, 1] = [b1]

    volo = ds.createVariable("volo", "f4", ("time",))
    volo.standard_name = "sea_water_volume"
    volo.units = "m3"
    volo.cell_methods = "area: time: mean"

    ds.table_id = "None"
    ds.frequency = "dec"
    ds.variable_id = "volo"

    results = check_time_squareness(ds, severity=BaseCheck.HIGH)
    assert len(results) == 1
    assert results[0].value[0] == results[0].value[1], results[0].msgs
    ds.close()


def test_monthly_tavg_still_passes_no_regression():
    # mon tokens are YYYYMM (no day) -- _parse_filename_start defaults
    # day=1, genuinely reconstructing the period start, not a midpoint.
    # The existing "start + half increment" derivation must still apply
    # here; TOKEN_IS_MIDPOINT_FREQ deliberately excludes "mon".
    cal = "proleptic_gregorian"
    units = "days since 1850-01-01 00:00:00"

    fname = "tas_tavg-h2m-hxy-u_mon_glb_g114_EC-Earth3-ESM-1-1_esm-hist_r1i1p1f1_185001-185012.nc"
    ds = _diskless_dataset(fname)
    ds.createDimension("time", 12)
    ds.createDimension("bnds", 2)

    time_var = ds.createVariable("time", "f8", ("time",))
    time_var.units = units
    time_var.calendar = cal
    time_var.bounds = "time_bnds"
    bnds_var = ds.createVariable("time_bnds", "f8", ("time", "bnds"))

    month_starts = [cftime.datetime(1850, m, 1, calendar=cal) for m in range(1, 13)] + [
        cftime.datetime(1851, 1, 1, calendar=cal)
    ]
    edges = [cftime.date2num(d, units=units, calendar=cal) for d in month_starts]
    time_var[:] = [0.5 * (edges[i] + edges[i + 1]) for i in range(12)]
    bnds_var[:, 0] = edges[:-1]
    bnds_var[:, 1] = edges[1:]

    tas = ds.createVariable("tas", "f4", ("time",))
    tas.standard_name = "air_temperature"
    tas.units = "K"
    tas.cell_methods = "area: time: mean"

    ds.table_id = "None"
    ds.frequency = "mon"
    ds.variable_id = "tas"

    results = check_time_squareness(ds, severity=BaseCheck.HIGH)
    assert len(results) == 1
    assert results[0].value[0] == results[0].value[1], results[0].msgs
    ds.close()
