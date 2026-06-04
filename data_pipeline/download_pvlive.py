"""
Download solar power generation data from Sheffield Solar's PV Live API.

This script retrieves hourly solar generation data for all Grid Supply Points
(GSPs) in the UK using the pvlive-api package. The data spans 2015–2022 and
is used as the target variable for the forecasting models.

Usage:
    python download_pvlive.py --start 2015-01-01 --end 2022-12-31 --output data.csv
"""

import argparse
from datetime import datetime

import pandas as pd
from pvlive_api import PVLive


def download_pvlive_by_gsp(
    start: str,
    end: str,
    include_national: bool = True,
    extra_fields: str = "",
) -> pd.DataFrame:
    """
    Download PV Live generation data for all GSPs in a given date range.

    Parameters
    ----------
    start : str
        Start date in ISO format (e.g. '2015-01-01').
    end : str
        End date in ISO format (e.g. '2022-12-31').
    include_national : bool
        Whether to include the national aggregate (gsp_id=0).
    extra_fields : str
        Comma-separated list of additional API fields to retrieve.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with generation data for all GSPs.
    """
    pvl = PVLive()
    min_gsp_id = 0 if include_national else 1
    frames = []

    for gsp_id in pvl.gsp_ids:
        if gsp_id < min_gsp_id:
            continue
        df = pvl.between(
            start=start,
            end=end,
            entity_type="gsp",
            entity_id=gsp_id,
            dataframe=True,
            extra_fields=extra_fields,
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PV Live GSP data")
    parser.add_argument("--start", default="2015-01-01", help="Start date (ISO format)")
    parser.add_argument("--end", default="2022-12-31", help="End date (ISO format)")
    parser.add_argument("--output", default="pvlive_gsp_data.csv", help="Output CSV path")
    parser.add_argument(
        "--no-national", action="store_true", help="Exclude national aggregate"
    )
    args = parser.parse_args()

    print(f"Downloading PV Live data from {args.start} to {args.end}...")
    data = download_pvlive_by_gsp(
        start=args.start,
        end=args.end,
        include_national=not args.no_national,
    )
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} rows to {args.output}")
