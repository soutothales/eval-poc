"""Results store queries with DuckDB over ``runs/*/cases.parquet`` (local Athena)."""

from __future__ import annotations

from typing import List

import duckdb
import pandas as pd

from agent import constants as ac
from eval import constants as c
from eval.io import read_json

HISTORY_METRICS = (
    c.M_CORRECTNESS,
    c.M_FAITHFULNESS,
    c.M_CONTEXTUAL_RECALL,
    c.M_HIT_RATE,
    c.M_BEHAVIOR_ACCURACY,
)


def query(sql: str) -> pd.DataFrame:
    """Run SQL against all scored runs, exposed as the view ``cases``.

    Parameters
    ----------
    sql: str
        Query referencing ``cases``.

    Returns
    -------
    pd.DataFrame
        Result.
    """
    pattern = (c.RUNS_DIR / "*" / c.CASES_TABLE_FILE).as_posix()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW cases AS SELECT * FROM read_parquet('{pattern}', union_by_name = true)")
    return con.execute(sql).df()


def history() -> pd.DataFrame:
    """Per-run trend table: config, model, quality means, cost and latency.

    Returns
    -------
    pd.DataFrame
        One row per scored run, oldest first.
    """
    runs = sorted(p.parent.name for p in c.RUNS_DIR.glob(f"*/{c.CASES_TABLE_FILE}"))
    if not runs:
        return pd.DataFrame()
    available: List[str] = query("SELECT * FROM cases LIMIT 0").columns.tolist()
    metric_cols = [m for m in HISTORY_METRICS if m in available]
    selects = ", ".join(f"round(avg({m}), 3) AS {m}" for m in metric_cols)
    df = query(
        f"SELECT {c.CT_RUN_ID}, count(*) AS n, round(avg(CAST({c.CT_ERROR} AS DOUBLE)), 3) AS error_rate, "
        + (selects + ", " if selects else "")
        + f"round(avg({c.CT_AGENT_COST}) * 1000, 4) AS agent_cost_per_1k_q, "
        + f"round(quantile_cont({c.CT_LATENCY_S}, 0.95), 2) AS latency_p95 "
        + f"FROM cases GROUP BY {c.CT_RUN_ID} ORDER BY {c.CT_RUN_ID}"
    )
    models = {}
    for rid in df[c.CT_RUN_ID]:
        m = read_json(c.RUNS_DIR / rid / c.RUN_MANIFEST_FILE)
        models[rid] = (m[c.RM_AGENT][ac.CFG_MODEL_ID], m[c.RM_DATASET][c.RM_DATASET_VERSION])
    df.insert(1, "model", df[c.CT_RUN_ID].map(lambda r: models[r][0]))
    df.insert(2, "dataset", df[c.CT_RUN_ID].map(lambda r: models[r][1]))
    return df
