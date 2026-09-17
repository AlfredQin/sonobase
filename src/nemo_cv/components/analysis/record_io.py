"""The verification contract for Stage-2 analysis records.

Why this exists
---------------
`t1_1_significance/*/analysis_report.json` once carried a `bootstrap_ci` block
whose A2/A4 entries recorded different quantities from the A2/A4 reports (HC18
SonoBase MAE 17.710 against the 2.49 that A2 itself reports). Nothing detected
it; it surfaced by chance while chasing an unrelated manuscript defect. An audit
written afterwards *still* would not have caught it, because that report has no
`models` key and the audit skipped anything it did not recognise.

The lesson is not "write a smarter audit". An audit that hard-codes what each
report looks like grows a special case per analysis, and the reports it silently
skips are invisible — a clean run is indistinguishable from a run that checked
nothing. The fix is to invert the direction: **each report declares how to
recompute its own statistics**, and the checker stays generic. A report that
declares nothing is then a detectable failure rather than a silent skip.

The contract
------------
Every `analysis_report.json` carries a `verification` block::

    "verification": {
      "dump": "per_sample.csv",        # row-level file beside the report, or null
      "checks": [
        {"path": ["models", "$model", "mae_mm"],
         "column": "abs_err_mm__{model}",
         "agg": "mean"},
        ...
      ]
    }

`path` walks into the report. A segment written ``$name`` is a wildcard: it
matches every key at that level and binds it to ``name``. `column` names the
dump column holding the row-level values, and may interpolate bound wildcards
(``abs_err_mm__{model}``) for dumps in wide layout.

Row selection is automatic and needs no declaration: for each bound wildcard,
if the dump has a column of that name, rows are filtered to the bound value.
That single rule covers both layouts in this codebase —

  * **wide** (A1/A2/A4/C2): one column per model, no ``model`` column to filter
    on, so the binding flows into the column template instead;
  * **long** (T3.1, B1, B3): a ``model`` column to filter on, one shared value
    column.

`agg` is one of ``mean``, ``sum``, ``count``, ``median``.

An analysis that legitimately has nothing to recompute — T2.3 skips itself when
ACOUSLIC ships no gestational-age metadata — declares ``dump: null`` with an
empty check list and a `skipped` reason. That is a *declared* absence, which the
checker reports separately from an undeclared one. The distinction matters: the
first is a documented property of the data, the second is a hole.
"""

from __future__ import annotations

import csv
import logging
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# `std` is the sample standard deviation (ddof=1), matching the `.std(ddof=1)`
# the analyses use for the "MAE ± Std" the manuscript prints.
AGGS = ("mean", "sum", "count", "median", "std")


def check(path: Sequence[str], column: str, agg: str = "mean") -> Dict[str, Any]:
    """One entry of a `verification.checks` list.

    ``path``   report keys to walk; ``$name`` binds a wildcard.
    ``column`` dump column, optionally interpolating bindings (``{model}``).
    ``agg``    how the column reduces to the stored statistic.
    """
    if agg not in AGGS:
        raise ValueError(f"agg must be one of {AGGS}, got {agg!r}")
    if not any(str(p).startswith("$") for p in path):
        # Not fatal, but a check with no wildcard verifies exactly one scalar and
        # is almost always a mistake in the declaration.
        logger.debug(f"verification check on {path} binds no wildcard")
    return {"path": list(path), "column": column, "agg": agg}


def external(path: Sequence[str], select: Dict[str, str], stat: str,
             file: str, column: str, agg: str = "mean") -> Dict[str, Any]:
    """A check against a row-level file belonging to a *different* analysis.

    T1.1 stores bootstrap CIs over MAEs it computed from A1/A2/A4/C2's
    per-sample CSVs. Those numbers live in T1.1's report but are owned
    elsewhere, so nothing inside T1.1's own directory can confirm them — which
    is precisely how its A2/A4 entries came to record a completely different
    quantity from the analyses they name, undetected.

    ``path``   walks to a *list* of dicts in the report (e.g. ``["bootstrap_ci"]``).
    ``select`` matches entries in that list; a ``$name`` value binds a wildcard.
    ``stat``   the key inside a matched entry holding the number to verify.
    ``file``   the row-level source, absolute or relative to the report's dir.
    ``column`` column in that file, may interpolate bindings.
    """
    if agg not in AGGS:
        raise ValueError(f"agg must be one of {AGGS}, got {agg!r}")
    return {"path": list(path), "select": dict(select), "stat": stat,
            "file": file, "column": column, "agg": agg}


def declare(dump: Optional[str], checks: Iterable[Dict[str, Any]],
            skipped_reason: Optional[str] = None,
            externals: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build the `verification` block to embed in an analysis_report.json.

    ``dump=None`` with ``skipped_reason`` set marks an analysis that genuinely
    produced no statistics, which is recorded as a declared absence rather than
    counted as unverifiable.
    """
    block: Dict[str, Any] = {"dump": dump, "checks": list(checks)}
    if externals:
        block["externals"] = list(externals)
    if skipped_reason:
        block["skipped_reason"] = skipped_reason
    return block


def write_dump(out_dir: pathlib.Path, rows: List[Dict[str, Any]],
               filename: str = "per_sample.csv",
               fieldnames: Optional[Sequence[str]] = None) -> Optional[pathlib.Path]:
    """Write the row-level records that produced a report's statistics.

    These are the rows *after* the analysis's own filtering, so that a checker
    reproduces the stored number exactly rather than approximately. Writing the
    pre-filter rows would leave every excluded row as an unexplained discrepancy.
    """
    if not rows:
        logger.warning(f"no rows to dump at {out_dir / filename}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    keys = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote {path} ({len(rows)} rows)")
    return path
