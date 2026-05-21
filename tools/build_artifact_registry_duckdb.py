#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".npz", ".npy"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def csv_shape_and_columns(path: Path) -> tuple[int | None, list[str]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return rows, header
    except Exception:
        return None, []


def infer_obs_id(path: Path) -> str | None:
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("obs") and len(lower) >= 6:
            token = lower.split("_")[0]
            if len(token) >= 6 and token[3:6].isdigit():
                return f"OBS-{token[3:6]}"

    name = path.name.lower()
    if name.startswith("obs") and len(name) >= 6 and name[3:6].isdigit():
        return f"OBS-{name[3:6]}"

    return None


def infer_campaign_from_path(path: Path) -> tuple[str | None, str | None, str]:
    """
    Return (corpus, campaign, layout).

    Supported layouts:

    New campaign layout:
      outputs/corpora/<corpus>/campaigns/<campaign>/...

    Legacy/root layout:
      outputs/index.csv
      outputs/trajectories/*.npz

    Legacy corpus/campaign are assigned from CLI args because they are not
    encoded in the root paths themselves.
    """
    parts = list(path.parts)

    try:
        i = parts.index("corpora")
        if parts[i + 2] == "campaigns":
            return parts[i + 1], parts[i + 3], "campaign"
    except Exception:
        pass

    if "trajectories" in parts or path.name == "index.csv":
        return None, None, "legacy_or_unknown"

    return None, None, "unknown"


def should_scan_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix not in ARTIFACT_SUFFIXES:
        return False
    if any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in path.parts):
        return False
    return True


def scan_artifacts(
    root: Path,
    *,
    legacy_corpus: str | None,
    legacy_campaign: str | None,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    rows = []

    for path in root.rglob("*"):
        if not should_scan_artifact(path):
            continue

        corpus, campaign, layout = infer_campaign_from_path(path)

        # Assign root-layout outputs/index.csv and outputs/trajectories to the
        # explicitly declared legacy corpus/campaign.
        if layout == "legacy_or_unknown":
            try:
                rel_to_root = path.relative_to(root)
                is_legacy_index = rel_to_root.parts == ("index.csv",)
                is_legacy_trajectory = (
                    len(rel_to_root.parts) >= 2
                    and rel_to_root.parts[0] == "trajectories"
                )
                if is_legacy_index or is_legacy_trajectory:
                    corpus = legacy_corpus
                    campaign = legacy_campaign
                    layout = "legacy_root"
            except Exception:
                pass

        csv_rows, cols = (None, [])
        if path.suffix == ".csv":
            csv_rows, cols = csv_shape_and_columns(path)

        rows.append(
            {
                "host_label": host_label,
                "scan_timestamp": scan_timestamp,
                "path": str(path),
                "suffix": path.suffix,
                "size_bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
                "sha1_head": sha1_file(path, max_bytes=1024 * 1024),
                "obs_id": infer_obs_id(path),
                "corpus": corpus,
                "campaign": campaign,
                "layout": layout,
                "rows": csv_rows,
                "columns_json": json.dumps(cols),
            }
        )

    return pd.DataFrame(rows)


def load_campaign_runs(
    outputs_root: Path,
    *,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    """
    Load progress snapshots from new campaign layout.
    """
    rows = []

    for progress in outputs_root.glob("corpora/*/campaigns/*/manifests/*_progress.json"):
        try:
            data = json.loads(progress.read_text(encoding="utf-8"))
        except Exception:
            continue

        root = Path(data.get("root", ""))
        corpus = None
        campaign = None

        parts = list(root.parts)
        try:
            i = parts.index("corpora")
            if parts[i + 2] == "campaigns":
                corpus = parts[i + 1]
                campaign = parts[i + 3]
        except Exception:
            pass

        # Resolve relative roots against the current working directory.
        resolved_root = root
        if not resolved_root.is_absolute():
            resolved_root = Path.cwd() / resolved_root

        traj_dir = resolved_root / "trajectories"
        trajectory_count = len(list(traj_dir.glob("*.npz"))) if traj_dir.exists() else 0

        index_path = resolved_root / "index.csv"
        index_rows = None
        if index_path.exists():
            index_rows, _ = csv_shape_and_columns(index_path)

        rows.append(
            {
                "host_label": host_label,
                "scan_timestamp": scan_timestamp,
                "corpus": corpus,
                "campaign": campaign,
                "layout": "campaign",
                "root": str(root),
                "run_name": data.get("run_name"),
                "host": data.get("host"),
                "pid": data.get("pid"),
                "updated_at": data.get("updated_at"),
                "total": data.get("total"),
                "done": data.get("done"),
                "failed": data.get("failed"),
                "running": data.get("running"),
                "pending": data.get("pending"),
                "percent": data.get("percent"),
                "elapsed_sec": data.get("elapsed_sec"),
                "throughput_jobs_per_min": data.get("throughput_jobs_per_min"),
                "eta_sec": data.get("eta_sec"),
                "last_completed": data.get("last_completed"),
                "last_error": data.get("last_error"),
                "trajectory_count": trajectory_count,
                "index_rows": index_rows,
            }
        )

    return pd.DataFrame(rows)


def load_legacy_campaign_run(
    outputs_root: Path,
    *,
    legacy_corpus: str | None,
    legacy_campaign: str | None,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    """
    Register root-layout legacy campaign:

      outputs/index.csv
      outputs/trajectories/*.npz

    This supports older C/Cp canonical runs that predate the campaign namespace.
    """
    if not legacy_corpus or not legacy_campaign:
        return pd.DataFrame()

    index_path = outputs_root / "index.csv"
    traj_dir = outputs_root / "trajectories"

    if not index_path.exists() and not traj_dir.exists():
        return pd.DataFrame()

    index_rows = None
    if index_path.exists():
        index_rows, _ = csv_shape_and_columns(index_path)

    trajectory_count = len(list(traj_dir.glob("*.npz"))) if traj_dir.exists() else 0

    done = index_rows if index_rows is not None else trajectory_count
    total = done

    row = {
        "host_label": host_label,
        "scan_timestamp": scan_timestamp,
        "corpus": legacy_corpus,
        "campaign": legacy_campaign,
        "layout": "legacy_root",
        "root": str(outputs_root),
        "run_name": f"{legacy_corpus}_{legacy_campaign}",
        "host": socket.gethostname(),
        "pid": None,
        "updated_at": scan_timestamp,
        "total": total,
        "done": done,
        "failed": None,
        "running": None,
        "pending": None,
        "percent": 1.0 if done is not None else None,
        "elapsed_sec": None,
        "throughput_jobs_per_min": None,
        "eta_sec": None,
        "last_completed": None,
        "last_error": None,
        "trajectory_count": trajectory_count,
        "index_rows": index_rows,
    }

    return pd.DataFrame([row])


def load_index_metrics_campaign_layout(
    outputs_root: Path,
    *,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    frames = []

    for index_path in outputs_root.glob("corpora/*/campaigns/*/index.csv"):
        try:
            df = pd.read_csv(index_path)
        except Exception:
            continue

        parts = list(index_path.parts)
        corpus = None
        campaign = None
        try:
            i = parts.index("corpora")
            if parts[i + 2] == "campaigns":
                corpus = parts[i + 1]
                campaign = parts[i + 3]
        except Exception:
            pass

        df.insert(0, "scan_timestamp", scan_timestamp)
        df.insert(0, "host_label", host_label)
        df.insert(0, "layout", "campaign")
        df.insert(0, "campaign", campaign)
        df.insert(0, "corpus_root", corpus)
        df.insert(0, "index_path", str(index_path))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_index_metrics_legacy_layout(
    outputs_root: Path,
    *,
    legacy_corpus: str | None,
    legacy_campaign: str | None,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    if not legacy_corpus or not legacy_campaign:
        return pd.DataFrame()

    index_path = outputs_root / "index.csv"
    if not index_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(index_path)
    except Exception:
        return pd.DataFrame()

    df.insert(0, "scan_timestamp", scan_timestamp)
    df.insert(0, "host_label", host_label)
    df.insert(0, "layout", "legacy_root")
    df.insert(0, "campaign", legacy_campaign)
    df.insert(0, "corpus_root", legacy_corpus)
    df.insert(0, "index_path", str(index_path))

    return df


def load_all_campaign_runs(
    outputs_root: Path,
    *,
    legacy_corpus: str | None,
    legacy_campaign: str | None,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    frames = [
        load_campaign_runs(
            outputs_root,
            host_label=host_label,
            scan_timestamp=scan_timestamp,
        ),
        load_legacy_campaign_run(
            outputs_root,
            legacy_corpus=legacy_corpus,
            legacy_campaign=legacy_campaign,
            host_label=host_label,
            scan_timestamp=scan_timestamp,
        ),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_index_metrics(
    outputs_root: Path,
    *,
    legacy_corpus: str | None,
    legacy_campaign: str | None,
    host_label: str,
    scan_timestamp: str,
) -> pd.DataFrame:
    frames = [
        load_index_metrics_campaign_layout(
            outputs_root,
            host_label=host_label,
            scan_timestamp=scan_timestamp,
        ),
        load_index_metrics_legacy_layout(
            outputs_root,
            legacy_corpus=legacy_corpus,
            legacy_campaign=legacy_campaign,
            host_label=host_label,
            scan_timestamp=scan_timestamp,
        ),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_empty_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame(columns=columns)


def export_parquet_tables(
    *,
    con: duckdb.DuckDBPyConnection,
    export_dir: Path,
    host_label: str,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "artifact_files": export_dir / f"{host_label}_artifact_files.parquet",
        "campaign_runs": export_dir / f"{host_label}_campaign_runs.parquet",
        "index_metrics": export_dir / f"{host_label}_index_metrics.parquet",
    }

    for table, path in targets.items():
        con.execute(
            f"copy {table} to ? (format parquet)",
            [str(path)],
        )
        print("exported", path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--db", default="metadata/pam_artifacts.duckdb")

    ap.add_argument(
        "--legacy-corpus",
        default=None,
        help=(
            "Corpus label for legacy root-layout artifacts at outputs/index.csv "
            "and outputs/trajectories/*.npz, e.g. C or Cp."
        ),
    )
    ap.add_argument(
        "--legacy-campaign",
        default="canonical_legacy",
        help="Campaign label for legacy root-layout artifacts.",
    )

    ap.add_argument(
        "--host-label",
        default=socket.gethostname(),
        help="Stable source label for this scan, e.g. macbook or macmini.",
    )
    ap.add_argument(
        "--export-dir",
        default="metadata/registry_exports",
        help="Directory for optional parquet registry exports.",
    )
    ap.add_argument(
        "--export-parquet",
        action="store_true",
        help="Export artifact_files, campaign_runs, and index_metrics to parquet files.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outputs_root = Path(args.outputs_root)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    scan_timestamp = utc_now_iso()
    host_label = args.host_label

    legacy_campaign = args.legacy_campaign if args.legacy_corpus else None

    artifact_df = scan_artifacts(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=legacy_campaign,
        host_label=host_label,
        scan_timestamp=scan_timestamp,
    )

    campaigns_df = load_all_campaign_runs(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=legacy_campaign,
        host_label=host_label,
        scan_timestamp=scan_timestamp,
    )

    index_df = load_all_index_metrics(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=legacy_campaign,
        host_label=host_label,
        scan_timestamp=scan_timestamp,
    )

    artifact_df = normalize_empty_frame(
        artifact_df,
        [
            "host_label",
            "scan_timestamp",
            "path",
            "suffix",
            "size_bytes",
            "modified_at",
            "sha1_head",
            "obs_id",
            "corpus",
            "campaign",
            "layout",
            "rows",
            "columns_json",
        ],
    )

    campaigns_df = normalize_empty_frame(
        campaigns_df,
        [
            "host_label",
            "scan_timestamp",
            "corpus",
            "campaign",
            "layout",
            "root",
            "run_name",
            "host",
            "pid",
            "updated_at",
            "total",
            "done",
            "failed",
            "running",
            "pending",
            "percent",
            "elapsed_sec",
            "throughput_jobs_per_min",
            "eta_sec",
            "last_completed",
            "last_error",
            "trajectory_count",
            "index_rows",
        ],
    )

    # Preserve the dynamic schema of index_metrics because different campaign
    # generations may contain slightly different metric columns.
    con = duckdb.connect(str(db_path))
    con.execute("create or replace table artifact_files as select * from artifact_df")
    con.execute("create or replace table campaign_runs as select * from campaigns_df")
    con.execute("create or replace table index_metrics as select * from index_df")

    print("wrote", db_path)
    print("host_label:", host_label)
    print("scan_timestamp:", scan_timestamp)
    print("artifact_files:", len(artifact_df))
    print("campaign_runs:", len(campaigns_df))
    print("index_metrics:", len(index_df))

    if len(campaigns_df):
        print()
        print(
            con.execute(
                """
                select host_label, corpus, campaign, layout, done, failed, pending,
                       trajectory_count, index_rows
                from campaign_runs
                order by host_label, corpus, campaign, layout
                """
            )
            .fetchdf()
            .to_string(index=False)
        )

    if args.export_parquet:
        export_parquet_tables(
            con=con,
            export_dir=Path(args.export_dir),
            host_label=host_label,
        )


if __name__ == "__main__":
    main()