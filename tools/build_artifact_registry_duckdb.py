#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".npz", ".npy"}


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

    Legacy corpus/campaign are assigned later from CLI args because they are not
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
        # Could be legacy or campaign; campaign would have matched above.
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
                if (
                    rel_to_root.parts == ("index.csv",)
                    or len(rel_to_root.parts) >= 2
                    and rel_to_root.parts[0] == "trajectories"
                ):
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


def load_campaign_runs(outputs_root: Path) -> pd.DataFrame:
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

        traj_dir = root / "trajectories"
        trajectory_count = len(list(traj_dir.glob("*.npz"))) if traj_dir.exists() else 0

        index_path = root / "index.csv"
        index_rows = None
        if index_path.exists():
            index_rows, _ = csv_shape_and_columns(index_path)

        rows.append(
            {
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

    # Infer total/done from index rows where possible.
    done = index_rows if index_rows is not None else trajectory_count
    total = done

    row = {
        "corpus": legacy_corpus,
        "campaign": legacy_campaign,
        "layout": "legacy_root",
        "root": str(outputs_root),
        "run_name": f"{legacy_corpus}_{legacy_campaign}",
        "host": None,
        "pid": None,
        "updated_at": None,
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


def load_index_metrics_campaign_layout(outputs_root: Path) -> pd.DataFrame:
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

    # Do not overwrite the original corpus column if present. Add corpus_root
    # and campaign as registry metadata.
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
) -> pd.DataFrame:
    frames = [
        load_campaign_runs(outputs_root),
        load_legacy_campaign_run(
            outputs_root,
            legacy_corpus=legacy_corpus,
            legacy_campaign=legacy_campaign,
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
) -> pd.DataFrame:
    frames = [
        load_index_metrics_campaign_layout(outputs_root),
        load_index_metrics_legacy_layout(
            outputs_root,
            legacy_corpus=legacy_corpus,
            legacy_campaign=legacy_campaign,
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


def main() -> None:
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
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_df = scan_artifacts(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=args.legacy_campaign if args.legacy_corpus else None,
    )

    campaigns_df = load_all_campaign_runs(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=args.legacy_campaign if args.legacy_corpus else None,
    )

    index_df = load_all_index_metrics(
        outputs_root,
        legacy_corpus=args.legacy_corpus,
        legacy_campaign=args.legacy_campaign if args.legacy_corpus else None,
    )

    artifact_df = normalize_empty_frame(
        artifact_df,
        [
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

    con = duckdb.connect(str(db_path))
    con.execute("create or replace table artifact_files as select * from artifact_df")
    con.execute("create or replace table campaign_runs as select * from campaigns_df")
    con.execute("create or replace table index_metrics as select * from index_df")

    print("wrote", db_path)
    print("artifact_files:", len(artifact_df))
    print("campaign_runs:", len(campaigns_df))
    print("index_metrics:", len(index_df))

    if len(campaigns_df):
        print()
        print(
            con.execute(
                """
                select corpus, campaign, layout, done, failed, pending,
                       trajectory_count, index_rows
                from campaign_runs
                order by corpus, campaign, layout
                """
            )
            .fetchdf()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
