import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_KEYS = {"F_raw", "H_joint", "K", "pi", "Hj_sm", "lags", "corrs"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["filename"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Validate Observatory trajectory .npz files against the expected schema."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/expected_trajectories.csv",
        help="Trajectory manifest CSV",
    )
    parser.add_argument(
        "--traj-dir",
        default="outputs/trajectories",
        help="Directory containing trajectory .npz files",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/manifests/trajectory_validation.csv",
        help="Validation report CSV",
    )
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    df = load_manifest(args.manifest)

    rows = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        path = traj_dir / filename

        status = "ok"
        error = ""
        keys_present = ""
        missing_keys = ""
        shape_summary = ""

        if not path.exists():
            status = "missing_file"
        else:
            try:
                data = np.load(path, allow_pickle=False)
                keys = set(data.files)
                keys_present = ",".join(sorted(keys))

                missing = sorted(REQUIRED_KEYS - keys)
                if missing:
                    status = "missing_keys"
                    missing_keys = ",".join(missing)
                else:
                    parts = []
                    for k in sorted(REQUIRED_KEYS):
                        try:
                            arr = data[k]
                            parts.append(f"{k}:{tuple(arr.shape)}")
                            if arr.size == 0:
                                status = "empty_array"
                        except Exception as exc:
                            status = "bad_array"
                            error = f"{type(exc).__name__}: {exc}"
                            break
                    shape_summary = ";".join(parts)

            except Exception as exc:
                status = "load_error"
                error = f"{type(exc).__name__}: {exc}"

        out = dict(row)
        out.update(
            {
                "status": status,
                "error": error,
                "keys_present": keys_present,
                "missing_keys": missing_keys,
                "shape_summary": shape_summary,
            }
        )
        rows.append(out)

    report = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    print(f"Wrote validation report: {out_csv}")
    print(report["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
