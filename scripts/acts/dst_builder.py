#!/usr/bin/env python3
"""
PROTOTYPE: Also hard to tell if the conversion is even possible

Convert track-finder CSV rows into a ROOT track ntuple.

This creates a flat ROOT TTree, not a native Fun4All/sPHENIX DST. Native DST
production requires the sPHENIX software stack and its event/node definitions.
"""

import argparse
import csv
import sys
from pathlib import Path


# Canonical output branch -> accepted input column names (case-insensitive).
ALIASES = {
	"event_id": ("event_id", "event", "evt"),
	"track_id": ("track_id", "track", "trk"),
	"x": ("x", "vx", "pos_x"),
	"y": ("y", "vy", "pos_y"),
	"z": ("z", "vz", "pos_z"),
	"px": ("px", "p_x", "mom_x"),
	"py": ("py", "p_y", "mom_y"),
	"pz": ("pz", "p_z", "mom_z"),
	"charge": ("charge", "q"),
	"chi2": ("chi2", "chisq"),
	"ndf": ("ndf", "n_dof"),
}


def convert_csv(input_path):
	"""Read tracks. Position, chi2, and ndf default to zero when absent."""
	columns = {name: [] for name in ALIASES}
	with input_path.open(newline="", encoding="utf-8-sig") as stream:
		reader = csv.DictReader(stream)
		if not reader.fieldnames:
			raise ValueError("input CSV must have a header row")
		headers = {header.strip().lower(): header for header in reader.fieldnames}
		missing = [
			name for name in ("px", "py", "pz", "charge")
			if not any(alias in headers for alias in ALIASES[name])
		]
		if missing:
			raise ValueError("missing required columns: " + ", ".join(missing))

		for line, row in enumerate(reader, start=2):
			if not any((value or "").strip() for value in row.values()):
				continue
			for name, aliases in ALIASES.items():
				raw = next(
					(row[headers[key]].strip() for key in aliases
					 if key in headers and row.get(headers[key], "").strip()),
					None,
				)
				if name == "event_id":
					value = 0 if raw is None else int(raw)
				elif name == "track_id":
					value = len(columns[name]) if raw is None else int(raw)
				elif name == "ndf":
					value = 0 if raw is None else int(raw)
				else:
					value = 0.0 if raw is None and name in ("x", "y", "z", "chi2") else (
						None if raw is None else float(raw)
					)
				if value is None:
					raise ValueError(f"line {line}: missing required value for {name}")
				columns[name].append(value)

	if not columns["track_id"]:
		raise ValueError("input CSV contains no tracks")
	return columns


def write_root(columns, output_path, tree_name):
	try:
		import numpy as np
		import uproot
	except ImportError as exc:
		raise RuntimeError("install ROOT-writing dependencies with: pip install numpy uproot") from exc

	arrays = {
		name: np.asarray(values, dtype=np.int64 if name in ("event_id", "track_id", "ndf") else np.float64)
		for name, values in columns.items()
	}
	with uproot.recreate(output_path) as root_file:
		root_file[tree_name] = arrays


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input_csv", type=Path)
	parser.add_argument("output_root", type=Path)
	parser.add_argument("--tree", default="tracks", help="output TTree name")
	args = parser.parse_args(argv)
	try:
		columns = convert_csv(args.input_csv)
		write_root(columns, args.output_root, args.tree)
	except (OSError, ValueError, RuntimeError) as exc:
		print(f"dst_builder: {exc}", file=sys.stderr)
		return 2
	print(f"Wrote {len(columns['track_id'])} tracks to {args.output_root}:{args.tree}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
