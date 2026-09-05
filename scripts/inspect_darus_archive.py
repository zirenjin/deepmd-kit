#!/usr/bin/env python3
"""List DaRUS archive members relevant to the silica thermodynamic tables."""

from __future__ import annotations

import argparse
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        print(f"members={len(names)}")
        for name in names:
            lowered = name.lower()
            if any(
                token in lowered
                for token in (
                    "thermo",
                    "free",
                    "gibbs",
                    "helmholtz",
                    "property",
                    "pbe-d3",
                    "pbe_d3",
                    "pbed3",
                    "tridymite",
                    "quartz",
                    "cristobalite",
                )
            ):
                info = archive.getinfo(name)
                print(f"{info.file_size:12d}  {name}")


if __name__ == "__main__":
    main()
