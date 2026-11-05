#!/usr/bin/env python3
"""Select a real OpenFOAM cell centre and verify its rank ownership.

The binary/ASCII reader handles the cell-centre volVectorField and labelLists
written by OpenFOAM v2606. It intentionally does not interpret the mesh or
invent a nearby coordinate: a selected centre belongs to a particular cell in
constant/polyMesh/cellProcAddressing.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

WS = b" \t\r\n"
DELIMS = b"();[]"


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.binary = False
        self.scalar_bytes = 8
        self.label_bytes = 4

    def skip(self) -> None:
        while self.pos < len(self.data):
            if self.data[self.pos] in WS:
                self.pos += 1
            elif self.data.startswith(b"//", self.pos):
                end = self.data.find(b"\n", self.pos)
                self.pos = len(self.data) if end < 0 else end + 1
            elif self.data.startswith(b"/*", self.pos):
                end = self.data.find(b"*/", self.pos + 2)
                if end < 0:
                    raise ValueError("unterminated comment")
                self.pos = end + 2
            else:
                break

    def token(self) -> bytes:
        self.skip()
        if self.pos >= len(self.data):
            raise ValueError("unexpected end of file")
        if self.data[self.pos:self.pos + 1] in b"(){}[];":
            self.pos += 1
            return self.data[self.pos - 1:self.pos]
        if self.data[self.pos:self.pos + 1] == b'"':
            end = self.pos + 1
            while end < len(self.data) and self.data[end:end + 1] != b'"':
                end += 2 if self.data[end:end + 1] == b"\\" else 1
            if end >= len(self.data):
                raise ValueError("unterminated quoted token")
            self.pos = end + 1
            return self.data[self.pos:end + 1]
        start = self.pos
        angle_depth = 0
        while self.pos < len(self.data):
            c = self.data[self.pos:self.pos + 1]
            if c == b"<":
                angle_depth += 1
            elif c == b">":
                angle_depth -= 1
            elif angle_depth == 0 and (c in WS or c in DELIMS or c in b"{}"):
                break
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"unexpected delimiter at byte {self.pos}")
        return self.data[start:self.pos]

    def expect(self, expected: bytes) -> None:
        actual = self.token()
        if actual != expected:
            raise ValueError(f"expected {expected!r}, got {actual!r} at {self.pos}")

    def header(self) -> None:
        match = re.search(rb"\bFoamFile\s*\{", self.data)
        if not match:
            raise ValueError("no FoamFile header")
        self.pos = match.end()
        entries = self.dictionary()
        self.binary = entries.get(b"format") == b"binary"
        arch = entries.get(b"arch", b"").decode("ascii", "ignore")
        if arch and "LSB" not in arch:
            raise ValueError("only LSB binary OpenFOAM fields are supported")
        self.scalar_bytes = 4 if "scalar=32" in arch else 8
        self.label_bytes = 8 if "label=64" in arch else 4

    def dictionary(self) -> dict[bytes, bytes]:
        entries: dict[bytes, bytes] = {}
        while True:
            key = self.token()
            if key == b"}":
                return entries
            value = []
            while True:
                self.skip()
                if self.data.startswith(b";", self.pos):
                    self.pos += 1
                    break
                value.append(self.token())
            entries[key] = b" ".join(value)

    def numeric_list(self, kind: str) -> np.ndarray:
        ncomp = {"scalar": 1, "vector": 3, "label": 1}[kind]
        self.expect(b"(")
        width = self.label_bytes if kind == "label" else self.scalar_bytes
        dtype = np.dtype(("<i%d" if kind == "label" else "<f%d") % width)
        count = self.n * ncomp
        if self.binary:
            nbytes = count * width
            values = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.pos).copy()
            self.pos += nbytes
            self.expect(b")")
        else:
            values = np.empty(count, dtype=dtype)
            for i in range(count):
                if ncomp > 1 and i % ncomp == 0:
                    self.expect(b"(")
                values[i] = int(self.token()) if kind == "label" else float(self.token())
                if ncomp > 1 and i % ncomp == ncomp - 1:
                    self.expect(b")")
            self.expect(b")")
        return values.reshape((self.n, ncomp)) if ncomp > 1 else values

    n = 0


def read_centres(path: pathlib.Path) -> np.ndarray:
    r = Reader(path.read_bytes())
    r.header()
    while True:
        key = r.token()
        if key == b"internalField":
            qualifier, list_type, count = r.token(), r.token(), r.token()
            if qualifier != b"nonuniform" or list_type != b"List<vector>":
                raise ValueError(f"{path}: C must have a nonuniform List<vector> internalField")
            r.n = int(count)
            break
        if key == b"}":
            raise ValueError(f"{path}: missing internalField")
        # Only scalar header entries precede internalField in writeCellCentres.
        while r.token() != b";":
            pass
    result = r.numeric_list("vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: non-finite cell centres")
    return result


def read_labels(path: pathlib.Path) -> np.ndarray:
    r = Reader(path.read_bytes())
    r.header()
    r.n = int(r.token())
    return r.numeric_list("label")


def parse_point(text: str) -> np.ndarray:
    vals = [float(v) for v in text.strip().strip("()").replace(",", " ").split()]
    if len(vals) != 3 or not np.isfinite(vals).all():
        raise ValueError(f"need three finite coordinates, got {text!r}")
    return np.asarray(vals)


def choose(centres: np.ndarray, point: np.ndarray) -> tuple[int, np.ndarray]:
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) == 0:
        raise ValueError("no cell centres")
    # Chunking bounds temporary memory for the 2.48M-cell G4 mesh.
    best_index = -1
    best_distance = float("inf")
    for start in range(0, len(centres), 100000):
        distance = np.einsum("ij,ij->i", centres[start:start + 100000] - point,
                             centres[start:start + 100000] - point)
        local = int(np.argmin(distance))
        if float(distance[local]) < best_distance:
            best_distance = float(distance[local])
            best_index = start + local
    return best_index, centres[best_index]


def verify(case: pathlib.Path, np_expected: int, centre: np.ndarray,
           centres_path: pathlib.Path) -> None:
    centres = read_centres(centres_path)
    distances = np.linalg.norm(centres - centre, axis=1)
    matches = np.flatnonzero(distances == 0)
    if len(matches) != 1:
        raise ValueError(f"selected centre matches {len(matches)} cells, expected exactly one")
    cell = int(matches[0])
    owners = []
    for rank in range(np_expected):
        if np_expected == 1:
            if not (case / "0/alpha.water").is_file():
                raise ValueError("serial case has no initialized 0/alpha.water")
            owners.append((0, cell))
            break
        mapping_path = case / f"processor{rank}/constant/polyMesh/cellProcAddressing"
        mapping = read_labels(mapping_path)
        local = np.flatnonzero(mapping == cell)
        if len(local) > 1:
            raise ValueError(f"rank {rank} maps global cell {cell} more than once")
        if len(local) == 1:
            owners.append((rank, int(local[0])))
    if len(owners) != 1:
        raise ValueError(f"global reference cell {cell} is on {len(owners)} domains, expected one")
    print(f"REFERENCE_OWNER globalCell={cell} rank={owners[0][0]} localCell={owners[0][1]} "
          f"np={np_expected} point=({centre[0]:.17g} {centre[1]:.17g} {centre[2]:.17g})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-decomposed", nargs=3,
                        metavar=("CASE", "NP", "CENTRE"))
    parser.add_argument("--centres", dest="global_centres", type=pathlib.Path,
                        help="global constant/C retained for ownership verification")
    parser.add_argument("centres", type=pathlib.Path, nargs="?")
    parser.add_argument("point", nargs="?", default="0,0,0.2")
    args = parser.parse_args()
    if args.verify_decomposed:
        case, np_text, centre_text = args.verify_decomposed
        if not args.global_centres:
            parser.error("--verify-decomposed requires --centres global-C")
        verify(pathlib.Path(case), int(np_text), parse_point(centre_text), args.global_centres)
        return
    if args.centres is None:
        parser.error("select mode requires constant/C and target point")
    centres = read_centres(args.centres)
    index, centre = choose(centres, parse_point(args.point))
    print(f"({centre[0]:.17g} {centre[1]:.17g} {centre[2]:.17g})")
    print(f"Selected global cell {index} nearest to {args.point}", file=sys.stderr)


if __name__ == "__main__":
    main()
