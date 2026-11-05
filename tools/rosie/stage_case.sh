#!/usr/bin/env bash
# Stage the initialized G4 case, without writing to the source sketch.
# Laptop / OpenFOAM v2606 only; this does not submit jobs or contact Rosie.
#   tools/rosie/stage_case.sh [--np '1 2 4 8'] [--steps 60] [--dt 1e-4] [--tar]
#   CASE=... OUT=... OF_ENV=... tools/rosie/stage_case.sh --np '2 4' --steps 3 --adaptive
# Defaults: the user's G4 sketch and /mnt/storage/brae-scratch/RosieKit/cases.
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SKETCH=$HOME/code/sketches/g4_tank_refill_interfoam
CASE=${CASE:-$DEFAULT_SKETCH/case}
[[ -d $CASE ]] || CASE=$DEFAULT_SKETCH
OUT=${OUT:-/mnt/storage/brae-scratch/RosieKit/cases}
OF_ENV=${OF_ENV:-/mnt/storage/brae-scratch/HaloOps/oracle/of_env.sh}
NPS=(1 2 4 8)
STEPS=60
DT=1e-4
ADAPTIVE=0
MAKE_TAR=0
DISABLE_UNPORTED_FO=${DISABLE_UNPORTED_FO:-1}
REFERENCE_POINT=${REFERENCE_POINT:-0,0,0.2}
SKIP_CENTRE_CHECK=${SKIP_CENTRE_CHECK:-0}
TAR_DIR=${TAR_DIR:-}

usage() {
    cat <<'EOF'
Usage: stage_case.sh [options]
  --np '1 2 4 8'  decompositions to stage (default: all four)
  --steps N       number of fixed timing steps (default: 60)
  --dt SECONDS    fixed deltaT (default: 0.0001)
  --adaptive      retain adjustTimeStep yes; --steps is then a run-time budget,
                  not a guaranteed number of iterations
  --tar           write OUT/g4_npN.tar.zst (one top-level g4_npN/ per archive)
  --ref-point X,Y,Z
                  choose the cell centre nearest this point (default: 0,0,0.2)
Environment: CASE (case dir or sketch), OUT, OF_ENV, TAR_DIR, ZST_LEVEL (3),
             DISABLE_UNPORTED_FO=0 (leave unported FOs enabled for OpenFOAM).
The source case is copied before every dictionary change. 0/ is used, never
0.orig/. For G4, 0/alpha.water must be a nonuniform initialized field. The
pressure reference is the cell centre nearest (0 0 0.2), not that axis vertex:
OpenFOAM otherwise reports it as found on multiple domains. The same actual
cell centre is used at every np, and its owner count is checked after each
decomposition. Generated C/Cx/Cy/Cz are removed from the staged runtime case.
The G4 `surfaces` FO is retained with `enabled false`: brae intentionally
ignores unported FO types, but OpenFOAM would execute it. Other FOs, including
probes and pulseStop, remain as in the source case. See README.md for this
correctness limitation and the stage-1 stop condition.
EOF
}
while (($#)); do
    case "$1" in
        --np) IFS=' ' read -r -a NPS <<< "$2"; shift ;;
        --steps) STEPS=$2; shift ;;
        --dt) DT=$2; shift ;;
        --adaptive) ADAPTIVE=1 ;;
        --tar) MAKE_TAR=1 ;;
        --ref-point) REFERENCE_POINT=$2; shift ;;
        --disable-unported-fo) DISABLE_UNPORTED_FO=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
[[ $STEPS =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be a positive integer" >&2; exit 2; }
[[ $DT =~ ^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ ]] || { echo "--dt must be a positive number" >&2; exit 2; }
[[ -d $CASE/system && -d $CASE/constant && -d $CASE/0 ]] || {
    echo "CASE must contain system/, constant/ and initialized 0/: $CASE" >&2; exit 1;
}
CASE=$(cd -- "$CASE" && pwd)
[[ -r $OF_ENV ]] || { echo "OpenFOAM v2606 environment not readable: $OF_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$OF_ENV"
for command_name in decomposePar foamDictionary postProcess python3 tar; do
    command -v "$command_name" >/dev/null || { echo "missing OpenFOAM/tool command: $command_name" >&2; exit 1; }
done
if ((MAKE_TAR)); then
    command -v zstd >/dev/null || { echo "--tar needs zstd" >&2; exit 1; }
    mkdir -p "${TAR_DIR:-$OUT}"
fi
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
TAR_DIR=${TAR_DIR:-$OUT}
mkdir -p "$TAR_DIR"
TAR_DIR=$(cd -- "$TAR_DIR" && pwd)

# Guard against accidentally using the dry 0.orig/alpha.water. Read only the
# field's text header; its payload can be binary and hundreds of megabytes.
python3 - "$CASE/0/alpha.water" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
data = p.read_bytes()[:65536]
text = data.decode("ascii", "ignore")
m = re.search(r"internalField\s+(uniform|nonuniform)\b", text)
if not m:
    raise SystemExit(f"{p}: no recognizable internalField; pass an initialized case explicitly")
if m.group(1) != "nonuniform":
    raise SystemExit(f"{p}: alpha.water is uniform (0.orig is NOT initialized); use the sketch's 0/")
print(f"Initialized alpha: {p} ({m.group(1)})")
PY

# All dictionary changes use OpenFOAM's own parser/writer. Preserve comments and
# unsupported entries wherever possible; never alter the sketch or its 0.orig.
configure_case() {
    local dir=$1 np=$2
    foamDictionary -entry startFrom -set startTime "$dir/system/controlDict" >/dev/null
    foamDictionary -entry startTime -set 0 "$dir/system/controlDict" >/dev/null
    foamDictionary -entry stopAt -set endTime "$dir/system/controlDict" >/dev/null
    foamDictionary -entry endTime -set "$END_TIME" "$dir/system/controlDict" >/dev/null
    foamDictionary -entry deltaT -set "$DT" "$dir/system/controlDict" >/dev/null
    foamDictionary -entry writeControl -set timeStep "$dir/system/controlDict" >/dev/null
    foamDictionary -entry writeInterval -set "$STEPS" "$dir/system/controlDict" >/dev/null
    foamDictionary -entry purgeWrite -set 1 "$dir/system/controlDict" >/dev/null
    foamDictionary -entry writeFormat -set binary "$dir/system/controlDict" >/dev/null
    foamDictionary -entry writeCompression -set off "$dir/system/controlDict" >/dev/null
    if ((ADAPTIVE)); then
        foamDictionary -entry adjustTimeStep -set yes "$dir/system/controlDict" >/dev/null
    else
        foamDictionary -entry adjustTimeStep -set no "$dir/system/controlDict" >/dev/null
    fi
    if [[ ! -f $dir/system/decomposeParDict ]]; then
        cat > "$dir/system/decomposeParDict" <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains 1;
method scotch;
DICT
    fi
    foamDictionary -entry numberOfSubdomains -set "$np" "$dir/system/decomposeParDict" >/dev/null
    foamDictionary -entry method -set scotch "$dir/system/decomposeParDict" >/dev/null
    python3 - "$dir/system/fvSolution" "$CELL_CENTRE" <<'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
updated, count = re.subn(r"(?m)^(\s*pRefPoint\s+)\([^;]*\)(\s*;)", r"\g<1>" + sys.argv[2] + r"\2", text)
if count != 1:
    raise SystemExit(f"{path}: expected exactly one pRefPoint, found {count}")
path.write_text(updated)
PY
    if [[ $DISABLE_UNPORTED_FO == 1 ]]; then
        # These are the only types that current brae silently ignores. If a
        # different CASE contains one, fail rather than silently dropping it.
        local fo
        while IFS= read -r fo; do
            foamDictionary -entry "functions.$fo.enabled" -set false "$dir/system/controlDict" >/dev/null
            printf 'Disabled unported function object: %s (explicitly, for OpenFOAM equivalence)\n' "$fo"
        done < <(python3 - "$dir/system/controlDict" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
# Match only immediate named blocks in the top-level functions dictionary.
# This intentionally reports only `surfaces`, the unported type in the G4 case.
for match in re.finditer(r"(?m)^    ([A-Za-z_][A-Za-z0-9_]*)\s*\n    \{", text):
    start = match.end()
    block = text[start:text.find("\n    }", start)]
    if re.search(r"\btype\s+surfaces\s*;", block):
        print(match.group(1))
PY
        )
    fi
}

END_TIME=$(python3 - "$STEPS" "$DT" "$ADAPTIVE" <<'PY'
import sys
steps, dt, adaptive = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
if not dt > 0:
    raise SystemExit("--dt must be positive")
if adaptive:
    print(format(steps * dt, ".17g"))
else:
    # Brae's fixed-step loop stops at endTime - 0.5*deltaT. The midpoint of
    # the final interval gives exactly N steps, without rounding at N*dt.
    print(format((steps - 0.5) * dt, ".17g"))
PY
)
CELL_CENTRE=${CELL_CENTRE_OVERRIDE:-}
BASE_NAME=${CASE_LABEL:-g4}
SELECTOR=$HERE/select_reference.py
[[ -f $SELECTOR ]] || { echo "missing $SELECTOR" >&2; exit 1; }

# A scratch tree is used only for cell-centre generation. The serial np=1 case
# becomes the pristine staged template; no centre field or 0.orig is retained.
WORK=$(mktemp -d "$OUT/.stage-${BASE_NAME}.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
cp -a "$CASE" "$WORK/base"
rm -rf "$WORK/base/0.orig" "$WORK/base"/processor*
if [[ ! -f $WORK/base/system/decomposeParDict ]]; then
    cp "$WORK/base/system/decomposeParDict."* "$WORK/base/system/decomposeParDict" 2>/dev/null || true
fi
# Do not carry a previous run's results or generated geometry/field artefacts.
rm -rf "$WORK/base/postProcessing" "$WORK/base"/log.*
rm -f "$WORK/base"/constant/C "$WORK/base"/constant/Cx "$WORK/base"/constant/Cy "$WORK/base"/constant/Cz
{
    echo "Generating cell centres with OpenFOAM v2606 (read-only geometry operation)..."
    foamDictionary -entry writeFormat -set binary "$WORK/base/system/controlDict" >/dev/null
    (cd "$WORK/base" && postProcess -func writeCellCentres -constant -noFunctionObjects \
        > "$WORK/log.writeCellCentres" 2>&1)
    [[ -f $WORK/base/constant/C ]] || { cat "$WORK/log.writeCellCentres" >&2; exit 3; }
    if [[ -z $CELL_CENTRE ]]; then
        CELL_CENTRE=$(python3 "$SELECTOR" "$WORK/base/constant/C" "$REFERENCE_POINT")
    fi
    # Retained outside the staged tree solely for global-cell ownership checks.
    CENTRES_FILE=$WORK/cell_centres.C
    mv "$WORK/base/constant/C" "$CENTRES_FILE"
    rm -f "$WORK/base/constant/Cx" "$WORK/base/constant/Cy" "$WORK/base/constant/Cz"
}
[[ $CELL_CENTRE =~ ^\([^()]+\)$ ]] || { echo "bad selected centre: $CELL_CENTRE" >&2; exit 3; }
echo "Pressure reference (a cell centre, inside exactly one cell): $CELL_CENTRE"

for np in "${NPS[@]}"; do
    [[ $np =~ ^[1-9][0-9]*$ ]] || { echo "invalid np: $np" >&2; exit 2; }
    dest=$OUT/${BASE_NAME}_np${np}
    [[ ! -e $dest ]] || { echo "refusing to overwrite existing $dest; move it away first" >&2; exit 6; }
    cp -a "$WORK/base" "$dest"
    configure_case "$dest" "$np"
    if ((np > 1)); then
        echo "Decomposing $dest with scotch, np=$np"
        (cd "$dest" && decomposePar -force -time 0 > "$OUT/log.decompose.${BASE_NAME}_np${np}" 2>&1)
        for ((rank = 0; rank < np; ++rank)); do
            [[ -f $dest/processor$rank/0/alpha.water ]] || {
                echo "missing decomposed initial alpha at rank $rank" >&2; exit 3;
            }
        done
        python3 "$SELECTOR" --verify-decomposed "$dest" "$np" "$CELL_CENTRE" --centres "$CENTRES_FILE" \
            > "$OUT/log.reference.${BASE_NAME}_np${np}"
        cat "$OUT/log.reference.${BASE_NAME}_np${np}"
        # decomposePar leaves a global mesh and 0/cellDist alongside the ranks.
        # They are decomposition outputs, not brae -parallel inputs.
        rm -rf "$dest/0" "$dest/constant/polyMesh"
    else
        python3 "$SELECTOR" --verify-decomposed "$dest" 1 "$CELL_CENTRE" --centres "$CENTRES_FILE" \
            > "$OUT/log.reference.${BASE_NAME}_np1"
        cat "$OUT/log.reference.${BASE_NAME}_np1"
    fi
    # Geometry inputs for regenerating a mesh are not runtime inputs and are
    # large. Keep dictionaries, physical properties, cellZones and the mesh.
    rm -rf "$dest/constant/triSurface"
    cat > "$dest/STAGED_RUN.txt" <<EOF
Source case: $CASE
Initial fields: source 0/ (initialized alpha.water; not 0.orig/)
Decomposition: scotch, np=$np, OpenFOAM v2606
Pressure reference: $CELL_CENTRE (cell centre nearest $REFERENCE_POINT, inside one cell)
Timing control: steps=$STEPS dt=$DT endTime=$END_TIME adjustTimeStep=$( ((ADAPTIVE)) && echo yes || echo no )
Write control: timeStep, interval=$STEPS, purgeWrite=1, binary, compression off
Function objects: preserved. Unported surfaces disabled when DISABLE_UNPORTED_FO=1.
Archives exclude triSurface, 0.orig, previous output and temporary cell centres.
EOF
    bytes=$(du -sb "$dest" | awk '{print $1}')
    printf 'STAGED np=%d path=%s bytes=%d (%.2f MiB)\n' "$np" "$dest" "$bytes" "$(awk -v b="$bytes" 'BEGIN { print b/1048576 }')"
    if ((MAKE_TAR)); then
        archive=$TAR_DIR/${BASE_NAME}_np${np}.tar.zst
        tmp_archive=$archive.partial
        rm -f "$tmp_archive"
        tar -C "$OUT" -cf - "${BASE_NAME}_np${np}" | zstd -q -"${ZST_LEVEL:-3}" -T0 -f -o "$tmp_archive"
        mv "$tmp_archive" "$archive"
        zstd -q -t "$archive"
        tar_bytes=$(stat -c %s "$archive")
        printf 'ARCHIVE np=%d path=%s bytes=%d (%.2f MiB)\n' "$np" "$archive" "$tar_bytes" \
            "$(awk -v b="$tar_bytes" 'BEGIN { print b/1048576 }')"
    fi
done

echo "STAGING COMPLETE: $OUT"
echo "At runtime every np uses the same initial fields and pRefPoint $CELL_CENTRE."
