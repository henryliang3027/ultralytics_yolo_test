#!/usr/bin/env bash
# Run record.py on every camera pair in one folder: each <cam1>_<date>_<start>_<end>.mp4 is paired with the
# <cam2>_ video of the same time window, and the output goes to <folder>/<id-type>/<date>_<start>_<end>/.
#
#   ./record_folder.sh /run/media/ubuntu/TRANS32_2/At_Least_1_Minute/Easy both
#   ./record_folder.sh <folder> [id-type: pid|tid|both, default both] [extra record.py args, e.g. --show]
#
# Env: CAM1 (default 51_203), CAM2 (default 51_205), PYTHON (default ../pvenv/bin/python).
# Pairs whose tracklets.jsonl already exists are skipped; pass --overwrite to redo them.
set -uo pipefail

DIR=${1:?usage: $0 <folder> [pid|tid|both] [record.py args...]}
TYPE=${2:-both}
shift $(( $# >= 2 ? 2 : 1 ))
CAM1=${CAM1:-51_203}
CAM2=${CAM2:-51_205}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-$HERE/../pvenv/bin/python}
OVERWRITE=0
for a in "$@"; do [[ $a == --overwrite ]] && OVERWRITE=1; done

shopt -s nullglob
done_ok=() skipped=() failed=()
for v1 in "$DIR"/"$CAM1"_*.mp4; do
    tag=$(basename "$v1" .mp4); tag=${tag#"$CAM1"_}
    v2=$DIR/${CAM2}_$tag.mp4
    if [[ ! -f $v2 ]]; then
        echo "[record_folder] no $CAM2 video for $tag, skipped"; skipped+=("$tag (no $CAM2)"); continue
    fi
    out=$DIR/$TYPE/$tag
    if [[ -f $out/tracklets.jsonl && $OVERWRITE == 0 ]]; then
        echo "[record_folder] $out/tracklets.jsonl exists, skipped (pass --overwrite to redo)"
        skipped+=("$tag (exists)"); continue
    fi
    echo "[record_folder] === $tag -> $out"
    if (cd "$HERE" && "$PYTHON" record.py "$v1" "$v2" --id-type "$TYPE" --out "$out" "$@"); then
        done_ok+=("$tag")
    else
        failed+=("$tag"); echo "[record_folder] FAILED: $tag"
    fi
done
for v2 in "$DIR"/"$CAM2"_*.mp4; do  # report videos that have no CAM1 partner
    tag=$(basename "$v2" .mp4); tag=${tag#"$CAM2"_}
    [[ -f $DIR/${CAM1}_$tag.mp4 ]] || { echo "[record_folder] no $CAM1 video for $tag, skipped"; skipped+=("$tag (no $CAM1)"); }
done

echo "[record_folder] done ${#done_ok[@]}: ${done_ok[*]:-}"
echo "[record_folder] skipped ${#skipped[@]}: ${skipped[*]:-}"
echo "[record_folder] failed ${#failed[@]}: ${failed[*]:-}"
(( ${#failed[@]} == 0 ))
