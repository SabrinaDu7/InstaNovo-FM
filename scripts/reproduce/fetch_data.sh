#!/bin/bash
# Fetch the InstaNovo-FM data and checkpoints onto nibi (started 2026-09-25):
#   $DATA/splits/mcfm/     every MCFM shard (train, validation, test)      about 52 GiB
#   $DATA/splits/lcfm/     the LCFM validation and test shards only        about 162 GiB (train is 305 GiB more)
#   $DATA/                 peptide_registry.parquet, the manifests, and the Hugging Face tree listing (sizes, sha256)
#   $CHECKPOINTS/          the eight v0.1.0 release checkpoints
# Resumable (curl -C -); one stream; log lines "done <path> <time>" or "FAILED <path> <time>", "ALL_DONE" at the end.
#   setsid nohup bash scripts/reproduce/fetch_data.sh > "$DATA/fetch.log" 2>&1 &
set -u
DATA="${DATA:-$HOME/projects/rrg-hsn/proteomies/data/proteometoolsI}"
CHECKPOINTS="${CHECKPOINTS:-$HOME/projects/rrg-hsn/proteomies/checkpoints/instanovofm}"
HF="https://huggingface.co/datasets/InstaDeepAI/InstaNovo"
mkdir -p "$DATA/splits/mcfm" "$DATA/splits/lcfm" "$DATA/manifests" "$CHECKPOINTS"

tree() {  # repo subtree -> JSON list of file entries (path, size, lfs sha256), saved for verification
  python3 - "$1" "$DATA/hf_tree_$(echo "$1" | tr '/' '_').json" <<'PY'
import json, sys, urllib.request
sub, out = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/api/datasets/InstaDeepAI/InstaNovo/tree/main/{sub}?recursive=true&expand=false"
entries = []
while url:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        entries += [e for e in json.load(r) if e.get("type") == "file"]
        link = r.headers.get("Link", "")
    url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
json.dump(entries, open(out, "w"), indent=1)
for e in entries: print(e["path"])
PY
}
fetch() {  # $1 repo path, $2 destination directory
  local name; name=$(basename "$1")
  if curl -sL --retry 5 --retry-delay 10 -C - -o "$2/$name" "$HF/resolve/main/$1"; then echo "done $1 $(date +%T)"; else echo "FAILED $1 $(date +%T)"; fi
}

echo "START $(date)"
for id in instanovo-fm-v0.1.0 instanovo-fm-mcfm-90k-v0.1.0 instanovo-fm-lcfm-ts-pa-v0.1.0 instanovo-fm-lcfm-sa-nopa-v0.1.0 \
          instanovo-fm-lcfm-sa-pa-v0.1.0 instanovo-fm-denovo-v0.1.0 instanovo-fm-denovo-frozen-v0.1.0 instanovo-fm-denovo-scratch-v0.1.0; do
  if curl -sL --retry 5 --retry-delay 10 -C - -o "$CHECKPOINTS/$id.ckpt" "https://github.com/instadeepai/InstaNovo-FM/releases/download/v0.1.0/$id.ckpt"; then echo "done checkpoint $id $(date +%T)"; else echo "FAILED checkpoint $id $(date +%T)"; fi
done
fetch peptide_registry.parquet "$DATA"
for f in $(tree manifests); do fetch "$f" "$DATA/manifests"; done
for f in $(tree splits/mcfm); do fetch "$f" "$DATA/splits/mcfm"; done
for f in $(tree splits/lcfm | grep -E "valid|test"); do fetch "$f" "$DATA/splits/lcfm"; done
echo "ALL_DONE $(date)"
