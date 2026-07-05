#!/usr/bin/env bash
# Robustly download torchvision pretrained weights into the torch hub cache with a
# hash-verified retry loop (works around the flaky ~throttled network that corrupts
# large downloads mid-flight). Idempotent: skips files already present & valid.
CACHE=/root/.cache/torch/hub/checkpoints
mkdir -p "$CACHE"
declare -A URLS=(
  [vgg16-397923af.pth]="https://download.pytorch.org/models/vgg16-397923af.pth"
  [resnet101-63fe2227.pth]="https://download.pytorch.org/models/resnet101-63fe2227.pth"
  [densenet121-a639ec97.pth]="https://download.pytorch.org/models/densenet121-a639ec97.pth"
)
for fname in "${!URLS[@]}"; do
  url="${URLS[$fname]}"
  want="${fname#*-}"; want="${want%.pth}"        # 8-hex sha256 prefix from filename
  final="$CACHE/$fname"
  tmp="$CACHE/.dl_$fname"
  # already valid?
  if [ -f "$final" ]; then
    got=$(sha256sum "$final" | cut -c1-8)
    [ "$got" = "$want" ] && { echo "OK cached  $fname"; continue; }
    echo "stale $fname (hash $got != $want) -> refetch"; rm -f "$final"
  fi
  ok=0
  for attempt in $(seq 1 30); do
    wget -c -q --tries=5 --timeout=40 --waitretry=5 "$url" -O "$tmp"
    [ -f "$tmp" ] || continue
    got=$(sha256sum "$tmp" | cut -c1-8)
    if [ "$got" = "$want" ]; then
      mv -f "$tmp" "$final"; echo "DONE $fname (attempt $attempt, hash ok)"; ok=1; break
    else
      echo "retry $fname attempt $attempt: hash $got != $want ($(stat -c%s "$tmp" 2>/dev/null) bytes) -> discard"
      rm -f "$tmp"                                 # corrupt: restart from scratch
    fi
  done
  [ "$ok" = 0 ] && echo "GIVEUP $fname after 30 attempts"
done
echo "PREFETCH_DONE"
