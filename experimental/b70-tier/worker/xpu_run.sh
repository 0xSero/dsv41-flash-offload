#!/bin/bash
# One Intel B70 lock by stable PCI identity; GPU0 and Qwen PCI03 are excluded.
# usage: xpu_run.sh <0000:c3:00.0|0000:84:00.0> <owner-tag> <command...>
set -euo pipefail
pci=$1;tag=$2;shift 2
case "$pci" in 0000:c3:00.0|0000:84:00.0) ;; *) echo "REFUSE protected/unknown PCI $pci" >&2;exit 64;; esac
node=
for path in /sys/class/drm/renderD*; do
 if [ "$(basename "$(readlink -f "$path/device")")" = "$pci" ];then node=/dev/dri/$(basename "$path");fi
done
[ -n "$node" ] && [ "$node" != /dev/dri/renderD131 ] || { echo "REFUSE absent/protected render node" >&2;exit 65; }
[ "$(basename "$(readlink -f /sys/bus/pci/devices/$pci/driver)")" = xe ] || { echo "REFUSE non-xe device" >&2;exit 65; }
key=${pci//[:.]/_};dir=$HOME/freetoken-exl3/locks;mkdir -p "$dir"
exec 5>"$dir/xpu-$key.lock";flock 5
printf '%s %s %s\n' "$tag" "$node" "$(date -Is)" > "$dir/xpu-$key.owner"
trap ': > "$dir/xpu-$key.owner"' EXIT
export DSV41_XPU_PCI=$pci DSV41_XPU_RENDER=$node
"$@"
