#!/usr/bin/env bash
# Hard-kill watcher: sleeps until TARGET_EPOCH, then terminates all python.exe.
# Designed so running sweeps die cleanly; resume-safe checkpointing preserves progress.
set -u
TARGET_EPOCH="${1:-}"
LOG="${2:-/tmp/alarm_kill.log}"
if [[ -z "$TARGET_EPOCH" ]]; then
  echo "usage: $0 <target_epoch_seconds> [logfile]" >&2; exit 2
fi
echo "[alarm] started $(date) target=$(date -d @"$TARGET_EPOCH") pid=$$" | tee -a "$LOG"
while :; do
  NOW=$(date +%s)
  REM=$((TARGET_EPOCH - NOW))
  if (( REM <= 0 )); then break; fi
  # Sleep in <=5min chunks so we can abort via external kill and get periodic heartbeats.
  if (( REM > 300 )); then sleep 300; else sleep "$REM"; fi
  echo "[alarm] heartbeat $(date) remaining=${REM}s" >> "$LOG"
done
echo "[alarm] FIRING $(date) — killing python.exe" | tee -a "$LOG"
taskkill //F //IM python.exe 2>&1 | tee -a "$LOG" || true
# Second pass for any stragglers after 10s.
sleep 10
taskkill //F //IM python.exe 2>&1 | tee -a "$LOG" || true
echo "[alarm] done $(date)" | tee -a "$LOG"
