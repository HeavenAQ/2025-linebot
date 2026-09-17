#!/usr/bin/env bash
# Process the whole manifest on a Mac in short-lived batches.
#
# Each batch is a fresh Python process handling at most BATCH_SIZE videos, so
# memory cannot build up across a 100-video run on a 16 GB machine. Finished
# items are skipped, so the loop stops once a batch has nothing left to do.
#
# Requires: OPENAI_API_KEY, Google credentials (GOOGLE_APPLICATION_CREDENTIALS
# or Application Default Credentials), and PYTHON pointing at an environment
# with badminton_analysis_ai's requirements plus rfdetr.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../../.." && pwd)"
python="${PYTHON:-python3}"
batch_size="${BATCH_SIZE:-4}"
videos_root="${VIDEOS_ROOT:-$HOME/dev/badminton-analysis/scoring_videos}"
log="${LOG_FILE:-$repo/research/gpt-validation/processing/run.log}"

export PYTHONPATH="$repo/badminton_analysis_ai:$repo/badminton_analysis_ai/generated"
cd "$repo"
crashes=0
while true; do
  start_line=$(( $(wc -l <"$log" 2>/dev/null || echo 0) + 1 ))
  "$python" research/gpt-validation/processing/process_videos.py \
    --manifest research/gpt-validation/processing/manifest.csv \
    --videos-root "$videos_root" \
    --limit "$batch_size" "$@" >>"$log" 2>&1 || true
  summary="$(tail -n +"$start_line" "$log" | grep '"validation run finished"' | tail -1 || true)"
  if [[ -z "$summary" ]]; then
    # The process died before finishing (e.g. killed for memory). The item it
    # was on stays "processing" and is picked up again by the next batch.
    crashes=$((crashes + 1))
    echo "batch ended without a summary (${crashes} in a row)" >>"$log"
    if (( crashes >= 3 )); then
      echo "giving up after repeated crashes" >>"$log"
      exit 1
    fi
    continue
  fi
  crashes=0
  if tail -n +"$start_line" "$log" | grep -q '"coaching unavailable; stopping run"'; then
    echo "GPT feedback unavailable (check OpenAI credits); stopping" >>"$log"
    exit 1
  fi
  processed="$(sed -E 's/.*"processed": ([0-9]+).*/\1/' <<<"$summary")"
  echo "batch finished: processed=${processed}" >>"$log"
  if [[ "$processed" == "0" ]]; then
    echo "nothing left to process" >>"$log"
    break
  fi
done
