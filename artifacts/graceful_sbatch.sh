#!/usr/bin/bash
max_enqueued=3

enqueued () {
  squeue --user=$(whoami) --noheader | wc -l
}

mapfile -t lines
for enqueuand in "${lines[@]}"; do
  while [ $(enqueued) -ge $max_enqueued ]; do
    sleep 5
  done
  sbatch $enqueuand
done
