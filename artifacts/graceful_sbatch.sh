#!/usr/bin/bash
max_enqueued=3

enqueued () {
  squeue --user=$(whoami) --noheader | wc -l
}

while read -r line; do
  read -r enqueuand < /dev/tty
  while [ $(enqueued) -ge $max_enqueued ]; do
    sleep 5
  done
  sbatch $enqueuand
done
