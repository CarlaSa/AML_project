#!/usr/bin/bash

set -e
source $HOME/lslurm/source.sh
cd $HOME/AML_project

train_unet="pipenv run python3 -m scripts.train_unet"
# abbrev=$($train_unet --get-abbrev-only "$@")
path=$($train_unet --get-path-only "$@")
# device_count=$($train_unet --get-cuda-device-count-only "$@")

mkdir "$path"
command="$train_unet --get-path-only $@"
stdbuf -oL bash -c "$command" 1> "$path/out.txt" 2> "$path/err.txt" &
echo "$!@$(hostname)" > "$path/pid.txt"
disown
