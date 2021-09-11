#!/usr/bin/bash
set -e

cluster='cluster'
node="${2:?'Please specify the computer node.'}"

sbatch='/opt/slurm/bin/sbatch'
squeue='/opt/slurm/bin/squeue'
remote_home=$(ssh "$cluster" pwd)
jupyter_script="$remote_home/lslurm/jupyter_notebook_gpu.sh"
repo="${1:?'Please specify the repo as the first argument.'}"
source='source lslurm/source.sh'
user="$(ssh $cluster whoami)"

get_free_port () {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); \
              print(s.getsockname()[1]); s.close()'
}
query_job_status() {
  read job_id job_status job_time <<<`ssh "$cluster" "$source;
    squeue --noheader --user=$user --name=jupyternb --format='%i %T %M' |
    head -n1"`
}
localport="$(get_free_port)"

echo "Using local port $localport."
echo
echo "Other SLURM jobs running:"
ssh "$cluster" "$squeue"
echo
echo "Starting Jupyter on $node."
ssh "$cluster" "$source; cd $repo && $sbatch $jupyter_script"

while [ "$job_status" != "RUNNING" ]; do
  query_job_status
  echo -ne "$job_id: $job_status for $job_time \033[0K\r"
  sleep 1
done
sleep 2
echo

echo "Reading remote Jupyter port and token."
read port token <<<`ssh -J "$cluster" "$node" \
  "$source; cd $repo && pipenv run jupyter notebook list" | \
  tail -n1 | sed -E 's~.*localhost:([0-9]+).*token=([a-z0-9]+).*~\1 \2~'`
re='^[0-9]+$'
if ! [[ $port =~ $re ]]; then
  echo "port is $port."
  echo "token is $token."
  echo "error: Jupyter server not running" >&2;
  exit 1
fi
url="http://localhost:$localport/?token=$token"
echo "Token: $token"

echo "Creating port forward."
ssh -NL "$localport":localhost:"$port" "$node" &

echo "Jupyter should be ready at the following URL:"
echo "  $url"
xdg-open "$url"

echo "Please remember to shut it down when your task is finished, or kill it:"
echo "  scancel $job_id"

wait
