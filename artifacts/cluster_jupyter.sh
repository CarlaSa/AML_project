set -e

cluster='cluster'
node='gpu08'

sbatch='/opt/slurm/bin/sbatch'
squeue='/opt/slurm/bin/squeue'
remote_home=$(ssh "$cluster" pwd)
jupyter_script="$remote_home/lslurm/jupyter_notebook_gpu.sh"
repo="$remote_home/AML_project"
fix_path='PATH=~/.pyenv/bin:~/.local/bin:/opt/slurm/bin:$PATH'

get_free_port () {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); \
              print(s.getsockname()[1]); s.close()'
}
localport="$(get_free_port)"

echo "Using local port $localport."
echo
ssh "$cluster" "$squeue"
echo
echo "Starting Jupyter on $node."
ssh "$cluster" bash -c "$fix_path; cd $repo && $sbatch $jupyter_script"
echo
ssh "$cluster" "$squeue"
echo
sleep 3
ssh "$cluster" "$squeue"
sleep 3
ssh "$cluster" "$squeue"
echo

echo "Reading remote Jupyter port and token."
read port token <<<`ssh -J "$cluster" "$node" \
  "$fix_path; cd $repo && pipenv run jupyter notebook list" | \
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
echo "$url"
xdg-open "$url"

echo "Please remember to shut it down when your task is finished."
