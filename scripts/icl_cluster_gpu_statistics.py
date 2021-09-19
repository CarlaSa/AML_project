import json
import subprocess
import pandas as pd
from tqdm import tqdm
from warnings import warn
from . import print_cluster_gpu_props_json

TIMEOUT = 20
OUTFILE = "artifacts/cluster_gpu_nodes.csv"


def main():
    with open(print_cluster_gpu_props_json.__file__, 'rb') as f:
        src = f.read()

    table: pd.DataFrame = pd.DataFrame()
    pwd: str = ""

    print("polling output to", OUTFILE)
    for i in tqdm(range(12)):
        node = f"gpu{i+1:02d}"
        if not pwd:
            command = ["ssh", node, "bash", "lslurm/nqron.sh", "pwd"]
            p = subprocess.Popen(command,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            outs, errs = p.communicate(input=src, timeout=TIMEOUT)
            if errs:
                warn(errs.decode("UTF-8"))
            else:
                pwd = outs.decode("UTF-8").strip()
                print(f"pwd is '{pwd}'")
        command = ["ssh", node, "bash", "lslurm/nqron.sh", "python3"]
        p = subprocess.Popen(command,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        outs, errs = p.communicate(input=src, timeout=TIMEOUT)
        table = table.append(dict(node=node, **json.loads(outs), errors=errs),
                             ignore_index=True)

    table.errors = table.errors.str.decode("UTF-8").str.replace(f"{pwd}/", "")
    table = table.convert_dtypes()
    table.to_csv(OUTFILE, index=False)


if __name__ == '__main__':
    main()
