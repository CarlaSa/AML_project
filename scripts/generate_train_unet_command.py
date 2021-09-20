import pandas as pd
# from scripts.train_unet import get_abbrev

# node_table = pd.read_csv('artifacts/cluster_gpu_nodes.csv')
job_table = pd.read_csv('artifacts/unet_training_runs.csv',
                        keep_default_na=False, na_values=[''])

for index, row in job_table[job_table["dir"].isna()].iterrows():
    command = (["--do-batch-norm"] if row.do_batch_norm is True else []) + [
        f"--batch-size={row.batch_size}",
        f"--n-blocks={row.n_blocks}",
        f"--n-initial-block-channels={row.n_initial_block_channels}",
        f"--learning-rate={row.learning_rate}",
        f"--epochs={row.epochs}",
        f"--adam-regul-factor={row.adam_regul_factor}",
        f"--p-dropout={row.p_dropout}",
        str(row.criterion),
        str(row.augmentation)
    ]
    # abbrev = get_abbrev(command)
    print(" ".join(command))
