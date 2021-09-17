import pandas as pd

table = pd.read_csv('artifacts/unet_training_runs.csv',
                    keep_default_na=False, na_values=[''])
for index, row in table[table["dir"].isna()].iterrows():
    command = (["--do-batch-norm"] if row.do_batch_norm is True else []) + [
        f"--batch-size={row.batch_size}",
        f"--n-blocks={row.n_blocks}",
        f"--n-initial-block-channels={row.n_initial_block_channels}",
        f"--learning-rate={row.learning_rate}",
        str(row.criterion),
        str(row.augmentation)
    ]
    job_name = f"{row.criterion}_{row.augmentation}_specials"
    arg_line = (f"--job-name={job_name} ../lslurm/run_gpu.sh "
                + "python3 -m scripts.train_unet " + " ".join(command))
    print(arg_line)
