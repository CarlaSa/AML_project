import os
from time import time, sleep
from dataclasses import dataclass
from argparse import ArgumentParser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from subprocess import check_output
from getpass import getuser
from typing import List


@dataclass
class CancellingEventHandler(FileSystemEventHandler):
    job_id: int
    file: str

    def on_created(self, event: FileCreatedEvent) -> None:
        file_created = os.realpath(event.src_path())
        print(f"(time={time()})  Found '{file_created}.'")
        if os.path.isfile(self.file):
            print(f"(time={time()})  CANCELLING {self.job_id}!")
            cmd = f"scancel {self.job_id}"
            print(cmd)
            os.system(cmd)


def get_running_slurm_jobs() -> List[int]:
    user = getuser()
    output = check_output(["squeue", f"--user={user}", "--noheader",
                           "--Format=jobid", "--states=RUNNING"])
    return [int(id) for id in output.splitlines()]


def running_slurm_job_id(raw: str) -> int:
    id = int(raw)
    if id not in get_running_slurm_jobs():
        raise RuntimeError(f"No running SLURM job {id} found.")
    return id


def hypothetical_file_path(raw: str) -> str:
    path = os.path.realpath(raw)
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        raise NotADirectoryError(f"{dirname} is not a directory.")
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists. "
                              + "Please kill your job manually.")
    return path


def get_args(*args):
    parser = ArgumentParser(
        description="Wait for a file to appear and cancel a SLURM job.")
    parser.add_argument("job_id", type=running_slurm_job_id)
    parser.add_argument("file", type=hypothetical_file_path)
    if len(args) > 0:
        return parser.parse_args(args)
    return parser.parse_args()


def main(*_args):
    args = get_args(*_args)
    event_handler = CancellingEventHandler()
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(args.file))
    observer.start()
    print(f"(time={time()})  SLURM job {args.job_id} shall be cancelled as "
          + f"soon as a file '{args.file}' will appear.")
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
