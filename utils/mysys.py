import os
import shutil
import subprocess
from typing import Optional


def quick_clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    print(f"Clear {path}")


def cmd(command: str, timeout: Optional[int] = None, verbose: bool = True):
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )
    result = proc.communicate(timeout=timeout)[0]
    if verbose:
        print(command)
        print(result)
    return result
