import os
import shutil


def clear_dir(dir_path):
    if isinstance(dir_path, list):
        for dir_p in dir_path:
            if dir_p is None:
                continue
            clear_dir(dir_p)

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
