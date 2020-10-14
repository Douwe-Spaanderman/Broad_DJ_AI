import sys
import os
import argparse
from pathlib import Path
import time

def main(path, name="Ongoing"):
    if not path.endswith("/"):
        path += "/"

    name += "/"
    path += name
    path_figures = path + "Figures/"
    path_model = path + "Model/"
    path_cache = path + "Cache/"

    path = Path(path)
    path_figures = Path(path_figures)
    path_model = Path(path_model)
    path_cache = Path(path_cache)
    
    path.mkdir(parents=True, exist_ok=True)
    path_figures.mkdir(parents=True, exist_ok=True)
    path_model.mkdir(parents=True, exist_ok=True)
    path_cache.mkdir(parents=True, exist_ok=True)

    print(f"Project is saved at: {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initiate directory for project")
    parser.add_argument("Path", help="path to where you want to create project")
    parser.add_argument("name", help="Project name")
    
    args = parser.parse_args()
    start = time.time()
    main(path=args.Path, name=args.name)
    end = time.time()
    print('completed in {} seconds'.format(end-start))