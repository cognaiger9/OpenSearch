import argparse
import json
import os
from runner.run_manager import RunManager

def load_dataset(data_path: str):
    with open(data_path, "r") as f:
        dataset = json.load(f)
    return dataset

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db_root_path", type=str, default="BULL")
    arg_parser.add_argument("--start", type=int, default=0)
    arg_parser.add_argument("--end", type=int, default=1000)
    args = arg_parser.parse_args()

    db_json = os.path.join(args.db_root_path, "data_preprocess", "dev.json")
    dataset = load_dataset(db_json)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(args.start, args.end, dataset)
    run_manager.run_tasks()