import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from runner.task import Task
from runner.database_manager import DatabaseManager
from pipeline.generate_db_schema import generate_db_schema
from pipeline.extract_col_value import extract_col_value
from pipeline.extract_query_noun import extract_query_noun
from pipeline.column_retrieve_and_other_info import column_retrieve_and_other_info
from pipeline.candidate_generate import candidate_generate

from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_WORKERS = 8   

class RunManager:
    RESULT_ROOT_PATH = "results"

    def __init__(self, args: Any):
        self.args = args
        self.result_directory = f"{self.RESULT_ROOT_PATH}/BULL"
        self.tasks: List[Task] = []
        self.total_number_of_tasks = 0
        self.processed_tasks = 0
        self.result_file = "sql_res.json"

        run_folder_path = Path(self.result_directory)
        run_folder_path.mkdir(parents=True, exist_ok=True)

        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)

    def initialize_tasks(self, start, end, dataset: List[Dict[str, Any]]):
        """
        Initializes tasks from the provided dataset.
        
        Args:
            dataset (List[Dict[str, Any]]): The dataset containing task information.
        """

        # Load task done from result file
        with open(self.result_file, "r") as f:
            self.processed_tasks = json.load(f)
        #print(f"Processed tasks: {self.processed_tasks}")

        for i, data in enumerate(dataset):
            if i < start:  # Skip elements before start
                continue
            if i >= end:  # Stop processing if exceeding end
                break
            task = Task(data)
            if str(task.question_id) in self.processed_tasks:
                print(f"Skipping task: {task.question_id}")
                continue
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self):
        """Runs the tasks using a pool of workers."""
        bert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_task = {executor.submit(self.worker, task, bert_model): task for task in self.tasks}
            for future in tqdm(as_completed(future_to_task), total=len(self.tasks), desc="Processing tasks"):
                task = future_to_task[future]
                try:
                    ans = future.result()
                    self.processed_tasks[task.question_id] = ans
                except Exception as exc:
                    print(f"Task {task.question_id} generated an exception: {exc}")

        # sql_res = {}
        
        # Initiate BERT model

        # for task in self.tasks:
        #     ans = self.worker(task, bert_model)
        #     sql_res[task.question_id] = ans
        #     #self.evaluate(task, ans)

        #self.compute_acc(sql_res)
        with open(self.result_file, "w") as f:
            json.dump(self.processed_tasks, f)

    def worker(self, task: Task, bert_model: SentenceTransformer) -> Tuple[Any, str, int]:
        """
        Worker function to process a single task.
        
        Args:
            task (Task): The task to be processed.
        
        Returns:
            tuple: The state of the task processing and task identifiers.
        """
        print(f"Processing task: {task.db_id} {task.question_id}")

        # Create singleton instance of DatabaseManager
        db_manager = DatabaseManager(db_root_path=self.args.db_root_path, db_id=task.db_id)
        
        db_info = generate_db_schema(task, bert_model)
        col_value = extract_col_value(task, db_info)
        query_noun = extract_query_noun(task, col_value)
        column_retrive_info = column_retrieve_and_other_info(task, db_info, query_noun, bert_model)
        candidate_info = candidate_generate(task, column_retrive_info)

        return candidate_info['SQL']
    
    def evaluate(self, task: Task, ans: str):
        ground_truth = task.SQL
        db_path = self.args.db_root_path + "/database_en/" + task.db_id + f"/{task.db_id}.sqlite"
        print("db_path: ", db_path)
        res = self.execute_sql(ans, ground_truth, db_path)
        self.corrected[task.question_id] = res

    def execute_sql(self, predicted_sql, gt_sql, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(gt_sql)
        ground_truth_res = cursor.fetchall()
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
        return res
    
    def compute_acc(self):
        acc = torch.sum(self.corrected) / self.total_number_of_tasks
        print(f"Accuracy: {acc}, total number of tasks: {self.total_number_of_tasks}")