from typing import Any, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import *
import json

def generate_db_schema(task: Any, bert_model: SentenceTransformer) -> Dict[str, Any]:
    print("In stage: generate_db_schema")

    db_manager = DatabaseManager()

    # Read parameters
    tables_info_dir = db_manager.db_tables
    sqllite_dir = db_manager.db_path
    db_dir = db_manager.db_directory_path
    chat_model = model_chose("deepseek")  # deepseek
    ext_file = Path(db_manager.db_root_path) / "db_schema.json"

    # Read existing data
    if os.path.exists(ext_file):
        with open(ext_file, 'r') as f:
            data = json.load(f) # The saved format was incorrect
    else:
        data ={}

    # Get database information agent
    DB_info_agent = db_agent_string(chat_model)
    
    # Check if this database has already been processed
    db = task.db_id
    existing_entry = data.get(db)

    if existing_entry:
        all_info, db_col = existing_entry
    else:
        print(f"no existing_entry for db: {db}")
        all_info, db_col = DB_info_agent.get_allinfo(db, sqllite_dir, db_dir, tables_info_dir, bert_model)
        data[db] = [all_info,db_col]
        with open(ext_file, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    response = {
        "db_list": all_info,
        "db_col_dic": db_col
    }
    #print(response)
    return response
