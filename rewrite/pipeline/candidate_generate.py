from typing import Any, Dict, List
from runner.database_manager import DatabaseManager
from pipeline.utils import make_newprompt
from llm.model import model_chose
from llm.db_conclusion import *
import json
from llm.prompts import *
from runner.check_and_correct import get_sql

def candidate_generate(task: Any, column_retrive_info: Dict[str, Any]) -> Dict[str, Any]:
    print("In stage: candidate_generate")
    db_manager = DatabaseManager()
    fewshot_path = db_manager.db_fewshot_path
    print("fewshot_path", fewshot_path)

    with open(fewshot_path) as f:## fewshot
        df_fewshot = json.load(f)

    chat_model = model_chose("deepseek")  # deepseek qwen-max gpt qwen-max-longcontext
    column = column_retrive_info["column"]
    foreign_keys = column_retrive_info["foreign_keys"]
    L_values = column_retrive_info["L_values"]
    q_order = column_retrive_info["q_order"]
    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    db=task.db_id

    key_col_des = "#Values in Database:\n" + '\n'.join(values)
    # key_col_des = ""
    
    new_db_info = f"Database Management System: SQLite\n#Database name: {db} \n{column}\n\n#Forigen keys:\n{foreign_keys}\n"
    # new_db_info=get_last_node_result(execution_history, "generate_db_schema")["db_list"]

    # question=rewrite_question(task.question)
    question=task.question
    fewshot=df_fewshot["questions"][task.question_id]['prompt']
    # fewshot=""
    # fewshot=fewshot.split("\n/* Given the following database schema: */")[0]
    new_prompt = make_newprompt(db_check_prompts().new_prompt, fewshot,
                            key_col_des, new_db_info, question,
                            task.evidence,q_order)
    
    print("cp1")
    
    single = "False".lower() == 'true'  # 将字符串转换为布尔值
    return_question="True".lower() == 'true' 
    SQL,_ = get_sql(chat_model, new_prompt, 0, return_question=return_question, n=21, single=single)

    print("cp2")
    
    response = {
        "rewrite_question" : question,
        "SQL" : SQL
        # "new_prompt":new_prompt
    }
    print(response)

    return response




def rewrite_question(question):
    if question.find(" / ")!=-1:
        question+=". For division operations, use CAST xxx AS REAL to ensure precise decimal results"
    return question
