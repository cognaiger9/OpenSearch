from typing import Any, Dict
from runner.database_manager import DatabaseManager
from llm.model import model_chose
import json
from llm.prompts import *


def extract_col_value(task: Any, db_info: Dict[str, Any]) -> Dict[str, Any]:
    db_manager = DatabaseManager()
    fewshot_path = db_manager.db_fewshot_path
    chat_model = model_chose("deepseek")

    with open(fewshot_path) as f:## fewshot
        df_fewshot = json.load(f)

    hint = task.evidence
    if hint == "":
        hint = "None"
    

    key_col_des_raw = get_des_ans(chat_model,
                                db_check_prompts().extract_prompt,
                                df_fewshot["extract"][task.question_id]['prompt'],
                                db_info,
                                task.question,
                                hint,
                                False,
                                temperature=0)

    response = {
        "key_col_des_raw": key_col_des_raw
    }
    #print(response)
    return response

def get_des_ans(chat_model,
                ext_prompt,
                fewshot,
                db,
                question,
                hint,
                debug,
                temperature=0):
    fewshot = fewshot.split("/* Answer the following:")[1:6]
    fewshot = "/* Answer the following:" + "/* Answer the following:".join(
        fewshot)
    ext_prompt = ext_prompt.format(fewshot=fewshot,
                                   db_info=db,
                                   query=question,
                                   hint=hint)
    #print(ext_prompt)

    if debug:
        print(ext_prompt)
    pre_col_values = chat_model.get_ans(ext_prompt, temperature,
                                        debug=debug).replace('```', '')

    return pre_col_values


