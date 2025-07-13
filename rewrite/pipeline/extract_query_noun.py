import re
from typing import Any, Dict
from llm.model import model_chose
from llm.prompts import *

def extract_query_noun(task: Any, activ: Dict[str, Any]) -> Dict[str, Any]:
    print("In stage: extract_query_noun")

    chat_model = model_chose("deepseek")
    key_col_des_raw = activ["key_col_des_raw"]
    #print(f"key_col_des_raw key: {key_col_des_raw}")
    
    noun_ext = chat_model.get_ans(db_check_prompts().noun_prompt.format(raw_question=task.question), temperature=0)
    values, col = parse_des(key_col_des_raw, noun_ext, debug=False)
    
    response = {
        "values" : values,
        "col" : col
    }
    print(response)
    return response

def parse_des(pre_col_values, nouns, debug):
    #print(f"pre_col_values: {pre_col_values}")
    pre_col_values = pre_col_values.split("/*")[0].strip()
    if debug:
        print(pre_col_values)
    col, values = pre_col_values.split('#values:')
    _, col = col.split("#columns:")
    col = strip_char(col)
    values = strip_char(values)

    if values == '':
        values = []
    else:
        values = re.findall(r"([\"'])(.*?)\1", values)
    nouns_all = re.findall(r"([\"'])(.*?)\1", nouns)
    values_noun = set(values).union(set(nouns_all))
    values_noun = [x[1] for x in values_noun]
    return values_noun, col


def strip_char(s):
    return s.strip('\n {}[]')