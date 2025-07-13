import re, json
from typing import Any, Dict
from runner.database_manager import DatabaseManager
from sentence_transformers import SentenceTransformer
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_MYSQL_like
from llm.prompts import *
from runner.extract import DES_new
from database_process.make_emb import load_emb
from runner.column_retrieve import ColumnRetriever
from runner.column_update import ColumnUpdater

def column_retrieve_and_other_info(task: Any, db_info: Dict[str, Any], query_noun: Dict[str, Any], bert_model: SentenceTransformer) -> Dict[str, Any]:
    print("In stage: column_retrieve_and_other_info")
    db_manager = DatabaseManager()
    emb_dir = db_manager.emb_dir
    tables_info_dir = db_manager.db_tables
    chat_model = model_chose("deepseek")

    all_db_col = db_info["db_col_dic"]
    origin_col = query_noun["col"]
    values = query_noun["values"]

    db = task.db_id

    emb_values_dic = {}
    if emb_values_dic.get(db):
        DB_emb, col_values = emb_values_dic[db]
    else:
        DB_emb, col_values = load_emb(db, emb_dir)
        emb_values_dic[db] = [DB_emb, col_values]

    db_col = {x: all_db_col[x][0] for x in all_db_col }  ## db string
    db_keys_col=all_db_col.keys()

    col_retrieve = ColumnRetriever(bert_model, tables_info_dir).get_col_retrieve(task.question, db,db_keys_col)

    foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(tables_info_dir, db)      
    cols = ColumnUpdater(db_col).col_pre_update(origin_col, col_retrieve, foreign_set)
    
    des = DES_new(bert_model, DB_emb, col_values)   
    
    cols_select, L_values = des.get_key_col_des(cols,
                                    values,
                                    debug=False,
                                    topk=10,
                                    shold=0.65)

    column = ColumnUpdater(db_col).col_suffix(cols_select)
    # values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    count=0
    while count<3:
        try:
            q_order = query_order(task.raw_question, chat_model, db_check_prompts().select_prompt, temperature=0)
            break
        except:
            count+=1

    # # q_order = f"The content of the SELECT statement should only include: {q_order}"
    # q_order=""

    response = {
        # "col_retrieve":list(col_retrieve),
        # "col_select":list(cols_select),
        "L_values" : L_values,
        "column" : column,
        "foreign_keys" : foreign_keys,
        "foreign_set" : foreign_set,
        "q_order" : q_order
    }
    #print(response)

    return response


def query_order(question, chat_model, select_prompt,temperature):
    select_prompt = select_prompt.format(question=question)
    ans = chat_model.get_ans(select_prompt, temperature=temperature)
    ans = re.sub("```json|```", "", ans)
    select_json = json.loads(ans)
    res, judge = json_ext(select_json)
    return res



def json_ext(jsonf):
    ans = []
    judge = False
    for x in jsonf:
        if x["Type"] == "QIC":
            Q = x["Extract"]["Q"].lower()
            if Q in ["how many", "how much", "which","how often"]:
                for item in x["Extract"]["I"]:
                    ans.append(x["Extract"]["Q"] + " " + item)
            elif Q in ["when", "who", "where"]:
                ans.append(x["Extract"]["Q"])
            else:
                ans.extend(x["Extract"]["I"])
        elif x["Type"] == "JC":
            ans.append(x["Extract"]["J"])
            judge = True
    return ans, judge

