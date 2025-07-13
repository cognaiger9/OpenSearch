import sys
import json
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            contents = json.loads(f.read())
        return contents
    except Exception as e:
        raise Exception(f"Unexpected error while loading JSON file {file_path}: {str(e)}")

def result_callback(result):
    exec_result.append(result)

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    #print(predicted_res)
    #print(ground_truth_res)
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
    #print(result)
    return result


def clean_sql_query(sql_str):
    """
    Clean SQL query by extracting only the SQL statement from markdown formatted text.
    
    Args:
        sql_str (str): Raw SQL string that may contain markdown formatting and explanations
        
    Returns:
        str: Clean SQL query
    """
    # If the query contains markdown code block
    if "```sql" in sql_str:
        # Extract content between ```sql and ```
        start = sql_str.find("```sql") + 6
        end = sql_str.find("```", start)
        if end != -1:
            sql_str = sql_str[start:end]
    
    # Remove any leading/trailing whitespace and newlines
    sql_str = sql_str.strip()
    
    return sql_str

def package_sqls(sql_file_path):
    """
    Package SQL queries and their corresponding database paths for evaluation.
    
    Args:
        sql_file_path (str): Path to the file containing SQL queries
        db_root_path (str): Root path to the database files
    
    Returns:
        tuple: (clean_sqls, db_path_list) containing:
            - clean_sqls: List of SQL queries
            - db_path_list: List of corresponding database paths
    """
    clean_sqls = []
    
    # Predict mode: Load predicted SQL queries from a JSON file
    sql_data = json.load(open(sql_file_path, 'r'))
    for idx, sql_str in sql_data.items():
        if type(sql_str) == str:
            sql = clean_sql_query(sql_str)
        else:
            sql = " "
        clean_sqls.append(sql)

    return clean_sqls

def get_gt_sqls(question_path, db_root_path):
    sql_gt_list = []
    db_path_list = []

    # Load questions from JSON file
    questions = load_json(question_path)
    for question in questions:
        sql_gt_list.append(question['sql_query'])
        db_path_list.append(db_root_path + question['db_name'] + '/' + question['db_name'] + '.sqlite')

    return sql_gt_list, db_path_list

def run_sqls_parallel(gt_queries, db_places, pred_queries: dict, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, predicted_sql in pred_queries.items():
        idx = int(i)
        ground_truth = gt_queries[idx]
        db_place = db_places[idx]
        print(f"idx: {idx}, predicted_sql: {predicted_sql}, ground_truth: {ground_truth}, db_place: {db_place}")
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_place, idx, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc(exec_results):
    num_queries = len(exec_results)
    #print(exec_results)
    results = [res['res'] for res in exec_results]
    
    all_acc = sum(results) / num_queries
    return all_acc * 100, num_queries

def print_data(acc, num_queries):
    print('======================================    ACCURACY    =====================================')
    print("Total Accuracy: {:.2f}%; Number of queries: {}".format(acc, num_queries))
    print('===========================================================================================')


if __name__ == '__main__':
    exec_result = []

    predict_sql_file = "./sql_res.json"
    question_path = "./BULL/data_preprocess/dev.json"
    db_root_path = "./BULL/database_en/"

    #pred_queries = package_sqls(predict_sql_file)
    # generate gt sqls:
    gt_queries, db_paths_gt = get_gt_sqls(question_path, db_root_path)
    pred_queries = load_json(predict_sql_file)
    query_pairs = list(zip(pred_queries,gt_queries))
    run_sqls_parallel(gt_queries, db_places=db_paths_gt, pred_queries=pred_queries, num_cpus=16, meta_time_out=30.0)
    exec_result = sort_results(exec_result)
    
    print('start calculate')
    acc, num_queries = compute_acc(exec_result)
    print_data(acc, num_queries)
    print('===========================================================================================')
    print("Finished evaluation")
    