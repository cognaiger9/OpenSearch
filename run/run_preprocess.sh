db_root_directory=BULL #root directory
#dataset_type=bull
#dev_json=BULL-en/dev.json
#train_json=BULL-en/train.json
#dev_table=BULL-en/tables.json  # 11 dev data
#train_table=BULL-en/tables.json  # 69 train data
dev_database=database_en #dev database directory
fewshot_llm=deepseek
DAIL_SQL=BULL/questions.json     #dailsql json file 
bert_model=all-MiniLM-L6-v2

#db_root_directory=Bird
dataset_type=bird
dev_json=dev/dev.json
train_json=train/train.json
dev_table=dev/dev_tables.json
train_table=train/train_tables.json

#python -u src/database_process/data_preprocess.py \
#    --dataset_type ${dataset_type} \
#    --db_root_directory "${db_root_directory}" \
#    --dev_json "${dev_json}" \
#    --train_json "${train_json}" \
#    --dev_table "${dev_table}" \
#    --train_table "${train_table}"

#python -u src/database_process/prepare_train_queries.py \
#    --db_root_directory "${db_root_directory}" \
#    --model "${fewshot_llm}"  \
#    --start 0 \
#    --end 3966

#python -u src/database_process/generate_question.py \
#    --db_root_directory "${db_root_directory}" \
#    --DAIL_SQL "${DAIL_SQL}"

python -u src/database_process/make_emb.py \
    --db_root_directory "${db_root_directory}" \
    --dev_database "${dev_database}" \
    --bert_model "${bert_model}"
