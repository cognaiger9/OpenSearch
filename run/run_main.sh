# Define variables
data_mode='dev' # Options: 'dev', 'train' 
db_root_path=BULL #root directory # UPDATE THIS WITH THE PATH TO THE TARGET DATASET
start=0 #inclusive
end=1  #exclusive
#pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation'
pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+evaluation'
# pipeline refers to the current workflow node combination
# checkpoint_nodes='generate_db_schema,extract_col_value,extract_query_noun'
# checkpoint_dir="./results/dev/generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation/Bird/2024-09-12-01-48-10"

# Nodes:
    # generate_db_schema
    # extract_col_value
    # extract_query_noun
    # column_retrieve_and_other_info
    # candidate_generate
    # align_correct
    # vote
    # evaluation

AK='your_ak' #set your ak in src/llm/model.py
engine1='deepseek'
engine6='finetuned_nl2sql'
engine8='finetuned_colsel'
engine9='finetuned_col_filter'

## n defaults to 21
#align_methods:style_align+function_align+agent_align
#temporarily no good way to comment
# pipeline_setup='{
#     "generate_db_schema": {                 #step to generate db_schema
#         "engine": "'${engine1}'",            #large model selection for generating db_schema                 
#         "bert_model": "/app/sentence-transformers/all-mpnet-base-v2/",  #bert_model selection
#         "device":"cpu"                                                  #bert_model loading method, currently this machine only supports cpu
#     },
#     "extract_col_value": {                    #get_des_ans gets key_col_des_raw
#         "engine": "'${engine1}'",             #large model
#         "temperature":0.0                     #large model generation parameter selection
#     },
#     "extract_query_noun": {                   #parse_des gets col and value
#         "engine": "'${engine1}'",             #large model used for parse_des
#         "temperature":0.0                     #large model generation parameter selection
#     },
#     "column_retrieve_and_other_info": {      #gets column descriptions and related information + query_order
#         "engine": "'${engine1}'",             #large model used for query_order
#         "bert_model": "/app/bge",        # bert_model selection
#         "device":"cpu",                          #bert_model loading method, currently this machine only supports cpu
#         "temperature":0.3,                        #generation parameters for query_order using large model
#         "top_k":10                                #top_k in get_key_col_des
#     },
#     "candidate_generate":{                    #generate candidate sql
#         "engine": "'${engine1}'",             #large model
#         "temperature": 0.7,                   #large model parameters
#         "n":21,                               #n, consistent with align step n
#         "return_question":"True",             #parameter in get_sql
#         "single":"False"                      #parameter in get_sql, different handling for n=1 and n!=1
#     },
#     "align_correct":{
#         "engine": "'${engine1}'",             #alignment and correction
#         "n":21,                                  #number of threads
#         "bert_model": "/app/bge",            
#         "device":"cpu",                           #bert_model loading method
#         "align_methods":"style_align+function_align+agent_align"   #alignment methods, separated by +
#     }
# }'  
pipeline_setup='{
    "generate_db_schema": {
        "engine": "'${engine1}'",
        "bert_model": "all-MiniLM-L6-v2",  
        "device":"cpu"
    },
    "extract_col_value": {
        "engine": "'${engine1}'",
        "temperature":0.0
    },
    "extract_query_noun": {
        "engine": "'${engine1}'",
        "temperature":0.0
    },
    "column_retrieve_and_other_info": {
        "engine": "'${engine1}'",
        "bert_model": "all-MiniLM-L6-v2",  
        "device":"cpu",
        "temperature":0.3,
        "top_k":10
    },
    "candidate_generate":{
        "engine": "'${engine1}'",
        "temperature": 0.7,  
        "n":21,
        "return_question":"True",
        "single":"False"
    },
    "align_correct":{
        "engine": "'${engine1}'",
        "n":21,
        "bert_model": "all-MiniLM-L6-v2",  
        "device":"cpu",
        "align_methods":"style_align+function_align+agent_align"
    }
}'  

python3 -u ./src/main.py --data_mode ${data_mode} --db_root_path ${db_root_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --start ${start} --end ${end} \
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
