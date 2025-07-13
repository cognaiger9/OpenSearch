            
def make_newprompt(new_prompt,
                   fewshot,
                   key_col_des,
                   db_info,
                   question,
                   hint="",
                   q_order=""):
    n_prompt = new_prompt.format(fewshot=fewshot,
                                 db_info=db_info,
                                 question=question,
                                 hint=hint,
                                 key_col_des=key_col_des,
                                 q_order=q_order)

    return n_prompt
