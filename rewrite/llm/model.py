import time
import re
from llm.prompts import prompts_fewshot_parse
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def model_chose(model="deepseek"):
    if model == "deepseek":
        return deep_seek(model)
    else:
        raise ValueError(f"Invalid model: {model}")

class req:

    def __init__(self, model) -> None:
        self.model= model

    def fewshot_parse(self, question, sql):
        s = prompts_fewshot_parse().parse_fewshot.format(question=question, sql=sql)
        ext = self.get_ans(s)
        ext= ext.replace('```','').strip()
        ext = ext.split("#SQL:")[0]  # prevent incorrect format generation, at least keep the SQL
        ans = self.convert_table(ext, sql)
        return ans
    
    def convert_table(self, s, sql):
        l = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
        x, v = s.split("#values:")
        t, s = x.split("#SELECT:")
        for li in l:
            s = s.replace(f"{li[1]}.", f"{li[0]}.")
        return t + "#SELECT:" + s + "#values:" + v

class deep_seek(req):

    def __init__(self, model) -> None:
        super().__init__(model)

    def get_ans(self, messages, temperature=0.0, debug=False):
        count = 0

        api_key = os.getenv("DEEPSEEK_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        while count < 5:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", 
                         "content": "You are an SQL expert, skilled in handling various SQL-related issues."},
                        {"role": "user", 
                         "content": messages}
                    ],
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=800
                )

                if debug:
                    print(response)
                ans = response.choices[0].message.content
                break
            except Exception as e:
                count += 1
                time.sleep(2)
                print(e, count, response)
        return ans
