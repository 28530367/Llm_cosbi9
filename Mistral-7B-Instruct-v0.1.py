import pandas as pd
import time
import re
import json
import os

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))

# 模型及參數調整
n_gpu_layers = -1
n_batch = 512
_model_path = f"/media/disk2/Llama/llama.cpp/models/ggml-mistral-7b-v0.1-Q4_K_M.gguf"

# 呼叫模型
llm = LlamaCpp(
    model_path=_model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,
    temperature=0.1,
    top_p=1,
    n_ctx=8192,
    verbose=True,
)


result = ""

def bdt_algorithm(node, task, patient):
    global result

    if "Yes" not in node and "No" not in node:
        result = node
        print("result: ", result)
        return 

    print("Question", node['Question'])

    prompt = PromptTemplate(
        template=f"""<|start_header_id|>system<|end_header_id|>You are an assistant for question-answering tasks. Answer the question according to the Task description and Patient description.
        Task description: {task},
        Patient description: {patient},
        Question: {{question}},
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    o1 = rag_chain.invoke(node['Question'])
    print("o1: ", o1)

    prompt = PromptTemplate(
        template=f"""<|start_header_id|>system<|end_header_id|>Response english letters "YES" or "NO" based on the given Task description, Patient description and Context.
        Task description: {task}, 
        Patient description: {patient},
        Context: {o1},
        Question: {{question}},
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    o2 = rag_chain.invoke(node['Question'])

    print("o2: ",o2)

    o2_FLU = o2.replace(" ", "")[0].upper() #o2 first letter upper
    print("o2_FLU = ", o2_FLU)
    if o2_FLU == 'Y':
        bdt_algorithm(node['Yes'], task, patient)
    elif o2_FLU == 'N':
        bdt_algorithm(node['No'], task, patient)
    else:
        result = "Something wrong!"
        return
     


if __name__ == "__main__":
    task_description = "Diagnose patient's disease."
    patient_description = """A 28-year-old female has a past medical history of kidney transplant and takes immunosuppression 
    drugs. She weights 65 kg and has tested positive for Covid-19. She has symptoms occurred for 6 days. She does not need hospitalization.
    And she does not have any chronic kidney disease with a GFR of 94 mL/min. Her immunosuppressive medications do not interact
    with paxlovid and she can hold few other home medicines while taking paxlovid. She can also take remdesivir at nearest infusion center."""

    with open(f"{current_dir}/json_file/COVID19_BDT_Guildelines.json", 'r', encoding='utf-8') as file:
        bdt_Guildelines = json.load(file)

    bdt_algorithm(bdt_Guildelines, task_description, patient_description)

    print("result: ", result)
    

