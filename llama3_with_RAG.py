from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 模型及參數調整
n_gpu_layers = 99
n_batch = 512
_model_path = f"/media/disk2/Llama/llama.cpp/models/ggml-meta-llama-3-8b-Q4_K_M.gguf"

# 呼叫模型
llm = LlamaCpp(
    model_path=_model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,
    temperature=0,
    top_p=1,
    n_ctx=8192
)

# PDF檔案輸入
loader = PyMuPDFLoader(f"/media/disk2/Llama/RAG_example/colon.pdf")
PDF_data = loader.load()
# print(PDF_data)

# 分割PDF內容
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits = text_splitter.split_documents(PDF_data)
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate(
    template="""<|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the context retrieved below to answer the question. If there is no answer in the retrieved context, answer the question with your own ideas. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result1 = rag_chain.invoke("What is Colon Cancer?")

print('result1:', result1)
