from langchain_community.llms import LlamaCpp

model = LlamaCpp(
    model_path="/media/disk2/Llama/llama.cpp/models/ggml-meta-llama-3-8b-Q4_K_M.gguf",
    n_gpu_layers=-1,
    verbose=True
)