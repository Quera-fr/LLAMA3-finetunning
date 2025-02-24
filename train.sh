git clone https://github.com/ggerganov/llama.cpp.git

python ./llama.cpp/convert_hf_to_gguf.py ./llama3-finetuned-merged

ollama create llama3-finetuned -f Modelfile

Ministral-3B-instruct-F16.gguf