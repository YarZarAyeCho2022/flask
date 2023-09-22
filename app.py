from flask import Flask, request, render_template
from dotenv import load_dotenv
from huggingface_hub import login
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
import torch
import os

global index

load_dotenv()
login (token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")



app = Flask(__name__, template_folder='Templates', static_folder='static')

@app.route("/")
def home():
    return render_template("index.html")
    # return os.getenv("OPENAI_API_KEY")

@app.route("/digest")
def digest_documents():
    documents = SimpleDirectoryReader("Data").load_data()
    embed_model = LangchainEmbedding(
        HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)


@app.route("/get",methods=["POST"])
def reply_message():
    question = request.form['question']
    query_engine = index.as_query_engine()
    answer = query_engine.query(question)
    # message = request.form['msg']
    return answer





if __name__ == "__main__":
    app.run(debug=True)