import streamlit as st
# from llama_index import download_loader
# from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
# from tempfile import NamedTemporaryFile
import os

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

INTRO = "この文章を３０字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"

if "qa" not in st.session_state:
    st.session_state.qa = {"pdf": "", "history": []}
#     st.session_state["qa"] = {"pdf": "", "history": [{"role": "Q", "msg": INTRO}]}

# Prompt
QA_PROMPT_TMPL = (
    "下記の情報が与えられています。 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "この情報を参照して次の質問に日本語で答えてください: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

@st.cache_resource
def load_vector_db(uploaded_file):
    storage_context = StorageContext.from_defaults(persist_dir="./storage/")
    index = load_index_from_storage(storage_context)
    return index

def store_del_msg():
    st.session_state.qa["history"].append({"role": "Q", "msg": st.session_state.user_input}) # store
    st.session_state.user_input = ""  # del

# View (User Interface)
## Sidebar
st.sidebar.title("年金さん")
user_input = st.sidebar.text_input("ご質問をどうぞ", key="user_input", on_change=store_del_msg)
## Main Content
if st.session_state.qa["history"]:
    for message in st.session_state.qa["history"]:
#         for message in st.session_state["qa"][1:]:
        if message["role"] == "Q": # Q: Question (User)
            st.info(message["msg"])
        elif message["role"] == "A": # A: Answer (AI Assistant)
            st.write(message["msg"])
        elif message["role"] == "E": # E: Error
            st.error(message["msg"])
chat_box = st.empty() # Streaming message

# Model (Business Logic)
index = load_vector_db(uploaded_file)
engine = index.as_query_engine(text_qa_template=QA_PROMPT, streaming=True, similarity_top_k=1)
if st.session_state.qa["history"]:
    query = st.session_state.qa["history"][-1]["msg"]
    try:
        response = engine.query(query) # Query to ChatGPT
        text = ""
        for next in response.response_gen:
            text += next
            chat_box.write(text)
        refer_pages = "\n\n参照：" + ", ".join([f"{node.extra_info['page_label']}ページ" for node in response.source_nodes])
        chat_box.write(text + refer_pages)
        st.session_state.qa["history"].append({"role": "A", "msg": text + refer_pages})
    except Exception as error_msg:
#             error_msg = "エラーが発生しました！　もう一度、質問して下さい。"
        st.error(error_msg)
        st.session_state.qa["history"].append({"role": "E", "msg": error_msg})
