import streamlit as st
# from llama_index import download_loader
# from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
# from tempfile import NamedTemporaryFile
from PIL import Image
import os

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

tab_qa, tab_doc = st.tabs(["Ｑ＆Ａ", "文書"])

PAGE_DIC = {page: f"page_{page:03d}.png" for page in range(1, 38)}
INTRO = "左側のテキストボックスに質問を入力し、エンターキーを押すとＡＩが回答します。"
# INTRO = "この文章を３０字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"

if "qa" not in st.session_state:
#     st.session_state.qa = {"pdf": "", "history": []}
    st.session_state["qa"] = {"history": [{"role": "A", "msg": INTRO}]}

if "page" not in st.session_state:
    st.session_state.page = 1

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
def load_vector_db():
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir="./storage/")
    index = load_index_from_storage(storage_context, service_context=service_context)
    return index

def store_del_msg():
    if st.session_state.user_input and st.session_state.qa["history"][-1]["role"] != "Q": # st.session_state.prev_q != st.session_state.user_input:
        st.session_state.qa["history"].append({"role": "Q", "msg": st.session_state.user_input}) # store
    st.session_state.user_input = ""  # del

# View (User Interface)
with tab_qa:
    ## Sidebar
    st.sidebar.title("補助金さん")
    st.sidebar.write("補助金・助成金についてお任せあれ")
    user_input = st.sidebar.text_input("ご質問をどうぞ", key="user_input", on_change=store_del_msg)

    # st.sidebar.markdown("---")
    ## Main Content
    # if st.session_state.qa["history"]:
    for message in st.session_state.qa["history"]:
    #     for message in st.session_state.qa["history"][1:]:
        if message["role"] == "Q": # Q: Question (User)
            st.info(message["msg"])
        elif message["role"] == "A": # A: Answer (AI Assistant)
            st.write(message["msg"])
        elif message["role"] == "E": # E: Error
            st.error(message["msg"])
    chat_box = st.empty() # Streaming message

    # Model (Business Logic)
    index = load_vector_db()
    engine = index.as_query_engine(text_qa_template=QA_PROMPT, streaming=True, similarity_top_k=1)

    if st.session_state.qa["history"][-1]["role"] == "Q":
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
            st.session_state.page = int(response.source_nodes[0].extra_info['page_label'])
        except Exception as error_msg:
    #             error_msg = "エラーが発生しました！　もう一度、質問して下さい。"
            st.error(error_msg)
            st.session_state.qa["history"].append({"role": "E", "msg": error_msg})

    image = Image.open(f"./pdf_png/{PAGE_DIC[st.session_state.page]}")
    st.sidebar.image(image, caption = '展示会出展助成事業（令和５年度　東京都）', use_column_width = "auto")

with tab_doc:
    col_l, col_prev, col_next, col_r = st.columns([1.5, 1, 1, 1.5])
    with col_prev:
        if st.button and st.session_state.page == 1:
            st.button("＜ 前ページ", disable=True)
        else:
            if st.button("＜ 前ページ"):
                st.session_state.page -= st.session_state.page
    with col_next:
        if st.button and st.session_state.page == 37:
            st.button("次ページ ＞", disable=True)
        else:
            if st.button("次ページ ＞"):
                st.session_state.page += st.session_state.page
    with col_l:
        pass
    with col_r:
        pass
    pdf_img = Image.open(f"./pdf_png/{PAGE_DIC[st.session_state.page]}")
    st.image(pdf_img, caption = '展示会出展助成事業（令和５年度　東京都）', use_column_width = "auto")
    
# st.session_state.qa
st.session_state.page
