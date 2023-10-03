import streamlit as st
# from llama_index import download_loader
from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
# from langchain import ChatOpenAI
from langchain import OpenAI
# from langchain.chat_models import ChatOpenAI
from PIL import Image
import fitz  # PyMuPDF
import os
# import base64

type_to_index = {"展示会出展助成事業":0, "BX2000":1, "MX1000/3000":2}
type_to_path = {"展示会出展助成事業":"./storage/", "BX2000":"./BX2000_DB/", "MX1000/3000":"./MX1000_DB/"}
docu_to_pdf_path = {"展示会出展助成事業":"./PDF/R5_tenjikaijyosei_boshuyoko_230403.pdf", "BX2000":"./PDF/BX2000_m.pdf", "MX1000/3000":"./PDF/MX1000_3000_m.pdf"}
# def show_pdf(file_path):
#     with open(file_path,"rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode()
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="800" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# show_pdf('./pdf-pages/R5_tenjikaijyosei_boshuyoko_230403-1.pdf')

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

PAGE_DIC = {page: f"page_{page:03d}.png" for page in range(1, 38)}
INTRO = "左側のテキストボックスに質問を入力しエンターキーを押すと、ＡＩが回答します。"
# INTRO = "この文章を３０字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"

if "qa" not in st.session_state:
#     st.session_state.qa = {"pdf": "", "history": []}
    st.session_state["qa"] = {"history": [{"role": "A", "msg": INTRO}]}

if "pdf_page" not in st.session_state:
    st.session_state.pdf_page = 1

if "docu_index" not in st.session_state:
    st.session_state.docu_index = 0

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
def load_vector_db(docu_type):
    db_path = type_to_path[docu_type]
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4", streaming=True))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir=db_path)
    index = load_index_from_storage(storage_context, service_context=service_context)
    return index

@st.cache_data
def get_pdf_image(docu_type, page):
    pdf_path = docu_to_pdf_path[docu_type]
    pdf_document = fitz.open(pdf_path)
    page_data = pdf_document.load_page(int(page))
    image = page_data.get_pixmap()
    image_data = image.tobytes("png")
    return image_data
    # return Image.open(f"./pdf_png/{PAGE_DIC[page]}")

def show_pdf(page):
    st.session_state.pdf_page = page

def store_del_msg():
    if st.session_state.user_input and st.session_state.qa["history"][-1]["role"] != "Q": # st.session_state.prev_q != st.session_state.user_input:
        st.session_state.qa["history"].append({"role": "Q", "msg": st.session_state.user_input}) # store
    st.session_state.user_input = ""  # del

# View (User Interface)
## Sidebar
st.sidebar.title("Documenter")
st.sidebar.write("あなたの文書からお答えします")
docu_index = st.session_state.docu_index
docu_type = st.sidebar.selectbox("文書を選んでください", ["展示会出展助成事業", "BX2000", "MX1000/3000"], index=docu_index)
st.session_state.docu_index = type_to_index[docu_type]
user_input = st.sidebar.text_input("ご質問をどうぞ", key="user_input", on_change=store_del_msg)
# st.sidebar.markdown("---")
if docu_type == "展示会出展助成事業":
    if st.sidebar.button("助成事業の目的"):
        st.session_state.qa["history"].append({"role": "Q", "msg": "助成事業の目的を教えて下さい。"})
    if st.sidebar.button("助成対象の経費"):
        st.session_state.qa["history"].append({"role": "Q", "msg": "助成対象の経費を教えて下さい。"})
    if st.sidebar.button("申請手順（表形式）"):
        st.session_state.qa["history"].append({"role": "Q", "msg": "申請手順を表にして下さい。"})
    st.sidebar.image(get_pdf_image(docu_type, 0), caption = '展示会出展助成事業', use_column_width = "auto")

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
pdf_page = st.container()

# Model (Business Logic)
index = load_vector_db(docu_type)
engine = index.as_query_engine(text_qa_template=QA_PROMPT, streaming=True, similarity_top_k=1)

if st.session_state.qa["history"][-1]["role"] == "Q":
    query = st.session_state.qa["history"][-1]["msg"]
    try:
        response = engine.query(query) # Query to ChatGPT
        text = ""
        for next in response.response_gen:
            text += next
            chat_box.write(text)
        page_int = int(response.source_nodes[0].node.extra_info['page_label'])
        refer_pages = f"\n\n参照：{page_int - 1} ページ\n\n\n"
        # refer_pages = "\n\n参照：" + ", ".join([f"{node..extra_info['page_label']}ページ" for node in response.source_nodes])
        chat_box.write(text + refer_pages)
        st.session_state.qa["history"].append({"role": "A", "msg": text + refer_pages})
        st.session_state.pdf_page = page_int
        with pdf_page:
            with st.expander(f"{page_int} ページを開く"):
                page = st.session_state.pdf_page
                pdf_image = get_pdf_image(st.session_state.docu_index, st.session_state.pdf_page)
                st.image(pdf_image, caption = f'{docu_type} {st.session_state.pdf_page}ページ', use_column_width = "auto")
    except Exception as error_msg:
#             error_msg = "エラーが発生しました！　もう一度、質問して下さい。"
        st.error(error_msg)
        st.session_state.qa["history"].append({"role": "E", "msg": error_msg})
        
# st.session_state.qa
# st.session_state.pdf_page
