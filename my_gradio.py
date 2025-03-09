import gradio as gr
from typing import TypedDict, Literal
from chat import get_ai_answer
# import fitz
# from docx import Document
from langchain.retrievers import EnsembleRetriever
from rag_utils import init_db_from_content , get_retriever, visualize_vectors

class MessageData(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str


class Playground:
    def __init__(self):
        self.chat_history = []
        self.chat_history_wo = []
        self.context_str = ""
        self.retriever = None
        
        # self.clean_history()

    def clean_history(self):
        """ 清空历史信息
        """
        self.chat_history.clear()
        self.chat_history_wo.clear()
        self.context_str = ""

def get_reference_text(retriever:EnsembleRetriever, query_str:str):
    relevant_documents = retriever.get_relevant_documents(query_str)
    return "\n".join([doc.page_content for doc in relevant_documents])

def submit(query_str , playground:Playground, chat_bot):
    references = get_reference_text(playground.retriever, query_str)
    playground.context_str += references
    # print('relevant_documents',references)
    _answer = get_ai_answer(chat_history=playground.chat_history , context_str=playground.context_str, query_str=query_str)
    # print('answer',_answer)
    playground.chat_history.append(MessageData(role='user' , content=query_str))
    playground.chat_history.append(MessageData(role='assistant' , content=_answer))
    chat_bot.append((query_str, _answer))
    query_str = ""
    return query_str , chat_bot 

def submit_wo(query_str , playground:Playground, chat_bot):
    _answer = get_ai_answer(chat_history=playground.chat_history_wo , context_str="", query_str=query_str)
    # print('answer',_answer)
    playground.chat_history_wo.append(MessageData(role='user' , content=query_str))
    playground.chat_history_wo.append(MessageData(role='assistant' , content=_answer))
    chat_bot.append((query_str, _answer))
    query_str = ""
    return query_str , chat_bot 

def clear_user_input():
    return gr.update(value='')

def clear_user_input_wo():
    return gr.update(value='')

def reset_state(playground, chat_bot):
    playground.clean_history()
    chat_bot = []

    return chat_bot

def reset_state_wo(playground, chat_bot_wo):
    playground.clean_history()
    chat_bot_wo = []

    return chat_bot_wo

# def read_pdf_with_pymupdf(file_path):
#     document = fitz.open(file_path)
#     text = ''
#     for page_num in range(len(document)):
#         page = document.load_page(page_num)
#         text += page.get_text()
#     return text

# def read_docx(file_path):
#     """读取单个DOCX文件的内容"""
#     doc = Document(file_path)
#     text = ''
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + '\n'
#     return text

def read_txt(file_path):
    """读取单个TXT文件的内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def start_state(file_upload, playground:Playground , chat_bot,Img, em_model):
    playground.clean_history()
    print(file_upload.name)
    _type = file_upload.name.split('.')[-1]
    # if _type == "pdf":
    #     content = read_pdf_with_pymupdf(file_upload.name)
    # elif _type == "docx":
    #     content = read_docx(file_upload.name)
    # else:
    content = read_txt(file_upload.name)
    db , documents = init_db_from_content(content, em_model)
    playground.retriever = get_retriever(db, documents)
    Img = visualize_vectors(db)
    chat_bot = []
    return chat_bot,Img


with gr.Blocks() as demo:
    playground = gr.State(value=Playground())
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 文档")  
            file_upload = gr.File(label="上传您的文件")
            gr.Markdown("# Embedding")
            em_model = gr.Dropdown(choices=["nomic-embed-text","text-embedding-v3"], label="Embedding")
            Img = gr.Image(value=None, visible=True)
            start_button = gr.Button(value="Start", variant="primary")
        with gr.Column():
            gr.Markdown("# AI 问答（RAG）")
            chat_bot = gr.Chatbot( value= [] , height=600)
            user_prompt = gr.Textbox(label="USER", placeholder="Enter a user message here.")
            with gr.Row():
                submit_button = gr.Button(value="Submit", variant="primary")
                clear_result_button = gr.Button("Clear History")
        with gr.Column():
            gr.Markdown("# AI 问答")
            chat_bot_wo = gr.Chatbot( value= [] , height=600)
            user_prompt_wo = gr.Textbox(label="USER", placeholder="Enter a user message here.")
            with gr.Row():
                submit_button_wo = gr.Button(value="Submit", variant="primary")
                clear_result_button_wo = gr.Button("Clear History")

    submit_button_wo.click(
        submit_wo, inputs=[user_prompt_wo , playground, chat_bot_wo], outputs= [user_prompt_wo, chat_bot_wo]
    )
    submit_button_wo.click(
        clear_user_input_wo, [], [user_prompt_wo]
    )
    clear_result_button_wo.click(reset_state_wo, inputs=[playground, chat_bot_wo], outputs=[chat_bot_wo], show_progress=True)


    submit_button.click(
        submit, inputs=[user_prompt , playground, chat_bot], outputs= [user_prompt, chat_bot]
    )
    submit_button.click(
        clear_user_input, [], [user_prompt]
    )
    clear_result_button.click(reset_state, inputs=[playground, chat_bot], outputs=[chat_bot], show_progress=True)

    start_button.click(
        start_state, inputs=[file_upload, playground, chat_bot,Img, em_model], outputs=[chat_bot,Img], show_progress=True
    )

if __name__ == "__main__":
    gr.close_all()
    demo.launch(server_port=40044, share=False, show_api=False)
