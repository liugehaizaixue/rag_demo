from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io
import os
from PIL import Image
EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")

from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key = "sk-dea50ab4b6e34bc0bf9c937b19c11d08"
)

def init_db_from_content(contents,em_model):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = [contents]
    documents = text_splitter.create_documents(texts)
    if em_model == "nomic-embed-text":
        db = FAISS.from_documents(documents, EMBEDDING)
    else:
        db = FAISS.from_documents(documents, embeddings)
    return db , documents


def get_retriever(db:FAISS, documents):
    dense_retriever = db.as_retriever(search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k =  1  # Retrieve top 2 results
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever],
                                       weights=[0.4, 0.6])
    return ensemble_retriever

def visualize_vectors(db):
    # 获取Faiss索引
    index = db.index

    # 获取向量数量
    num_vectors = index.ntotal

    # 获取向量的维度（假设所有向量维度相同）
    d = index.d  # 向量的维度

    print(f"向量数量: {num_vectors}")
    print(f"向量维度: {d}")

    # 打印每个向量的值
    for i in range(num_vectors):
        vector = np.zeros(d, dtype='float32')
        index.reconstruct(i, vector)
        print(f"Vector {i}: {vector}")

    # 检查是否有足够的向量和特征
    if num_vectors < 2 or d < 2:
        raise ValueError("样本数或特征数过少，无法进行PCA降维。")

    # 提取所有向量
    vectors = np.zeros((num_vectors, d), dtype='float32')
    for i in range(num_vectors):
        index.reconstruct(i, vectors[i])

    # 使用PCA将向量降维到2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # 创建一个Figure和Axes对象
    fig, ax = plt.subplots()

    # 绘制散点图
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

    # 设置标题和标签
    ax.set_title("2D PCA Visualization of Vectors")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # 使用 PIL 打开该 BytesIO 数据并传递给 gradio
    img = Image.open(buf)
    return img

    # 显示图像
    plt.show()

    # 你也可以保存图像
    fig.savefig("pca_visualization.png")


if __name__ == "__main__":
    def read_txt(file_path):
        """读取单个TXT文件的内容"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    content = read_txt("./test.txt")
    db,docs = init_db_from_content(content)

    # 获取Faiss索引
    index = db.index

    # 获取向量数量
    num_vectors = index.ntotal

    # 获取向量的维度（假设所有向量维度相同）
    d = index.d  # 向量的维度

    # 打印每个向量的值
    # for i in range(num_vectors):
    #     vector = np.zeros(d, dtype='float32')
    #     index.reconstruct(i, vector)
    #     print(f"Vector {i}: {vector}")

    # 检查是否有足够的向量和特征
    if num_vectors < 2 or d < 2:
        raise ValueError("样本数或特征数过少，无法进行PCA降维。")

    # 提取所有向量
    vectors = np.zeros((num_vectors, d), dtype='float32')
    for i in range(num_vectors):
        index.reconstruct(i, vectors[i])

    # 使用PCA将向量降维到2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # 绘制2D图
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    plt.title("2D PCA Visualization of Vectors")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()