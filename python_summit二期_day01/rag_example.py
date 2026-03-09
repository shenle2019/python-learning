"""
RAG (检索增强生成) 实现
基于 LangChain + FAISS + Qwen
"""
import os
import re
import torch
import numpy as np
from typing import List, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA


class ChineseTextSplitter(CharacterTextSplitter):
    """中文文本分块器"""
    
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        
        # 中文句子分隔符
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["'"'"』"'"']]{0,2}|(?=["'"'"『"'"']]{1,2}|$))'
        )
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


def load_file(filepath: str, encoding: str = "utf-8"):
    """
    加载文件并分块
    
    Args:
        filepath: 文件路径
        encoding: 文件编码
    Returns:
        分块后的文档列表
    """
    loader = TextLoader(filepath, autodetect_encoding=True)
    textsplitter = ChineseTextSplitter(pdf=False)
    docs = loader.load_and_split(textsplitter)
    
    # 保存加载的临时文件用于检查
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath: str, docs: List[Document]):
    """保存加载检查文件"""
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write(f"filepath={filepath},len={len(docs)}\n")
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def separate_list(ls: List[int]) -> List[List[int]]:
    """将连续的索引分组"""
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


class FAISSWrapper(FAISS):
    """增强版 FAISS 向量数据库，支持自定义参数"""
    
    chunk_size: int = 250
    chunk_content: bool = True
    score_threshold: float = 0

    def similarity_search_with_score_by_vector(
        self, 
        embedding: List[float], 
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32), 
            k
        )
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                continue
            
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            
            if not self.chunk_content:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            
            id_set.add(i)
            docs_len = len(doc.page_content)
            
            # 扩展相关块
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        
        if not self.chunk_content:
            return docs
        
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        
        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            
            doc_score = min([
                scores[0][id] 
                for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]
            ])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        
        return docs


def create_rag_system(
    llm,
    embedding_model_name: str = "text2vec",
    embedding_model_path: str = None,
    embedding_device: str = "cuda",
    prompt_template: str = None,
    vector_search_top_k: int = 3,
    chain_type: str = "stuff"
):
    """
    创建 RAG 系统
    
    Args:
        llm: LangChain LLM 实例
        embedding_model_name: Embedding 模型名称
        embedding_model_path: Embedding 模型路径
        embedding_device: 设备 (cuda/cpu)
        prompt_template: 自定义提示词模板
        vector_search_top_k: 检索返回数量
        chain_type: 链类型
    Returns:
        RAG 检索问答链
    """
    
    if prompt_template is None:
        prompt_template = """已知信息:
{context_str}

基于以上已知信息，请简洁而专业地回答用户的问题。如果无法从已知信息中找到答案，请回复"根据给定信息无法回答此问题"，请勿编造信息。请用中文回答。
问题: {question}"""
    
    # 初始化 Embedding
    if embedding_model_path:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': embedding_device}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': embedding_device}
        )
    
    # 创建 Prompt
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context_str", "question"]
    )
    
    chain_type_kwargs = {
        "prompt": prompt, 
        "document_variable_name": "context_str"
    }
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=None,  # 后续设置
        chain_type_kwargs=chain_type_kwargs
    )
    
    return qa, embeddings


# 使用示例
if __name__ == "__main__":
    # 配置参数
    FILEPATH = "your_file_path"  # 替换为你的文件路径
    EMBEDDING_MODEL = "text2vec"  # 或使用其他 Embedding 模型
    EMBEDDING_DEVICE = "cuda"
    VECTOR_SEARCH_TOP_K = 3
    
    # 加载文档
    docs = load_file(FILEPATH)
    
    # 使用示例：
    # 1. 创建 LLM 实例
    # from qwen_llm import Qwen
    # llm = Qwen()
    
    # 2. 创建 RAG 系统
    # qa, embeddings = create_rag_system(llm)
    
    # 3. 构建向量库
    # docsearch = FAISSWrapper.from_documents(docs, embeddings)
    # qa.retriever = docsearch.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})
    
    # 4. 提问
    # query = "请介绍一下大语言模型"
    # print(qa.run(query))
    
    print("请根据实际情况修改配置后使用")
