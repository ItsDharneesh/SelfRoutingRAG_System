"""Vector store embeddings for document embeddings and retrieval"""

# from typing import List
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document


# class VectorStore:
#     """Manages vectore store appplications"""

#     def __init__(self):
#         self.embedding=OpenAIEmbeddings()
#         self.vectostore=None
#         self.retriever=None

#     def create_retreiver(self, documents: List[Document]):
#         """
#         create vector store from docuemnts
        
#         Args:
#             documents: List of documents to embed
#         """
#         self.vectostore=FAISS.from_documents(documents,self.embedding)
#         self.retriever=self.vectostore.as_retriever

#     def get_retreiver(self):
#         """
#         Get the retriver instance
        
#         Returns: Retriever instance
#         """
#         if self.retriever is None:
#             raise ValueError("Vectore store not initialised. Call create_vectorstore first.")
#         return self.retriever
    
#     def retrieve(self,query:str,k: int=4)-> List[Document]:
#         """
#         reteive relevant documents for a query

#         Args:
#             query: Search query
#             k: Number of documents to retrieve
        
#         Returns:
#             List of relevant documents
#         """

#         if self.retriever is None:
#             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
#         return self.retriever.invoke(query)

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStore:
    """Manages vector store embeddings and retrieval"""

    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vectostore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        self.vectostore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectostore.as_retriever()

    def get_retriever(self):
        """Get the retriever instance"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    
    
    
    
    
