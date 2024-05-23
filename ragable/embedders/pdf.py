from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from ragable.adapters.interfaces.vector_store_adapter import VectorStoreAdapter

class PdfEmbedder:
    chunk_size: int
    chunk_overlap: int

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def train(self, store :VectorStoreAdapter, pdf_paths :List[str]):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        for pdf_path in pdf_paths:
            text_blob = ""
            with open(pdf_path, "rb") as pdfFile:
                reader = PdfReader(pdfFile)
                for page in reader.pages:
                    text_blob += page.extract_text()

            for chunk in text_splitter.split_text(text_blob):
                store.add_document(text=chunk)
