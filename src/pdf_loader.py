from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document

def extract_text_from_pdfs(pdf_docs) -> List[Document]:
    """
    Extracts text page-by-page from uploaded PDF file objects 
    and converts them into LangChain Document instances with metadata.
    """
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = getattr(pdf, "name", "uploaded_doc")
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_name, "page": page_num}
                )
                documents.append(doc)
    return documents
