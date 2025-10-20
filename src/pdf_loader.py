from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document


def extract_text_from_pdfs(pdf_docs) -> List[Document]:
    documents = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        pdf_name = getattr(pdf, "name", "uploaded_document.pdf")
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"source": pdf_name, "page": page_num},
                    )
                )
    return documents
