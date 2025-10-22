from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


GROQ_DEFAULT_MODEL = "qwen/qwen3.6-27b"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

GROQ_MODEL_OPTIONS = [
    "qwen/qwen3.6-27b",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

GROQ_MODEL_LABELS = {
    "qwen/qwen3.6-27b":      "Qwen3 27B (default)",
    "groq/compound":         "Groq Compound",
    "groq/compound-mini":    "Groq Compound Mini",
    "openai/gpt-oss-120b":   "GPT-OSS 120B",
    "openai/gpt-oss-20b":    "GPT-OSS 20B",
    "allam-2-7b":            "Allam 7B",
}


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'document')}, Page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content}"
        for doc in docs
    )


def detect_provider(api_key: str) -> str:
    key = api_key.strip()
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-"):
        return "openai"
    raise ValueError(
        f"Unrecognized API key format. Expected 'sk-' for OpenAI or 'gsk_' for Groq. "
        f"Received prefix: '{key[:6]}...'"
    )


def build_rag_chain(vectorstore: FAISS, api_key: str, model_override: str = None):
    clean_key = api_key.strip()
    provider = detect_provider(clean_key)

    if provider == "groq":
        from langchain_groq import ChatGroq
        model_name = model_override or GROQ_DEFAULT_MODEL
        llm = ChatGroq(model_name=model_name, groq_api_key=clean_key, temperature=0.2)
    else:
        from langchain_openai import ChatOpenAI
        model_name = model_override or OPENAI_DEFAULT_MODEL
        llm = ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=clean_key)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a conversation history and the latest user question, "
            "rewrite the question as a self-contained search query. "
            "Do not answer the question. Return only the rewritten query."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_chain = contextualize_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a document Q&A assistant. Your ONLY job is to answer questions using the document excerpts provided in the Context section below. "
            "You MUST NOT use any knowledge from your training data. If the answer is not explicitly found in the context, respond with exactly: 'The documents do not contain information about this topic.'\n\n"
            "STRICT RULES:\n"
            "- Answer ONLY from the provided context. Never guess, infer beyond the text, or use external knowledge.\n"
            "- Be specific: quote exact figures, names, and details from the text.\n"
            "- Cite the page number inline at the end of every sentence that references the document, using format (Page N).\n"
            "- Write in clear, direct prose. No bullet points unless explicitly asked.\n"
            "- Do NOT start with 'Based on the context', 'The document states', or similar filler phrases.\n"
            "- Do NOT include <think>, <draft>, or any reasoning blocks in your output.\n\n"
            "Context from the indexed documents:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def contextual_retriever(input_dict):
        if input_dict.get("chat_history"):
            query = history_chain.invoke(input_dict)
        else:
            query = input_dict["input"]
        return retriever.invoke(query)

    def full_chain(input_dict):
        docs = contextual_retriever(input_dict)
        formatted_context = format_docs(docs)
        response_text = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": formatted_context,
            "chat_history": input_dict.get("chat_history", []),
            "input": input_dict["input"],
        })
        return {"answer": response_text, "context": docs}

    class RAGChain:
        def invoke(self, inputs):
            return full_chain(inputs)

    return RAGChain()
