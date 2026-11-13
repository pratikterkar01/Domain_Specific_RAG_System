# rag_prompt.py
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from Utils import Config


class RagPrompt:
    """
    Prompt, context formatting and answer post-processing for the building-code RAG.

    Usage with rag_chain.py (your hybrid retrievers with .search()):
        from Prompt import RagPrompt
        rag = ComplianceRAG(retriever, RagPrompt())

    Usage on its own (a LangChain retriever whose .invoke() returns Documents):
        chain = RagPrompt().build_chain(retriever, llm)
        out = chain.invoke("your question")
    """

    def __init__(self, settings=None, system_prompt: str | None = None,
                 human_prompt: str | None = None, abstain_text: str | None = None):
        # Everything comes from your config unless you pass an override.
        self.settings = settings or Config.get_settings()
        self.system_prompt = system_prompt or self.settings.SYSTEM_PROMPT
        self.human_prompt = human_prompt or self.settings.HUMAN_PROMPT
        self.abstain_text = abstain_text or self.settings.ABSTAIN_TEXT
        self.prompt = self._build_prompt()

    # ---- 1. Prompt template ----
    @staticmethod
    def _escape_braces(text: str) -> str:
        """Literal { } in the system text must not be read as template variables."""
        return text.replace("{", "{{").replace("}", "}}")

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self._escape_braces(self.system_prompt)),  # no template vars in system text
            ("human", self.human_prompt),                          # uses {context} and {question}
        ])

    # ---- 2. Format retrieved docs as numbered clauses ----
    @staticmethod
    def format_context(docs) -> str:
        lines = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            sec = d.metadata.get("section", "")
            lines.append(f"[{i}] ({src}, {sec}) {d.page_content.strip()}")
        return "\n\n".join(lines)

    # ---- 3. Post-processing: abstain flag + citation check ----
    def postprocess(self, inputs: dict) -> dict:
        answer = inputs["answer"].strip()
        n_docs = len(inputs["docs"])
        cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer)}
        abstained = self.abstain_text.lower().rstrip(".") in answer.lower()
        return {
            "answer": answer,
            "abstained": abstained,
            "cited_ids": sorted(cited),
            "invalid_citations": sorted(c for c in cited if c < 1 or c > n_docs),
            "no_citation": (not abstained) and len(cited) == 0,
            "sources": [d.metadata for d in inputs["docs"]],
        }

    # ---- 4. Chain (for a LangChain retriever; rag_chain.py builds its own chain) ----
    def build_chain(self, retriever, llm):
        """
        retriever: LangChain retriever returning the final top-k parent sections (Documents)
        llm: any LangChain chat model, e.g. ChatOllama / ChatOpenAI-compatible vLLM endpoint
        """
        answer_chain = (
            RunnablePassthrough.assign(context=lambda x: self.format_context(x["docs"]))
            | self.prompt
            | llm
            | StrOutputParser()
        )
        return (
            {"question": RunnablePassthrough()}
            | RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
            | RunnablePassthrough.assign(answer=answer_chain)
            | RunnableLambda(self.postprocess)
        )


# ---- Example usage ----
if __name__ == "__main__":
    from langchain_ollama import ChatOllama   # or ChatOpenAI(base_url=<vLLM url>)

    llm = ChatOllama(model="qwen3:4b", temperature=0, num_predict=512)
    retriever = ...  # a LangChain retriever (Qdrant hybrid + rerank + parent expansion)
    chain = RagPrompt().build_chain(retriever, llm)

    out = chain.invoke("<your question>")
    print(out["answer"])
    print("Abstained:", out["abstained"], "| Cited:", out["cited_ids"])