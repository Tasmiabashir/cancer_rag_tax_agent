# cancer_rag_tax_agent

This project shows how a RAG agent and a ReAct agent work together — one agent that decides which of two tools to use depending on the question: a RAG tool over a research PDF, and a Pakistan FBR income tax calculator tool. Build the ONE agent with BOTH tools.

## Structure

- `rag_tax_agent.py` — builds the FAISS vector store from `cancer_treatment.pdf`, defines two tools (`cancer_pdf_qa`, `fbr_tax_calculator`), and runs a ReAct agent that picks the right tool per question.
- `requirements.txt` — pinned/required dependencies.
- `cancer_treatment.pdf` — source paper used for the RAG tool ("New approaches and procedures for cancer treatment: Current perspectives", SAGE Open Medicine, 2021).

## Setup

```bash
pip install -r requirements.txt --break-system-packages
export GROQ_API_KEY=your_groq_key_here   # or create a .env file (see .env.example)
python rag_tax_agent.py
```

## Notes

The agent decides per-question whether to retrieve from the PDF, calculate tax, or both — demonstrating multi-tool ReAct routing in LangChain.
