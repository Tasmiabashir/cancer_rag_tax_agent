"""
RAG + Multi-Tool ReAct Agent
----------------------------
A LangChain agent that answers questions from a PDF (via RAG) and calculates
Pakistan FBR income tax (via a custom tool), choosing which tool to use at runtime.
"""

import os
import re

# LLM - free, no credit card needed (Groq's hosted Llama models)
from langchain_groq import ChatGroq

# Document loading + chunking - reads the PDF, splits it into small pieces
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + vector store - turns text chunks into searchable "meaning vectors"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Retrieval chain - combines the vector store + LLM into a Q&A pipeline
from langchain.chains import RetrievalQA

# Agent - lets the LLM decide which tool to call and when
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate


# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PASTE_YOUR_GROQ_KEY_HERE")
PDF_PATH = "cancer_treatment.pdf"  # change to your local file path

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================
# STEP 1: LLM
# ============================================================
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ============================================================
# STEP 2: Load PDF and split into chunks
# ============================================================
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)


# ============================================================
# STEP 3: Embed chunks and build the vector store
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================
# STEP 4: Helper - clean up messy tool input
# ============================================================
def clean_tool_input(raw: str) -> str:
    """Some LLMs write 'param_name = value' instead of just 'value' - strip that off."""
    raw = raw.strip()
    match = re.match(r'^\w+\s*=\s*(.*)$', raw)
    if match:
        raw = match.group(1).strip()
    return raw.strip('"').strip("'")


# ============================================================
# STEP 5: Tool 1 - RAG Q&A over the PDF
# ============================================================
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


@tool
def cancer_pdf_qa(question: str) -> str:
    """Answer questions about the uploaded research paper using retrieval-augmented generation."""
    question = clean_tool_input(question)
    result = qa_chain.invoke({"query": question})
    return result["result"]


# ============================================================
# STEP 6: Tool 2 - FBR income tax calculator (pure Python, no LLM call)
# ============================================================
@tool
def fbr_tax_calculator(annual_income: str) -> str:
    """Calculate Pakistan FBR income tax for a salaried individual. Input: annual income in PKR."""
    try:
        cleaned = clean_tool_input(annual_income)
        match = re.search(r'[\d,]+\.?\d*', cleaned)
        if not match:
            return f"Could not find a valid number in input: {annual_income!r}"
        income = float(match.group().replace(",", ""))
    except Exception as e:
        return f"Error parsing income: {e}"

    slabs = [
        (0, 600_000, 0.0),
        (600_000, 1_200_000, 0.025),
        (1_200_000, 2_200_000, 0.11),
        (2_200_000, 3_200_000, 0.23),
        (3_200_000, 4_100_000, 0.30),
        (4_100_000, float("inf"), 0.35),
    ]
    tax = 0.0
    for lower, upper, rate in slabs:
        if income > lower:
            tax += (min(income, upper) - lower) * rate
        else:
            break

    return (
        f"For an annual income of PKR {income:,.0f}, the estimated FBR income tax "
        f"(FY2025-26 salaried slabs) is PKR {tax:,.0f} per year, "
        f"or about PKR {tax / 12:,.0f} per month."
    )


# ============================================================
# STEP 7: Build the ReAct agent with both tools
# ============================================================
tools = [cancer_pdf_qa, fbr_tax_calculator]

REACT_PROMPT = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)


# ============================================================
# STEP 8: Run a query
# ============================================================
if __name__ == "__main__":
    result = agent_executor.invoke(
        {
            "input": (
                "According to the PDF, what role do cancer stem cells play in tumor treatment? "
                "Also, what is the FBR income tax for an annual salary of PKR 2500000?"
            )
        }
    )
    print(result["output"])
