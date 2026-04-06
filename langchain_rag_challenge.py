"""
=============================================================
  PYTHON CODING CHALLENGE
  Topic   : LangChain v1 · RAG Agents · pgvector ·
            Embeddings · LangSmith
  Level   : Intermediate
  Tasks   : 20  (project-style, grouped by topic)
=============================================================

SETUP — install dependencies before you begin
----------------------------------------------
  pip install langchain langchain-openai langchain-community
              langchain-core langsmith psycopg2-binary numpy
              python-dotenv

ENVIRONMENT VARIABLES — create a .env file or export these:
  OPENAI_API_KEY       = "sk-..."
  LANGCHAIN_API_KEY    = "ls__..."        # LangSmith
  LANGCHAIN_TRACING_V2 = "true"
  LANGCHAIN_PROJECT    = "rag-challenge"
  PG_CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/vectordb"

TOPIC SECTIONS
--------------
  Section A — LangChain Core         (Tasks  1 – 4)
  Section B — Embeddings             (Tasks  5 – 8)
  Section C — pgvector               (Tasks  9 – 13)
  Section D — RAG Agents             (Tasks 14 – 17)
  Section E — LangSmith              (Tasks 18 – 20)

RULES
-----
  - Implement every function stub below.
  - Do NOT add extra libraries beyond those listed in Setup.
  - Keep function signatures exactly as given.
  - For tasks that call an LLM, handle API errors gracefully
    with try/except.
=============================================================
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# SECTION A — LangChain Core  (Tasks 1 – 4)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 1 — Basic LCEL Chain with PromptTemplate
# ─────────────────────────────────────────────────────────────
"""
TASK 1: Basic LCEL Chain
--------------------------
Build a simple LangChain Expression Language (LCEL) chain that:
  1. Accepts a topic as input.
  2. Fills it into a PromptTemplate.
  3. Sends the prompt to ChatOpenAI (gpt-4o-mini).
  4. Parses the output as a plain string.
  5. Returns the result.

Use the pipe operator  |  to chain components.

Expected usage:
  result = basic_lcel_chain("quantum computing")
  print(result)
  # "Quantum computing uses quantum bits (qubits)..."

HINT:
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain_core.output_parsers import StrOutputParser

  chain = prompt | llm | parser
  chain.invoke({"topic": "..."})
"""

def basic_lcel_chain(topic: str) -> str:
    """Returns a one-paragraph explanation of the given topic."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass  # Remove this line when you start coding

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 2 — Sequential Chain (Multi-Step Pipeline)
# ─────────────────────────────────────────────────────────────
"""
TASK 2: Sequential Chain (Multi-Step Pipeline)
------------------------------------------------
Build a two-step LCEL chain:
  Step 1 — given a topic, generate a short 3-sentence summary.
  Step 2 — given the summary, translate it into French.

Return a dict with keys:
  {"summary": "...", "translation": "..."}

HINT:
  - Use RunnablePassthrough or RunnableParallel to pass
    intermediate outputs to the next step.
  - from langchain_core.runnables import RunnablePassthrough
  - Chain: (prompt1 | llm | parser) then feed output
    into (prompt2 | llm | parser).
  - You can use two separate .invoke() calls if you prefer
    to keep it simple and readable.
"""

def sequential_chain(topic: str) -> dict:
    """Returns {'summary': ..., 'translation': ...} for the topic."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 3 — Conversation Chain with Memory
# ─────────────────────────────────────────────────────────────
"""
TASK 3: Conversation Chain with Memory
----------------------------------------
Build a conversational chain that:
  - Maintains chat history across multiple turns.
  - Uses ChatPromptTemplate with a MessagesPlaceholder
    for the history.
  - Returns a list of (role, content) tuples representing
    the full conversation after all turns.

Simulate this 3-turn conversation:
  Turn 1 — user: "My name is Alex. What is machine learning?"
  Turn 2 — user: "Can you give me a real-world example?"
  Turn 3 — user: "What is my name?"   ← tests memory

Expected (partial):
  [("human", "My name is Alex..."),
   ("ai",    "Machine learning is..."),
   ...
   ("ai",    "Your name is Alex.")]

HINT:
  from langchain_core.chat_history import InMemoryChatMessageHistory
  from langchain_core.runnables.history import RunnableWithMessageHistory
  Use session_id to scope the history.
"""

def conversation_with_memory() -> list:
    """Runs a 3-turn conversation and returns the full message history."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 4 — Agent with Custom Tools
# ─────────────────────────────────────────────────────────────
"""
TASK 4: Agent with Custom Tools
---------------------------------
Create a LangChain agent that uses two custom tools:
  Tool 1 — word_count(text: str) → int
            Returns the number of words in a text.
  Tool 2 — reverse_text(text: str) → str
            Returns the text reversed word-by-word.

Build the agent using the @tool decorator and
create_react_agent, then run it with AgentExecutor.

Test query:
  "How many words are in 'The quick brown fox'?
   Also reverse it."

HINT:
  from langchain.agents import create_react_agent, AgentExecutor
  from langchain.tools import tool
  from langchain import hub
  prompt = hub.pull("hwchase17/react")
"""

def agent_with_tools(query: str) -> str:
    """Runs a ReAct agent with custom tools and returns the final answer."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION B — Embeddings  (Tasks 5 – 8)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 5 — Generate and Inspect Embeddings
# ─────────────────────────────────────────────────────────────
"""
TASK 5: Generate and Inspect Embeddings
-----------------------------------------
Use OpenAIEmbeddings (text-embedding-3-small) to embed a list
of sentences. Return a dict with:
  {
    "num_sentences" : int,
    "embedding_dim" : int,
    "first_5_values": list[float],   # first 5 values of sentence[0]
    "vectors"       : list[list[float]]
  }

sentences = [
  "LangChain simplifies LLM application development.",
  "pgvector adds vector search to PostgreSQL.",
  "RAG grounds language models with external knowledge.",
]

HINT:
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vectors = embeddings.embed_documents(sentences)
  A single vector is a plain Python list of floats.
"""

def generate_embeddings(sentences: list) -> dict:
    """Embeds a list of sentences and returns metadata + vectors."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 6 — Cosine Similarity (from scratch, then with numpy)
# ─────────────────────────────────────────────────────────────
"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.

Both should return a float between -1 and 1.

Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"

Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)

HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""

def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


def compare_word_pairs() -> dict:
    """
    Embeds dog/puppy and dog/automobile, returns:
    {
      "dog_vs_puppy"      : float,
      "dog_vs_automobile" : float,
      "more_similar_pair" : str
    }
    """
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 7 — Batch Embedding with Chunking
# ─────────────────────────────────────────────────────────────
"""
TASK 7: Batch Embedding with Chunking
----------------------------------------
Given a long text document, split it into overlapping chunks
using RecursiveCharacterTextSplitter, then embed all chunks
in a single batch call.  Return:
  {
    "num_chunks"   : int,
    "chunk_size"   : int,   # configured chunk size
    "overlap"      : int,   # configured overlap
    "embedding_dim": int,
    "chunks"       : list[str]
  }

Use chunk_size=200, chunk_overlap=40.

HINT:
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200, chunk_overlap=40
  )
  chunks = splitter.split_text(long_text)
  vectors = embeddings.embed_documents(chunks)
"""

SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""

def batch_embed_with_chunks(text: str, chunk_size: int, overlap: int) -> dict:
    """Splits text into chunks, embeds them, and returns metadata."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 8 — Compare Two Embedding Models
# ─────────────────────────────────────────────────────────────
"""
TASK 8: Compare Two Embedding Models
--------------------------------------
Embed the same sentence using two different OpenAI models:
  Model A: text-embedding-3-small   (1536 dims)
  Model B: text-embedding-3-large   (3072 dims)

For the sentence:  "Vector databases power semantic search."

Return a dict:
  {
    "sentence"   : str,
    "model_a"    : {"model": str, "dims": int, "first_3": list[float]},
    "model_b"    : {"model": str, "dims": int, "first_3": list[float]},
    "dim_ratio"  : float   # model_b_dims / model_a_dims
  }

HINT:
  OpenAIEmbeddings(model="text-embedding-3-small")
  OpenAIEmbeddings(model="text-embedding-3-large")
  embeddings.embed_query(sentence) → single vector (list of floats)
"""

def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION C — pgvector  (Tasks 9 – 13)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 9 — Create pgvector Table via psycopg2
# ─────────────────────────────────────────────────────────────
"""
TASK 9: Create pgvector Table via psycopg2
-------------------------------------------
Connect directly to PostgreSQL using psycopg2 and:
  1. Enable the pgvector extension.
  2. Drop then recreate a table called "documents" with:
       id       SERIAL PRIMARY KEY
       content  TEXT
       metadata JSONB
       embedding vector(1536)
  3. Return True on success, raise on error.

Prereq — PostgreSQL must be running with pgvector installed:
  CREATE EXTENSION IF NOT EXISTS vector;

HINT:
  import psycopg2, json
  conn = psycopg2.connect(os.environ["PG_CONNECTION_STRING_RAW"])
  # PG_CONNECTION_STRING_RAW = "host=... dbname=... user=... password=..."
  cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          content TEXT,
          metadata JSONB,
          embedding vector(1536)
      )
  ''')
"""

def setup_pgvector_table() -> bool:
    """Creates the pgvector extension and documents table. Returns True on success."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 10 — Insert Document Embeddings into pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 10: Insert Document Embeddings
--------------------------------------
Given a list of (content, metadata) tuples, embed each document
using OpenAIEmbeddings and insert them into the "documents"
table created in Task 9.  Return the count of inserted rows.

documents = [
  ("LangChain enables LLM pipelines.", {"source": "docs", "page": 1}),
  ("pgvector stores vector embeddings.", {"source": "docs", "page": 2}),
  ("RAG retrieves relevant context.",   {"source": "paper", "page": 5}),
  ("LangSmith traces LLM calls.",       {"source": "blog",  "page": 1}),
]

HINT:
  import json
  vector = embeddings.embed_query(content)
  # Convert list to string for psycopg2:  str(vector)  or  json.dumps(vector)
  cursor.execute(
      "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
      (content, json.dumps(metadata), str(vector))
  )
"""

def insert_documents(documents: list) -> int:
    """Embeds and inserts documents. Returns count of inserted rows."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 11 — Similarity Search with pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 11: Similarity Search with pgvector
------------------------------------------
Embed a query string and find the top-k most similar documents
using cosine distance (<=>).  Return a list of dicts:
  [{"content": str, "metadata": dict, "distance": float}, ...]

HINT:
  vector_str = str(embeddings.embed_query(query))
  cursor.execute('''
      SELECT content, metadata, embedding <=> %s AS distance
      FROM documents
      ORDER BY distance ASC
      LIMIT %s
  ''', (vector_str, top_k))
  rows = cursor.fetchall()
"""

def similarity_search(query: str, top_k: int = 3) -> list:
    """Returns top-k similar documents for the query."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 12 — Metadata Filtering in pgvector
# ─────────────────────────────────────────────────────────────
"""
TASK 12: Metadata Filtering
------------------------------
Extend the similarity search to filter by a metadata field.
Only return documents whose metadata->>'source' matches the
given source value.

Example:
  results = filtered_search("LLM tracing", source_filter="blog", top_k=2)

HINT:
  Add a WHERE clause using JSONB operators:
  WHERE metadata->>'source' = %s
  Parameters: (vector_str, source_filter, top_k)
"""

def filtered_search(query: str, source_filter: str, top_k: int = 3) -> list:
    """Returns top-k similar docs filtered by metadata source."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 13 — LangChain PGVector VectorStore Integration
# ─────────────────────────────────────────────────────────────
"""
TASK 13: LangChain PGVector VectorStore
-----------------------------------------
Use LangChain's built-in PGVector vectorstore to:
  1. Create a PGVector store from the document list in Task 10.
  2. Run a similarity_search_with_score for a query.
  3. Return a list of (Document, score) tuples.

Use collection_name="lc_documents".

HINT:
  from langchain_community.vectorstores import PGVector
  from langchain_core.documents import Document

  docs = [Document(page_content=c, metadata=m) for c, m in documents]

  store = PGVector.from_documents(
      documents=docs,
      embedding=embeddings,
      collection_name="lc_documents",
      connection_string=os.environ["PG_CONNECTION_STRING"],
  )
  results = store.similarity_search_with_score(query, k=top_k)
"""

def langchain_pgvector_search(documents: list, query: str, top_k: int = 3) -> list:
    """Creates a PGVector store and runs a scored similarity search."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION D — RAG Agents  (Tasks 14 – 17)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 14 — Basic RAG Pipeline
# ─────────────────────────────────────────────────────────────
"""
TASK 14: Basic RAG Pipeline
------------------------------
Build an end-to-end RAG chain that:
  1. Loads documents from a list of strings.
  2. Stores them in a PGVector vectorstore.
  3. Creates a retriever (top-3 results).
  4. Passes retrieved context + question to ChatOpenAI.
  5. Returns the final answer string.

Use the LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

HINT:
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  prompt = ChatPromptTemplate.from_template(
      "Answer using only this context:\n{context}\n\nQuestion: {question}"
  )
"""

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def basic_rag_pipeline(documents: list, question: str) -> str:
    """Indexes documents and answers the question using RAG."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 15 — RAG with Source Attribution
# ─────────────────────────────────────────────────────────────
"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }

HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:

  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""

def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""

def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 17 — RAG Agent (Tool-based Retrieval)
# ─────────────────────────────────────────────────────────────
"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.

Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.

HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""

def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION E — LangSmith  (Tasks 18 – 20)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 18 — Trace a Chain with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 18: LangSmith Tracing
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.

Expected return:
  {"answer": str, "run_id": str}

HINT:
  from langchain_core.tracers.context import collect_runs

  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""

def traced_chain(topic: str) -> dict:
    """Runs a chain with LangSmith tracing. Returns answer and run_id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 19 — Create a LangSmith Dataset
# ─────────────────────────────────────────────────────────────
"""
TASK 19: Create a LangSmith Dataset and Add Examples
------------------------------------------------------
Use the LangSmith SDK to:
  1. Create a dataset named "rag-eval-dataset".
  2. Add 3 question-answer example pairs to it.
  3. Return the dataset id as a string.

Examples to add:
  Q: "What does RAG stand for?"
     A: "Retrieval-Augmented Generation"
  Q: "What PostgreSQL extension enables vector search?"
     A: "pgvector"
  Q: "What LangChain tool provides observability?"
     A: "LangSmith"

HINT:
  from langsmith import Client
  client = Client()

  dataset = client.create_dataset("rag-eval-dataset")
  client.create_examples(
      inputs=[{"question": q} for q in questions],
      outputs=[{"answer": a} for a in answers],
      dataset_id=dataset.id
  )
"""

def create_langsmith_dataset() -> str:
    """Creates a LangSmith dataset with 3 examples. Returns dataset id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.

Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}

HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator

  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}

  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""

def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    pass

    # ── END OF YOUR CODE ─────────────────────────────────────


# =============================================================
#  MAIN — run and print results for each task
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LANGCHAIN · RAG · PGVECTOR · EMBEDDINGS · LANGSMITH")
    print("20-Task Coding Challenge")
    print("=" * 60)

    # ── Section A ─────────────────────────────────────────────
    print("\n── SECTION A: LangChain Core ──────────────────────────\n")

    print("[Task 1] Basic LCEL Chain")
    result = basic_lcel_chain("vector databases")
    print(result)

    print("\n[Task 2] Sequential Chain")
    seq = sequential_chain("transformer architecture")
    print("Summary    :", seq.get("summary", ""))
    print("Translation:", seq.get("translation", ""))

    print("\n[Task 3] Conversation with Memory")
    history = conversation_with_memory()
    for role, msg in history:
        print(f"  [{role.upper()}] {msg[:80]}")

    print("\n[Task 4] Agent with Custom Tools")
    ans = agent_with_tools("How many words are in 'The quick brown fox'? Also reverse it.")
    print(ans)

    # ── Section B ─────────────────────────────────────────────
    print("\n── SECTION B: Embeddings ──────────────────────────────\n")

    sentences = [
        "LangChain simplifies LLM application development.",
        "pgvector adds vector search to PostgreSQL.",
        "RAG grounds language models with external knowledge.",
    ]

    print("[Task 5] Generate Embeddings")
    emb_info = generate_embeddings(sentences)
    print(f"  Sentences : {emb_info.get('num_sentences')}")
    print(f"  Dimensions: {emb_info.get('embedding_dim')}")
    print(f"  First 5   : {emb_info.get('first_5_values')}")

    print("\n[Task 6] Cosine Similarity")
    word_pairs = compare_word_pairs()
    print(f"  dog vs puppy      : {word_pairs.get('dog_vs_puppy', ''):.4f}")
    print(f"  dog vs automobile : {word_pairs.get('dog_vs_automobile', ''):.4f}")
    print(f"  More similar      : {word_pairs.get('more_similar_pair')}")

    print("\n[Task 7] Batch Embedding with Chunking")
    chunk_info = batch_embed_with_chunks(SAMPLE_DOCUMENT, 200, 40)
    print(f"  Chunks     : {chunk_info.get('num_chunks')}")
    print(f"  Embed dims : {chunk_info.get('embedding_dim')}")

    print("\n[Task 8] Compare Embedding Models")
    model_cmp = compare_embedding_models("Vector databases power semantic search.")
    print(f"  Model A dims : {model_cmp.get('model_a', {}).get('dims')}")
    print(f"  Model B dims : {model_cmp.get('model_b', {}).get('dims')}")
    print(f"  Dim ratio    : {model_cmp.get('dim_ratio')}")

    # ── Section C ─────────────────────────────────────────────
    print("\n── SECTION C: pgvector ────────────────────────────────\n")

    docs_to_insert = [
        ("LangChain enables LLM pipelines.", {"source": "docs", "page": 1}),
        ("pgvector stores vector embeddings.", {"source": "docs", "page": 2}),
        ("RAG retrieves relevant context.",   {"source": "paper", "page": 5}),
        ("LangSmith traces LLM calls.",       {"source": "blog",  "page": 1}),
    ]

    print("[Task 9] Setup pgvector Table")
    ok = setup_pgvector_table()
    print(f"  Table created: {ok}")

    print("\n[Task 10] Insert Documents")
    inserted = insert_documents(docs_to_insert)
    print(f"  Rows inserted: {inserted}")

    print("\n[Task 11] Similarity Search")
    results = similarity_search("How does LangSmith help?", top_k=2)
    for r in results:
        print(f"  [{r.get('distance', 0):.4f}] {r.get('content')}")

    print("\n[Task 12] Filtered Search")
    fresults = filtered_search("LLM tracing", source_filter="blog", top_k=2)
    for r in fresults:
        print(f"  [{r.get('distance', 0):.4f}] {r.get('content')}")

    print("\n[Task 13] LangChain PGVector Integration")
    lc_results = langchain_pgvector_search(docs_to_insert, "vector embeddings", top_k=2)
    for doc, score in lc_results:
        print(f"  [score={score:.4f}] {doc.page_content}")

    # ── Section D ─────────────────────────────────────────────
    print("\n── SECTION D: RAG Agents ──────────────────────────────\n")

    print("[Task 14] Basic RAG Pipeline")
    rag_ans = basic_rag_pipeline(RAG_DOCUMENTS, "What is LCEL?")
    print(" ", rag_ans)

    print("\n[Task 15] RAG with Source Attribution")
    rag_src = rag_with_sources(RAG_DOCUMENTS, "What distance metrics does pgvector support?")
    print("  Answer  :", rag_src.get("answer", ""))
    print("  Sources :")
    for s in rag_src.get("sources", []):
        print(f"    [{s.get('score', 0):.4f}] {s.get('content', '')[:60]}")

    print("\n[Task 16] Conversational RAG")
    conv_answers = conversational_rag(RAG_DOCUMENTS)
    print("  Turn 1:", conv_answers[0][:80] if conv_answers else "")
    print("  Turn 2:", conv_answers[1][:80] if len(conv_answers) > 1 else "")

    print("\n[Task 17] RAG Agent")
    agent_ans = rag_agent("What distance metrics does pgvector support?")
    print(" ", agent_ans)

    # ── Section E ─────────────────────────────────────────────
    print("\n── SECTION E: LangSmith ───────────────────────────────\n")

    print("[Task 18] Traced Chain")
    traced = traced_chain("embeddings")
    print(f"  Answer : {str(traced.get('answer', ''))[:80]}")
    print(f"  Run ID : {traced.get('run_id')}")

    print("\n[Task 19] Create LangSmith Dataset")
    dataset_id = create_langsmith_dataset()
    print(f"  Dataset ID: {dataset_id}")

    print("\n[Task 20] Run LangSmith Evaluation")
    eval_summary = run_langsmith_evaluation()
    print(f"  Dataset     : {eval_summary.get('dataset')}")
    print(f"  # Examples  : {eval_summary.get('num_examples')}")
    print(f"  Pass rate   : {eval_summary.get('pass_rate')}")

    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)
