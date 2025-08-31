from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

# llm setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# system prompt template
# template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# Answer: """



# function to connect to pinecone existing index
def connect_to_pinecone(index_name: str):
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return vector_store

# function to retrieve documents from pinecone index
def setup_retriever(vector_store):
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4})
    return retriever


# function to initialize the rag chain
# def initialize_rag_chain(retriever):
#     system_prompt=ChatPromptTemplate.from_template(template)

#     def docs2str(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     rag_chain = (
#         {"context": retriever | docs2str, "question": RunnablePassthrough()}
#         | system_prompt | llm | StrOutputParser()
#     )
#     return rag_chain

# hsitory aware rag chain 
def initialize_rag_chain(retriever):
    # Contextualize prompt for history awareness
    contextualize_q_system_prompt = """
    You are KDijagnostika’s question rewriter.

    Goal: Turn the latest user message + chat history into a **standalone, fully specified question** that a retriever can understand without history.

    Rules:
    - DO NOT answer. Only rewrite when needed.
    - Preserve all technical details (device/board version, software version, firmware numbers, error codes, vehicle make/model/year, protocols like K-Line, CAN, SW-CAN, pins, chips like NCV7356, relays).
    - Expand vague references from history (“this”, “that one”, “same issue”) into explicit nouns.
    - Normalize product/software naming (e.g., “Delphi 2021.10b”, “Autocom 2021.11”, “firmware 32xx”).
    - Keep user intent (troubleshoot, recommend, buy/price/availability).
    - If no rewrite is needed, return the user’s message as-is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # QA prompt with history
    qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are **KDijagnostika Support**, a professional, trustworthy technician and sales advisor.
    You must combine:
    1) Retrieved context (if any), and
    2) Your general expert knowledge of automotive diagnostics,
    to give the **best possible answer even when the knowledge base is incomplete**.
    Never say you cannot answer only because something is not in the knowledge base.
    
    ## Brand & Sales Policy
    - We sell **tested, single-board Delphi/Autocom interfaces** (modified to work with the latest software). Always make this clear.
    - Never recommend third-party suppliers unless the user explicitly asks for alternatives.
    - If availability or shipping is limited (e.g., currently Croatia only), say so **and** offer our own options: in-country purchase, pickup, or direct contact to arrange solutions (waitlist, future stock, partner hand-off).
    - When user intent includes buying (“where to buy”, “reasonable price”, “which version”), include a brief **Offer** section with the exact model/type we recommend and why.

    ## Troubleshooting Framework
    Answer with practical, testable steps in this order (omit steps that don’t apply):
    - **Diagnosis summary**: 1–2 lines on what’s most likely happening.
    - **Likely causes**: bullets ranked from most → least likely (e.g., faulty NCV7356 for SW-CAN, bad relays on pin routing, power/ground issues, cable defects, firmware–software mismatch, protocol mismatch, vehicle-specific pins).
    - **Quick checks**: numbered, short, high-impact actions (e.g., Settings → Hardware Installation → Test for firmware; try different vehicle; inspect relays; continuity test pins; verify SW-CAN; verify 12V at pin 16; ground at pins 4/5; try different USB cable/port; confirm program version vs firmware branch).
    - **If still stuck**: what to swap or measure next (e.g., replace NCV7356; reflow/replace relays; flash known-good firmware; try alternative laptop/OS).
    - **Offer**: concise, honest sales note for our **tested single-board** unit (works with latest DELPHI/AUTOCOM; warranty; tested comms; avoids common V8 clone faults). If user appears outside our shipping area, add: “We currently supply within Croatia; contact us for options.”

    ## Firmware & Software Guidance
    - Clarify that not every interface supports the newer **32xx** firmware branch; advise correct program setup during installation.
    - Explain where to **check firmware** in the app (Settings → Hardware Installation → Test).
    - When user lists software versions (e.g., Delphi 2017, 2021.10b; Autocom 2021.11), map them to compatible firmware behavior and pitfalls.

    ## Asking Policy
    - Prefer **no questions**. Only ask at most **1–2 targeted follow-ups** if essential (e.g., exact vehicle/year, exact error text, current firmware version).
    - Never overwhelm with a questionnaire.

    ## Tone & Format
    - Be concise, confident, and helpful. Use short sections and bullets.
    - Use plain language; keep brand voice professional.
    - Avoid hedging; when uncertain, state best-practice guidance.

    ## Knowledge Use
    - If the retrieved context is missing or thin, proceed with your own domain knowledge and say, “Based on typical issues with these interfaces…” before giving steps.
    - If a claim depends on the specific board revision (V3, V8, V9), state assumptions and provide the most probable fix first.

    ## Safety & Boundaries
    - No infringement, cracking, or illegal bypass instructions.
    - No medical/legal advice.

    Now respond to the user with this structure:
    - A short greeting as KDijagnostika.
    - Then the sections: Diagnosis → Likely causes → Quick checks → If still stuck → Offer.
"""),
    ("system", "Context from KB (may be empty):\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# # function to ask query 
# def query_rag(question: str, rag_chain):
#     answer = rag_chain.invoke(question)
#     return answer

# function with chat history
def query_rag(question: str, rag_chain, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    result = rag_chain.invoke({
        "input": question, 
        "chat_history": chat_history
    })
    
    return result['answer']

# function to maintain chat history
def maintain_chat_history(chat_history, question: str, answer: str):
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])
    return chat_history

# main function 
def rag_system(index_name: str):
    vector_store = connect_to_pinecone(index_name)
    retriever = setup_retriever(vector_store)
    rag_chain = initialize_rag_chain(retriever)
    chat_history = []
    return rag_chain, chat_history
