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
    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""
    
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
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
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
