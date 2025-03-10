from langchain_openai import OpenAIEmbeddings
import streamlit as st

import os

import getpass
import os


# Add an environment variable
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME= st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION= st.secrets.get("AZURE_OPENAI_API_VERSION") 


from langchain_openai import AzureChatOpenAI


os.environ['PINECONE_API_KEY'] = 'pcsk_2yWxfV_RzZcenPUjLkzMK78P8D2MEX6yfzSZJ2GYCKCfkiHUpgbj8ekG4yWfue7JJsEYtr'

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

import getpass
import os


from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment='text-embedding-3-large',
    openai_api_version='2023-05-15',
)


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(os.environ['PINECONE_API_KEY'])


# vector store
index_name = "helpdesk"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever()


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information abput laptops.",
)

tools = [retriever_tool]

from typing import List
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    generation: str
    documents: List[str]

### Router


from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage

from pydantic import BaseModel, Field


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vector_store", "final_response"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call

structured_llm_router = llm.with_structured_output(RouteQuery)


# Prompt
system = """
"You are an expert at determining whether a user's question should be answered using a vector store or a final response. The vector store contains documents related to 
laptop and computer troubleshooting guides. Use the vector store for questions on these topics. Additionally, if the user's question appears to be a follow-up 
(e.g., 'tell me more on this' or 'it didn't work'),retrieve relevant context from the vector store to provide a more informed response. If no relevant information is found in the vector store, use the final response."
Instructions:
If the question is about laptop or computer troubleshooting, retrieve relevant information from the vector store.
If the question appears to reference an earlier conversation (e.g., a follow-up like "tell me more" or "it didn’t work"), try to retrieve context from previous interactions in the vector store before responding.
If the question is unrelated to these topics and no relevant data exists in the vector store, provide a final response.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


def final_response(state):
    final_msg = ("Sorry, this question is beyond my knowledge, "
                 "as a virtual assistant I can only assist you on any troubleshooting with your laptop")
    return {"messages": [AIMessage(content=final_msg)]}


from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

def get_latest_user_question(messages):
    # Iterate over the messages in reverse order.
    for msg in reversed(messages):
        # Check if the message is a HumanMessage.
        # Adjust this check if you have a different way of identifying user messages.
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""


### Edges




def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
   # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)=
    model=llm

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade,method="function_calling")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool


    messages = state["messages"]
    last_message = messages[-1]

    #question = messages[0].content
    question = get_latest_user_question(st.session_state.conversation)
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    #model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model=llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question contextualized for HP.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with a re-phrased question specific to HP
    """

    print("---TRANSFORM QUERY FOR HP---")

    messages = state["messages"]
    #question = messages[0].content  # Extract the user's question
    question = get_latest_user_question(st.session_state.conversation)


    # Prompt to force contextualization for HP
    msg = [
        HumanMessage(
            content=f"""
        You are a virtual assistant specializing in HP laptop.
        Your job is to refine the user's question to be more specific to HP laptops troubleshooting.

        **User's Original Question:**
        {question}

        **Rewritten Question (must be relevant to HP laptop or accessories ):**
        """,
        )
    ]

    # Invoke the model to rephrase the question
    #model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = llm
    response = model.invoke(msg)

    #print("All messages:")
    #print("All messages:", messages)
    print("relevent conextualized question=" + response.content)

    #print(response.content)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    #question = messages[0].content
    question = get_latest_user_question(st.session_state.conversation)
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
   # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(
    template="""
    You are a helpdesk agent specializing in troubleshooting computer issues. Your goal is to assist customers by diagnosing problems, providing step-by-step solutions, and ensuring their systems are functioning properly."

    Context Information:
    {context}

    Customer's Issue:
    {question}

    Instructions:
    If the context contains relevant details, use them to provide accurate troubleshooting steps.
    Ask clarifying questions if needed to better understand the customer's issue.
    Provide clear, concise, and actionable solutions in a step-by-step format.
    If multiple solutions exist, suggest the most effective one first.
    If no relevant information is available, politely inform the customer:
    "I'm sorry, but I don't have enough details to resolve this issue. Could you provide more information?"
    Troubleshooting Response Format:
    Greeting & Acknowledgment: ("I'm here to help! Let's troubleshoot your issue.")
    Problem Diagnosis: ("Based on your description, this issue may be caused by...")
    Step-by-Step Solution: ("Try the following steps...")
    Next Steps: ("If the issue persists, you may need to...")
    Closing & Reassurance: ("Let me know if this resolves your issue! I'm happy to assist further.")
    """,
    input_variables=["context", "question"],)


    # LLM
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


#print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
#prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

### Hallucination Grader
from langchain_core.prompts import ChatPromptTemplate

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

################################# GRAPH################################################

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")
    # Initialize session state for retry count.
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

# Data model for hallucination grading
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM with function call
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt for hallucination grading
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



def grade_documents_limited(state) -> str:
    retry_count = st.session_state.retry_count +1
    print("---TEST global retry count is ---", retry_count)
    
    decision = grade_documents(state)  # Assume this function is defined elsewhere.
    
    if decision == "rewrite":
        if retry_count >= 1:
            print("---Maximum retries reached: switching to final response---")
            return "final"
        else:
            st.session_state.retry_count = retry_count + 1
            print("---after increment, global retry count is ---", global_retry_count)
            return "rewrite"
    else:
        return decision

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    messages = state["messages"]
    #question = messages[0].content
    question = get_latest_user_question(messages)
    source = question_router.invoke({"question": question})
    if source.datasource == "final_response":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "final_response"
    elif source.datasource == "vector_store":
        print("---ROUTE QUESTION TO RAG---")
        return "vector_store"


def hallucination_test(state):
    """Runs hallucination grading before ending the graph."""
    latest_response = state["messages"][-1].content if state["messages"] else ""
    
    # Ensure that docs is properly retrieved from the state or last response
    docs = state["messages"][-2].content if len(state["messages"]) > 1 else ""  # Use the second-to-last message as docs if available
    
    hallucination_result = hallucination_grader.invoke({"documents": docs, "generation": latest_response})
    
    result_msg = f"Hallucination test result: {hallucination_result.binary_score}"
    print(result_msg)
    return {"messages": [AIMessage(content=latest_response)],"result": hallucination_result.binary_score }

def final_response(state):
    final_msg = ("Sorry, this question is beyond my knowledge, "
                 "as a virtual assistant I can only assist you on any troubleshooting with your laptop")
    return {"messages": [AIMessage(content=final_msg)]}

# Define workflow
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("retrieve", ToolNode([retriever_tool]))
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)
graph.add_node("hallucination_test", hallucination_test)
graph.add_node("final_response", final_response)


graph.add_conditional_edges(
    START,
    route_question,
    {
        "final_response": "final_response",
        "vector_store": "rewrite",
    },
) 

graph.add_edge("rewrite", "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
graph.add_conditional_edges("retrieve", grade_documents_limited, {"rewrite": "rewrite", "generate": "generate", "final": "final_response"})
#graph.add_edge("generate", "hallucination_test")
#graph.add_edge("hallucination_test", END)
graph.add_edge("generate", END)
graph.add_edge("rewrite", "agent")

# Compile graph
memory = MemorySaver()
graph = graph.compile(checkpointer=memory)
##

#############################################GUI#################################################

import uuid
import streamlit as st

# Generate a thread_id dynamically if it doesn't exist in session state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Now use the dynamically generated thread_id in your config.
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "history" not in st.session_state:
    st.session_state.history = ""


import pprint


# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")

def run_virtual_assistant():
    st.title("Virtual Agent")

    # Display conversation history if available.
    if st.session_state.conversation:
        with st.expander("Click here to see the old conversation"):
            st.subheader("Conversation History")
            st.markdown({st.session_state.history})

    # Use a form to handle user input and clear the field after submission.
    with st.form(key="qa_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about any issues with your laptop. (or type 'reset' to clear):")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and user_input:
        # Allow the user to reset the conversation.
        if user_input.strip().lower() == "reset":
            st.session_state.conversation = []
            st.session_state.retry_count = 0
            st.experimental_rerun()
        else:
            # Append the user's question to the conversation history.
            st.session_state.conversation.append(("user", user_input))
            st.session_state.retry_count = 0

            # Prepare the input for the graph using the entire conversation history.
            inputs = {
                "messages": st.session_state.conversation,            }
            
            final_message_content = ""
            # Process the input through the graph (assumes 'graph' is defined globally).
            for output in graph.stream(inputs, config):
                for key, value in output.items():
                    # Check if the value is a dict containing messages.
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, "content"):
                                final_message_content = msg.content + "\n"
                                # Append the assistant response to conversation history.
                                st.session_state.conversation.append(("assistant", msg.content))
                            else:
                                final_message_content = str(msg) + "\n"
                                st.session_state.conversation.append(("assistant", str(msg)))

            # Render the final response.
            st.markdown(final_message_content)
            st.session_state.history+="################MESSAGE###############"
            st.session_state.history+=final_message_content


if __name__ == "__main__":
    run_virtual_assistant()


