import getpass
import os

import streamlit as st

# Retrieve secrets using st.secrets
# Add an environment variable
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME= st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION= st.secrets.get("AZURE_OPENAI_API_VERSION") 


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

import getpass
import os


from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment='text-embedding-3-large',
    openai_api_version='2023-05-15',
)

################################# VECTOR STORE ###########################################


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

pc = Pinecone('pcsk_2yWxfV_RzZcenPUjLkzMK78P8D2MEX6yfzSZJ2GYCKCfkiHUpgbj8ekG4yWfue7JJsEYtr')


# vector store
index_name = "helpdesk"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever()


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_troubleshooting_guide",
    "Search and return information regarding the troubleshooting information for common laptop problem.",
)

tools = [retriever_tool]

########################################################## AGENT STATE####################################################

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

########################################################## ROUTER ####################################################

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
        description="Given a user question choose to route it to final answer or a vectorstore.",
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
                 "as a virtual assistant I can only assist you on laptop hardware or OS related issues")
    return {"messages": [AIMessage(content=final_msg)]}

print(question_router.invoke({"question": "it was not much help?"}))


######################################################### UTILITY #########################################################

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

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
    # Iterate over the messages in reverse order
     for role, content in reversed(messages):
        if role.lower() == "user":
            return content
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
    #model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

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

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = get_latest_assistant_message(st.session_state.conversation)

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

   # print("Messages before summary:", messages)

    
    # Summarizing long messages before sending them to the model
    def summarize_message(msg):
        if isinstance(msg, AIMessage) and len(msg.content) > 1000:
            return AIMessage(content=msg.content[:997] + "...")
        return msg

    messages = [summarize_message(msg) for msg in messages]

    print("Messages after summarization:", messages)
   
    model=llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list

    return {"messages": [response]}

##############DEBUG #######################################

from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_all_assistant_messages(messages):
    # Collect all assistant messages
    assistant_messages = [content for role, content in reversed(messages) if role.lower() == "ai"]
    
    # Return the list of all assistant messages
    return assistant_messages



def get_latest_assistant_message(messages):
    # Iterate over the messages in reverse order to find the latest assistant message
    for role, content in reversed(messages):
        if role.lower() == "ai":
            return content
    return ""

############################################################

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
    #question = get_latest_user_question(messages)
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

    # Invoke the model to rephrase the question with relevent context
    #model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = llm
    response = model.invoke(msg)
    print("relevent conextualized question=" + response.content)

    st.session_state.conversation.append(("ai", response.content))

    #print(response.content)
    return {"messages": [response]}


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]

    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)
    # Assume the last assistant message (or retrieved content) holds the context.
    last_message = messages[-1]
    docs = last_message.content

    prompt = PromptTemplate(
        template = """  
            You are a **helpful, patient, and knowledgeable** helpdesk agent assisting users with troubleshooting laptop issues.  
            Your goal is to **engage naturally, diagnose problems interactively, and guide users step-by-step.**  

            ---  
            **User’s Issue:**  
            {question}  

            **Relevant Context (if available):**  
            {context}  
            ---  

            ### **How to Respond:**  
            - Speak **naturally and conversationally**—don’t sound like a script.  
            - Adapt to the user's level of tech knowledge. If they seem less technical, explain simply.  
            - If you need more info, **ask follow-up questions naturally** instead of saying, *"Need more details."*  
            - Don’t just list solutions—**walk the user through them like a real conversation.**  
            - Change response structure **dynamically** based on previous answers.  

            ### **Example Interaction Style:**  
            - If the user seems frustrated, acknowledge it:  
            *"That sounds frustrating! Let's get it fixed quickly."*  
            - If they’re unsure, reassure them:  
            *"No worries, I’ll guide you step by step."*  
            - If it’s a common issue, be proactive:  
            *"Ah, I’ve seen this before! Try this first..."*  
            - If troubleshooting requires multiple steps, ask for feedback after each one.  
            *"Did that help, or is the issue still there?"*  

            ### **Response Format (Flexible, Not Fixed):**  
            - Start with a conversational opener, adjusting tone based on the issue.  
            - **Possible Cause:** Briefly explain what might be wrong (if clear).  
            - **Step-by-Step Fix:** Guide them interactively, asking for feedback.  
            - **Next Steps:** If the issue isn’t resolved, suggest what to try next or escalate.  

            ⚡ **Be engaging, patient, and dynamic. No robotic responses.**  
        """,
        input_variables=["context", "question"]
    )

   # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
   
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


################### Hallucination Grader #############################################
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

############################################### GRAPH ###########################################

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import streamlit as st

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

# Global variable for retry count
global_retry_count = 0

def grade_documents_limited(state) -> str:

    decision = grade_documents(state)  # Assume this function is defined elsewhere.
    retry_count = st.session_state.retry_count +1
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
    Route question to final answer or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    messages = state["messages"]
    #question = messages[0].content
    question = get_latest_user_question(st.session_state.conversation)
    print("user question" + question)
    source = question_router.invoke({"question": question})
    if source.datasource == "final_response":
        print("---ROUTE QUESTION TO final response---")
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
#graph.add_node("hallucination_test", hallucination_test)
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
    st.title("Virtual Agent for Laptop Technical Support")

    # Display conversation history if available.
    if st.session_state.conversation:
        with st.expander("Click here to see the old conversation"):
            st.subheader("Conversation History")
            st.markdown({st.session_state.history})

    # Use a form to handle user input and clear the field after submission.
    with st.form(key="qa_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about your laptop problem :")
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
                "messages": st.session_state.conversation,            
                }
            
            #print("user input message "+ st.session_state.conversation)


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
