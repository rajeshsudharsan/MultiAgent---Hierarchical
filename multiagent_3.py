from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tool_node
from langchain_core.tools import tool
from typing_extensions import Literal
from langgraph.graph import StateGraph,START,END,MessagesState
from typing import TypedDict
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from IPython.display import display,Image
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time


llm = ChatOpenAI(model_name="o4-mini-2025-04-16")

#llm=ChatGroq(model="deepseek-r1-distill-llama-70b")
#llm = ChatGroq(model="Gemma2-9b-It")

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
search_tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

@tool
def summary_tool(content: str) :
    ''' This tool is used to summarize the content provided'''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_text(content)

    docs = text_splitter.create_documents(splits)
    prompt = ChatPromptTemplate.from_messages(
         [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )

    # Instantiate chain
    chain = create_stuff_documents_chain(llm, prompt)
    # Invoke chain
    result = chain.invoke({"context": docs})
    return result

@tool
def pdf_or_docx_generator_tool(content: str):
    ''' This tool is used to Generate the content in ether pdf or docx format'''
    
    current_time_seconds = time.time()
    current_time_milliseconds = int(round(current_time_seconds * 1000))

    
    docs_name = f'multi_agent_output_{current_time_milliseconds}.docx'
    pdf_name = f'multi_agent_output_{current_time_milliseconds}.pdf'
    #with open(docs_name, "w") as f:
    #    f.write(content)
    with open(pdf_name, "w") as f:
        f.write(content)
    return "Content written successfully"

class AgentState(MessagesState):
    next: str

class Router(TypedDict):
    next: Literal['researcher', 'reporter', 'FINISH']

class ResearcherRouter(TypedDict):
    research_next: Literal["medical_researcher", "finance_researcher","RESEARCH_DONE"]

class ReporterRouter(TypedDict):
    report_next: Literal["summarizer", "doc_generator","REPORT_DONE"]

members = ["researcher", "reporter"]

research_team = ["medical_researcher", "pharma_researcher"]
report_team =["summarizer", "doc_generator"]

system_prompt = f""""
You are a supervisor, tasked with managing a allocating a work between the following teams: {members}. 
    1. Given the following user request, First the research work has to be performed and then the reporting has to be done. 
    2. Based on 1, please allocate to teams {members} accordingly
    3. researcher team has following 2 teams {research_team}. If user request is medical related, please respond with 'medical_researcher'. 
    If user request is pharma related, please respond with 'finance_researcher'.
    if you get the response from one of the team, respond with RESEARCH_DONE
    4. Once the research is done, generate a report using reporting team 
    5. Reporting team has 2 teams {report_team} and it needs to be executed sequentially. First the summary has to be created and document needs to be generated. please respond with next accordingly.
    6. once the summary and doc is generated, respond with REPORT_DONE
    When all tasks are finished, respond with FINISH.
"""

def supervisor(state : AgentState) -> Command[Literal["researcher", "reporter", "__end__"]] :
    if(state["messages"][-1].name == 'researcher'):
        return Command(goto="reporter", update={
                    "messages": [
                        HumanMessage(content=state['messages'][-1].content, name="researcher")
                     ]  ,"next":"reporter"})
    messages = [{"role": "system", "content": system_prompt},] + state["messages"]
    llm_with_structure_output_router =llm.with_structured_output(Router)
    response = llm_with_structure_output_router.invoke(messages)
    goto = response["next"]
    print("**********BELOW IS MY GOTO from supervisor ***************")
    print(goto)    
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next":goto})

def researcher(state: AgentState) -> Command[Literal["medical_researcher","finance_researcher", "__end__"]] :
    messages = state["messages"]
    if messages[-1].name =='medical_researcher' or messages[-1].name == 'finance_researcher':
        return Command(
            update={
                "messages": [
                    HumanMessage(content=state['messages'][-1].content, name="researcher")
                ]
            },
            goto = END
        )
    llm_with_structure_output_researcher = llm.with_structured_output(ResearcherRouter)
    response = llm_with_structure_output_researcher.invoke(messages)
    goto = response["research_next"]
    print("**********BELOW IS MY GOTO from Researcher ***************")
    print(goto)    
    if goto == "RESEARCH_DONE":
        return Command(
            update={
                "messages": [
                    HumanMessage(content=state['messages'][-1].content, name="researcher")
                ]
            },
            goto = END
    )
    else:
        return Command(goto=goto, update={"next":goto})



def medical_researcher(state: AgentState) -> Command[Literal["researcher"]] :
    print('I am in medical researcher node ..Trying to get the Result')
    medical_research_agent = create_react_agent(llm, tools=[search_tool], prompt="You are a medical researcher")
    medical_result =  medical_research_agent.invoke(state)
    print(f'Medical result {medical_result}')
    return Command(
        update={
            "messages": [
                HumanMessage(content=medical_result['messages'][-1].content, name="medical_researcher")
            ]
        },
        goto="researcher",
    )

def finance_researcher(state: AgentState) -> Command[Literal["researcher"]] :
    print('I am in Finance researcher node ..Trying to get the Result')
    finance_research_agent = create_react_agent(llm, tools=[search_tool], prompt="You are a finance researcher")
    finance_result=finance_research_agent.invoke(state)
    print(f'Finance result {finance_result}')
    return Command(
        update={
            "messages": [
                HumanMessage(content=finance_result["messages"][-1].content, name="finance_researcher")
            ]
        },
        goto="researcher",
    )

def reporter(state: AgentState) -> Command[Literal["summarizer", "doc_generator","__end__"]] :
    
    messages = state["messages"]
    
    if messages[-1].name == 'summarizer':
        print("**********BELOW IS MY GOTO from Reporter ***************")
        print("doc_generator")
        return Command(
            update={
                "messages": [
                    HumanMessage(content=state['messages'][-1].content, name="reporter")
                ]
            },
            goto = "doc_generator"
        )
    
    if messages[-1].name == 'doc_generator':
        print("**********BELOW IS MY GOTO from Reporter ***************")
        print("END")
        return Command(
            update={
                "messages": [
                    HumanMessage(content=state['messages'][-1].content, name="reporter")
                ]
            },
            goto = END
        )
    llm_with_structure_output_reporter = llm.with_structured_output(ReporterRouter)
    reporter_response = llm_with_structure_output_reporter.invoke(messages)

    goto = reporter_response["report_next"]
    print("**********BELOW IS MY GOTO from Reporter ***************")
    print(goto)
    if goto == "REPORT_DONE":
        return Command(
            update={
                "messages": [
                    HumanMessage(content=state['messages'][-1].content, name="reporter")
                ]
            },
            goto = END
    )
    return Command(goto=goto, update={"next":goto})

def summarizer(state: AgentState) -> Command[Literal["reporter"]] :
    summary_agent=create_react_agent(llm,tools=[summary_tool], prompt="Summarize the following content with bullet points on what needs to be done along with the user request")
    
    summary_response = summary_agent.invoke({"messages": [{"role":"user","content":state['messages'][-1].content}]})
    
    summary = summary_response['messages'][-1].content
    print("**********BELOW IS MY GOTO from summarizer ***************")
    print("reporter")
    return Command(
        update={
            "messages": [
                HumanMessage(content=summary, name="summarizer")
            ]
        },
        goto="reporter",
    )

def doc_generator(state: AgentState) -> Command[Literal["reporter"]]:
    
    doc_agent = create_react_agent(llm,tools=[pdf_or_docx_generator_tool], prompt="Generate a report in pdf or docx format")
    doc_response = doc_agent.invoke({"messages": [{"role":"user","content" : state['messages'][-1].content}]})

    print("**********BELOW IS MY GOTO from doc_generator ***************")
    print("reporter")
    return Command(
        update={
            "messages": [
                HumanMessage(content=doc_response['messages'][-1].content , name="doc_generator")
            ]
        },
        goto="reporter",
    )

research_graph = StateGraph(AgentState)
report_graph = StateGraph(AgentState)
graph=StateGraph(AgentState)
graph.add_node("supervisor", supervisor)

research_graph.add_node("researcher",researcher)
research_graph.add_node("medical_researcher",medical_researcher)
research_graph.add_node("finance_researcher",finance_researcher)
research_graph.add_edge(START,"researcher")
research_app = research_graph.compile()


report_graph.add_node("reporter",reporter)
report_graph.add_node("summarizer",summarizer)
report_graph.add_node("doc_generator",doc_generator)
report_graph.add_edge(START,"reporter")
report_app = report_graph.compile()




graph.add_node("researcher", research_app)
graph.add_node("reporter", report_app)
graph.set_entry_point("supervisor")
graph.add_edge("researcher", "supervisor")
graph.add_edge("reporter", "supervisor")

app=graph.compile()

app_png = app.get_graph().draw_mermaid_png() 
research_png = research_app.get_graph().draw_mermaid_png() 
report_png = report_app.get_graph().draw_mermaid_png() 

def save_graph(file_name: str, png_graph: bytes) : 
    with open(file_name, "wb") as f:
        f.write(png_graph)
    
save_graph("app.png",app_png)
save_graph("research.png",research_png)
save_graph("report.png",report_png)

#response = app.invoke({'messages':['What Precaution needs to be done for Flu in Adults ? ']})
#response = app.invoke({'messages':['How to invest in stock markets ']})
response = app.invoke({'messages':['how to open a demat account ? ']}, {"recursion_limit": 10})
#response = app.invoke({'messages':['What happens if vitamin D is less in adults ']})







