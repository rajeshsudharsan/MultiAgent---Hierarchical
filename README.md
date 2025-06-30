Problem Statement:
1) Given a supervisor with 2 teams.
     a) Research team will do the search based on the question asked
     b) Reporting Team will do the summarization based on the answer from team one and generate the summarization content in pdf or word doc
2) Research team  - is split into 2
     a) Medical team - if its medical related questions, this team will respond
     b) finance team - if its financial related questions, this team will respond
3) Reporting Team - is split into 2
     a) first team - summarize the content
     b) second team - Generate the report

 Architecture:
   used:  langgraph - multi_agent (Multi-agent architectures -Hierarchical)
   Reference: https://langchain-ai.github.io/langgraph/concepts/multi_agent/

Png Files Attached are the workflow.

Required Resources to run the code:

1) Tavily API KEY
2) OPEN AI - Payment Required (OR) GROQ API KEY - Free Model
3) Python 3.12

How to run:
python multyagent_3.py




