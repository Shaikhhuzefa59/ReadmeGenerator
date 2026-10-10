import os
import sys
import shutil
import tempfile
from typing import Dict, TypedDict
from git import Repo
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END


# --------- 1. Graph State Schema ----------
class GraphState(TypedDict):
    repo_url : str
    repo_dir : str
    file_tree : str
    key_file_contents : Dict[str, str]
    code_explanation : str
    generate_linkedin : bool
    linkedin_post : str
    final_readme : str

# --------- initialize LLM -----------
endpoint = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen2.5-Coder-32B-Instruct',
    task = 'text-generation',
    max_new_tokens=2048,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

llm = ChatHuggingFace(llm=endpoint)

# ------------- Node Defination ------------

def fetch_repository(state: GraphState) -> Dict:
    """Clones the repository into an isolated temporary directory."""
    temp_dir = tempfile.mkdtemp()
    Repo.clone_from(state['repo_url'], temp_dir, depth=1)
    return {'repo_dir': temp_dir}

def parse_codebase(state: GraphState) -> Dict:
    """Extracts the file tree and reads key archetechtural files"""
    repo_path = state['repo_dir']
    tree_list = []
    contents = {}

    ignored_dirs = {'.git', 'node_modules', 'venv', '__pycache__', 'dist','build'}
    priority_file = {
        'package.json','requirements.txt', 'pyproject.toml', 'dockerfile', 'main.py','app.py','index.js','src/index.ts'
    }

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        rel_root = os.path.relpath(root, repo_path)

        for file in files:
            re_file_path = os.path.normpath(os.path.join(rel_root, file)) if rel_root != "." else file
            tree_list.append(re_file_path)

            if file.lower() in priority_file or re_file_path.lower() in priority_file:
                full_path = os.path.join(root, file)
                try:
                    with open(full_path,'r', encoding='utf-8',errors='ignore') as f:
                        contents[re_file_path] = f.read()[:3000]
                except Exception:
                    continue

    return {
        'file_tree': '\n'.join(tree_list[:60]),
        'key_file_contents': contents
        }

def explain_code(state:GraphState)->Dict:
    """analyzes components and dependencies to describe app functionality."""
    prompt = f"""You are a senior architect. Analyze this codebase data:
    repository file tree : {state['file_tree']}

    key configuration & entrypoints Files:
    {chr(10).join([f'---{path} ----\n{content}' for path, content in state['key_file_contents'].items()])}

    Task:
    1. Identify the core tech stack and frameworks.
    2. Explain the purpose and flow of the application.
    3. list the main components and what their responsibilities are.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {'code_explanation':response.content}

def generate_readme(state: GraphState) -> Dict:
    """Formats findings into a github README.md."""
    prompt = f"""You are an open source technical writer.
    Generate a comprehension, production ready README.md based on this technical analysis
    
    {state['code_explanation']}
    project file tree : {state['file_tree']}
    
    Include standard sections:
    - Project title & High level overview"
    - Features & Heighlights
    - Tech Stack
    - Project Direcotry Structure
    - Getting started (prerequisities, installation, and running)
    - contributing & License placeholders"""

    response = llm.invoke([
        SystemMessage(content='You only output clean, formatted markdown without wrapping in conversational filler.'),
        HumanMessage(content=prompt)
    ])

    if os.path.exists(state['repo_dir']):
        shutil.rmtree(state['repo_dir'], ignore_errors=True)

    return {'final_readme':response.content}

def generate_linkedin_post(state: GraphState) -> Dict:
    """Generate a LinkedIn post using code Explanation"""
    prompt = f"""You are an expert tech content creator and developer advocate on LinkedIn.
    Based on the architectural analysis and code explanation provided below, write an engaging, 
    high-impact LinkedIn post showcasing this project to recruiters, fellow developers, and tech leads
    also add tags releated to these fields.
    code_explanation : {state['code_explanation']}
    Requirements:
    Hook: Start with a strong, curiosity-inducing opening line about the problem this project solves or what was built (no generic greetings like "Hey everyone" or "Excited to share").
    The Problem & Solution: Briefly state the practical challenge and how this application solves it.
    Core Features & Tech Stack: Highlight 3–4 standout technical capabilities using clean bullet points and mention the main frameworks/libraries used.
    Key Takeaway/Engineering Insight: Share one clear insight or technical highlight from building/architecting this system.
    Call-to-Action (CTA): Invite developers to check out the project, provide feedback, or discuss the architectural approach.
    Tone & Formatting: Keep paragraphs short (1–2 sentences), scannable, professional yet energetic, and include 3–5 relevant tech hashtags at the bottom.
    """
    response = llm.invoke(prompt)

    return {'linkedin_post':response.content}

def route_after_explanation(state:GraphState)->str:
    """Checks user flag to determine next step"""
    if state.get('generate_linkedin', False):
        return 'generate_linkedin'
    return 'generate_readme'

# ------ 3. Build the Langgraph ----------

workflow = StateGraph(GraphState)

workflow.add_node('fetch_repository', fetch_repository)
workflow.add_node('parse_codebase', parse_codebase)
workflow.add_node('explain_code', explain_code)
workflow.add_node('generate_readme', generate_readme)
workflow.add_node('generate_linkedin', generate_linkedin_post)

workflow.add_edge(START, 'fetch_repository')
workflow.add_edge('fetch_repository', 'parse_codebase')
workflow.add_edge('parse_codebase', 'explain_code')



# Conditional Routing
workflow.add_conditional_edges(
    'explain_code',route_after_explanation,{
        'generate_linkedin':'generate_linkedin',
        'generate_readme':'generate_readme'
    }
)
workflow.add_edge('generate_readme', END)
workflow.add_edge('generate_linkedin', END)

app = workflow.compile()

if __name__=='main':
    print('='*55)
    print(' Langraph REPOSITORY CLI')   
    print('='*55)
    repo_url = input('\n Enter Github Repository URL: ').strip()
    if not repo_url:
        print('[!] URL cannot be empty.')
        sys.exit(1)

    print("\n select outputmode:")
    print(" [1] Generate README.md")
    print(" [2] Generate Linkedin Post")
    choice = input('Enter choice (1/2 default 1 ): ').strip()

    want_linkedin = choice == '2'

    initial_state = {
        'repo_url': repo_url,
        'generate_linkedin': want_linkedin,
    }

    try:
        final_state = app.invoke(initial_state)

        if want_linkedin:
                print("\n" + "=" * 50)
                print("         GENERATED LINKEDIN POST         ")
                print("=" * 50)
                print(final_state["linkedin_post"])
        else:
            output_file = "GENERATED_README.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_state["final_readme"])

            print("\n" + "=" * 50)
            print(f"[✓] Success! README saved to {output_file}")
            print("=" * 50)

    except Exception as e:
        print(f"\n[!] Workflow failed: {e}")