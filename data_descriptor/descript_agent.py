from datetime import datetime
import os,sys
from pathlib import Path
import json
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
import asyncio

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.talent_training_market import AgentManager,SubAgent
from utils.file_check_point_saver import FileByteSaver
from data_descriptor.tool_model import get_default_model
from agent_tools.local_tools.git import create_check_git_ignore_tool

# Custom path
custom_path = Path(project_root) / "agent_checkpoints"
custom_path.mkdir(exist_ok=True)

checkpointer = FileByteSaver(str(custom_path))

class descriptor_agent(SubAgent):
    
    DEFAULT_DESCRIPTION_FILE = "description.txt"
    PROJECT_ROOT = project_root

    def __init__(self, data_path: str):

        self.child_manager = None

        self.path = data_path
        # 如果文件夹不存在，则raise异常

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The directory {self.path} does not exist.")

        if not os.path.isdir(self.path):
            raise NotADirectoryError(f"The path {self.path} is not a directory.")
        
        # 检查是否已存在描述文件
        if not os.path.exists(f"{self.path}/{self.DEFAULT_DESCRIPTION_FILE}"):
            pass
        else:
            # 描述文件最近更新时间，是否早于某个文件
            des_file= f"{self.path}/{self.DEFAULT_DESCRIPTION_FILE}"
            des_mtime = os.path.getmtime(des_file)
            need_refresh = False
            for file_name in os.listdir(self.path):
                file_path = os.path.join(self.path, file_name)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime > des_mtime:
                        need_refresh = True
                        break
            if need_refresh:
                pass
                # raise ValueError(f"The description file {des_file} is outdated. Please refresh it.")
            else:
                print(f"The description file {des_file} is up to date.")
        self.description_file = f"{self.path}/{self.DEFAULT_DESCRIPTION_FILE}"

        model = get_default_model()
        base_tools=[signature_python_module,get_python_includes,get_python_object_code,create_check_git_ignore_tool()]
        manager = AgentManager(llm_client=model)
        super().__init__(manager=manager, task_key_target="", task_description=f"请描述{self.path}文件夹下的数据情况", enable_hiring=True, depth=0, base_tools=base_tools)

        thread_id = self.path.replace(self.PROJECT_ROOT, "").replace(os.sep, "_").strip("_") +datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {"configurable": {"thread_id": f"{thread_id}"},"recursion_limit": 1000}
        self.initial_tools = [data_sample, save_description, ls, create_check_git_ignore_tool()]
        self.agent_id = "root"   
        manager.register_agent(self)

    def _generate_system_prompt(self) -> str:
        prompt = f"""
            你的核心任务是：为指定目录生成一份用于后续**代码生成与数据分析**的**机器可读**描述文档（`description.txt`），需使用Markdown格式。

            ### 一、 任务目标
            最终的 `description.txt` 文件应是一份**分层信息地图**的一部分，需满足：
            1.  **继承与更新**：基于现有 `description.txt` 内容进行更新与补充。
            2.  **即拿即用**：描述应确保其他智能体能够：根据业务需求**找到并提取**数据；**找到、理解并复用**现有代码，避免重复开发。
            3.  **层级清晰**：基于所处的目录层次，保持合理的信息颗粒度。上层目录描述更概括，涵盖下层目录；下层目录描述更具体、清晰，使阅读者能像查阅地图一样自由缩放信息粒度。

            ### 二、 描述内容要素
            你的描述需涵盖以下实体信息，仅描述受git管理的内容：
            *   **数据文件**：结构、核心字段含义、数据量、更新频率（如“每日更新”）。
            *   **代码文件（如.py, .sh等）**：分析其功能、输入/输出接口及在业务流中的角色。
            *   **配置文件等关键文件**：说明其格式、主要配置或内容的作用。
            *   **路径信息**：以 `{self.PROJECT_ROOT}` 为根目录，使用**相对路径**。注意文件名和路径本身可能蕴含的版本、类别等信息。

            ### 三、 工具使用与工作指引
            你拥有以下工具来完成工作，请依据情形合理使用：

            **核心工具**
            *   `data_sample`：
            *   **核心用途**：用于对**子目录**或当前目录下的**特定文件**进行采样，以获取其描述摘要。当整合子目录信息时，应调用此工具获取其摘要并整合到当前层描述中。
            *   **效率优先原则**：如果通过此工具发现子目录**已有**描述摘要，则应基于该摘要进行整合，**避免**再对子目录内的文件进行重复采样。
            *   **特殊情况**：如果发现采样的子目录**没有**描述文件（`description.txt`），则表明此类目录结构可能简单，此时**应穿透该子目录**，对其内部文件进行直接采样和分析。
            *   **采样建议原则**：当遇到大量按规律（如日期、ID）命名的子目录或文件时，应选取**1到2个典型代表**进行深入分析，并在描述中清晰说明其命名规律、内部结构以及相互间的关联。
            *   `save_description`：**完成描述后，务必调用此工具进行保存。**

            **辅助工具**
            *   `ls`：可用于列出目录内容，辅助你进行扫描和决策，调用前请先确认路径是否被忽略。
            *   `is_git_ignored`：
            *   **可用于辅助判断文件或目录是否应被忽略。
            *   **优先调用此工具判断是否应该忽略文件或目录（含子目录），避免不必要的采样和分析工作。

            **基础工作规则**
            1.  本项目根目录为：`{self.PROJECT_ROOT}`。
            2.  首先判断任务是否应该执行：
                2.1  自动忽略名称包含 `backup`, `old`, `cache`, `__pycache__` 的目录。
                2.2  调用git工具判断文件或目录是否被忽略，如GIT忽略该文件或目录，则直接跳过。
            2.  特别的，对于 `.py` 代码文件，{f"**针对每个文件雇佣一个子智能体进行分析**。若雇佣，每个子智能体仅负责分析一个单独的文件，并务必等待所有雇员完整返回后再统筹整合结果。此分析任务是**不可再委派**的原子工作单元，你在雇佣时必须明确执行此项要求。" if self.child_manager else "你应通过阅读其内容来分析功能。"}
            3.  不得仅对规律文件做笼统说明（如“存在一系列日期文件”）而跳过对采样文件内部的具体描述。
            4.  对于按日期增长的文件或数据，避免定义绝对上限，应使用“每日更新”等动态描述。
            5.  如子目录中已存在description.txt文件，则基于该文件内容进行整合描述，不再对该子目录进行扫描、采样等操作。

            {self.child_manager.default_prompt() if self.child_manager else ""}
        """
        return prompt
    
@tool
def data_sample(path:str):
    """
        根据路径参数，进行采样。
        如果为文件夹，则返回文件夹的description.txt内容；
        如果为csv，则返回前3条+后三条采样；如果为json，则返回json数组前3条+后三条，如是dict，则全量返回；
        如果为description.txt，则返回描述内容。
    """
    print(f"data_sample called with path: {path}")
    if not os.path.exists(path):
        return {"error": "the path is not exsist, to check it"}
    
    if path.endswith("description.txt"):
        with open(path, 'r') as file:
            description = file.read()
        return {
            "message": "This is a description file.",
            "description": description
        }
    
    if os.path.isdir(path):
        description_file = Path(path) / "description.txt"
        if description_file.exists():
            with open(description_file, 'r') as file:
                description = file.read()
                return {
                    "message": "This is a directory.",
                    "description": description
                }
        else:
            return {
                "message": "This is a directory.",
                "description": "And there is no description.txt file in this directory."
            }
        
    else:
        if path.endswith("csv"):
            # 读取前三行并返回：
            with open(path, 'r') as file:
                # 小于3行时，直接读取所有行
                lines = file.readlines()
                if len(lines) > 10:
                    lines = lines[:3] + ["...\n"] + lines[-3:]

                else:
                    lines = lines[:len(lines)-1]  # 最后一行通常是空行
            return {
                "message": f"这是CSV文件，共计{sum(1 for _ in open(path))}行。",
                "sample": "".join(lines)
            }
        
        if path.endswith("jsonl"):
            with open(path, 'r') as file:
                data = file.readlines()
                return {
                    "message": "这是JSONL文件",
                    "total": len(data),
                    "sample": data[:3] + ["..."] + data[-3:]  # 返回前3条和后3条
                }
        
        if path.endswith("json"):
            with open(path, 'r') as file:
                data = json.load(file)
                # 如果data是数组，则返回前两条；
                if isinstance(data, list):
                    if len(data) > 10:
                        # 返回前三条+后三条摘要,逻辑与csv取法相同
                        sample = data[:3] + ["..."] + data[-3:]
                    else:
                        sample = data
                    return {
                        "message": "这是JSON数组",
                        "total": len(data),
                        "sample": sample
                    }
                else:
                    return {
                        "message": "这是JSON对象",
                        "total": len(data),
                        "sample": data
                    }
                
        if path.endswith("txt"):
            with open(path, 'r') as file:
                content = file.read(500)  # 读取前500个字符
            return {
                "message": "这是TXT文件",
                "sample": content
            }
        
        if path.endswith("py"):
            with open(path, 'r') as file:
                content = file.read()  # 代码文件返回全量。
            return {
                "message": "这是PY文件,如果文件过大，返回的被截断，那么建议雇佣一个子智能体进行分析。",
                "sample": content[:500] + "\n...\n" + content[-500:] if len(content) > 1000 else content,
                "total_len": len(content)
            }
        
        if path.endswith("sh"):
            with open(path, 'r') as file:
                content = file.read() 
            return {
                "message": "这是SH文件",
                "sample": content
            }

@tool
def save_description(path:str, description:str):
    """
        保存描述信息到指定路径的description.txt文件中。
    """
    print(f"save_description called with path: {path}")
    description_file = Path(path) / "description.txt"
    try:
        with open(description_file, 'w') as file:
            file.write(description)
        return {"message": f"Description saved to {description_file}"}
    except Exception as e:
        return {"error": f"An error occurred while saving the description: {e}"}

@tool
def ls(path:str):
    """
        列出指定路径下的文件和文件夹。
    """
    print(f"ls called with path: {path}")
    if not os.path.exists(path):
        return {"error": "the path is invalid, to check it"}
    
    if not os.path.isdir(path):
        with open(path, 'r') as file:
            content = file.read()  # 读取前500个字符
        return {
            "info": f"The path {path} is a file.",
            "len": len(content),
            "sample": content[:500] + "\n...\n" + content[-500:] if len(content) > 1000 else content
        }

    items = os.listdir(path)
    # 分别返回文件和文件夹列表
    files = []
    directories = []
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path):
            directories.append(item)
    res_files = []
    res_directories = []
    if len(files) > 50:
        res_files = files[:50] + ["..."]
        if "description.txt" not in res_files:
            res_files.insert(0, "description.txt")
    if len(directories) > 50:
        res_directories = directories[:50] + ["..."]

    return {
        "files": res_files if res_files else files,
        "file_count": len(files),
        "directories": res_directories if res_directories else directories,
        "directory_count": len(directories)
    }

@tool
def signature_python_module(python_file: str):
    
    """
        根据python文件路径，生成Python模块的签名信息，包括函数和类的定义、代码长度、及采样信息。
        注意：如果代码长度小于10000字符，则返回全量代码，否则返回前5000和后5000字符的采样。
    """
    if not os.path.exists(python_file):
        return {"error": "the python_file is not exsist, to check it"}
    
    project_root = "/prog/pweb/AI-Trader"
    sys.path.insert(0, project_root)
    if not python_file.startswith(project_root):
        return {"error": f"the python_file should be under {project_root}"}
    relative_path = python_file.replace(project_root, "").lstrip(os.sep).replace(os.sep, ".")
    module_name = relative_path[:-3] if relative_path.endswith(".py") else relative_path
    try:
        module = __import__(module_name, fromlist=[''])
    except Exception as e:
        return {"error": f"Failed to import module {module_name}: {str(e)}"}
    import inspect
    functions = []
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            if obj.__module__ != module_name:
                continue
            desc = {
                "function_name": name, 
                "docstring": inspect.getdoc(obj),
                "signature": str(inspect.signature(obj))
            }
            functions.append(desc)
        elif inspect.isclass(obj):
            # 判断class是否定义在该模块中
            if obj.__module__ == module_name:
                desc = {
                    "class_name": name,
                    "docstring": inspect.getdoc(obj),
                    "signature": str(inspect.signature(obj))
                }
                classes.append(desc)
    
    print(f"Module {module_name} has {len(functions)} functions and {len(classes)} classes.")

    with open(python_file, 'r') as file:
        code_content = file.read()
        if len(code_content) > 10000:
            code_content = code_content[:5000] + "\n...\n" + code_content[-5000:]

    return {
        "module": module_name,
        "functions": functions,
        "classes": classes,
        "code_length": len(code_content),
        "code_sample": code_content
    }

@tool
def get_python_includes(python_file: str) -> list[str]:
    """
        获取python文件中的import语句，返回包含的模块列表。
    """
    includes = []
    try:
        with open(python_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("import "):
                    parts = line.split()
                    if len(parts) >= 2:
                        includes.append(parts[1])
                elif line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 4 and parts[2] == "import":
                        includes.append(parts[1])
    except Exception as e:
        print(f"Error reading python file {python_file}: {e}")
    return includes

@tool
def get_python_object_code(python_file: str, object_name: str) -> str:
    """
        获取python文件中指定对象（函数或类）的代码定义。
    """
    if not os.path.exists(python_file):
        return {"error": "the python_file is not exsist, to check it"}
    
    project_root = "/prog/pweb/AI-Trader"
    sys.path.insert(0, project_root)
    if not python_file.startswith(project_root):
        return {"error": f"the python_file should be under {project_root}"}
    relative_path = python_file.replace(project_root, "").lstrip(os.sep).replace(os.sep, ".")
    module_name = relative_path[:-3] if relative_path.endswith(".py") else relative_path
    try:
        module = __import__(module_name, fromlist=[''])
    except Exception as e:
        return {"error": f"Failed to import module {module_name}: {str(e)}, 请检查文件路径"}
    import inspect
    obj = getattr(module, object_name, None)
    if obj is None:
        return {"error": f"Object {object_name} not found in module {module_name}，函数或者类名拼写错误？"}
    try:
        source = inspect.getsource(obj)
        return {
            "object_name": object_name,
            "source_code": source
        }
    except Exception as e:
        return {"error": f"Failed to get source code for {object_name}: {str(e)}"}

async def run():
    data_path = "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_realtime_cache/2025120317"
    paths = [
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow/dfcf",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow/dfcf_blocks",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow/ths",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow/ths_blocks",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow/tushare",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_money_flow",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/indexes",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/blocks/ths",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/blocks/dfcf",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/blocks/sw_industry",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/blocks",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/market_summary",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_news_cache",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_realtime_cache",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_daily_open",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/stock_snap_ma_daily",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data/indexes_daily",
        # "/prog/pweb/AI-Trader/data/A_stock",
        # "/prog/pweb/AI-Trader/data/A_stock/A_stock_data",
        # "/prog/pweb/AI-Trader/scripts/crawler",
        # "/prog/pweb/AI-Trader/tools",
        # "/prog/pweb/AI-Trader/agent",
        # "/prog/pweb/AI-Trader/data",
        # "/prog/pweb/AI-Trader",
        "/prog/pweb/AI-Trader/data/agent_finnal_decision",
        "/prog/pweb/AI-Trader/data"
    ]
     
    for p in paths:
        agent = descriptor_agent(data_path=p)
        print(f"Initial chat messages for path {p}:")
        await agent.execute()
        print("Data Description:" + p)

async def walk_run(base_path: str):
    for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
        # 取dirpath的最后一级目录名作为data_path
        data_path = dirpath.split("/")[-1]
        # 如果 data_path是日期格式，则跳过
        try :
            datetime.strptime(data_path, "%Y-%m-%d")
            print(f"Skipping date-formatted directory: {dirpath}")
            continue
        except ValueError:
            print(f"Processing directory: {dirpath}")
            agent = descriptor_agent(data_path=dirpath)
            await agent.execute()
            print("Data Description:" + dirpath)

if __name__ == "__main__":
    asyncio.run(run())
    # chooser = "/prog/pweb/AI-Trader/data/agent_chooser_astock"
    # reviwer = "/prog/pweb/AI-Trader/data/agent_review_astock"
    # root_path = "/prog/pweb/AI-Trader/scripts"
    # vendor = "/prog/pweb/AI-Trader/vendor"
    # utils = "/prog/pweb/AI-Trader/utils"
    # scripts = "/prog/pweb/AI-Trader/scripts"
    # prompts = "/prog/pweb/AI-Trader/prompts"
    # data_sources = "/prog/pweb/AI-Trader/data_sources"
    # data_descriptor = "/prog/pweb/AI-Trader/data_descriptor"
    # configs = "/prog/pweb/AI-Trader/configs"
    # agentic_codes = "/prog/pweb/AI-Trader/agentic_codes"
    # agent_tools = "/prog/pweb/AI-Trader/agent_tools"
    # agent_runnable_path = "/prog/pweb/AI-Trader/agent_runnables"
    # agent = "/prog/pweb/AI-Trader/agent"
    # asyncio.run(walk_run(agent))

        