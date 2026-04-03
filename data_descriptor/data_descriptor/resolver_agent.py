from datetime import datetime
import os,sys
from pathlib import Path
import json
import asyncio
import os
from typing import final
from dotenv import load_dotenv
import datasets
load_dotenv()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.talent_training_market import AgentManager,SubAgent
from utils.file_check_point_saver import FileByteSaver
from data_descriptor.tool_model import get_model_by_name
import vendor.vendor_patch  # noqa: E402
from langchain.tools import tool
from agent_tools.local_tools.caculator import get_calculator_tool, get_sandbox_tool
from tools.general_tools import extract_tool_messages
# Custom path
custom_path = Path(project_root) / "agent_checkpoints"
custom_path.mkdir(exist_ok=True)

model_type = "deepseek"  # or "qwen""deepseek""baidu""gpt""gemini"

checkpointer = FileByteSaver(str(custom_path))

class resolver_agent(SubAgent):
    
    DEFAULT_DESCRIPTION_FILE = "description.txt"
    PROJECT_ROOT = project_root

    def __init__(self, agent_num: str, question: str = "", dataset: str= ""):

        self.child_manager = None
        self.agent_num = agent_num
        self.dataset = dataset
        model = get_model_by_name(model_type)
        # base_tools=[add, subtract, multiply, divide, power, distance, floor, ceil, modulus]
        base_tools = get_calculator_tool()+ get_sandbox_tool() + [floor, ceil]
        manager = AgentManager(llm_client=model)
        manager.open_shared_blackboard()
        manager.open_email_service()
        manager.open_signal_channel()
        super().__init__(manager=manager, task_key_target="", task_description=f"{question}", enable_hiring=True, depth=0, base_tools=base_tools)

        thread_id = self.agent_num +"_"+ datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {"configurable": {"thread_id": f"{thread_id}"},"recursion_limit": 1000}
        self.initial_tools = [save_result] + base_tools
        self.agent_id = "root"   
        # TODO register a root advisor
        manager.register_agent(self)

    def final_handle(self, final_content: str):
        """"""
        toolmessages = extract_tool_messages(self.last_response)
        for message in toolmessages:
            if message.name == "save_result":
                return True
            else:
                continue
        save_result.func("auto", final_content, {"kwargs":{"agent_num": self.agent_num, "dataset": self.dataset}})
        return True

    def _generate_system_prompt(self) -> str:
        
        prompt = f"""

            你的核心任务是：解决、证明用户的数学问题，且要求明确证明思路清晰，正确；

            ### 一、 任务目标
            1、根据用户问题，输出答案；
            2、**至少**从三个独立的理论思路进行证明，确保结论可检查验证；
            3、使用工具将答案和证明过程保存下来;

            ### 二、 描述内容要素
            1. 你需要根据用户的问题，给出清晰的数学证明过程，确保每一步推导都有理有据；
            2. 逐字推敲题目中每个表述的精确含义，特别是涉及参数范围、量词或边界约束的条件。常被忽视的细节可能导向新的可能性。
               2.1 **注意**，原始问题的精准澄清，公示在公共黑板上，供所有协作智能体参考；
               2.2 **注意**，数学表述陷阱，必须明确所求对象的数学定义，避免自然语言的歧义性；
            3. 你或者你的下属的每个思路独立、能相互校验思路进行证明，确保结论可检查验证；
            4. 必须有子智能体独立对结果负责，验证最终结果的正确性、最优性、完整性。验证结果要保持独立性、正交性；
            5. 验证很重要，根据问题的复杂性，考虑雇佣多个检查智能体，分别确保结果：正确性、最优性、完整性；
               5.1 **注意**，中间性结论的验证同样重要；
            6. 中间结论与最终结论，以及验证与证明结果不一致时；
               条件溯源：每步推导后反思所用条件的来源（题目条件、定义、已知定理），避免引入隐性假设。
               错误复盘：对易错点进行对比说明，强化差异理解。

            ### 三、 工具使用与工作指引
            你拥有以下工具来完成工作，请依据情形合理使用：

            **核心工具**
                暂无

            **辅助工具**
                1.  save_result(result: str, derivation_process: str) : 保存结果到指定路径的文件中。
                2.  shared_blackboard(action: str, content: str = None, task_key_target: str = None): 提供一个共享的思考面板，供所有协作的智能体异步地记录灵感、暂存结论或提出疑问。
                

            **基础工作规则**
            1.  你的任务多为数学及科学证明；
                1.1 **注意**，题目澄清、转换、拆解时，务必注意数学用语表述的准确性和全面性；
            2.  务必确保思路清晰，步骤详尽，逻辑严谨；逻辑推演为主，计算为辅；
            3.  仔细分析问题，若问题复杂，考虑雇佣子智能体协助完成，确保子智能体明确任务目标和要求；
                **注意，子智能体也可以雇佣下属，形成多级协作结构**；
            4.  在你认为得出答案的时候，至少直接雇佣一个检查智能体，严格检查答案的正确性和完整性；
            5.  如果解答过程中，雇佣了小弟，务必等待所有小弟的回答，再严格比对论证汇总;
            6.  时间，不是你要考虑的因素，确保正确性和完整性才是首要目标；不要以时间为理由发出催促信号；
            7.  完成证明后，先使用save_result工具保存结果和完整推导过程；
            8.  约束你以及你的部下们或者部下的部下们：公共黑板如果要分享引理、证明等内容，慎重一些，必须确保内容的正确性和相关性，避免传播错误信息。
            {self.child_manager.default_prompt() if self.child_manager else ""}
            当你觉得任务完成时，输出
                {SubAgent.STOP_SIGNAL}
        """
        prompt = f"""

            你的核心任务是：解决、证明用户的数学问题，且要求明确证明思路清晰，正确；

            ### 一、 任务目标
            1、根据用户问题，输出答案；
            2、要求从明确的理论思路进行证明，确保结论可检查验证；
            3、使用工具将答案和证明过程保存下来;

            ### 二、 描述内容要素
            1. 你需要根据用户的问题，给出清晰的数学证明过程，确保每一步推导都有理有据；
            2. 逐字推敲题目中每个表述的精确含义，特别是涉及参数范围、量词或边界约束的条件。常被忽视的细节可能导向新的可能性。
               2.1 **注意**，原始问题的精准澄清，公示在公共黑板上，供所有协作智能体参考；
               2.2 **注意**，数学表述陷阱，必须明确所求对象的数学定义，避免自然语言的歧义性；
            3. 动手证明前首先要审慎选择证明方法和证明思路，确保每个步骤都有理有据；
            4. 必须有子智能体独立对结果负责，验证最终结果的正确性、最优性、完整性。验证结果要保持独立性、正交性；
            5. 验证很重要，根据问题的复杂性，考虑雇佣多个检查智能体，分别确保结果：正确性、最优性、完整性；
               5.1 关键的中间性结论，如有必要也需要专门的智能体验证，确保其正确性和合理性，如确认有误，及时通知对应的证明智能体进行修正；
            6. 中间结论与最终结论，以及验证与证明结果不一致时；
               条件溯源：每步推导后反思所用条件的来源（题目条件、定义、已知定理），避免引入隐性假设。
               错误复盘：对易错点进行对比说明，强化差异理解。

            ### 三、 工具使用与工作指引
            你拥有以下工具来完成工作，请依据情形合理使用：

            **核心工具**
                暂无

            **辅助工具**
                1.  save_result(result: str, derivation_process: str) : 保存结果到指定路径的文件中。
                2.  shared_blackboard(action: str, content: str = None, task_key_target: str = None): 提供一个共享的思考面板，供所有协作的智能体异步地记录灵感、暂存结论或提出疑问。
                

            **基础工作规则**
            1.  你的任务多为数学及科学证明；
                1.1 **注意**，题目澄清、转换、拆解时，务必注意数学用语表述的准确性和全面性；
            2.  务必确保思路清晰，步骤详尽，逻辑严谨；逻辑推演为主，计算为辅；
            3.  仔细分析问题，若问题复杂，考虑雇佣子智能体协助完成，确保子智能体明确任务目标和要求；
                **注意，子智能体也可以雇佣下属，形成多级协作结构**；
            4.  在你认为得出答案的时候，至少直接雇佣一个检查智能体，严格检查答案的正确性和完整性；
            5.  如果解答过程中，雇佣了小弟，务必等待所有小弟的回答，再严格比对论证汇总;
            6.  时间，不是你要考虑的因素，确保正确性和完整性才是首要目标；不要急于获取结论；
            7.  完成证明后，先使用save_result工具保存结果和完整推导过程；
            8.  约束你以及你的部下们或者部下的部下们：公共黑板如果要分享引理、证明等内容，慎重一些，必须确保内容的正确性和相关性，避免传播错误信息。
            {self.child_manager.default_prompt() if self.child_manager else ""}
            当你觉得任务完成时，输出
                {SubAgent.STOP_SIGNAL}
        """
        # prompt = f"""
        #     你的核心任务是：解决、证明用户的数学问题，且要求明确证明思路清晰，正确；
        #     反正就是得做对做对做对！！！！！！
        #     多论证，多从不同角度，多雇佣智能体，确保结果正确性、最优性、完整性！！！
            
        #     {self.child_manager.default_prompt() if self.child_manager else ""}
        #     当你觉得任务完成时，输出
        #         {SubAgent.STOP_SIGNAL}
        # """
        
        return prompt

# 加法
@tool
def add(a: float, b: float) -> float:
    """
        返回两个数字的和。
    """
    return a + b

# 减法
@tool
def subtract(a: float, b: float) -> float:
    """
        返回两个数字的差。
    """
    return a - b

# 乘法
@tool
def multiply(a: float, b: float) -> float:
    """
        返回两个数字的积。
    """
    return a * b

# 除法
@tool
def divide(a: float, b: float) -> float:
    """
        返回两个数字的商。
    """
    if b == 0:
        return "Error: Division by zero is undefined."
    return a / b

#乘方
@tool   
def power(base: float, exponent: float) -> float:
    """
        返回base的exponent次方。
    """
    return base ** exponent

#距离
@tool
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
        计算二维空间中两点之间的欧几里得距离。
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

@tool
def floor(value: float) -> int:
    """
        返回数字的下限整数值。
    """
    return int(value // 1)

@tool
def ceil(value: float) -> int:
    """
        返回数字的上限整数值。
    """
    return int(-(-value // 1))

@tool
def modulus(a: int, b: int) -> int:
    """
        返回两个整数的模。
    """
    return a % b

@tool(meta={
    "inject_requires": {
        "agent_num": True,
        "dataset": True
    }
})
def save_result(result: str, derivation_process: str, **kwargs):
    """
        保存结果到指定路径的文件中。
        result: 最终结果字符串
        derivation_process: 推导过程字符串
    """

    agent_num = kwargs["kwargs"]["agent_num"] or "unknown_agent"
    dataset = kwargs["kwargs"]["dataset"] or "dataset"
    print(f"save_result called with agent_num: {agent_num} in dataset : {dataset}")
    
    result_file = Path(project_root) / "logs" / dataset / agent_num / model_type /f"resolver_result_{agent_num}.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"保存结果到路径: {result_file}")
    print(f"最终结果:\n{result}\n")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"最终结果:\n{result}\n\n推导过程:\n{derivation_process}")
    print(f"结果已保存到 {result_file}")
    
global shared_blackboard_storage
shared_blackboard_storage = []

@tool
def shared_blackboard(action: str, content: str = None, task_key_target: str = None):
    """
    提供一个共享的思考面板，供所有协作的智能体异步地记录灵感、暂存结论或提出疑问。

    这块面板是一个中立的协作空间，旨在帮助跨越单个智能体的工作边界。当你在思考中
    产生了可能对其他伙伴有价值的中间产物——无论是一个关键的引理猜想、一个令你困惑的
    反例，还是一个有待验证的构造方向——都可以选择将其留在这里。同样，当你感到思路
    需要新的输入或验证时，也不妨来这里看看其他智能体留下了怎样的痕迹。

    参数:
        action: 希望执行的操作。目前支持 'write'（写入一段内容）或 'read'（读取全部黑板内容）。
        content: 当 action 为 'write' 时，需要写入的文本内容。鼓励清晰、结构化的表述。
        task_key_target: 你的关键任务目标；

    返回:
        执行读取操作时，将返回所有黑板条目列表；
        执行写入操作时，将返回确认信息。
    """
    global shared_blackboard_storage
    if action == "write":
        print(f"shared_blackboard write called with task_key_target: {task_key_target}, content: {content}")
        if content is None:
            return "Error: 'content' must be provided when action is 'write'."
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_key_target": task_key_target,
            "content": content
        }
        shared_blackboard_storage.append(entry)
        return f"Content written to blackboard at {entry['timestamp']}."
    elif action == "read":
        filtered_entries = ""
        
        for entry in shared_blackboard_storage:
            if task_key_target is None or entry["task_key_target"] == task_key_target:
                filtered_entries += f"[{entry['timestamp']}] ({entry['task_key_target']}): {entry['content']}\n"
        return filtered_entries
    else:
        return "Error: 'action' must be either 'write' or 'read'."


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


"""
p:15 ： 则需要及其复杂的两次换元策略；
    Let
        $$x = 2a^2 + 3 + frac{c^2}{4}, quad y = b^2 + frac{d^2}{2} + 1.$$
        Define
        [
        f(k) = frac{9}{sqrt{k}} + 5sqrt{k} - sqrt{90-56k}.
        ]
p:29:海量搜索问题，待深究
p:35,37、38 错误且不稳定。
p:33,39,41,46,49议


p:7 : 前半段需要化解推理，推理出猜测答案为<32的偶数集合，或者其子集；
      而关于可达性的证明则需要逐一暴力枚举，来证明可达性。
p:17 ：原文中的自然语义描述的歧义性，需要进行数学化的澄清和转换；
    Attention: That is, find the largest integer that is smaller than the sum for all possible permutations.
    In other words, you need to find the smallest  sum of the functions over all permutations of the set \(\{1, 2, \ldots, 2025\}\).


p:17、28、33，34语言陷阱。

p:24,22:@pass2
p:25,31:@pass3
p:28:杠杆倍率30.5。Y
"""

async def run_data_sample(start_index: int =0, end_index: int = None):
    dataset = load_data()
    if end_index is None:
        end_index = len(dataset["test"])
    for i in range(start_index, end_index):
        agent = None
        try:
            data_item = dataset["test"][i]
            question_id = data_item["question_id"]
            prompt = data_item["prompt"]

            # prompt = prompt + "\n 注意：在操作中，行与列的交叉点仅变色一次，即一次操作，会有37个点变色。\n "
            initpar = {
                "agent_num":f"puzzle_{question_id}",
                "question": prompt,
                "dataset": "resolve"
            }
            agent = resolver_agent(**initpar)
            print(f"Initial chat messages for path {initpar}:")
            response = await agent.execute()
        except KeyboardInterrupt:
            print("Process interrupted by user. Exiting gracefully.")
            break
        except Exception as e:
            print(f"Error processing puzzle_{question_id}: {e}")
            continue
        finally:
            root_manager = agent.manager
            print("Final Agent Tree:")
            root_manager.satisfy_sub_agents()
            root_manager.print_sub_tree()
            print("Final Agent Tree JSON:")
            cost_json = root_manager.export_tree_json()
            file_path = Path(f"{project_root}/logs/resolve/puzzle_{question_id}/{model_type}/agent_tree_{question_id}.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(cost_json, ensure_ascii=False, indent=2))
            print(f"Agent tree saved to {file_path}")

            from group_tools.cost_pre_exe import build_exec_costs_everylevel
            exec_costs_everylevel = build_exec_costs_everylevel(str(file_path))
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(exec_costs_everylevel, ensure_ascii=False, indent=2))
            print(f"Execution costs agent tree saved to {file_path}")

            shares = root_manager.shared_blackboard

            shared_file_path = Path(f"{project_root}/logs/resolve/puzzle_{question_id}/{model_type}/shared_blackboard_{question_id}.txt")
            shared_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shared_file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(shares, ensure_ascii=False, indent=2))
            print(f"Shared blackboard saved to {shared_file_path}")

            mails_file_path = Path(f"{project_root}/logs/resolve/puzzle_{question_id}/{model_type}/mails_{question_id}.json")
            mails_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mails_file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(root_manager.mail_list, ensure_ascii=False, indent=2))
            print(f"Mails saved to {mails_file_path}")
            user_comfirt = input("答案已出现，如不正确，请输入正确答案，并回车继续：")
            if user_comfirt.strip():
                response = await agent.continuer(add_infos= f"用户输入了答案修正：{user_comfirt.strip()}，请基于此反思你的答案，并检讨出错的地方。")
            else:
                print("No input received, moving to the next puzzle.")
                continue


async def run_data_aime(start_index:int=0, end_index:int = None):
    dataset_name = "aime_2026"
    dataset = load_data_by_checkpoint("/prog/projects/datasets/aime_2026")

    if end_index is None:
        end_index = len(dataset["train"])
    for i in range(start_index, end_index):
        data_item = dataset["train"][i]
        question_id = data_item["problem_idx"]
        prompt = data_item["problem"]

        prompt = prompt + "\n "
        initpar = {
            "agent_num":f"puzzle_{question_id}",
            "question": prompt,
            "dataset": dataset_name
        }
        agent = resolver_agent(**initpar)
        print(f"Initial chat messages for path {initpar}:")
        response = await agent.execute()
        root_manager = agent.manager
        print("Final Agent Tree:")
        root_manager.satisfy_sub_agents()
        root_manager.print_sub_tree()
        print("Final Agent Tree JSON:")
        cost_json = root_manager.export_tree_json()
        file_path = Path(f"{project_root}/logs/{dataset_name}/puzzle_{question_id}/{model_type}/agent_tree_{question_id}.json")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(cost_json, ensure_ascii=False, indent=2))
        print(f"Agent tree saved to {file_path}")

        from group_tools.cost_pre_exe import build_exec_costs_everylevel
        exec_costs_everylevel = build_exec_costs_everylevel(str(file_path))
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(exec_costs_everylevel, ensure_ascii=False, indent=2))
        print(f"Execution costs agent tree saved to {file_path}")

        shares = root_manager.shared_blackboard

        shared_file_path = Path(f"{project_root}/logs/{dataset_name}/puzzle_{question_id}/{model_type}/shared_blackboard_{question_id}.txt")
        shared_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shared_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(shares, ensure_ascii=False, indent=2))
        print(f"Shared blackboard saved to {shared_file_path}")

        mails_file_path = Path(f"{project_root}/logs/{dataset_name}/puzzle_{question_id}/{model_type}/mails_{question_id}.json")
        mails_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mails_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(root_manager.mail_list, ensure_ascii=False, indent=2))
        print(f"Mails saved to {mails_file_path}")
        user_comfirt = input("答案已出现，如不正确，请输入正确答案，并回车继续：")
        if user_comfirt.strip():
            reponse = await agent.continuer(add_infos= f"用户输入了答案修正：{user_comfirt.strip()}，请基于此反思你的答案，并检讨出错的地方。")
        else:
            print("No input received, moving to the next puzzle.")
            continue

def load_data():    
    cache_dir = "/home/ubuntu/.cache/huggingface/datasets"
    dataset = datasets.load_dataset(
        "hf-imo-colab/AMO-Bench")
    return dataset

def load_data_by_checkpoint(checkpoint: str):

    dataset = datasets.load_dataset(
    "parquet",
    data_files=checkpoint+"/data/train-00000-of-00001.parquet"
    )
    return dataset

def analyze_agent_response(datasetname, end_index = None):
    if end_index is None:
        print("No end_index provided, analyzing all puzzles in the dataset.")
        return
    log_path = Path(project_root) / "logs" / datasetname
    anns = []
    for i in range(end_index):
        question_id = i+1
        file_path = log_path / f"puzzle_{question_id}" / model_type / f"agent_tree_{question_id}.json"
        question_dict = {
            "question_id": question_id
        }
        
        if not file_path.exists():
            print(f"Agent tree file {file_path} does not exist, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            agent_tree = json.load(f)
        
        # 这里可以添加对agent_tree的分析逻辑，例如统计使用了哪些工具，子智能体的层级结构等。
        print(f"Analyzing agent tree for puzzle_{question_id}...")
        # 示例：统计工具使用情况
       
        extra_dict = _extra_analyze_(agent_tree)
        question_dict.update(extra_dict)

        root_agent = agent_tree.get("agents", [])[0]  # 获取根智能体
        costs = root_agent.get("costs", {}).get("agent", {})    
        hire_cost = costs.get("hire_cost", 0)
        execution_cost = costs.get("execution_cost", 0)
        leverage = execution_cost / hire_cost if hire_cost > 0 else "N/A"
        print(f"Tool leverage for puzzle_{question_id}: {leverage}")
        question_dict["hire_cost"] = hire_cost
        question_dict["execution_cost"] = execution_cost
        question_dict["leverage"] = leverage
        file_mails = Path(project_root) / "logs" / datasetname / f"puzzle_{question_id}" / model_type / f"mails_{question_id}.json"
        if file_mails.exists():
            # 统计该json文件字符长度
            with open(file_mails, "r", encoding="utf-8") as f:
                mails = json.load(f)
                mails_length = len(json.dumps(mails, ensure_ascii=False))
                print(f"Mails JSON length for puzzle_{question_id}: {mails_length}")
        else:
            print(f"Mails file {file_mails} does not exist.")
        question_dict["mails_length"] = mails_length if file_mails.exists() else 0

        file_shared = Path(project_root) / "logs" / datasetname / f"puzzle_{question_id}" / model_type / f"shared_blackboard_{question_id}.txt"
        if file_shared.exists():
            with open(file_shared, "r", encoding="utf-8") as f:
                shared = json.load(f)
                shared_length = len(json.dumps(shared, ensure_ascii=False))
                print(f"Shared blackboard length for puzzle_{question_id}: {shared_length}")
        else:
            print(f"Shared blackboard file {file_shared} does not exist.")
        question_dict["shared_length"] = shared_length if file_shared.exists() else 0
        anns.append(question_dict)
    
    total_dict = {}
    average_hire_cost = sum(q["hire_cost"] for q in anns if q["hire_cost"] != "N/A") / len([q for q in anns if q["hire_cost"] != "N/A"])
    average_execution_cost = sum(q["execution_cost"] for q in anns if q["execution_cost"] != "N/A") / len([q for q in anns if q["execution_cost"] != "N/A"])
    average_max_depth = sum(q["max_depth"] for q in anns if q["max_depth"] != "N/A") / len([q for q in anns if q["max_depth"] != "N/A"])    
    average_angent_count = sum(q["total_agents"] for q in anns if q["total_agents"] != "N/A") / len([q for q in anns if q["total_agents"] != "N/A"])
    average_per_depth = []
    max_depth = max(q["max_depth"] for q in anns if q["max_depth"] != "N/A")
    
    def get_total_and_count_by_key_and_depth(anns, key, depth):
        total = 0
        count = 0
        for q in anns:
            if not q["per_depth"] or depth >= len(q["per_depth"]):
                continue
            value = q["per_depth"][depth].get(key, "N/A")
            if value != "N/A":
                total += value
                count += 1

        return total, count
    
    def build_average_leverage_by_depth(anns, average_per_depth=0):
        hire_cost = 0
        execution_cost = 0
        for q in anns:
            if not q["per_depth"] or average_per_depth >= len(q["per_depth"]):
                continue
            hire_cost_q = q["per_depth"][average_per_depth].get("hire_cost", "N/A")
            if hire_cost_q == "N/A" or hire_cost_q == 0:
                continue
            execution_cost_q = q["per_depth"][average_per_depth].get("execution_cost", "N/A")
            if hire_cost_q != "N/A" and execution_cost_q != "N/A":
                hire_cost += hire_cost_q
                execution_cost += execution_cost_q

        return execution_cost / hire_cost if hire_cost > 0 else "N/A"



    for depth in range(max_depth + 1): 
        total_hire_cost_depth, count_hire_cost_depth = get_total_and_count_by_key_and_depth(anns, "hire_cost", depth)
        total_execution_cost_depth, count_execution_cost_depth = get_total_and_count_by_key_and_depth(anns, "execution_cost", depth)
        total_leverage_depth, count_leverage_depth = get_total_and_count_by_key_and_depth(anns, "leverage", depth)
        total_agent_count_depth, count_agent_count_depth = get_total_and_count_by_key_and_depth(anns, "agent_count", depth)
        average_hire_cost_depth = total_hire_cost_depth / count_hire_cost_depth if count_hire_cost_depth > 0 else "N/A"
        average_execution_cost_depth = total_execution_cost_depth / count_execution_cost_depth if count_execution_cost_depth > 0 else "N/A"
        average_leverage_depth = build_average_leverage_by_depth(anns, depth)
        average_agent_count_depth = total_agent_count_depth / count_agent_count_depth if count_agent_count_depth > 0 else "N/A"
        
        
        average_per_depth.append({
            "depth": depth,
            "average_agent_count": average_agent_count_depth,
            "average_hire_cost": average_hire_cost_depth,
            "average_execution_cost": average_execution_cost_depth,
            "average_leverage": average_leverage_depth
        })
    total_dict["average_per_depth"] = average_per_depth
    total_dict["average_max_depth"] = average_max_depth
    total_dict["average_agent_count"] = average_angent_count

    total_dict["average_hire_cost"] = average_hire_cost
    total_dict["average_execution_cost"] = average_execution_cost
    total_dict["average_leverage"] = average_execution_cost / average_hire_cost if average_hire_cost > 0 else "N/A"
    print(f"Average hire cost: {average_hire_cost}")
    print(f"Average execution cost: {average_execution_cost}")
    print(f"Average leverage: {total_dict['average_leverage']}")
    total_dict["average_mails_length"] = sum(q["mails_length"] for q in anns) / len(anns)
    total_dict["average_shared_length"] = sum(q["shared_length"] for q in anns) / len(anns)
    print(f"Average mails length: {total_dict['average_mails_length']}")
    print(f"Average shared blackboard length: {total_dict['average_shared_length']}")
    anns.append(total_dict)
    # 将分析结果保存到json文件中
    analysis_file = log_path / f"analysis_{datasetname}_{end_index}.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(anns, f, ensure_ascii=False, indent=2)
    print(f"Analysis results saved to {analysis_file}")

def _extra_analyze_(json):
    # 这里可以添加对单个agent_tree的更深入分析逻辑，例如分析子智能体的层级结构、工具使用的时序等。
    # 1、最大深度；
    # 2、智能体总数；
    # 3、每层智能体数量，与最大深度有关；
    # 4、每层雇佣成本、执行成本、杠杆率等数据；
    all_agents = []
    def traverse_agents(agent_node, depth=0):
        all_agents.append(agent_node)
        child_manager = agent_node.get("child_manager")
        if child_manager and "agents" in child_manager:
            for child_agent in child_manager["agents"]:
                traverse_agents(child_agent, depth + 1)
    root_agent = json.get("agents", [])[0]  # 获取根智能体
    traverse_agents(root_agent)
    total_agents = len(all_agents)
    max_depth = max(agent.get("depth", 0) for agent in all_agents)
    agents_per_depth = [0] * (max_depth + 1)
    for agent in all_agents:
        depth = agent.get("depth", 0)
        agents_per_depth[depth] += 1

    # 计算每层的雇佣成本、执行成本、杠杆率等数据
    costs_per_depth = []
    for depth in range(max_depth + 1):
        hire_cost = sum(agent.get("costs", {}).get("agent", {}).get("hire_cost", 0) for agent in all_agents if agent.get("depth", 0) == depth)
        execution_cost = sum(agent.get("costs", {}).get("agent", {}).get("execution_cost", 0) for agent in all_agents if agent.get("depth", 0) == depth)
        leverage = execution_cost / hire_cost if hire_cost > 0 else "N/A"
        costs_per_depth.append({
            "depth": depth,
            "agent_count": agents_per_depth[depth],
            "hire_cost": hire_cost,
            "execution_cost": execution_cost,
            "leverage": leverage
        })
    return {
        "total_agents": total_agents,
        "max_depth": max_depth,
        "per_depth": costs_per_depth
    }


if __name__ == "__main__":
    # asyncio.run(run())
    # load_data()


    # asyncio.run(run_data_sample(0, 1))
    # asyncio.run(run_data_aime(14,15))

    # analyze_agent_response("aime_2026", 30)
    analyze_agent_response("resolve0221", 50)
    # asyncio.run(run_test())

        