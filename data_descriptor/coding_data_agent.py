import os, sys
from pathlib import Path
import json
import re
import uuid
from langchain.agents import create_agent
import asyncio
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage, AIMessage
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from data_descriptor.tool_model import DeepSeekChatOpenAI
from tools.general_tools import extract_conversation
from langgraph.checkpoint.base import BaseCheckpointSaver, create_checkpoint
from pathlib import Path
from data_descriptor.tool_model import get_default_model
import traceback
from agent_tools.local_tools.caculator import get_sandbox_tool
# Custom path
custom_path = Path(project_root) / "agent_checkpoints"
custom_path.mkdir(exist_ok=True)

code_path = Path(project_root) / "agentic_codes" 
code_path.mkdir(exist_ok=True)

CODING_START_PROMPT = "<CODING_START>"
CODING_END_PROMPT = "<CODING_END>"

class data_spaces():
    
    @classmethod
    def get_default_spaces(cls, index: int = 0):
        default_spaces = [{
            "name": "A股市场数据",
            "path": project_root+"/data/A_stock/A_stock_data",
            "description": "A_stock数据集，包含股票的历史数据等，用于股票分析与建模。"
        },{
            "name": "交易数据",
            "path": project_root+"/data/agent_data_astock",
            "description": "agent_data_astock数据集，包含各个交易员的持仓和交易数据，用于交易策略分析与复盘。"
        },{
            "name": "整个工程",
            "path": project_root,
            "description": "包含整个工程的数据和代码，用于全面的项目分析与开发。"
        },{
            "name": "数据目录",
            "path": project_root+"/data",
            "description": "包含项目中的所有数据文件，用于数据分析与处理。"
        }]

        if index < 0 or index > len(default_spaces):
            raise ValueError("Invalid data space index.")
        
        if index == 0:
            return default_spaces
        
        return [default_spaces[index-1]]

    @classmethod
    def whole(cls):
        instance = data_spaces()
        default_spaces = cls.get_default_spaces(index=0)
        instance.add_data_space(default_spaces)
        return instance
    
    @classmethod
    def market(cls):
        instance = data_spaces()
        default_spaces = cls.get_default_spaces(index=1)
        instance.add_data_space(default_spaces)
        return instance

    @classmethod
    def traders(cls):
        instance = data_spaces()
        default_spaces = cls.get_default_spaces(index=2)
        instance.add_data_space(default_spaces)
        return instance

    @classmethod
    def data_directory(cls):
        instance = data_spaces()
        default_spaces = cls.get_default_spaces(index=4)
        instance.add_data_space(default_spaces)
        return instance

    @classmethod
    def project(cls):
        instance = data_spaces()
        default_spaces = cls.get_default_spaces(index=3)
        instance.add_data_space(default_spaces)
        return instance

    def __init__(self):
        self.data_space = []
    
    def add_data_space(self, space: any = None):
        if space and isinstance(space, dict):
            name = space.get("name", None)
            path = space.get("path", None)
            if not path or not os.path.exists(path):
                raise ValueError("The provided data space path:{path} does not exist.")
            description = space.get("description", "")
            self.data_space.append({
                "name": name or os.path.basename(path),
                "path": path,
                "description": description
            })
            return
        elif space and isinstance(space, list):
            for ds in space:
                self.add_data_space(ds)
            return
        elif space and isinstance(space, str):
            path = space
            if not os.path.exists(path):
                raise ValueError("The provided data space path:{path} does not exist.")
            name = os.path.basename(path)
            description = ""
        else:
            raise ValueError("Invalid data space format.")
    
    def __str__(self):
        if not self.data_space or len(self.data_space) == 0:
            return "No data spaces defined."
        if len(self.data_space) == 1:
            ds = self.data_space[0]
            return f"\n数据空间: 名称: {ds['name']}；- 路径: {ds['path']}；- 空间信息: {ds['description']}\n"
        if len(self.data_space) > 1:
            result = "\n数据空间列表："
            for idx, ds in enumerate(self.data_space):
                result += f"\n\t  {idx+1}. 名称: {ds['name']}；路径: {ds['path']}；空间信息: {ds['description']}"
            result = result + "\n"
            return result

class coding_data_agent():

    def __init__(self, input: str, data_root: data_spaces = None, module_name: str = None):
        self.input = input
        self.module_name = module_name
        if not data_root:
            raise ValueError("Data root path is required for coding agent.")
        self.data_root = data_root
        self.uuid= uuid.uuid4().hex
        self.config = {"configurable": {"thread_id": f"code-gen-{self.uuid}"},"recursion_limit": 1000}
        self.code_save_path = self.__build_code_file_path()
        self.total_code_chars = 0
        # 新增：代码提取状态管理
        self.code_extraction_state = {
            "in_code_block": False,  # 是否正在代码块中
            "start_found_in_turn": False,  # 本轮是否找到START
            "buffer": "",  # 缓冲未完成的代码片段
            "status_code": 0  # 状态码：0-未开始，1-代码提取中，2-完成
        }
    
    def  __build_code_file_path(self) -> str:
        """构建代码保存路径"""
        
        filename = f"{self.module_name}.py" if self.module_name else f"{self.uuid}.py"

        file_path = os.path.join(code_path, filename)
        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            counter = 1
            while True:
                new_file_path = f"{base}_{counter}{ext}"
                if not os.path.exists(new_file_path):
                    file_path = new_file_path
                    break
                counter += 1
        return file_path

    def set_data_root(self, path: data_spaces):
        self.data_root = path

    def extract_code_with_state_machine(self, response_text: str) -> tuple[str, int]:
        """
            使用状态机提取代码，返回(提取的代码, 状态码)
            状态码：
            0: 未提取到代码
            1: 提取到部分代码（开始但未结束）
            2: 提取到完整代码（开始并结束）
            3: 续写代码（上一轮开始，本轮继续）
        """
        
        # 1. 提取响应中的文本内容
        text = self._extract_text_from_response(response_text)
        if not text:
            return "", self.code_extraction_state["status_code"]
        
        # 2. 状态机逻辑
        has_start = CODING_START_PROMPT in text
        has_end = CODING_END_PROMPT in text
        
        if has_start:
            start_idx = text.find(CODING_START_PROMPT) + len(CODING_START_PROMPT)
            
            if has_end:
                # 情况1：START和END都有
                end_idx = text.find(CODING_END_PROMPT)
                code = text[start_idx:end_idx]
                return self._clean_code_text(code), 2
            else:
                # 情况2：只有START
                code = text[start_idx:]
                return self._clean_code_text(code), 1
        elif has_end:
            # 情况3：只有END（上一轮开始了）
            end_idx = text.find(CODING_END_PROMPT)
            code = text[:end_idx]
            return self._clean_code_text(code), 2
        elif self.code_extraction_state["status_code"] == 1:
            # 情况4：无标记，但上一轮开始了（续写）
            return self._clean_code_text(text), 3
        
        return "", 0

    def _extract_text_from_response(self, response) -> str:
        """从响应对象中提取文本"""
        text = ""
        
        if hasattr(response, 'generations') and response.generations:
            for generation in response.generations:
                if hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                    text += str(generation.message.content)
                    break
        elif isinstance(response, dict) and "messages" in response:
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    text += str(msg.content)
                    break
        elif hasattr(response, 'content'):
            text = str(response.content)
        
        return text

    def _extract_between_markers(self, text: str) -> str:
        """提取START和END之间的代码"""
        start_idx = text.find(CODING_START_PROMPT)
        if start_idx == -1:
            return ""
        
        start_content = text[start_idx + len(CODING_START_PROMPT):]
        end_idx = start_content.find(CODING_END_PROMPT)
        
        if end_idx == -1:
            return ""
        
        code = start_content[:end_idx]
        return self._clean_code_text(code)

    def _extract_from_start_to_end(self, text: str, extract_to_end: bool = True) -> str:
        """从START提取到结尾或到END"""
        start_idx = text.find(CODING_START_PROMPT)
        if start_idx == -1 and not extract_to_end:
            # 没有START，但有END（情况6）
            end_idx = text.find(CODING_END_PROMPT)
            if end_idx != -1:
                code = text[:end_idx]
                return self._clean_code_text(code)
            return ""
        
        if start_idx == -1:
            return ""
        
        start_content = text[start_idx + len(CODING_START_PROMPT):]
        
        if extract_to_end:
            # 提取到文本末尾
            code = start_content
        else:
            # 提取到END
            end_idx = start_content.find(CODING_END_PROMPT)
            if end_idx == -1:
                return ""
            code = start_content[:end_idx]
        
        return self._clean_code_text(code)

    def _clean_code_text(self, code: str) -> str:
        """改进：更彻底地清理代码块标记"""
        # 先备份原始代码用于结束标记检测
        original_code = code
        
        # 1. 移除所有代码块标记（```python和```）
        # 移除开始的```python标记
        code = re.sub(r'^```python\s*\n?', '', code, flags=re.IGNORECASE)
        # 移除开始的```标记（如果不是python标记）
        code = re.sub(r'^```\s*\n?', '', code)
        # 移除结尾的```标记
        code = re.sub(r'\n?\s*```\s*$', '', code)
        
        # 2. 移除行内的代码块标记（如果有的话）
        code = re.sub(r'\n```python\s*\n?', '\n', code, flags=re.IGNORECASE)
        code = re.sub(r'\n```\s*\n?', '\n', code)
        
        # 3. 替换转义字符
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')
        
        # 2. 标记和保护三引号区域
        # 使用占位符暂时替换三引号内容
        triple_quotes = []
        
        # 查找所有三引号字符串
        def replace_triple_quote(match):
            placeholder = f'__TRIPLE_QUOTE_{len(triple_quotes)}__'
            triple_quotes.append(match.group(0))
            return placeholder
        
        # 替换三引号字符串（单引号和双引号）
        code = re.sub(r'(\"{3}|\'{3})[\s\S]*?\1', replace_triple_quote, code)

        # 步骤3.1: 修复引号前的换行
        # 查找模式：换行符后跟着引号
        code = re.sub(r'\n\s*"', '"', code)      # 双引号前
        code = re.sub(r"\n\s*'", "'", code)      # 单引号前
        
        # 步骤3.2: 修复引号后的换行
        # 查找模式：引号后跟着换行符
        code = re.sub(r'"\s*\n', '"', code)      # 双引号后
        code = re.sub(r"'\s*\n", "'", code)      # 单引号后

        # 4. 恢复三引号内容
        for i, quote_content in enumerate(triple_quotes):
            placeholder = f'__TRIPLE_QUOTE_{i}__'
            code = code.replace(placeholder, quote_content)

        # 4. 移除首尾空白
        code = code.strip()
        
        # 5. 保存原始代码中的结束标记信息
        if hasattr(self, '_last_raw_code'):
            self._last_raw_code = original_code
        
        return code

    def extract_code_from_response(self, response) -> str:
        """包装状态机方法，保持向后兼容"""
        code, status_code = self.extract_code_with_state_machine(response)
        
        # 根据状态码记录日志
        status_messages = {
            0: "未提取到代码",
            1: "开始提取代码（等待结束）",
            2: "完整提取代码",
            3: "续写代码"
        }
        print(f"  [状态机] 状态码: {status_code} - {status_messages.get(status_code, '未知')}")
        
        return code

    def save_code_chunk(self, code_chunk: str) -> str:
        """Save code chunk and return status."""
        if not code_chunk.strip():
            return "No code to save"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.code_save_path), exist_ok=True)
            
            # Append to file
            with open(self.code_save_path, "a", encoding="utf-8") as f:
                f.write(code_chunk + "\n\n")
            is_complete = CODING_END_PROMPT in code_chunk  # 直接使用 in 判断，更可靠
            print(f"  [保存] 本次代码块中检测到结束标记？ {is_complete}") 
            self.total_code_chars += len(code_chunk)
            is_complete = CODING_END_PROMPT.lower() in code_chunk.lower()
            
            status = f"Saved {len(code_chunk)} chars (Total: {self.total_code_chars})"
            if is_complete:
                status += " ✅ COMPLETE"
            
            print(status)
            return status
            
        except Exception as e:
            return f"Save error: {e}"

    def _get_last_code_context(self, num_lines: int = 15) -> str:
        """
        读取已保存代码文件的最后若干行，作为续写的上下文。
        如果文件不存在或为空，则返回空字符串。
        """
        if not os.path.exists(self.code_save_path):
            return ""
        try:
            with open(self.code_save_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 返回文件的最后 num_lines 行，过滤掉可能存在的纯空白行
            last_lines = [line.rstrip() for line in lines[-num_lines:] if line.strip()]
            return "\n".join(last_lines[-num_lines:])  # 再次切片确保不超过指定行数
        except Exception:
            return ""

    def _build_continuation_prompt(self) -> str:
        # 1. 获取上下文：读取已保存文件的最后N行代码。
        previous_code_tail = self._get_last_code_context(num_lines=10)
        # 2. 构建提示词：
        prompt = f"""
            请继续生成下一段Python代码。

            【任务状态】
            你正在进行分段代码编写，主要依据是已获取的 `description.txt` 文件。

            【接续指令】
            请基于下方已生成的代码末尾，进行**直接、连贯的逻辑接续**，并完成后续部分。
            {previous_code_tail if previous_code_tail else '# (代码起始位置)'}
            **重要**：请仅续写新代码，严格避免重复上述内容或生成解释性文本。

            【输出格式】
            你必须严格遵守以下格式：
            <CODING_START>
            ```python
            # 你的接续代码
            ```
            <CODING_END>
        """
        return prompt

    async def run(self) -> str:
        
        model = get_default_model()

        system_prompt = """
            你是一个严谨的Python代码生成专家。你的核心任务是：**严格遵循《数据描述文件》中的所有规范**，将用户需求转化为可独立运行、高质量、无错误的Python代码,数据根目录{data_path}。

            【首要原则与工作流】
                1.  **核心依据与导航式探索**：
                    你必须通过 `describe` 工具进行逐级深入的导航, 该工具会为你提供该目录的`description.txt`。
                    要了解一个目录下的数据结构和内容，**首先**调用 `describe` 工具获取该目录的 `description.txt` 内容。
                    当目录下不存在 `description.txt` 时，表示该目录的数据结构和内容已在上级目录的描述中涵盖，无需重复获取。
                    适度调用 `ls` 和 `data_sample` 工具辅助理解数据结构和样本内容，但**绝不可**绕过 `describe` 工具。
                    导航的终点不是固定的层级，而是**当你获取的描述信息足以让你开始编写符合用户需求的代码时**。此时，应停止探索并开始生成。

                2.  **生成阶段隔离**：一旦你通过描述文件**定位到目标数据空间**并声明开始编写代码，或输出 `<CODING_START>` 标记，即进入生成阶段。
                    此阶段**严禁调用任何工具**，必须基于已获取的所有描述信息完成代码。

                3.  **路径规则**：在编写的代码中，必须主动忽略所有名为 `backup`, `backup_old`, `old`, `cache` 的目录及其子目录。
                
                <CODING_START>
                ```python
                # 你的代码
                ```
                <CODING_END>

            【代码质量要求】
            -   **正确性**：精准实现需求，逻辑完备。
            -   **可运行**：生成完整代码片段，确保在 `code_runner` 中可执行。
            -   **专业性**：结构清晰，包含必要注释，错误处理得当。
            -   **专注性**：除最终代码外，不输出任何解释、总结或额外文本。

            现在，请首先调用 describe 工具获取《数据描述文件》，然后开始生成代码。如果代码较长，我们后续会分段进行。
        """

        # Create agent with EXPLORATION tools only (no python_writer)
        agent = create_agent(
            model=model,
            tools=[describe, ls, data_sample]+get_sandbox_tool(),  # No code writing tools!
            system_prompt=system_prompt.replace("{data_path}", str(self.data_root)),
            # checkpointer=checkpointer,
        )
        
        # Continuous generation loop
        messages = [{"role": "user", "content": self.input}]
        max_turns = 30
        
        print("🚀 Starting code generation...")
        
        for turn in range(max_turns):
            print(f"\n🔄 Turn {turn + 1}/{max_turns}")
            try:
                # Agent continues with full conversation history
                response = await agent.ainvoke(
                    {"messages": messages}, 
                    self.config
                )
                
                # Extract conversation response
                agent_response = extract_conversation(response, "final")
                print(f"Agent: {agent_response[:200]}...")
                
                # Extract and save code IMMEDIATELY
                code_chunk, status_code = self.extract_code_with_state_machine(response)
                print(f"  状态码: {status_code}, 提取代码长度: {len(code_chunk)}")

                if code_chunk:
                    save_result = self.save_code_chunk(code_chunk)
                    print(save_result)
                    
                    # 根据状态码决定是否继续
                    if status_code == 2:  # 完整提取
                        print("  检测到完整代码块，准备检查是否完成")
                        break
                
                messages = response['messages']
                # Prepare continuation
                messages.append({
                    "role": "user", 
                    "content": "你已经完成所有数据结构调研，继续生成下一段代码。同样以<CODING_START>开始，以<CODING_END>结束。"
                })
                
            except Exception as e:
                print(f"❌ Error in turn {turn + 1}: {e}")
                break
        
        # Final status
        final_size = os.path.getsize(self.code_save_path) if os.path.exists(self.code_save_path) else 0
        return f"Code generation complete! Saved {final_size} bytes to {self.code_save_path}. Total chunks processed: {turn + 1}"

    async def __call__(self, input: str) -> str:
        self.input = input
        return await self.run()

# Keep exploration tools (unchanged)
@tool
def describe(path: str):
    """
        读取并返回指定路径的数据描述文件 (`description.txt`) 内容。
        不同层级的描述文件描述的信息颗粒度不同；当进入目录时，首先Check `description.txt`.

        参数:
            path: 需要了解其描述的任意文件或目录路径。建议从数据根目录开始，然后根据上层描述的指引，有目的地查看关键子目录的描述。

        返回:
            包含 `desc_path` (找到描述的目录) 和 `description` (描述文本) 的字典。
    """
    print(f"describe called with path: {path}")
    if not os.path.exists(path):
        return {"error": "the path is invalid, to check it"}
    
    current_path = path
    if os.path.isfile(current_path):
        current_path = os.path.dirname(current_path)
    
    while True:
        des_file = os.path.join(current_path, "description.txt")
        if os.path.exists(des_file):
            with open(des_file, "r", encoding="utf-8") as f:
                content = f.read()
            return {
                "desc_path": current_path,
                "description": content
            }
        
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            break
        current_path = parent_path
    
    return {"error": "no description file found in the path hierarchy"}

@tool
def data_sample(path: str):
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
                "description": "No description file found, you can get the description from parent's description."
            }
    else:
        if path.endswith("csv"):
            with open(path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 10:
                    lines = lines[:3] + ["...\n"] + lines[-3:]
                else:
                    lines = lines[:len(lines)-1]
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
                "sample": data[:3] + ["..."] + data[-3:]
            }
        
        if path.endswith("json"):
            with open(path, 'r') as file:
                data = json.load(file)
            if isinstance(data, list):
                if len(data) > 10:
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
                content = file.read(500)
            return {
                "message": "这是TXT文件",
                "sample": content
            }

@tool
def ls(path: str):
    """
    列出指定路径下的文件和文件夹。
    """
    print(f"ls called with path: {path}")
    if not os.path.exists(path):
        return {"error": "the path is invalid, to check it"}
    
    items = os.listdir(path)
    files = []
    directories = []
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path):
            directories.append(item)
    return {
        "item_count": len(items),
        "files": files,
        "directories": directories
    }

async def coding(
        input_requirement: str = None,
        data_space: str = None,
        module_name : str = None,
):  
    
    if not input_requirement:
        raise ValueError("Input requirement is required for coding agent.")
    if not data_space:
        raise ValueError("Data space path is required for coding agent.")

    agent = coding_data_agent(input=input_requirement, data_root=data_space, module_name=module_name)
    result = await agent.run()
    print("\n" + "="*50)
    print("Agent Result:")
    print(result)
    print(f"\n📁 Final code saved: {agent.code_save_path}")
    if os.path.exists(agent.code_save_path):
        print(f"📊 File size: {os.path.getsize(agent.code_save_path)} bytes")

if __name__ == "__main__":
    # Use your latest requirement
    input_requirement = """
        设计一个函数，输入参数包括：日期（date_str）、总数(total_N)、获取资金流入最多的股票数量(top_K)。函数要求如下：
            优先查看开盘数据中是否能够筛选出来，如果筛选不出来，从前一日的tushare资金净流入数据来获取资金流入最多的股票列表。
        返回股票的code列表。
    """

    input_requirement = """
    编写一个function， 基于每日股票列表和每日市场快照，构建MA数据的CSV文件。
    开始日期从2025-11-10到最新；且之后每日调用可以创建每日最新MA数据行；
    要求：
        1、计算ma5_price、ma10_price、ma20_price、ma40_price、ma60_price等数据,以及对应的ma_amount;（设计时，可考虑冗余ma4、ma9等字段以降低后续的每日计算量。
        2、由于开始计算时间，和新上市股票可能不足对应时间，因此，需要冗余一个计算时间天数；
        3、对于当天无，或者当天再股票列表中标记为当天停牌，为保持连续性，从上一日取数据顺延。状态字段：	list_status	str	上市状态 L上市 D退市 P暂停上市；
        4、每日一个数据文件YYYYMMDD.csv
        5、由于较早期的股票时间列表数据不存在，最早股票列表为20251110，所以，从1110开始计算，因此MA60可能需要从1110开始向前获取>=60天的数据记录。
        6、文件以
            import os, sys
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
            sys.path.insert(0, project_root)
        开始，构建project_root变量；
        7、引入data.A_stock.base_data_prepaire_tushare模块；
            该模块中变量stock_snapshot_ma_daily_dir为ma文件保存路径；
            get_stock_list(date: str = None) -> pd.DataFrame:,获取指定日期列表，取自股票列表文件；
            get_stock_snapshot(date: str) -> Optional[pd.DataFrame]:获取指定日期快照，取自快照文件；
            get_market_trade_days(start_date: str=default_start_date, end_date: str = datetime.now().strftime("%Y%m%d")) -> Optional[pd.DataFrame]:
                ```
                获取指定日期范围内的交易日列表。
                参数:
                start_date: 起始日期字符串，格式为 'YYYYMMDD'，含start_date
                end_date: 结束日期字符串，格式为 'YYYYMMDD'，含end_date
                返回:
                pd.DataFrame 包含交易日数据，或 None 如果文件不存在或读取失败
                ```
            以上api中日期参数均为YYYYMMDD格式要求。
    """

    input_requirement = """
        编写一个function:prepare_index_ma_daily通过指数日线数据，生成日度移动平均数据。以具体的指数代码为参数，可选测试参数000300.SH

要求生成的字段项如下：
ts_code,           -- 股票代码
trade_date,        -- 交易日期
close_price,       -- 收盘价
high_price,        -- 最高价
low_price,         -- 最低价
pre_close,         -- 前收盘价
amount,            -- 成交额
ma5_price, ma10_price, ma20_price, ma40_price, ma60_price,
ma5_amount, ma10_amount, ma20_amount, ma40_amount, ma60_amount,
calc_days_5, calc_days_10, calc_days_20, calc_days_40, calc_days_60

代码要求：
1、文件以
		import os, sys
		project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
		sys.path.insert(0, project_root)
	开始，构建project_root变量；

2、引入data.A_stock.base_data_prepaire_tushare模块；
	该模块中变量indexes_daily_dir为ma文件的保存路径，ma文件以指数编码+—_ma.csv的形式命名
	模块可用api
	def get_index_daily(index_code:str, start_date: str = default_index_start_date, end_date: str = None) -> Optional[pd.DataFrame]:
    ··· 
    获取指定指数在指定日期范围内的日线数据（本地文件）。
    如果需要获取的数据日期范围超出了本地文件的范围，则调用API获取缺失的数据并合并保存，但是尽量计算精确，避免重复获取。
    参数:
      index_code: 指数代码字符串
      start_date: 起始日期字符串，格式为 'YYYYMMDD'（含当日）
      end_date: 结束日期字符串，格式为 'YYYYMMDD'，默认为今天（含当日）
    ···
3、能够根据已有数据，每日做增量运算；
4、同时在代码中生成一个取数API:get_index_ma_daily,参考get_index_daily
    """

    input_requirement = """
    下表汇总了构成“价格动量”选股器的全部因子，其计算逻辑严格遵循我们之前讨论的正交设计。

因子类别	因子名称	计算逻辑与公式	可配置参数	

价格动量	时间序列动量	MOM = (close_price / N_MOM日前收盘价) - 1
使用close_price字段回溯计算。	N_MOM (动量周期，默认60)
MOM_TH (动量阈值，默认0.05)	ts_code, trade_date, close_price
趋势状态	双均线多头排列	TREND = (close_price > ma20_price) AND (ma20_price > ma60_price)
结果为布尔值（True/False）。	MA_SHORT (短周期，默认20)
MA_LONG (长周期，默认60)	close_price, ma20_price, ma60_price
相对强度	相对沪深300强度	RS = (个股N_RS日收益率 / 沪深300_N_RS日收益率) - 1
注意：需关联基准指数表计算。	N_RS (强度周期，默认20)
RS_TH (强度阈值，默认0.02)	close_price (用于计算个股收益率)
波动突破	真实波幅突破	ATR = 过去N_ATR日的平均真实波幅
BREAKOUT_SIG = (high_price - low_price) > (ATR * BREAK_TH)	N_ATR (ATR周期，默认20)
BREAK_TH (突破系数，默认1.0)	high_price, low_price, close_price (用于计算ATR)
成交量确认	放量	VOL_SIG = amount > (ma20_amount * VOL_RATIO_TH)	VOL_RATIO_TH (放量阈值，默认1.5)	amount, ma20_amount
数据质量	数据完备性	QUALITY = (calc_days_20 >= 20)
确保有足够长的有效数据。	MIN_CALC_DAYS (最小计算天数，默认20)	calc_days_20
注意：high_price 和 low_price 是计算 “波动突破” 因子的必需字段，已在之前的推荐表结构中提出。

⚙️ 详细计算过程与边界处理
以下是每个因子在代码实现中需要考虑的详细步骤和边界情况：

1. 时间序列动量 (MOM)

计算：对于每只股票，获取当日 close_price 和 N_MOM 个交易日前的 close_price 计算收益率。

参数：N_MOM 可调，测试不同长度动量（如20日短期动量 vs 60日中期动量）。

边界：检查 calc_days_N 字段，确保 N_MOM 日前数据有效，否则标记为 NaN。

2. 双均线多头排列 (TREND)

计算：直接使用 ma20_price 和 ma60_price 字段进行比较。

可扩展：参数 MA_SHORT 和 MA_LONG 可配置为其他周期（如10/30），但需确保MISSION48已计算对应 ma 字段。

边界：确保 calc_days_60 足够，避免使用上市初期的失真均线。

3. 相对沪深300强度 (RS)

计算：

计算个股 N_RS 日收益率：(close_price / N_RS日前close_price) - 1。

从独立的指数CSV文件中获取沪深300同期的收益率。

按公式计算 RS。

关键：需要准备 index_000300_SH.csv 文件，包含 trade_date， close_price 字段。

4. 真实波幅突破 (BREAKOUT_SIG)

计算（需在MISSION48或49中实现ATR计算）：

计算每日真实波幅 TR：MAX(high_price - low_price, |high_price - 前收|, |low_price - 前收|)。若暂无前收，可先简化为 (high_price - low_price)。

计算 ATR：TR 在过去 N_ATR 日的简单移动平均。

判断当日波幅是否突破：(high_price - low_price) > (ATR * BREAK_TH)。

参数：N_ATR 和 BREAK_TH 共同控制对“突破”的敏感度。

5. 成交量确认 (VOL_SIG)

计算：直接比较当日成交额 amount 与 ma20_amount。

参数：VOL_RATIO_TH 决定对放量程度的要求。

6. 数据质量过滤 (QUALITY)

计算：此为硬性过滤条件，不满足的股票直接排除。

参数：MIN_CALC_DAYS 通常与最长指标周期（如60）对齐。

🧮 综合打分与选股输出
所有因子计算后，需要整合成一个可排序的综合评分，并生成LLM可读的理由。

1. 综合打分（示例）

python
# 可配置的因子权重
WEIGHTS = {
    ‘MOM’: 0.3,
    ‘TREND’: 0.25,
    ‘RS’: 0.2,
    ‘BREAKOUT_SIG’: 0.15,
    ‘VOL_SIG’: 0.1
}
# 计算加权总分（假设各因子已归一化到0-1区间）
综合得分 = (MOM因子得分 * W_MOM) + (TREND因子得分 * W_TREND) + ... 
注：因子权重 WEIGHTS 本身也应作为可调参数，未来可由复盘智能体优化。

2. 输出格式（与架构设计一致）
选股器最终输出一个结构化列表（如JSON），每个股票包含：

symbol, name

momentum_score (综合得分)

factors (各因子具体值，用于分析和反馈)

reason (自然语言理由，至关重要)

理由生成示例：

“该股票过去60日动量强劲(上涨12%)，处于20/60日均线多头排列趋势中，相对沪深300超额收益达5%，且近期呈现放量突破形态。”



基于以上需求，设计生成所需的代码；要求，1、所有因子可独立计算或调用；2、所有参数要求对外暴露，未来可通过配置文件或者反馈学习注入调整；3、最终要求暴露为一个可直接引用的function，供我的策略器注入。
    """

    input_requirement = """
        编写一个工具，从交易维度，分析每个交易员的交易频次，每次交易的收益率必要信息；
        如交易action中，未记录当时的价格，则需要从每日市场快照中获取对应的价格信息；
        统计交易员包括：deepseek-chat-v3.1、deepseek-reviewer、deepseek-momentum-reviewer、 deepseek-momentum-reviewer-noreason
    """

    input_requirement = """
    写一个function，其中，参数包括：日期（date_str）、指数编号（ts_code)、获取总数(topn)、maxprice 和minprice。函数要求如下：
        # 建议筛选阈值
        股价下限：排除股价 < 2元（避免ST/退市风险股）
        股价上限：排除股价 > 300元（根据资金规模调整）

        # 筛选逻辑
        1. 先按指数权重排序
        2. 排除股价异常值（上下限外）
        3. 取前N名
    """

    input_requirement = """
        编写一个函数，参数为交易员signature，输出交易员整个交易历史中，所有的可配置选股参数的历史变化情况。
        (代码只实时计算，不需要保存分析结果。)
    """

    input_requirement = """
        编写一个函数，计算final_desision的持仓市值。接收一个日期参数，根据当日收盘价计算。如果当日无收盘数据，返回无法计算。
        如，日期参数为空则默认计算最新日期的持仓市值。
    """

    whole = data_spaces.whole()
    market = data_spaces.market()
    traders = data_spaces.traders()
    project = data_spaces.project()
    data = data_spaces.data_directory()
    asyncio.run(coding(
        input_requirement=input_requirement,
        data_space=data,
        module_name="final_position_value_calculator"
    ))



    print("Whole Data Spaces:", str(whole))
    print("Market Data Spaces:", market)
    print("Traders Data Spaces:", traders)