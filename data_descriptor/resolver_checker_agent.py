from datetime import datetime
import re
import os,sys
from pathlib import Path
import json
import asyncio
import os
from unittest import result
from dotenv import load_dotenv
import datasets
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import vendor.vendor_patch  # noqa: E402
from langchain.tools import tool
load_dotenv()
from utils.talent_training_market import AgentManager,SubAgent
from data_descriptor.resolver_agent import model_type, load_data
from data_descriptor.tool_model import get_default_model
from agent_tools.local_tools.caculator import get_calculator_tool, get_sandbox_tool

class resolver_checker_agent(SubAgent):
    
    def __init__(self, question_index: int, dataset: str= "", checker_str: str = ""):
        
        self.child_manager = None
        self.question_index = question_index
        self.agent_num = f"puzzle_{question_index+1}"
        self.dataset = dataset
        model = get_default_model()
        # base_tools=[add, subtract, multiply, divide, power, distance, floor, ceil, modulus]
        base_tools = get_calculator_tool()+ get_sandbox_tool() 
        manager = AgentManager(llm_client=model)
        manager.open_shared_blackboard()
        manager.open_email_service()
        super().__init__(manager=manager, task_key_target="", task_description=f"{checker_str}", enable_hiring=True, depth=0, base_tools=base_tools)

        thread_id = self.agent_num +"_"+ datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {"configurable": {"thread_id": f"{thread_id}"},"recursion_limit": 1000}
        self.initial_tools = [save_result] + base_tools
        self.agent_id = "root"   
        # TODO register a root advisor
        manager.register_agent(self)

    def _generate_system_prompt(self):

        prompt = f"""

            你的核心任务是：根据题目、标准答案，检查用户答案的正确性；

            ### 一、 任务目标
            1、检查用户答案的正确性，并做出解释；如答案是数学表达式等，需验证其等价性；
            2、调用保存工具保存你的检查结果；

            {self.child_manager.default_prompt() if self.child_manager else ""}
            当你觉得任务完成时，输出
                {SubAgent.STOP_SIGNAL}
        """
        return prompt

@tool(meta={
    "inject_requires": {
        "agent_num": True,
        "dataset": True
    }
}) 
def save_result(check_result: bool, descript: str, **kwargs):
    """
        保存结果到指定路径的文件中。
        check_result: 最终结果布尔值
        descript: 描述字符串
    """

    agent_num = kwargs["kwargs"]["agent_num"] or "unknown_agent"
    dataset = kwargs["kwargs"]["dataset"] or "dataset"
    print(f"save_result called with agent_num: {agent_num} in dataset : {dataset}")
    
    result_file = Path(project_root) / "logs" / dataset / model_type /f"check_result_batch.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    # 读取现有结果
    if result_file.exists():
        with open(result_file, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}
    # 更新结果
    results[agent_num] = {
        "check_result": check_result,
        "description": descript
    }
    # 保存结果
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"结果已保存到 {result_file}")
    
def load_user_answer(question_index: int, dataset: str):
    agent_num = f"puzzle_{question_index+1}"
    result_file = Path(project_root) / "logs" / dataset / agent_num / model_type /f"resolver_result_{agent_num}.txt"
    if result_file.exists():
        with open(result_file, "r") as f:
            user_answer = f.read()
        return user_answer
    else:
        print(f"用户答案文件 {result_file} 不存在")
        return None
    
async def run_checker_agent(datasetname = "", start_index:int = 0, end_index:int = None):
    
    dataset = load_data()
    if end_index is None:
        end_index = len(dataset["test"])
    for i in range(start_index, end_index):

        user_answer = load_user_answer(question_index=i, dataset=datasetname)
        if user_answer is None:
            print(f"跳过检查器运行，因用户答案不存在，问题索引：{i}")
            continue
        question = dataset["test"][i]["prompt"]
        answer = dataset["test"][i]["answer"]
        checker_str = f"题目：{question}\n标准答案：{answer}\n用户答案：{user_answer}\n\n请检查用户答案的正确性，并提供详细的解释和修正建议；"
        agent = resolver_checker_agent(question_index=i, dataset=datasetname, checker_str=checker_str)
        await agent.execute()

if __name__ == "__main__":
    asyncio.run(run_checker_agent(datasetname="resolve", start_index=0, end_index=None))


