from socket import AI_ADDRCONFIG
import os,sys,asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from data_descriptor.resolver_agent import load_data

stop_signal = "<FINISH_SIGNAL>"

model_type = "deepseek"  # or "qwen""deepseek""baidu""gpt""gemini"
from data_descriptor.tool_model import get_model_by_name 
model = get_model_by_name(model_type)
dataset_name = "resolve_direct"  # or "aime_2026"
async def run_data_sample(start_index: int =0, end_index: int = None):
    dataset = load_data()
    if end_index is None:
        end_index = len(dataset["test"])
    for i in range(start_index, end_index):
        print(f"Processing sample {i+1}/{len(dataset['test'])}")
        final_answer = ""
        agent_num = f"puzzle_{(i+1)}"
        data_item = dataset["test"][i]
        question_id = data_item["question_id"]
        prompt = data_item["prompt"]
        messages = [
            SystemMessage(content=f"You are a helpful assistant for solving puzzles. Please provide a clear and concise answer to the following question. If you need to think step by step, please do so. When you finish your answer, please end with the signal '{stop_signal}' to indicate completion."),   
            HumanMessage(content=prompt)
        ]
        response = await model.ainvoke(messages)
        final_answer += response.content
        continuer_message = HumanMessage(content=f"请继续回答，直到你认为完成了问题的解答，并且在最后加上完成信号'{stop_signal}'。")
        while stop_signal not in response.content:
            AI_MESSAGE = AIMessage(content=response.content)
            messages.append(AI_MESSAGE)
            messages.append(continuer_message)
            response = await model.ainvoke(messages)
            final_answer += response.content

        print(f"Question ID: {question_id}")
        result_file = Path(project_root) / "logs" / dataset_name / agent_num / model_type /f"resolver_result_{agent_num}.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(final_answer)
        print(f"Response saved to {result_file}")

async def run_data_sample(start_index: int = 0, end_index: int = None):
    # 加载数据集（只加载一次，避免重复IO）
    dataset = load_data()
    if end_index is None:
        end_index = len(dataset["test"])

    # 获取所有待处理的索引列表
    indices = list(range(start_index, end_index))
    total = len(indices)

    # 定义处理单个样本的内部异步函数
    async def process_one_sample(i):
        # i 是实际的数据索引
        print(f"Processing sample {i+1}/{len(dataset['test'])}")
        final_answer = ""
        agent_num = f"puzzle_{(i+1)}"
        data_item = dataset["test"][i]
        question_id = data_item["question_id"]
        prompt = data_item["prompt"]
        messages = [
            SystemMessage(content=f"You are a helpful assistant for solving puzzles. Please provide a clear and concise answer to the following question. If you need to think step by step, please do so. When you finish your answer, please end with the signal '{stop_signal}' to indicate completion."),   
            HumanMessage(content=prompt)
        ]
        response = await model.ainvoke(messages)
        final_answer += response.content
        continuer_message = HumanMessage(content=f"请继续回答，直到你认为完成了问题的解答，并且在最后加上完成信号'{stop_signal}'。")
        while stop_signal not in response.content:
            AI_MESSAGE = AIMessage(content=response.content)
            messages.append(AI_MESSAGE)
            messages.append(continuer_message)
            response = await model.ainvoke(messages)
            final_answer += response.content

        print(f"Question ID: {question_id}")
        result_file = Path(project_root) / "logs" / dataset_name / agent_num / model_type / f"resolver_result_{agent_num}.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(final_answer)
        print(f"Response saved to {result_file}")

    # 分批并发处理，每批最多5个
    batch_size = 10
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        tasks = [process_one_sample(i) for i in batch_indices]
        # 并发执行当前批次的所有任务，等待它们完成后再进入下一批
        await asyncio.gather(*tasks)
        print(f"Batch {batch_start//batch_size + 1} completed.")

if __name__ == "__main__":
    "3/9/14/20/26/22"
    asyncio.run(run_data_sample(20))