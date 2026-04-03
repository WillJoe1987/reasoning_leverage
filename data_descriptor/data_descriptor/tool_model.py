from datetime import datetime
from langchain_openai import ChatOpenAI
from typing import Optional
import json
from pathlib import Path
from langchain.messages import ToolMessage
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from langchain_google_genai import ChatGoogleGenerativeAI  
import os

load_dotenv()

ds_context_limit = 131072  # DeepSeek API context length limit in characters

class DeepSeekChatOpenAI(ChatDeepSeek):

        def _log_interaction(self, direction: str, method: str, payload) -> None:
            """Append a plain-text record to self.interaction_log_file.

            Format:
            <ISO TS> | <method> | <direction>
            <repr(payload)>

            direction: 'request' or 'response'
            """
            log_date_str = datetime.now().strftime("%Y%m%d")
           
            interaction_log_file = f"/prog/pweb/AI-Trader/logs/sepc_call_llm_{log_date_str}.log"
            try:
                ts = datetime.now().isoformat()
                p = Path(interaction_log_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(f"{ts} | {method} | {direction}\n")
                    f.write(repr(payload) + "\n\n")
            except Exception:
                # 日志写入失败不应影响主流程，静默忽略
                pass

        """
        Custom ChatOpenAI wrapper for DeepSeek API compatibility.
        Handles the case where DeepSeek returns tool_calls.args as JSON strings instead of dicts.
        """

        def _create_message_dicts(self, messages: list, stop: Optional[list] = None) -> list:
            """Override to handle response parsing"""
            message_dicts = super()._create_message_dicts(messages, stop)
            return message_dicts

        def _generate(self, messages: list, stop: Optional[list] = None, **kwargs):
            """Override generation to fix tool_calls format in responses"""
                # 1) 记录 request（在任何处理前）,MESSAGES取最后一条
            self._log_interaction("request", "_generate", messages[-1])

            # Call parent's generate method
            result = super()._generate(messages, stop, **kwargs)

            # Fix tool_calls format in the generated messages
            for generation in result.generations:
                for gen in generation:
                    if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                        tool_calls = gen.message.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            for tool_call in tool_calls:
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    args = tool_call["function"]["arguments"]
                                    # If arguments is a string, parse it
                                    if isinstance(args, str):
                                        try:
                                            tool_call["function"]["arguments"] = json.loads(args)
                                        except json.JSONDecodeError:
                                            pass  # Keep as string if parsing fails
                # 4) 记录 response（完整的 result）
            self._log_interaction("response", "_generate", result)

            return result

        async def _agenerate(self, messages: list, stop: Optional[list] = None, **kwargs):
            """Override async generation to fix tool_calls format in responses"""
            self._log_interaction("request", "_agenerate", messages[-1])

            total_length = sum(len(m.text) if isinstance(m.text, str) else 0 for m in messages)
            # 转为百分比 
            percent =(float(total_length) / ds_context_limit) * 100
            print(f"当前消息总长度占比: {percent:.2f}% of {ds_context_limit} characters limit.")

            if isinstance(messages[-1], ToolMessage):
                content = messages[-1].content
                if isinstance(content, str) :
                    pass
                else:
                    messages[-1].content = content[0][content[0]['type']]
            
            messages[-1].content += f"\n\n当前消息总长度占比: {percent:.2f}% of {ds_context_limit} characters limit."

            # Call parent's async generate method
            result = await super()._agenerate(messages, stop, **kwargs)

            # Fix tool_calls format in the generated messages
            for generation in result.generations:
                for gen in generation:
                    if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                        tool_calls = gen.message.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            for tool_call in tool_calls:
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    args = tool_call["function"]["arguments"]
                                    # If arguments is a string, parse it
                                    if isinstance(args, str):
                                        try:
                                            tool_call["function"]["arguments"] = json.loads(args)
                                        except json.JSONDecodeError:
                                            pass  # Keep as string if parsing fails

            self._log_interaction("response", "_agenerate", result)
            return result



class ChatGemini(ChatGoogleGenerativeAI):
        
        def _log_interaction(self, direction: str, method: str, payload) -> None:
            pass

        def _generate(self, messages: list, stop: Optional[list] = None, **kwargs):
            """Override generation to fix tool_calls format in responses"""

            # 如果kwargs包含generation_config，则记录下来，
                # 1) 记录 request（在任何处理前）,MESSAGES取最后一条
            self._log_interaction("request", "_generate", messages[-1])

            # Call parent's generate method
            result = super()._generate(messages, stop, **kwargs)

            # Fix tool_calls format in the generated messages
            for generation in result.generations:
                for gen in generation:
                    if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                        tool_calls = gen.message.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            for tool_call in tool_calls:
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    args = tool_call["function"]["arguments"]
                                    # If arguments is a string, parse it
                                    if isinstance(args, str):
                                        try:
                                            tool_call["function"]["arguments"] = json.loads(args)
                                        except json.JSONDecodeError:
                                            pass  # Keep as string if parsing fails
                # 4) 记录 response（完整的 result）
            self._log_interaction("response", "_generate", result)

            return result

        async def _agenerate(self, messages: list, stop: Optional[list] = None, **kwargs):
            """Override async generation to fix tool_calls format in responses"""
            
            self._log_interaction("request", "_agenerate", messages[-1])

            total_length = sum(len(m.text) if isinstance(m.text, str) else 0 for m in messages)
            # 转为百分比 
            percent =(float(total_length) / ds_context_limit) * 100
            print(f"当前消息总长度占比: {percent:.2f}% of {ds_context_limit} characters limit.")

            if isinstance(messages[-1], ToolMessage):
                content = messages[-1].content
                if isinstance(content, str) :
                    pass
                else:
                    messages[-1].content = content[0][content[0]['type']]
            
            messages[-1].content += f"\n\n当前消息总长度占比: {percent:.2f}% of {ds_context_limit} characters limit."

            # Call parent's async generate method
            result = await super()._agenerate(messages, stop, **kwargs)

            # Fix tool_calls format in the generated messages
            for generation in result.generations:
                for gen in generation:
                    if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                        tool_calls = gen.message.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            for tool_call in tool_calls:
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    args = tool_call["function"]["arguments"]
                                    # If arguments is a string, parse it
                                    if isinstance(args, str):
                                        try:
                                            tool_call["function"]["arguments"] = json.loads(args)
                                        except json.JSONDecodeError:
                                            pass  # Keep as string if parsing fails

            self._log_interaction("response", "_agenerate", result)
            return result


def get_baidu_model():

    return ChatOpenAI(
        model="ernie-5.0-thinking-preview",
        base_url="https://qianfan.baidubce.com/v2",
        api_key=os.getenv("BAIDU_API_KEY"),
        max_retries=3,
        timeout=180,
        extra_body={"enable_thinking": False}
    )

def get_default_model():

    return DeepSeekChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DS_API_KEY"),
        max_retries=3,
        timeout=180,
        temperature=0,
    )

def get_qwen_model():
    return ChatOpenAI(
        # model="qwen3-max",
        model="qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("QWEN_API_KEY"),
        max_retries=3,
        timeout=1800,
    )

def _reasoning_model():
    return DeepSeekChatOpenAI(
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DS_API_KEY"),
        max_retries=3,
        timeout=180,
        temperature=1.5,
        extra_body={"enable_thinking": True}
    )

def get_chat_model():
    return get_default_model()

def get_reasoning_model():
    return _reasoning_model()

def get_gpt_model():
    return ChatOpenAI(
        model="gpt-5.2",
        base_url="https://cn.getgoapi.com/v1",
        api_key=os.getenv("GPT_API_KEY"),
        max_retries=3,
        timeout=180,
        temperature=0.2,
    )

def get_gemini_model():
    return ChatOpenAI(
        model="gemini-3-flash-preview",
        base_url="https://cn.getgoapi.com",
        api_key=os.getenv("GPT_API_KEY"),
        max_retries=3,
        timeout=180,
        temperature=0.2,
        extra_body={
      'extra_body': {
        "google": {
          "thinking_config": {
            "thinking_level": "low",
            "include_thoughts": True
          }
        }
      }
    }
        
    )

def get_model_by_name(name: str):
    """ 
        default deepseek
        "qwen""deepseek""baidu""gpt""gemini"
    """
    name = name.lower()
    if name == "qwen":
        return get_qwen_model()
    elif name == "deepseek":
        return get_default_model()
    elif name == "baidu":
        return get_baidu_model()
    elif name == "gpt":
        return get_gpt_model()
    elif name == "gemini":
        return get_gemini_model()
    else:
        return get_default_model()


async def async_test():
    from langchain_core.messages import HumanMessage, SystemMessage

    puzzle_text = """
Let $a, b,$ and $n$ be positive integers with both $a$ and $b$ greater than or equal to $2$ and less than or equal to $2n$. Define an $a \times b$ cell loop in a $2n \times 2n$ grid of cells to be the $2a + 2b - 4$ cells that surround an $(a - 2) \times (b - 2)$ (possibly empty) rectangle of cells in the grid. For example, the following diagram shows a way to partition a $6 \times 6$ grid of cells into $4$ cell loops.

| P P P P | Y Y |
| P | R R | P | Y | Y |
| P | R R | P | Y | Y |
| P P P P | Y | Y |
| G G G G | Y | Y |
| G G G G | Y Y |

Find the number of ways to partition a $10 \times 10$ grid of cells into $5$ cell loops so that every cell of the grid belongs to exactly one cell loop.
    """


    model = get_gemini_model()
    messages = [
        HumanMessage(content=puzzle_text)
    ]

    # 直接调用模型
    response = await model.invoke(messages)
    print("=== 直接调用结果 ===")
    print(response.content)


if __name__ == "__main__":
    
    import asyncio
    asyncio.run(async_test())