from langchain.agents.middleware.types import wrap_model_call
from langchain.messages import AIMessage, HumanMessage, ToolMessage


_filter_tools = ["shared_blackboard","yellow_page_book"]

_filter_tools_args = {
    "shared_blackboard": {
        "action" : ["read"]
    }
}

def indexOf(a, b):
    "Return the first index of b in a."
    for i, j in enumerate(a):
        if j is b or j == b:
            return i
    else:
        return -1

def _get_first_tool_call(tool_name, messages):
    # 基于OPENAI的协议，所以，toolcall和toolmessage是严格配对的；
    # 所以，这里就简化一下逻辑吧，不在搞什么复杂的栈结构了；
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            index = indexOf(messages, msg)
            pre_msg = messages[index - 1] if index > 0 else None
            if not _check_tool_call_args(pre_msg):
                continue
            if index > 0:
                return index - 1
    return None

def _check_and_remove_tool_calls(tool_name, messages):
    while True:
        index = _get_first_tool_call(tool_name, messages)
        if index is not None:
            # 移除 tool call 和 tool message
            # TODO ，后续在考虑对tool_call_id进行校验
            messages.pop(index)       # tool call
            messages.pop(index)       # tool message (after deletion, same index)
        else:
            break
    return messages

def _check_tool_call_args(aimessage):
    """Check if the tool call arguments match the filter criteria."""
    if not isinstance(aimessage, AIMessage):
        return False
    additional_kwargs = getattr(aimessage, "additional_kwargs", {})
    tool_calls = aimessage.tool_calls
    for tool_call in tool_calls:
        args = tool_call.get("args", {})
        tool_name = tool_call.get("name", "")
        if tool_name in _filter_tools_args:
            required_args = _filter_tools_args[tool_name]
            for key, expected_value in required_args.items():
                actual_value = args.get(key)
                if actual_value in expected_value:
                    return True
        elif tool_name in _filter_tools and tool_name not in _filter_tools_args:
            return True
    return False

def _filter_request(request):
    """Filter the model call request if needed."""
    # 判断最近一条数据，是否是 ToolMessage，且name为：shared_blackboard;
    messages = request.messages if hasattr(request, "messages") else request.get("messages", [])
    if isinstance(messages, list) and len(messages) > 0:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage) and last_message.name in _filter_tools:
            # 如果消息太短，直接返回原 request
            if len(messages) <= 2:
                return request
            
            # 判断倒数第二条消息，是否是对应的 tool call，且参数匹配；
            second_last_message = messages[-2]
            if not (_check_tool_call_args(second_last_message)):
                return request

            # 对除最后两条外的前缀进行过滤，然后把前缀与最后两条拼回去
            prefix = messages[:-2]
            # 移除消息列表中，除最后一组call和toolmessage意外的其他shared_blackboard的call和toolmessage
            filtered_messages = _check_and_remove_tool_calls(last_message.name, list(prefix))
            return request.override(messages = filtered_messages + messages[-2:])
    return request

def model_call_middleware(context:object):
    
    @wrap_model_call
    async def modify_model_call(request, handler):
        """Middleware to modify model calls to fix tool_calls format in responses."""
        request = _filter_request(request)
        response = await handler(request)
        return response

    return modify_model_call