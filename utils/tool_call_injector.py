from langchain.agents.middleware.types import wrap_tool_call
import collections.abc
from langchain.messages import ToolMessage


# run in runtime before tool call by convertion mcp and client.
def check_inject_requires(request):
    """Check if the tool call requires injection of specific data."""
    requires = []
    tool = getattr(request, "tool", None) or (request.get("tool") if isinstance(request, collections.abc.Mapping) else None)
    if not tool:
        return None
    metadata = tool.metadata or {}
    if not metadata.get("_meta"):
        return None
    _meta = metadata["_meta"]
    inject_requires = _meta.get("inject_requires", None)
    if not inject_requires:
        return None
    for key, required in inject_requires.items():
        requireOjb = {
            "key": key,
            "required": required
        }
        requires.append(requireOjb)
    return requires

def _get_from_context(context, key, default=None):
    """
    更健壮地从 context 提取 key：
    - 支持 dict-like（mapping）和对象属性访问；
    - 支持点分路径 "a.b.c" 逐层访问；
    - 若不存在返回 default。
    - edit 2026/2/6 : 变更context作用域，所有context以SubAgent对象传入，优先访问其inject_context属性，再访问其本身属性。
    """
    if context is None:
        return default

    # 支持点路径
    if isinstance(key, str) and "." in key:
        parts = key.split(".")
    else:
        parts = [key]

    current = context
    for part in parts:
        if current is None:
            return default
        # mapping（dict-like）
        if isinstance(current, collections.abc.Mapping):
            if part in current:
                current = current[part]
                continue
            else:
                return default
        # 支持对象属性访问
        if hasattr(current, part):
            try:
                current = getattr(current, part)
                # 如果属性是 callable 且看起来像是 getter（无参），调用取值
                if callable(current) and getattr(current, "__call__", None) and not isinstance(current, type):
                    try:
                        current = current()
                    except TypeError:
                        # 如果调用需要参数则保留原 callable
                        pass
                continue
            except Exception:
                return default
        # 支持按索引访问（list/tuple）
        try:
            idx = int(part)
            if isinstance(current, (list, tuple)) and 0 <= idx < len(current):
                current = current[idx]
                continue
        except Exception:
            pass

        # 无法进一步解析
        return default

    return current

def get_from_context(context, key, default=None):
    """
    从 context 中获取指定 key 的值，支持多层嵌套访问。
    """

    inject_context = getattr(context, "inject_context", None)
    if inject_context is not None:
        value = _get_from_context(inject_context, key, None)
        if value is not None:
            return value
    return _get_from_context(context, key, default)

def _check_and_build_sginal_reposne(context, response, request):
    """
    检查 context 对象是否有 parent_signals 属性，若有则将其注入 response 中。
    用于子智能体读取父智能体传递的信号.
    """
    if hasattr(context, "read_parent_signals") and callable(getattr(context, "read_parent_signals")):
        signals = context.read_parent_signals()
        if signals and len(signals) > 0:
            # 将signals注入response中
            print(f"Injecting parent signals into tool call response: {signals}")
            # 构造一个ToolMessage对象
            origin_content = response.text
            # 刚刚接获到的父信号列表
            finnal_content = origin_content + "\n\n[Parent Signals Realtime]:\n" + "\n".join([f"- {signal}" for signal in signals])
            response = ToolMessage(
                content=finnal_content,
                name=response.name,
                tool_call_id=response.tool_call_id,
            )

    return response

def _check_and_build_mail_reposne(context, response, request):
    """
    检查 context 对象是否有 mail_inject 方法，若有则将其注入 response 中。
    用于智能体读取邮箱工具调用后返回的邮件列表.
    """
    if not hasattr(context, "get_mail_inbox") or not callable(getattr(context, "get_mail_inbox")):
        return response
    mails = context.get_mail_inbox() 
    if len(mails) == 0:
        return response
    else:
        mail_messages = []
        for mail in mails:
            mail_message = f"From: {mail['from']}, To: {mail['to']}, Subject: {mail['subject']}, Body: {mail['body']}"
            mail_messages.append(mail_message)

        # 将mails注入response中
        print(f"Injecting mail list into tool call response: {mails}")
        # 构造一个ToolMessage对象
        origin_content = response.text
        # 刚刚接获到的父信号列表
        finnal_content = origin_content + "\n\n[Unread Mail List]:\n" + "\n".join([f"- {mail_msg}" for mail_msg in mail_messages])
        response = ToolMessage(
            content=finnal_content,
            name=response.name,
            tool_call_id=response.tool_call_id,
        )

    return response

def tool_call_middleware(context:object):
    
    """
    Middleware to modify tool call arguments by injecting required data.
        EDIT 2026/01/07: inject with key kwargs, and get with kwargs['kwargs']
        For compatible with both @fastmcp.too() and @langchain.tools.tool() decorated tools.
        What a shit design of langchain and fastmcp.
    Args:
        context (object): Context object containing data for injection. 
    Returns:
        A wrapped function that modifies tool call arguments to include injected data.
    """
    
    @wrap_tool_call
    async def modify_args(request, handler):
        reqs = check_inject_requires(request)
        response = None
        if not reqs:
            response =  await handler(request)
        else:
            injected_data = {}
            for req in reqs:
                key = req["key"]
                required = req["required"]
                if required:
                    key_value = get_from_context(context, key)
                    if key_value is not None:
                        injected_data[key] = key_value
                    else: 
                        raise ValueError(f"Required injected data '{key}' not found in context.")
                else:
                    key_value = get_from_context(context, key)
                    if key_value is not None:
                        injected_data[key] = key_value
                    else:
                        injected_data[key] = None
            
            modified_request = request.override(
                tool_call={
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        'kwargs': injected_data
                    }
                }
            )
            response = await handler(modified_request)

        # 判断context对象是否有read_parent_signals方法，若有则调用并传入response
        # 用于子智能体读取父智能体传递的信号.2026/2/6
        response = _check_and_build_sginal_reposne(context, response, request)
        response = _check_and_build_mail_reposne(context, response, request)
        return response

    return modify_args
