
def get_key_injects(kwargs:dict, key:str):
    """
        检查请求中的工具调用，获取需要注入的元数据要求列表。
        返回格式：[{ "key": key, "required": bool }, ...]
        若无注入要求，返回空列表。
    """
    return get_all_injects(kwargs).get(key, None)

def get_all_injects(kwargs:dict):
    """
        检查请求中的工具调用，获取所有需要注入的元数据要求。
        返回格式：{ key: value, ... }
    """
    args = kwargs.get("kwargs", {})
    return args