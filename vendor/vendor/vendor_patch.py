
import sys,os
import importlib.abc
import importlib.machinery

class VendorLoader(importlib.abc.Loader):
    def __init__(self, vendor_path):
        self.vendor_path = vendor_path
        self._module = None
    
    def create_module(self, spec):
        # 返回None让Python使用默认创建机制
        return None
    
    def exec_module(self, module):
        if self._module is not None:
            return self._module
        
        # 直接从文件读取并执行
        with open(self.vendor_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 在模块的命名空间中执行代码
        exec_globals = module.__dict__
        exec(code, exec_globals)
        
        self._module = module
        return module

class VendorFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_name, vendor_path):
        self.module_name = module_name
        # 从当前文件位置的目录拼接相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.vendor_path = os.path.join(current_dir, vendor_path)
    
    def find_spec(self, fullname, path, target=None):
        if fullname == self.module_name:
            return importlib.machinery.ModuleSpec(
                fullname,
                VendorLoader(self.vendor_path),
                origin=self.vendor_path
            )
        return None
"""
    a little pactch to relaod vendor modules from specific path.
"""
pathed_modules = [
    ('fastmcp.tools.tool','fastmcp/tools/tool.py'),
]


for module_name, vendor_path in pathed_modules:
    finder = VendorFinder(module_name, vendor_path)
    sys.meta_path.insert(0, finder)

from typing import Any, Callable, Optional
try:
    import langchain.tools as _lt
    from langchain.tools import tool as _original_tool
    _has_langchain = True
except Exception:
    _has_langchain = False
if _has_langchain:
    # 防止重复打补丁
    if getattr(_lt, "_patched_tool_with_metadata", False):
        # already patched
        pass
    else:
        def _deep_merge_dict(a: dict, b: dict) -> dict:
            """递归深度合并字典：b 覆盖/合并到 a，返回新字典（不修改输入）。"""
            result = {}
            # 先复制 a 的内容
            for k, v in (a or {}).items():
                result[k] = v
            # 合并 b
            for k, v in (b or {}).items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge_dict(result[k], v)
                else:
                    # 对于非 dict 的值，直接覆盖（包括 list、tuple、其他类型）
                    result[k] = v
            return result
        
        def _merge_metadata(target_obj: Any, new_meta: dict):
            if new_meta is None:
                return
            try:
                existing = getattr(target_obj, "metadata", None) or {}
                # 如果传入的 new_meta 看起来像直接的 inject_requires（即未包装到 _meta），则做兼容
                if " _meta" not in new_meta and "_meta" not in new_meta and "inject_requires" in new_meta:
                    new_meta = {"_meta": new_meta}
                merged = _deep_merge_dict(existing, new_meta)
                setattr(target_obj, "metadata", merged)
            except Exception:
                try:
                    setattr(target_obj, "metadata", new_meta)
                except Exception:
                    pass

        def tool(*t_args, **t_kwargs):
            """
            Replacement decorator that supports:
            - @tool
            - @tool(...)
            - @tool(metadata={...}) 或 @tool(meta={...})
            兼容原始 langchain.tools.tool 的行为，并在装饰后对象上写入 .metadata。
            """
            # 支持传入 meta 参数名或 metadata 参数名
            meta = t_kwargs.pop("meta", None)
            metadata = t_kwargs.pop("metadata", None)
            # 如果用户直接传入 meta 为 inject_requires dict（未包裹 _meta），自动包裹
            if metadata is None and meta is not None:
                if isinstance(meta, dict) and "_meta" not in meta and "inject_requires" in meta:
                    metadata = {"_meta": meta}
                else:
                    metadata = meta

            # 情形一：直接作为 @tool 使用，t_args[0] 为函数
            if t_args and callable(t_args[0]) and not t_kwargs:
                func = t_args[0]
                decorated = _original_tool(func)
                _merge_metadata(decorated, metadata)
                return decorated

            # 情形二：以参数形式使用，返回真正的 decorator
            def decorator(func: Callable):
                decorated = _original_tool(*t_args, **t_kwargs)(func)
                _merge_metadata(decorated, metadata)
                return decorated

            return decorator

        # 替换模块内的 tool 名称
        _lt.tool = tool
        # 标记为已打补丁
        _lt._patched_tool_with_metadata = True