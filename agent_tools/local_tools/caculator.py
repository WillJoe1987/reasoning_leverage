import math
import random
import io
from typing import Dict, Any
from contextlib import redirect_stdout
import ast
import os,sys
import multiprocessing
import traceback
import time
from langchain.tools import tool
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class SafeMathEvaluator:
    """安全的数学表达式计算器"""
    
    def __init__(self):
        # 安全的白名单函数
        self.safe_functions = {
            # 基础数学
            'abs': abs, 'round': round, 'pow': pow, 'divmod': divmod,
            'max': max, 'min': min,
            
            # 数学运算
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
            'log10': math.log10, 'log2': math.log2,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'degrees': math.degrees, 'radians': math.radians,
            'ceil': math.ceil, 'floor': math.floor,
            'factorial': math.factorial, 'gcd': math.gcd,
            'isclose': math.isclose,
            
            # 常量
            'pi': math.pi, 'e': math.e, 'inf': math.inf, 'nan': math.nan,
        }
        
        # 支持的运算符
        self.allowed_operators = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.And, ast.Or, ast.Not, ast.In, ast.NotIn
        }
    
    def evaluate(self, expression: str, variables: Dict[str, Any] = None) -> Any:
        """
        安全地计算数学表达式
        
        Args:
            expression: 数学表达式字符串
            variables: 可选变量字典
            
        Returns:
            计算结果
        """
        try:
            # 解析为AST
            tree = ast.parse(expression.strip(), mode='eval')
            
            # 检查AST是否安全
            self._check_node(tree.body)
            
            # 合并变量和安全函数
            namespace = {**self.safe_functions}
            if variables:
                namespace.update(variables)
            
            # 编译并执行
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {'__builtins__': {}}, namespace)
            
            return result
            
        except Exception as e:
            return f"表达式计算错误: {e}，请检查表达式的合法性和安全性。"
    
    def _check_node(self, node):
        """递归检查AST节点是否安全"""
        if isinstance(node, ast.Call):
            # 检查函数调用
            if not isinstance(node.func, ast.Name):
                raise ValueError("不允许的属性访问调用")
            if node.func.id not in self.safe_functions:
                raise ValueError(f"不允许的函数: {node.func.id}")
                
            # 检查参数
            for arg in node.args:
                self._check_node(arg)
            for kwarg in node.keywords:
                self._check_node(kwarg.value)
                
        elif isinstance(node, ast.Name):
            # 变量名在运行时检查，这里只做基本检查
            pass
            
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.allowed_operators:
                raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
            self._check_node(node.left)
            self._check_node(node.right)
            
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.allowed_operators:
                raise ValueError(f"不允许的一元运算符: {type(node.op).__name__}")
            self._check_node(node.operand)
            
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in self.allowed_operators:
                    raise ValueError(f"不允许的比较运算符: {type(op).__name__}")
            self._check_node(node.left)
            for comparator in node.comparators:
                self._check_node(comparator)
                
        elif isinstance(node, ast.BoolOp):
            if type(node.op) not in self.allowed_operators:
                raise ValueError(f"不允许的布尔运算符: {type(node.op).__name__}")
            for value in node.values:
                self._check_node(value)
                
        elif isinstance(node, ast.IfExp):
            self._check_node(node.test)
            self._check_node(node.body)
            self._check_node(node.orelse)
            
        # 允许的常量类型
        elif isinstance(node, (ast.Constant, ast.Num, ast.Str, ast.List, ast.Tuple, ast.Dict, ast.Set)):
            pass
            
        else:
            raise ValueError(f"不允许的语法结构: {type(node).__name__}")

class IterativeTaskSolver:
    """迭代任务求解器。使用 SafeMathEvaluator 作为底层计算引擎，安全地执行声明式迭代任务。"""
    
    def __init__(self, evaluator: SafeMathEvaluator):
        self.evaluator = evaluator
        self.max_iterations = 10000  # 安全上限，防止无限循环
    
    def solve(self, task: dict) -> dict:
        """
        执行一个迭代计算任务。
        
        Args:
            task (dict): 任务描述字典，包含以下键：
                - `expression`: (str) 要计算的表达式，可包含变量如 `x`。
                - `variable`: (str) 迭代变量的名称，如 `"n"`。
                - `range`: (dict) 定义迭代范围。支持两种格式：
                    1) `{"start": 0, "stop": 10000, "step": 1}` (stop不包含)
                    2) `{"values": [0, 1, 2, 3, ...]}` (直接指定值列表)
                - `condition`: (str, 可选) 一个条件表达式。只有满足此条件的迭代结果才会被收集。
                - `aggregate`: (str, 可选) 聚合方式。可以是 `"list"`(默认，返回所有结果),
                              `"count"`(只计数), `"sum"`, `"min"`, `"max"`等。
        
        Returns:
            dict: 包含 `results` (列表或聚合值) 和 `summary` (元数据) 的字典。
        """
        try:
            # 1. 提取任务参数
            expr = task['expression']
            var_name = task['variable']
            range_spec = task['range']
            condition = task.get('condition')
            aggregate = task.get('aggregate', 'list')
            
            # 2. 生成迭代值序列
            if 'values' in range_spec:
                values = range_spec['values']
                if len(values) > self.max_iterations:
                    values = values[:self.max_iterations]
            else:
                start = range_spec.get('start', 0)
                stop = range_spec.get('stop')
                step = range_spec.get('step', 1)
                if stop is None:
                    raise ValueError("range 必须包含 'stop' 或 'values'")
                # 计算迭代次数，确保不超过上限
                num_steps = (stop - start) // step
                if num_steps > self.max_iterations:
                    stop = start + self.max_iterations * step
                values = list(range(start, stop, step))
            
            # 3. 执行迭代计算
            results = []
            for value in values:
                # 构造变量环境
                variables = {var_name: value}
                
                # 如果有条件，先检查条件
                if condition:
                    try:
                        cond_result = self.evaluator.evaluate(condition, variables)
                        if not cond_result:
                            continue
                    except Exception as e:
                        # 条件计算错误，跳过此次迭代
                        continue
                
                # 计算主表达式
                try:
                    result = self.evaluator.evaluate(expr, variables)
                    results.append({
                        'value': value,
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'value': value,
                        'result': f"Error: {e}",
                        'error': True
                    })
            
            # 4. 根据聚合方式处理结果
            final_result = None
            if aggregate == 'list':
                final_result = [item['result'] for item in results if not item.get('error')]
            elif aggregate == 'count':
                final_result = len([item for item in results if not item.get('error')])
            elif aggregate == 'sum':
                valid_results = [item['result'] for item in results if not item.get('error') 
                                 and isinstance(item['result'], (int, float))]
                final_result = sum(valid_results)
            elif aggregate == 'min':
                valid_results = [item['result'] for item in results if not item.get('error') 
                                 and isinstance(item['result'], (int, float))]
                final_result = min(valid_results) if valid_results else None
            elif aggregate == 'max':
                valid_results = [item['result'] for item in results if not item.get('error') 
                                 and isinstance(item['result'], (int, float))]
                final_result = max(valid_results) if valid_results else None
            else:
                final_result = [item['result'] for item in results if not item.get('error')]
            

            # 检查final_result是否包含错误信息，
            # 抽检第一条和最后一条，如果都是错误信息，则只返回一条错误信息即可
            error_prefix = "表达式计算错误"
            if isinstance(final_result, list) and len(final_result) > 1:
                first_item = final_result[0]
                last_item = final_result[-1]
                if (isinstance(first_item, str) and first_item.startswith(error_prefix) and
                    isinstance(last_item, str) and last_item.startswith(error_prefix)):
                    final_result = [first_item]

            # 5. 返回结构化结果
            return {
                'success': True,
                'aggregated_result': final_result,
                'details': {
                    'total_iterations': len(values),
                    'successful_calculations': len([r for r in results if not r.get('error')]),
                    'sample_results': results[:10]  # 只返回前10个作为示例
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class PureMathSandbox:
    """
    纯数学代码沙盒。用于安全地执行一段专注于数学计算与算法验证的Python代码片段。
    此沙盒严格禁止任何文件、网络、系统操作，并限制资源使用。
    """
    
    def __init__(self, timeout=2, max_iterations=100000, max_recursion_depth=10):
        self.timeout = timeout          # 执行超时（秒）
        self.max_iterations = max_iterations  # 最大循环迭代总数
        self.max_recursion_depth = max_recursion_depth # 最大递归深度
        
        # 允许导入的模块白名单（仅包含安全的数学模块）
        self.allowed_modules = {
            'math': math,
            'random': random,
        }
        
        # 允许的 AST 节点类型（白名单）
        self.allowed_nodes = {
            # 基础结构
            ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
            ast.Return, ast.Pass, ast.Break, ast.Continue, ast.Attribute,ast.JoinedStr, ast.FormattedValue,
            
            # 控制流
            ast.If, ast.While, ast.For,
            
            # 函数定义（允许，但受递归深度限制）
            ast.FunctionDef, ast.Lambda, ast.arguments, ast.arg,
            
            # 数据类型
            ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.GeneratorExp,
            
            # 运算
            ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare,
            
            ast.LtE, ast.Lt, ast.GtE, ast.Gt, ast.Eq, ast.NotEq, ast.Mod,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow,
            ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,ast.IfExp,ast.comprehension,
            ast.Slice,

            # 变量与常量
            ast.Name, ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant,
            ast.Load, ast.Store,
            
            # 其他必要节点
            ast.Call, ast.Subscript, ast.Index, ast.Slice,
        }
        
        # 允许的运算符
        self.allowed_operators = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub, ast.Not, ast.Invert,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.And, ast.Or,
        }

        self.dangerous_nodes = {
            ast.Import, ast.ImportFrom,     # 禁止导入
            ast.Global, ast.Nonlocal,       # 禁止全局声明
            ast.With,                       # 禁止上下文管理器
            ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith,  # 禁止异步
            ast.ClassDef,                   # 禁止类定义
            ast.Assert,                     # 禁止断言
            ast.Raise,                      # 禁止抛出异常
            ast.Try, ast.ExceptHandler,   # 禁止异常处理
            ast.Yield, ast.YieldFrom,       # 禁止生成器
        }
    
    def _execute(self, code: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        在沙盒中执行代码片段。
        
        Args:
            code: 要执行的Python代码字符串。应包含一个名为 `run` 的函数，或直接是可执行的语句。
            inputs: 输入变量字典，将在执行环境中可用。
            
        Returns:
            包含执行结果、输出和可能的错误信息的字典。
        """
        start_time = time.time()
        iteration_count = [0]  # 用于闭包中修改
        
        # 准备执行环境
        env = {
            '__builtins__': {
                # 仅保留最安全的几个内置函数
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'bool': bool,
                'str': str,
                'list': list,
                'tuple': tuple,
                'dict': dict,
                'set': set,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'pow': pow,
                'divmod': divmod,
                'all': all,
                'any': any,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'print': print,  # 允许打印输出
            },
            **self.allowed_modules,
        }
        
        if inputs:
            env.update(inputs)
        
        # 安全检查
        try:
            tree = ast.parse(code)
            self._check_ast(tree, iteration_count)
        except Exception as e:
            return {
                'success': False,
                'error_type': '安全检查失败',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
        
        # 执行代码
        try:
            # 动态监测循环的封装函数
            def execute_with_limits():
                local_iterations = [0]
                 # 创建字符串缓冲区捕获print输出
                output_buffer = io.StringIO()
                 # 重定向标准输出到缓冲区
                with redirect_stdout(output_buffer):
                    # 替换原生的 while/for 循环检测逻辑（此处简化演示）
                    # 实际实现需要更复杂的AST重写或字节码注入来计数循环
                    exec(compile(tree, '<sandbox>', 'exec'), env, env)
                    
                 # 获取捕获的print输出
                captured_print = output_buffer.getvalue()
                
                # 将捕获的print输出存入环境，方便后续访问
                env['_captured_print'] = captured_print
                # 原有的优先级逻辑
                if 'run' in env and callable(env['run']):
                    run_result = env['run']()
                    if run_result is not None:
                        return run_result
                    # 如果run返回None但修改了result，使用result
                    elif 'result' in env and env['result'] is not None:
                        return env['result']
                    
                if 'result' in env and env['result'] is not None:
                    return env['result']
                
            
            # 超时控制（简化版，实际需使用信号或线程）
            result = execute_with_limits()
            
            # 收集输出（约定输出变量为 `output` 或函数返回值）
            output = env.get('output', result)
            
            # 获取捕获的print输出
            captured_print = env.get('_captured_print', '')
            
            # 处理print输出长度限制（2048字符）
            if captured_print and len(captured_print) > 2048:
                # 取最后2048个字符
                captured_print = captured_print[-2048:]
                truncated = True
            else:
                truncated = False

            return {
                'success': True,
                'result': result,
                'output': output,
                'captured_print': captured_print,  # 原始的print输出
                'print_truncated': truncated,  # 是否被截断
                'execution_time': time.time() - start_time,
                'iterations': iteration_count[0],
                'env_keys': list(env.keys())  # 调试用
            }
            
        except TimeoutError:
            return {
                'success': False,
                'error_type': 'TimeoutError',
                'error_message': f'执行超过 {self.timeout} 秒限制',
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute(self, code: str, inputs: dict = None) -> dict:
        """
        在沙盒中执行代码片段，支持超时与print捕获。
        """
        start_time = time.time()
        iteration_count = [0]

        env = {
            '__builtins__': {
                'range': range, 'len': len, 'int': int, 'float': float, 'bool': bool,
                'str': str, 'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
                'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                'pow': pow, 'divmod': divmod, 'all': all, 'any': any, 'sorted': sorted,
                'enumerate': enumerate, 'zip': zip, 'isinstance': isinstance, 'print': print,
            },
            **self.allowed_modules,
        }
        if inputs:
            env.update(inputs)

        try:
            tree = ast.parse(code)
            self._check_ast(tree, iteration_count)
            compiled_code = compile(tree, '<sandbox>', 'exec')
        except Exception as e:
            return {
                'success': False,
                'error_type': '安全检查失败',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }

        def target(queue, code, env):
            try:
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):
                    exec(code, env, env)
                    run_result = None
                    if 'run' in env and callable(env['run']):
                        run_result = env['run']()
                        if run_result is None and 'result' in env:
                            run_result = env['result']
                    elif 'result' in env:
                        run_result = env['result']
                    else:
                        run_result = None
                captured_print = output_buffer.getvalue()
                output = env.get('output', run_result)
                queue.put({
                    'success': True,
                    'result': run_result,
                    'output': output,
                    'captured_print': captured_print,
                    'print_truncated': len(captured_print) > 2048,
                    'execution_time': time.time() - start_time,
                    'iterations': iteration_count[0],
                    'env_keys': list(env.keys())
                })
            except Exception as e:
                queue.put({
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': traceback.format_exc()
                })
        try:
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=target, args=(queue, compiled_code, env))
            p.start()
            p.join(self.timeout)
            if p.is_alive():
                p.terminate()
                return {
                    'success': False,
                    'error_type': 'TimeoutError',
                    'error_message': f'执行超过 {self.timeout} 秒限制',
                }
            if not queue.empty():
                result = queue.get()
                # 处理print截断
                if result.get('captured_print') and result.get('print_truncated'):
                    result['captured_print'] = result['captured_print'][-2048:]
                return result
            else:
                return {
                    'success': False,
                    'error_type': 'UnknownError',
                    'error_message': '未知错误，未获取到结果',
                }
        except Exception as e:
            return {
                'success': False,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }


    def _check_ast(self, node, iteration_count):
        """递归检查AST，确保只包含允许的节点和操作"""
        if type(node) in self.dangerous_nodes:
            raise ValueError(f'不允许的AST节点类型: {type(node).__name__}')
        
            # 检查循环（保持不变）
        if isinstance(node, (ast.While, ast.For)):
            iteration_count[0] += 1000  # 预估每次循环
        
        if iteration_count[0] > self.max_iterations:
            raise ValueError(f'预估循环次数超过限制: {self.max_iterations}')
        
        # 递归检查子节点
        for child in ast.iter_child_nodes(node):
            self._check_ast(child, iteration_count)
    
# 使用示例
global SME 

def get_calculator_tool():
    if 'SME' not in globals():
        global SME
        SME = SafeMathEvaluator()
    solver = IterativeTaskSolver(SME)

    @tool
    def tool_calculator(expression_line: str) -> Any:
        """
            安全计算**单行**数学表达式。支持多种数学运算和函数。

            ### 支持的运算符:
            - 算术: +, -, *, /, //, %, **
            - 比较: ==, !=, <, <=, >, >=
            - 逻辑: and, or, not
            - 位运算: &, |, ^, ~, <<, >> (可选添加)
            
            ### 支持的数学函数:
            - 基础: abs, round, pow, divmod, max, min, sum
            - 指数对数: sqrt, exp, log, log10, log2
            - 三角函数: sin, cos, tan, asin, acos, atan
            - 双曲函数: sinh, cosh, tanh
            - 其他: ceil, floor, factorial, gcd, radians, degrees
            
            ### 支持的常量:
            - pi, e
            
            ### 数据类型:
            - 数字: 整数(42), 浮点数(3.14), 复数(3+4j)
            - 序列: 列表([1,2,3]), 元组((1,2,3)), 字符串("hello")
            
            ### 限制:
            1. 不支持赋值语句(如 x=5)
            2. 不支持循环和条件语句(除三元表达式外)
            3. 不支持导入模块
            4. 只允许调用白名单中的函数
            5. 变量需要在调用前通过其他方式定义
            
            Args:
                expression: 数学表达式字符串，如 "3 + 4 * 2"
                
            Returns:
                计算结果，可能是数字、布尔值或其他数据类型
                
            Raises:
                ValueError: 表达式包含不安全操作或语法错误
        """

        print(f"Calculating expression=======: {expression_line}")

        return SME.evaluate(expression_line)

    @tool
    def tool_iterative_solve(task_description: str) -> str:
        """
        安全地执行一个迭代计算数学表达式。适用于需要扫描参数范围、验证大量数值或搜索满足条件的解的场景。
        
        例如：验证当 n 从 0 到 10000 时，某个表达式是否始终满足条件；或者搜索在某个范围内使表达式取极值的参数。
        
        Args:
            task_description (str): 一个 JSON 字符串，描述迭代任务。结构示例：
                {
                    "expression": "n*(n+1)/2",          # 要计算的表达式
                    "variable": "n",                    # 迭代变量名
                    "range": {                          # 迭代范围（二选一）
                        "start": 0, "stop": 100, "step": 1   # 或 "values": [1,2,3,5,7]
                    },
                    "condition": "result < 1000",       # 可选：过滤条件
                    "aggregate": "list"                 # 可选：聚合方式 (list, count, sum, min, max)
                }
        
        Returns:
            str: 格式化的结果报告，包括聚合结果和摘要统计。
        """
        try:
            import json
            task = json.loads(task_description)
            print(f"Received iterative task: {task}")
            # 验证必要字段
            required = ['expression', 'variable', 'range']
            for field in required:
                if field not in task:
                    return f"错误：任务描述中缺少必要字段 '{field}'。"
            
            # 执行任务
            result = solver.solve(task)
            print(f"Iterative task result: {result}")
            
            # 格式化输出
            if not result['success']:
                return f"任务执行失败：{result['error']}"
            
            output_lines = [
                "=" * 50,
                "迭代任务完成报告",
                "=" * 50,
                f"表达式: {task['expression']}",
                f"变量: {task['variable']}",
                f"聚合方式: {task.get('aggregate', 'list')}",
                "",
                f"聚合结果: {result['aggregated_result']}",
                "",
                "执行摘要:",
                f"  总迭代次数: {result['details']['total_iterations']}",
                f"  成功计算次数: {result['details']['successful_calculations']}",
            ]
            
            # 如果有条件，显示条件
            if 'condition' in task:
                output_lines.insert(4, f"条件: {task['condition']}")
            
            # 显示样本结果（前几个）
            sample = result['details'].get('sample_results', [])
            if sample:
                output_lines.append("\n样本结果（前10个）:")
                for item in sample:
                    status = "✓" if not item.get('error') else "✗"
                    output_lines.append(f"  {status} {task['variable']}={item['value']} → {item['result']}")
            
            output_lines.append("=" * 50)
            return "\n".join(output_lines)
            
        except json.JSONDecodeError:
            print("Failed to decode JSON task description.")
            return "错误：无法解析任务描述，请确保它是有效的 JSON 格式。"
        except Exception as e:
            print(f"Error during iterative task solving: {e}")
            return f"工具调用出错：{e}"

    # return tool_iterative_solve

    return [tool_calculator, tool_iterative_solve]


def get_sandbox_tool():
    sandbox = PureMathSandbox(timeout=30, max_iterations=100000, max_recursion_depth=20)
    
    @tool
    def tool_math_sandbox(code: str, inputs_json: str = '{}') -> str:
        """
        在严格受限的安全沙盒中执行一段数学计算或算法验证代码。
        
        此沙盒仅用于验证数学逻辑和算法正确性，禁止任何文件、网络或系统操作。
        代码应专注于纯计算，并返回明确的结果。
        可调用的模块仅限于 `math` 、`random`和基础内置函数；
        可用资源：
            - 内置函数：'range','len','int','float','bool','str','list','tuple','dict','set','sum','min','max','abs','round','pow','divmod','all','any','sorted','enumerate','zip','isinstance','print'
            - 数学模块：math (无需导入，直接使用)
            - 随机数模块：random (无需导入，直接使用random.randint()等)
        
        ❌ 禁止操作（以下相关代码会直接导致抛出异常）：
            - 任何形式的导入（import/from ... import ...）
            - 文件、网络、系统操作
            - 类定义、异常处理、异步操作、生成器
            - 禁止使用print函数输出内容（可通过返回值传递结果）

        Args:
            code (str): Python代码片段。**必须**定义一个 `run()` 函数或直接计算并赋值给 `result` 变量，。
            inputs_json (str): 可选，JSON格式的输入变量字典。例如：'{"n": 100, "values": [1,2,3]}'
        
        Returns:
            str: 格式化的执行结果报告。包含成功/失败状态、计算结果或错误详情。
            
        **代码示例**：
            ======================= 📊 完整示例 =======================
    
            ✅ 正确示例1（推荐使用run函数）：
                def run():
                    total = 0
                    for i in range(1, 101):
                        total += i
                    return total  # 🚨 必须有return！
            
            ✅ 正确示例2（使用result变量）：
                result = 42  # 直接设置result变量
            
            ✅ 正确示例3（覆盖输出）：
                def run():
                    return 42
                output = "我覆盖了run函数的返回值"  # ⚠️ 这会覆盖run()的返回值
            
            ❌ 错误示例1（无输出）：
                total = 0
                for i in range(1, 101):
                    total += i  # ❌ 计算了结果但没有赋值给result或通过run返回
            
            ❌ 错误示例2（只有print）：
                print(42)  # ❌ print不是输出方式，仅用于调试

            ❌ 错误示例3（使用import语句）：
                import random  # ❌ 禁止使用import语句，random和math模块已内置，可直接使用。
            
        
        **限制**：
        - 执行时间上限：3秒
        - 循环总次数上限：100000次
        - 递归深度上限：20层
        - 仅允许 math、random 模块和基础内置函数
        """
        try:
            import json
            inputs = json.loads(inputs_json) if inputs_json.strip() else {}
        except json.JSONDecodeError:
            return "错误：inputs_json 参数必须是有效的JSON字符串"
        
        print("Starting math sandbox execution..."+code)
        print(f"Executing code in math sandbox with inputs: {inputs}")

        result = sandbox.execute(code, inputs)
        
        print(f"Math sandbox execution result: {result}")

        # 格式化输出
        output_lines = [
            "=" * 40,
            "数学沙盒执行报告",
            "=" * 40,
            f"状态: {'成功' if result['success'] else '失败'}",
            f"执行时间: {result.get('execution_time', 0):.3f}秒",
        ]
        
        if result['success']:
            output_lines.extend([
                f"返回值: {result.get('result', 'None')}",
                f"输出内容: {result.get('output', 'None')}",
                f"预估循环次数: {result.get('iterations', 0)}",
            ])
        else:
            output_lines.extend([
                f"错误类型: {result.get('error_type', 'Unknown')}",
                f"错误信息: {result.get('error_message', 'Unknown')}",
            ])
            if 'traceback' in result:
                output_lines.append(f"追踪信息:\n{result['traceback']}")
        
        output_lines.append("=" * 40)
        return "\n".join(output_lines)
    
    return [tool_math_sandbox]

if __name__ == "__main__":
    tool = get_sandbox_tool()[0]

    code = """
def grundy_by_depth(n):
    n = math.pow(2, n)
    if n <= 4:
        return 0
    depth = 0
    while n >= 5:
        if n % 2 == 0:
            n = n // 5
        else:
            n = n // 5 + 1
        depth += 1
    return 1 if depth % 2 == 1 else 0

def run():
    results = [grundy_by_depth(i) for i in range(101, 201)]
    return {'first_100': results}
"""
    result = tool.func(code=code)
    print(result)