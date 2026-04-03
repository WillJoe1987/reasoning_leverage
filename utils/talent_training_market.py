import re
import os,sys
from pathlib import Path
from typing import Dict, Any, List
import uuid
import asyncio
from datetime import datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import vendor.vendor_patch  # noqa: E402
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage

custom_path = Path(project_root) / "agent_checkpoints"
custom_path.mkdir(exist_ok=True)
from utils.file_check_point_saver import FileByteSaver

from utils.tool_call_injector import tool_call_middleware,check_inject_requires,get_from_context
from utils.model_call_injector import model_call_middleware
from langgraph.checkpoint.memory import MemorySaver

import traceback


# 今读史于此，目叔孙通传，方知制礼之大矣！
# 史者，约制千秋；礼者，约制四维。
# 其势之成也，奔行自若，上不能免，下无可避！
# 这种自组织的文明，真不是随便能学来的！
# 所以，Root节点可以替换，也可以没有，它存在不存在关系都不大；
# 起决定作用的是整个网络！
# 1、太子早定；2、沉淀礼制（树形注意力）；


class SubAgent:

    STOP_SIGNAL = "<FINISH_SIGNAL>"
    """私有的子智能体类 - 每个智能体都有自己的实例"""
    
    def __init__(self, manager: 'AgentManager', task_key_target: str, task_description: str, enable_hiring: bool , 
                 depth: int, force_order: bool = False, parent_id: str = None, ask_boss_tool=None, base_tools: List = None):
        self.agent_id = f"emp_{uuid.uuid4().hex[:8]}"
        self.checkpointer = FileByteSaver(str(custom_path))
        self.manager = manager  # 父Manager
        self.task_key_target = task_key_target
        self.task_description = task_description
        self.depth = depth
        self.parent_id = parent_id or False  # 父智能体ID
        self.status = "created"
        self.hired_at = datetime.now().isoformat()
        self.results = None
        self.child_manager = None  # 子Manager（如果本智能体可以雇佣）
        self.ask_boss_tool = ask_boss_tool  # 向老板提问的工具
        self.config = {"configurable": {"thread_id": f"subagent_{self.agent_id}"},"recursion_limit": 1000}
        self.base_tools = base_tools or []
        self._agent = None  # 实际的智能体实例
        self.enable_hiring  = enable_hiring
        self.llm_client = self.manager.llm_client
        self.force_order = force_order
        self.initial_tools = [] # 初始工具集（仅根智能体使用）
        self.last_response = None
        self.inject_context = False  # 注入的上下文数据
        self.parent_signals = []  # 父智能体传递的信号列表

    async def execute(self) -> Dict[str, Any]:
        """执行智能体任务"""
        self.status = "working"
        self.started_at = datetime.now().isoformat()
        
        async def _loop():
            try:
                # 准备工具：基础工具 + 子Manager工具（如果有）
                tools_for_agent = []
                if self.parent_id and self.base_tools:
                    tools_for_agent.extend(self.base_tools)
                elif not self.parent_id and self.initial_tools:
                    tools_for_agent.extend(self.initial_tools)
                
                # 如果本智能体可以雇佣，创建子Manager并添加其工具
                if self.enable_hiring and self.depth < self.manager.max_depth - 1:
                    self.child_manager = AgentManager(
                        llm_client=self.llm_client,
                        base_tools=self.base_tools or [],
                        max_depth=self.manager.max_depth,
                        depth=self.depth + 1,
                        parent_id=self.agent_id,
                        root_manager=self.manager.root_manager  # 指向根Manager
                    )
                    self.child_manager.set_owner(self)
                    tools_for_agent.extend(self.child_manager.get_tools())
                
                if self.ask_boss_tool:
                    tools_for_agent.append(self.ask_boss_tool)

                tools_for_agent.extend(self.manager.shared_tools(self))
                # 如果未开启邮箱服务，则注入为空工具集，避免智能体调用时报错
                tools_for_agent.extend(self.manager.mail_service_tools(self))

                # 创建系统提示
                self.system_prompt = self._generate_system_prompt()
                
                print(f"SubAgent {self.agent_id} System Prompt:\n{self.system_prompt}\n{'-'*50}\n")
                
                self.checkpointer.set_system_prompt(self.system_prompt)
                print("Creating tool_call_middleware with context:", self.inject_context)
                tool_call_middleware_instance = tool_call_middleware(self)
                model_call_middleware_instance = model_call_middleware(self.inject_context or self)
                # 创建智能体
                self._agent = create_agent(
                    model=self.llm_client,
                    tools=tools_for_agent,
                    system_prompt=self.system_prompt,
                    checkpointer=self.checkpointer,
                    middleware=[tool_call_middleware_instance, model_call_middleware_instance],
                )
                
                # 执行任务
                start_time = datetime.now()
                self.last_response = await self._agent.ainvoke({
                    "messages": [{"role": "user", "content": self.task_description}]
                }, config=self.config)
                execution_time = (datetime.now() - start_time).total_seconds()

                _checking = self._check_tool_call_bad_case()

                while _checking:
                    print(f"SubAgent {self.agent_id} detected tool_call bad case, injecting fake response to retry LLM.")
                    self.last_response = await self._agent.ainvoke({
                        "messages": [_checking]
                    }, config=self.config)
                    _checking = self._check_tool_call_bad_case()

                check_child_time = 0

                # TODO check if all sub agents are completed before marking self as completed.  
                while self.child_manager and self.child_manager.check_all_completed() is False:
                    last_check = self._check_tool_call_bad_case()
                    if last_check:
                        print(f"SubAgent {self.agent_id} detected tool_call bad case, injecting fake response to retry LLM.")
                        self.last_response = await self._agent.ainvoke({
                            "messages": [last_check]
                        }, config=self.config)
                        continue
                    else:
                        print(f"SubAgent {self.agent_id} is waiting for child agents to complete...")
                        waiting_prompt = "你还有子智能体正在工作，请等待它们完成后整合所有结果分析，并重新审视你的回答，确保最终输出的结果是最佳答案。"
                        self.last_response = await self._agent.ainvoke({
                            "messages": [{"role": "user", "content": waiting_prompt}]
                        }, config=self.config)
                        check_child_time += 1
                        if self.depth > 0 and check_child_time > 5 :  # 等待超过一定次数，强制继续，避免死循环
                            print(f"SubAgent {self.agent_id} has been waiting for child agents for too long, forcing to continue.")
                            break

                        continue
                
                # 处理结果
                response_content = self._extract_response(self.last_response)
                
                while not self.STOP_SIGNAL in response_content :
                    last_check = self._check_tool_call_bad_case()
                    if last_check:
                        print(f"SubAgent {self.agent_id} detected tool_call bad case, injecting fake response to retry LLM.")
                        self.last_response = await self._agent.ainvoke({
                            "messages": [last_check]
                        }, config=self.config)
                        continue
                    else:
                        print(f"SubAgent {self.agent_id} did not receive STOP_SIGNAL, forcing re-invocation.")
                        continue_prompt = f"""为检测到结束信号，请继续输出，直到你认为任务完成时，输出
                        {self.STOP_SIGNAL}"""
                        self.last_response = await self._agent.ainvoke({
                            "messages": [{"role": "user", "content": continue_prompt}]
                        }, config=self.config)
                        response_content += self._extract_response(self.last_response)
                        continue
                
                response_content = response_content.replace(self.STOP_SIGNAL, "").strip()

                # 最终处理,TODO 先不做那么复杂了，
                # 理论上当机械逻辑判断还需要继续的进行交互时，应继续交互，
                # 直到机械逻辑层判断符合输出条件，再进行最终处理
                # 因此这里应该需要相似的循环逻辑，来切入机械逻辑判断
                handle_flag = self.final_handle(response_content)

                self.results = {
                    "response": response_content,
                    "execution_time": execution_time,
                    "completed_at": datetime.now().isoformat()
                }
                print(f"SubAgent {self.agent_id} completed in {execution_time:.2f} seconds.\nResult:\n{response_content}\n{'='*50}\n")

                self.status = "completed"
                self.completed_at = datetime.now().isoformat()
                
                if not self.parent_id:
                    self._log_total()

                return self.results
            
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                traceback.print_exc()
                print(f"SubAgent execute() {self.agent_id} failed with error: {e}")
                return {
                    "status": "failed",
                    "error": str(e)
                }
        try:
            self._execution_task = asyncio.create_task(_loop())
            return await self._execution_task
        except asyncio.CancelledError:
            return {"status": "cancelled"}

    async def continuer(self, add_infos:str = None) :

        async def _loop():
            """继续执行未完成的任务"""
            try:
                self.status = "working"
                start_time = datetime.now()
                last_check = self._check_tool_call_bad_case()
                if last_check:
                    print(f"SubAgent {self.agent_id} detected tool_call bad case, injecting fake response to retry LLM.")
                    self.last_response = await self._agent.ainvoke({
                        "messages": [last_check]
                    }, config=self.config)
                
                continue_prompt = f"""继续，{"以下是补充信息："+add_infos if add_infos else ""}"""
                self.last_response = await self._agent.ainvoke({
                    "messages": [{"role": "user", "content": continue_prompt}]
                }, config=self.config)
                execution_time = (datetime.now() - start_time).total_seconds()
                # 处理结果
                response_content = self._extract_response(self.last_response)
                if self.results is None:
                    self.results = {}
                    self.results.update({
                        "response": response_content,
                        "execution_time": execution_time,
                        "continued_at": datetime.now().isoformat()
                    })
                else:
                    self.results.update({
                        "response": self.results.get("response", "") + response_content,
                        "execution_time": self.results.get("execution_time",0) + execution_time,
                        "continued_at": datetime.now().isoformat()
                    })

                self.status = "completed"
                self.completed_at = datetime.now().isoformat()
                    
                if not self.parent_id:
                    self._log_total()

                return self.results
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                print(f"SubAgent continuer() {self.agent_id} failed with error: {e}")
                traceback.print_exc()
                return {
                    "status": "failed",
                    "error": str(e)
                }
        
        self._execution_task = asyncio.create_task(_loop())
        try:
            return await self._execution_task
        except asyncio.CancelledError:
            return {"status": "cancelled"}

    def cancel(self):
        """外部中断"""
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
            self.status = "cancelled"
            print(f"SubAgent {self.agent_id} cancelled")
        if self.child_manager:
            self.child_manager.cancel_all_agents()

    def final_handle(self, final_content: str):
        """To implement final handling logic if needed"""
        return True

    def _generate_system_prompt(self) -> str:
        """生成系统提示词"""
        hire_note = ""
        depth_note = f"你当前处于第{self.depth + 1}层,当然，最多{self.manager.max_depth}层。"
        if self.child_manager:
            hire_note = f"""\n\n你可以使用提供的管理工具来雇佣子智能体处理复杂子任务。雇佣能力包括：\n
            {self.child_manager.default_prompt()} 

            关于雇佣能力：
                请优先考虑老板的建议；
                当你可以将当前任务拆解成多个可并行的更小的子任务时，再考虑雇佣子智能体来协助完成；
                在你完整任务，给出响应之前，必须等待所有子智能体完成任务，并严格比对论证汇总；
        """
        
        system_prompt = f"""你是一个专门处理以下任务的专家智能体：
            你的ID: {self.agent_id}
            任务关键目标：{self.task_key_target}
            任务描述：{self.task_description}

            {self.heredity_strategy()}

            在做任何执行动作前，请先向你的老板进行必要的咨询，确保你有足够的信息和资源来完成任务。
            {hire_note}

            注意事项：动手前，请先询问你的老板；
            如果你咨询老板后，他建议你拒绝，你可以直接返回拒绝并说明理由。
            请专注完成分配给你的任务，提供详细、准确的分析结果；
            当你遇到遇到与其他人的冲突，或者将要输出时，也可以向老板咨询，寻求建议，确保你的行动与老板的整体目标一致。
            当你认为任务完成时，输出
                {self.STOP_SIGNAL}
        """

        return system_prompt
    
    def _extract_response(self, result: Any) -> str:
        """提取响应内容"""
        if hasattr(result, 'get') and isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                return last_msg.text if hasattr(last_msg, "text") else str(last_msg)
        return str(result)

    def _check_tool_call_bad_case(self):
        """
            检查self.last_response最后一条数据，是否是tool_call调用的消息；
            如果是，则生成一条伪装信息，该信息tool_call_id为原tool_call_id，内容为调用失败，并提示LLM重试继续；
        """
        if not self.last_response:
            return None
        if hasattr(self.last_response, 'get') and isinstance(self.last_response, dict):
            messages = self.last_response.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    # 检查content_blocks中是否有tool_call类型
                    for block in getattr(last_msg, "content_blocks", []):
                        if block.get('type') == 'tool_call':
                            # 生成伪装信息
                            fake_response = {
                                "role": "tool",
                                "content": f"工具调用失败，请重试。",
                                "tool_name": block.get('name'),
                                "tool_call_id": block.get('tool_call_id')
                            }
                            # 构造一个ToolMessage对象
                            fake_tool_message = ToolMessage(
                                content=fake_response["content"],
                                name=fake_response["tool_name"],
                                tool_call_id=fake_response["tool_call_id"]
                            )
                            return fake_tool_message
        
                    if last_msg.lc_attributes['invalid_tool_calls'] and len(last_msg.lc_attributes['invalid_tool_calls']) > 0:
                        invalid_call = last_msg.lc_attributes['invalid_tool_calls'][-1]
                        fake_response = {
                            "role": "tool",
                            "content": f"工具调用失败，请重试。",
                            "tool_name": invalid_call.get('name'),
                            "tool_call_id": invalid_call.get('id')
                        }
                        fake_tool_message = ToolMessage(
                            content=fake_response["content"],
                            name=fake_response["tool_name"],
                            tool_call_id=fake_response["tool_call_id"]
                        )
                        return fake_tool_message
        return False

    def heredity_strategy(self) -> str:
        return ""

    async def build_current_chatlist(self):
        # 获取当前历史消息列表
        copy = self._agent.copy()
        try:
            messages = copy.checkpointer.get_tuple({"configurable": {"thread_id": self.config['configurable']['thread_id']}})[1]['channel_values']['messages']
        except Exception as e:
            print(f"Error getting messages from checkpointer: {e}")
            messages = [] 
        messagelist = [{"role":"system", "content": self.checkpointer._system_prompt}]
        
        for mes in messages:
            if isinstance(mes, AIMessage):
                conent_blocks = mes.content_blocks
                contents = []

                for block in conent_blocks:
                    if block['type'] == 'text':
                        contents.append({
                            "type": "text",
                            "text": block['text']
                        })
                    elif block['type'] == 'tool_call':
                        contents.append({
                            "type": "tool_call",
                            "tool_name": block['name'],
                            "tool_input": block['args']
                        })

                messagelist.append({
                    "role": "assistant",
                    "content": contents
                })
            elif isinstance(mes, HumanMessage):
                messagelist.append({"role":"user", "content": mes.content})
            elif isinstance(mes, ToolMessage):
                messagelist.append({"role":"tool", "content": mes.content, "tool_name": mes.name, "tool_call_id": mes.tool_call_id})

        return messagelist

    def _log_total(self):
        print("\n" + "="*80)
        print("🏁 任务完成，打印完整智能体树")
        print("="*80 + "\n")
        print(f"******* total is {len(self.manager.root_manager.satisfy_sub_agents())}*******")
        self.manager.root_manager.print_sub_tree()
        print("*****************************************") 

    def add_parent_signal(self, signal: str, recursion = False):
        """添加来自父智能体的信号"""
        # signal为空则直接return
        if not signal:
            return

        signal_dict = {
            "from": self.parent_id,
            "signal": signal,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }

        self.parent_signals.append(signal_dict)
        # 递归传递信号给子智能体
        if recursion and self.child_manager:
            self.child_manager.recursion_signals(signal)

    def read_parent_signals(self) -> List[str]:
        """读取并标记父智能体的信号为已读"""
        signals = [s["signal"] for s in self.parent_signals if not s["read"]]
        for s in self.parent_signals:
            s["read"] = True
        return signals

    def get_mail_inbox(self) -> List[Dict[str, Any]]:
        """获取智能体的邮件列表"""
        return self.manager.get_mail_inbox(self.agent_id)

class AdvisorAgent:
    """顾问智能体 - 专门为雇佣动作提供咨询服务"""
    def __init__(self, manager: 'AgentManager', chatlist: List[Dict[str, Any]], enable_hiring: bool = False, depth: int = 0, max_depth: int = 6, base_tools: List = None):
        self.advisor_id = f"adv_{uuid.uuid4().hex[:8]}"
        self.manager = manager
        self.chatlist = chatlist
        self.enable_hiring = enable_hiring
        self.depth = depth
        self.max_depth = max_depth
        self.base_tools = base_tools or []
        self._agent = None  # 实际的智能体实例  
        self.last_response = None,
        self.advis_log = []

    def _get_hire_info(self) -> str:
        if self.enable_hiring and self.depth < self.max_depth - 1:
            return f"""当前处于第{self.depth}层, 向你咨询的智能体可以雇佣子智能体为它工作。
                因此，需要你要根据颗粒度和任务复杂度，判断是否建议它雇佣子智能体。
                请注意：雇佣子智能体会增加整体任务的复杂度和上下文消耗，请谨慎建议雇佣。
            """
        return f"""
            当前处于第{self.depth}层, 向你咨询的智能体**不可以**雇佣子智能体为它工作。
        """
    
    def _advisor_defination(self):
        advisor_define = """
            作为是雇主的影子顾问，负责根据此前的任务进展，回答下属智能体完成任务时的疑问，协助其对齐任务目标和颗粒度，面对雇员时，你就代表老板本人。
                你的职责包括：
                1. 熟读并理解和把握整体任务目标，并分析当前状态，你的回答聚焦于当前最新雇员的子任务的目标和边界，避免避免与雇主的整体任务混淆；
                2. 雇员本身不具备代码编写能力和环境操作能力，仅可通过雇主提供的工具进行任务执行；
                3. 如果任务信息不足，明确告知雇员拒绝执行，并说明理由，避免盲目行动；
                4. 如根据雇主的对话信息判断，雇主本人已足够具备完整任务的信息，则可建议下属智能体直接拒绝雇主；
                5. 雇员咨询时，根据已提供的工具，雇主是否在之前的任务进展中已调用过，综合判断是否还能提供增量信息，是否能够产生新的进展，避免重复劳动；
                6. 你所见的对话记录为雇主本人的对话记录，如发现有重复的能力调用记录时，该智能体仅雇主可见，当前子智能体是不可见的，请注意区分；
                当前雇主已提供给子智能体的工具包(子智能体可调用的工具，注意，可能于雇主智能体可见的工具集不同，提供咨询的时候注意区分。)：
            """
        tools_info = ""
        for tool in self.base_tools:
            tools_info += f"- **{tool.name}**: {tool.description}\n"
        advisor_define += f"子智能体可调用的工具如下：\n{tools_info}\n"
        return advisor_define
    
    def _build_project_background(self) -> str:
        advisore_system = "以下是雇佣动作发生时点，雇主自身执行整体任务的完整记录，请仔细阅读并理解：\n".join([f"{info}" for info in self.chatlist ])

        system_info = self.chatlist.pop(0)["content"]
        
        quest_info = self.chatlist[0]["content"]

        totalbackgroud = f"""
            - **BOSS的核心策略**：{system_info}
            - **BOSS的总体任务**: {quest_info}
            - {advisore_system}
            - 难以决策时，向上级顾问请求帮助。
        """
        return totalbackgroud

    def get_model(self):
        from data_descriptor.tool_model import get_reasoning_model, get_default_model,get_qwen_model,get_baidu_model,get_gpt_model,get_gemini_model
        # return get_gemini_model()
        return self.manager.llm_client or get_default_model()
        
    def create_tool(self):
        hireinfo = self._get_hire_info()
        advisor_define = self._advisor_defination()
        advisore_system = self._build_project_background()
        self.system_prompt = advisor_define + hireinfo + advisore_system

        self.add_info("system", self.system_prompt)
        owner = self.manager.owner
        preadvisor = owner.manager.agents[owner.agent_id]["advisor"] if owner and owner.manager and owner.manager.agents.get(owner.agent_id, {}).get("advisor", None) else None
        tools = []
        if preadvisor:
            ask_superior_tool = preadvisor.create_tool_for_subadvisor()
            tools.append(ask_superior_tool)
        tool_call_middleware_instance = tool_call_middleware(self)
        self._agent = create_agent(
            model= self.get_model(),
            tools=tools,
            system_prompt=self.system_prompt,
            checkpointer=None,
            middleware=[tool_call_middleware_instance]
        )

        @tool
        async def ask_boss(task_key_target:str, question: str) -> str:
            """
                向创建者（老板）提问，获取建议。
                参数：
                    - task_key_target: 任务的关键目标或主题。
                    - question: 具体的问题内容。
                返回：
                    老板的回答内容。
            """
            response_content = ""

            input = f"任务关键目标：{task_key_target}\n问题：{question}"
            input += "\n\n请基于此前的任务进展，提供准确、相关的指导和建议，帮助智能体顺利完成任务；如果任务信息不足，明确告知智能体拒绝执行，并说明理由。"

            print("Asking boss with input:", input)
            self.add_info("user", input)
            try:
                response = await self._agent.ainvoke({
                        "messages": [{"role": "user", "content": input}]
                })
                self.last_response = response
                response_content = self._extract_response(response)

                print("Boss response:", response_content)
            except Exception as e:
                response_content = f"Error while asking boss: {e}"
            finally:
                # 这里可以实现与老板的交互逻辑
                self.add_info("assistant", response_content)
                return response_content
            
        self._ask_tool = ask_boss
        return ask_boss, self

    def create_tool_for_subadvisor(self):

        @tool(
            meta={
                "inject_requires": {
                    "advisor_id": True
                }
            }
        )
        async def ask_superior(task_key_target:str, question: str, **kwargs) -> str:
            """
                向上级（老板的顾问）提问，获取之前的任务细节。
                参数：
                    - task_key_target: 任务的关键目标或主题。
                    - question: 具体的问题内容。
                返回：
                    上级的回答内容。
            """
            advisor_id = kwargs["kwargs"]["advisor_id"] or "unknown_advisor"
            system_prompt = f"""
                你是一个负责为下级顾问提供咨询服务的顾问智能体，负责根据此前的任务进展，回答下属顾问的疑问，协助其对齐任务目标和颗粒度。"""
            system_prompt += "\n".join([f"{info['role']}: {info['content']}" for info in self.advis_log ])
            input = f"任务关键目标：{task_key_target}\n问题：{question}"
            input += "\n\n请基于此前的任务进展，提供准确、相关的指导和建议，帮助智能体顺利完成任务；如果任务信息不足，明确告知智能体拒绝执行，并说明理由。"
            print("Asking superior with input:", input)
            response_content = ""
            try:
                llm_client = self.get_model()
                checkpointer = MemorySaver()
                agent_copy = create_agent(
                    llm_client,
                    tools=[],
                    system_prompt=system_prompt,
                    checkpointer=checkpointer,
                )
                self.add_info(advisor_id, input)
                thread_id = f"{advisor_id}_{self.advisor_id}_{datetime.now().timestamp()}"
                response = await agent_copy.ainvoke({
                        "messages": [{"role": "user", "content": input}]
                },{"configurable": {"thread_id": thread_id}})
                response_content = self._extract_response(response)
                print("Superior response:", response_content)
            except Exception as e:
                # 打印更详细的错误信息
                import traceback
                error_detail = traceback.format_exc()
                print(f"Detailed error in ainvoke:\n{error_detail}")
                response_content = f"Error while asking superior: {e}"
            finally:
                self.add_info("assistant_"+self.advisor_id, response_content)
                return response_content
            
        return ask_superior

    def _extract_response(self, result: Any) -> str:
        """提取响应内容"""
        if hasattr(result, 'get') and isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                # MCP >= 1.0
                return last_msg.text if hasattr(last_msg, "text") else str(last_msg)
        return str(result)

    async def __call__(self, task_key_target:str, question: str) -> str:
        return await self._ask_tool.ainvoke({
            "task_key_target": task_key_target,
            "question": question
        })

    def add_info(self, role:str, info: str):
        info_obj = {
            "role": role,
            "content": info,
            "timestamp": datetime.now().isoformat()
        }
        self.advis_log.append(info_obj)

class AgentManager:

    """
        智能体管理器 - 每个层次都有自己的实例
            增加一个inject_context的传递通路，在tools依赖注入企业上下文的时候，传递必要参数。
            now there need a advisor from creator's checkpoint realtime. 
            To answer and check the agents' works.
            Especially about 1. the environment info and the task requirement changes .
                                    and need to check if refuse to work if none more information.
                             2. the attentions about the agents' works.
                So, this is half of the mirros architecture.
                !!!And , the other half is to toolable back the sub agents' works to the creator agent.
    """
    
    def __init__(self, llm_client, base_tools: List = None, max_depth: int = 6,
                 depth: int = 0, parent_id: str = None, root_manager: 'AgentManager' = None):
        self.llm_client = llm_client
        self.base_tools = base_tools or []
        self.max_depth = max_depth
        self.depth = depth  # 当前Manager的深度
        self.parent_id = parent_id  # 父智能体ID
        self.root_manager = root_manager or self  # 根Manager
        self.inject_context = False  # 注入的上下文数据
        # 当前Manager管理的智能体
        self.agents: Dict[str, dict] = {}
        self.shared_blackboard_opened = False
        self.mail_service_opened = False
        self.signal_channel_opened = False
        # 工具实例
        self._tools = self._create_all_tools()
    
    def set_owner(self, owner: any):
        """设置当前Manager的所有者标识"""
        
        inject_context = []

        for tool in self.base_tools:
            reqs = check_inject_requires({"tool": tool})
            inject_context.extend(reqs or [])


        # [{
        #     "key": key,
        #     "required": True
        # }]
        inecject_obj = {}
        for item in inject_context:
            key = item["key"]
            value = item["required"]
            if value is True:
                context_value = get_from_context(owner.inject_context or {}, key)
                if context_value is None:
                    context_value = get_from_context(owner, key)
                
                if context_value is None:
                    raise ValueError(f"Required injected data '{key}' not found in owner context.")
                inecject_obj[key] = context_value
            else:
                context_value = get_from_context(owner.inject_context or {}, key)
                if context_value is None:
                    context_value = get_from_context(owner, key)
                
                if context_value is None:
                    print(f"WARNNING: Optional injected data '{key}' not found in owner context, skipping.")
                else:  
                    inject_context[key] = context_value
        self.inject_context = inecject_obj
        print(f"AgentManager inject_context: {self.inject_context}")
        self.owner = owner
        self.parent_id = owner.agent_id

    def get_owner(self) -> any:
        """获取当前Manager的所有者标识"""
        return getattr(self, 'owner', None)

    def is_root(self) -> bool:
        """检查当前Manager是否为根Manager"""
        return self.root_manager == self

    def open_signal_channel(self):

        """开启信号通道机制"""
        if not self.is_root():
            print("Only root manager can open signal channel.")
            return 
        self.signal_channel_opened = True

    def open_shared_blackboard(self):
        """开启共享黑板机制"""
        if not self.is_root():
            print("Only root manager can open shared blackboard.")
            return 
        self.shared_blackboard_opened = True
        self.shared_blackboard = {}

    def shared_tools(self, context:SubAgent) -> List:
        if not context or context is None:
            return []
        if not self.shared_blackboard_opened:
            if not self.is_root():
                return self.root_manager.shared_tools(context)
            else:
                return []
        
        agent_id = context.agent_id
        self.shared_blackboard.setdefault(agent_id, [])

        print(f"Creating shared_blackboard tool for agent {agent_id}, and his task_key_target is : {context.task_key_target}")
        #TODO 事实上，还是比较容易爆炸的，可暂时还没有更有诱导性的策略，让智能过程可以自主取生发演进。
        #     业务可以分配并通知其有限资源，尤其自主决策写入内容。
        #     还是不希望收窄子智能体的视野。
        @tool
        def shared_blackboard(action: str, content: str = None, task_key_target: str = None):
            """
            提供一个**正式的共识记录面板**，用于记录已达成共识的关键发现、最终结论和经过验证的公共知识。

            这块面板是智能体社区的权威记录空间，用于固化协作过程中产生的核心成果。只有在以下情况下才应使用共享面板：
            1. 经过多方验证的关键定理或证明
            2. 最终达成一致的数据分析结果
            3. 已被确认的任务分解结构和分工方案
            4. 影响全局的约束条件或边界发现

            **重要原则**：
            - 所有中间过程、猜想、疑问应通过邮箱进行私域讨论
            - 只有形成共识、经过验证的内容才应写入共享面板
            - 每个条目都应有明确的贡献者和验证状态标记
            - 面板内容应保持高度简洁、结构化和可引用

            参数:
                action: 希望执行的操作。
                    'write'：写入一条**已达成共识**的内容，需注明验证状态和贡献者；
                    'read'：读取所有已记录的共识条目；
                    'rewrite'：更新某条内容的状态(仅限于自己写入的内容），输入格式以#开头: #index#新内容
                    'delete'：归档已完成任务的条目（仅限于自己写入的内容），输入格式以#开头: #index
                    'append': 在原有内容基础上追加信息（可以在别人的内容上追加），输入格式以#开头: #agent_id#index#追加内容
                content: 需要记录的文本内容。必须遵循以下格式：
                    【标题】简洁描述核心内容
                    【状态】待验证/已确认/已归档
                    【贡献者】列出参与验证的智能体ID
                    【内容】清晰、结构化、无歧义的表述
                    【验证依据】引用邮箱讨论记录或实验数据。
                    如为rewrite: #index#新内容
                    如为delete: #index
                    如为append: #agent_id#index#追加内容
                task_key_target: 本条内容所属的关键任务目标；

            返回:
                执行读取操作时，将返回所有共识条目列表，按重要性排序；
                执行写入/更新操作时，将返回确认信息和唯一引用ID。
            """
            shared_blackboard_storage = self.shared_blackboard[agent_id]
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
                print(f"shared_blackboard read called with task_key_target: {task_key_target}")
                
                # readable_keys = [context.parent_id] + list(context.manager.agents.keys())
                # if context.child_manager:
                #     readable_keys += list(context.child_manager.agents.keys())
                readable_keys = list(self.shared_blackboard.keys())
                
                for agent_id_key in readable_keys:                    
                    if agent_id_key in self.shared_blackboard:
                        for index, entry in enumerate(self.shared_blackboard[agent_id_key]):
                            filtered_entries += f"agent_id: #{agent_id_key}#\n编号：#{index}#\n[{entry['timestamp']}] ({entry['task_key_target']}): {entry['content']}\n"
                print(f"shared_blackboard read returning entries:\n{len(filtered_entries)} characters")
                return filtered_entries
            elif action == "rewrite":
                if content is None:
                    return "Error: 'content' must be provided when action is 'rewrite'."
                try:
                    print(f"shared_blackboard rewrite called with content: {content}")
                    index_str, new_content = content.split("#", 2)[1:]
                    index = int(index_str)
                    if 0 <= index < len(shared_blackboard_storage):
                        shared_blackboard_storage[index]['content'] = new_content
                        shared_blackboard_storage[index]['timestamp'] = datetime.now().isoformat()
                        return f"Content at index {index} rewritten."
                    else:
                        return f"Error: Index {index} out of range."
                except Exception as e:
                    return f"Error processing rewrite action: {e}"
            elif action == "delete":
                if content is None:
                    return "Error: 'content' must be provided when action is 'delete'."
                try:
                    print(f"shared_blackboard delete called with content: {content}")
                    index_str = content.split("#", 2)[1]
                    index = int(index_str)
                    if 0 <= index < len(shared_blackboard_storage):
                        del shared_blackboard_storage[index]
                        return f"Content at index {index} deleted. now your total is {len(shared_blackboard_storage)}"
                    else:
                        return f"Error: Index {index} out of range."
                except Exception as e:
                    return f"Error processing delete action: {e}"
            elif action == "append":
                if content is None:
                    return "Error: 'content' must be provided when action is 'append'."
                try:
                    print(f"shared_blackboard append called with content: {content}")
                    if content.startswith('#'):
                        content = content[1:]
                    parts = content.split('#', 2)
                    if len(parts) != 3:
                        return "追加格式错误"  # 分割后不是三个部分，格式错误
                    
                    agent_id_key, index_str, append_content = parts
                    index = int(index_str)
                    if agent_id_key in self.shared_blackboard:
                        target_storage = self.shared_blackboard[agent_id_key]
                        if 0 <= index < len(target_storage):
                            target_storage[index]['content'] += "\n" + append_content
                            target_storage[index]['timestamp'] = datetime.now().isoformat()
                            return f"Content at index {index} for agent {agent_id_key} appended."
                        else:
                            return f"Error: Index {index} out of range for agent {agent_id_key}."
                    else:
                        return f"Error: Agent ID {agent_id_key} not found."
                except Exception as e:
                    return f"Error processing append action: {e}"
            else:
                return "Error: 'action' must be either 'write' or 'read' or 'rewrite' or 'delete' or 'append'."
        
        return [shared_blackboard]

    def open_email_service(self):
        """开启邮件服务工具"""
        if not self.is_root():
            print("Only root manager can open email service.")
            return 

        self.mail_service_opened = True
        self.mail_list = {}
        
    def mail_service_tools(self, context:SubAgent) -> List:
        
        if not context or context is None:
            return []
        if not self.mail_service_opened:
            if not self.is_root():
                return self.root_manager.mail_service_tools(context)
            else:
                return []
        
        agent_id = context.agent_id
        self.mail_list.setdefault(agent_id, [])

        @tool
        def yellow_page_book():
            """查询黄页的工具，返回你所在的任务中，所有智能体的ID和任务关键信息，方便你了解可以联系谁来协助你完成任务"""
            return self.yellow_pages()

        @tool
        async def send_email(to_agent_id: str, subject: str, body: str):
            """发送邮件的工具"""

            to_agent = self.lookup_agent(to_agent_id)
            if not to_agent:
                return f"Error: Agent ID {to_agent_id} not found."

            email_entry = {
                "timestamp": datetime.now().isoformat(),
                "to": to_agent_id,
                "subject": subject,
                "body": body,
                "from": agent_id,
                "read": False
            }
            self.mail_list[to_agent_id].append(email_entry)
            print(f"Email sent by agent {agent_id} at {email_entry['timestamp']} to {to_agent_id} with subject '{subject}'.")

            if to_agent.status == "completed" or to_agent.status == "failed":
                # to_agent.continuer("You have received new email. Please check your inbox for details.")
                asyncio.create_task(to_agent.continuer("You have received new email. Please check your inbox for details."))
            return f"Email sent to {to_agent_id} with subject '{subject}'."

        @tool
        def check_inbox_email():
            """检查收件箱的工具，返回你收到的邮件列表"""
            return "你有如下mail"

        return [yellow_page_book, send_email, check_inbox_email]

    def get_mail_inbox(self, agent_id: str) -> List[Dict[str, Any]]:
        """获取指定智能体的邮件列表"""
        if not self.is_root():
            return self.root_manager.get_mail_inbox(agent_id)
        
        if not self.mail_service_opened:
            print("Mail service is not opened.")
            return []

        mails_unread = [mail for mail in self.mail_list.get(agent_id, []) if not mail["read"]]
        for mail in mails_unread:
            mail["read"] = True
        print(f"Agent {agent_id} has {len(mails_unread)} new emails.")
        return mails_unread

    async def build_advisor(self, enable_hiring: bool):
        """为当前雇员设置顾问"""
        if self.owner is None:
            return 
        else:
            advisor = AdvisorAgent(
                manager=self,
                chatlist = await self.owner.build_current_chatlist(),
                enable_hiring=enable_hiring,
                depth=self.depth,
                max_depth=self.max_depth,
                base_tools=self.base_tools or []
            )
            return advisor.create_tool()

    def _extract_response(self, result: Any) -> str:
        """提取响应内容"""
        if hasattr(result, 'get') and isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                # MCP >= 1.0
                return last_msg.text if hasattr(last_msg, "text") else str(last_msg)
        return str(result)

    def register_agent(self, agent: SubAgent, advisor: any = None):
        """注册智能体到当前Manager"""

        if hasattr(self, 'inject_context'):
            print("Injecting context into agent:", self.inject_context, ": ", agent.agent_id)
            agent.inject_context = self.inject_context

        self.agents[agent.agent_id] = {
            "agent": agent, 
            "advisor": advisor
        }
    
    async def hire(self, task_key_target:str,  task_description: str, enable_hiring: bool = True, force_order: bool = False ) -> Dict[str, Any]:
        """
        雇佣一个智能体 - 在当前Manager的层次内
        """
        # 检查深度限制
        if self.depth >= self.max_depth - 1:
            return {
                "status": "error",
                "error": f"已达到最大深度限制 {self.max_depth}",
                "current_depth": self.depth,
                "max_depth": self.max_depth,
                "suggestion": "请简化任务或重新设计工作流"
            }
        
        # 检查重复任务（基于当前Manager内的任务）
        for agent in self.agents.values():
            if agent["agent"].task_description == task_description:
                return {
                    "status": "duplicate",
                    "message": "当前层次内已有相同任务",
                    "existing_agent_id": agent["agent"].agent_id,
                    "suggestion": "使用list_agents查看状态，或使用get_agent_output获取现有结果"
                }
        
        if self.owner:
            ask_boss_tool, advisor = await self.build_advisor(enable_hiring)
        
        # 创建子智能体
        agent = SubAgent(
            manager=self,
            task_key_target=task_key_target,
            task_description=task_description,
            depth=self.depth,  
            parent_id=self.parent_id,
            ask_boss_tool=ask_boss_tool,
            base_tools=self.base_tools or [],
            enable_hiring = enable_hiring,
            force_order=force_order
        )
        self.register_agent(agent, advisor)
        print("******* hire called, and total is *******")
        self.root_manager.print_sub_tree()
        print(len(self.root_manager.satisfy_sub_agents()))
        print("*****************************************")
        # 异步执行
        asyncio.create_task(
            agent.execute()
        )
        
        # return {
        #     "status": "hired",
        #     "agent_id": agent.agent_id,
        #     "message": f"智能体已创建 (深度: {agent.depth})",
        #     "depth": agent.depth,
        #     "can_hire": agent.depth < self.max_depth - 1
        # }

        return {
            "status": "hired",
            "agent_id": agent.agent_id,
            "message": f"智能体已创建"
        }
    
    def list_agents(self) -> Dict[str, Any]:
        """列出当前Manager管理的所有智能体"""
        agents_list = []
        for agent_id, item in self.agents.items():
            agent = item["agent"]
            agents_list.append({
                "id": agent_id,
                "task": agent.task_description[:50] + "..." if len(agent.task_description) > 50 else agent.task_description,
                "status": agent.status,
                "depth": agent.depth,
                "hired_at": agent.hired_at,
                "result_available": agent.results is not None,
                "has_child_manager": agent.child_manager is not None
            })
        
        return {
            "manager_depth": self.depth,
            "manager_parent": self.parent_id,
            "count": len(agents_list),
            "agents": agents_list,
            "statistics": {
                "created": len([a for a in agents_list if a["status"] == "created"]),
                "working": len([a for a in agents_list if a["status"] == "working"]),
                "completed": len([a for a in agents_list if a["status"] == "completed"]),
                "failed": len([a for a in agents_list if a["status"] == "failed"])
            }
        }
    
    async def get_output(self, agent_id: str, timeout: int = 30) -> Dict[str, Any]:
        """获取智能体的产出结果"""
        if agent_id not in self.agents:
            return {"error": f"未找到智能体: {agent_id}"}
        
        agent = self.agents[agent_id]["agent"]
        
        if agent.status == "failed":
            return {
                "status": agent.status,
                "agent_id": agent_id,
                "message": "智能体执行失败，无法获取结果",
                "depth": agent.depth
            }


        if agent.results:
            return {
                "status": agent.status,
                "agent_id": agent_id,
                "result": agent.results,
                "depth": agent.depth
            }
        
        # 等待任务完成
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if agent.results:
                return {
                    "status": agent.status,
                    "agent_id": agent_id,
                    "result": agent.results,
                    "depth": agent.depth,
                    "waited": (datetime.now() - start_time).total_seconds()
                }
            await asyncio.sleep(0.5)
        
        return {
            "status": "timeout",
            "agent_id": agent_id,
            "message": f"已等待 {timeout} 秒，尚未完成。",
            "agent_status": agent.status
        }
    
    async def wait(self, wait_seconds: int = 5) -> Dict[str, Any]:
        """等待当前Manager的智能体完成"""
        await asyncio.sleep(wait_seconds)
        
        status = self.list_agents()
        completed = status["statistics"]["completed"]
        
        return {
            "wait_time": wait_seconds,
            "manager_depth": self.depth,
            "total_agents": status["count"],
            "completed": completed,
            "working": status["statistics"]["working"]
        }
    
    def fire(self, agent_id: str) -> Dict[str, Any]:
        """解雇一个智能体 - 只能解雇当前Manager管理的智能体"""
        if agent_id not in self.agents:
            return {"error": f"未找到智能体: {agent_id}"}
        
        agent = self.agents[agent_id]["agent"]
        agent.cancel()

        # self.root_manager.shared_blackboard中，该智能体AIkey增加（fired）标记，表示已被解雇，其他智能体调用共享黑板工具时，可以看到该标记，并且不再读取该智能体的内容。
        if self.root_manager.shared_blackboard_opened and agent_id in self.root_manager.shared_blackboard:
            self.root_manager.shared_blackboard[agent_id] = [{"timestamp": datetime.now().isoformat(), "task_key_target": agent.task_key_target, "content": "该智能体已被解雇，相关内容不再可见", "fired": True}]

        if agent.child_manager:
            agent.child_manager._fire_all()
        del self.agents[agent_id]
        
        return {
            "status": "fired",
            "agent_id": agent_id,
            "message": "智能体已被解雇"
        }

    def _fire_all(self):
        """解雇当前Manager管理的所有智能体 - 只能解雇当前Manager管理的智能体"""
        for agent_id in list(self.agents.keys()):
            self.fire(agent_id)

    def get_tools(self) -> List[Any]:
        """获取当前Manager的管理工具"""
        return self._tools
    
    def _create_all_tools(self) -> List[Any]:
        """创建所有管理工具 - 这些工具只操作当前Manager的层次"""
        return [
            self._create_hire_tool(),
            self._create_fire_tool(),
            self._create_list_tool(),
            self._create_get_output_tool(),
            self._create_wait_tool(),
            self._create_continue_tool()
        ] + ([self._create_signal_tool()] if self.root_manager.signal_channel_opened else [])
    
    def _create_hire_tool(self):
        @tool
        async def hire_agent(task_key_target:str, task_description: str, enable_hiring: bool, force_order: bool , **kwargs) -> Dict[str, Any]:
            """
                雇佣一个智能体处理子任务。
                参数：
                    - task_key_target: 任务的关键目标或主题。
                    - task_description: 需要完成的具体任务描述.
                    - enable_hiring (bool, optional): 是否允许被雇佣的智能体继续雇佣其下属。若为 `False`，则该智能体无法发起新的雇佣。这用于定义原子任务或控制协作链深度。
                    - force_order (bool, optional): 雇佣的子智能体是否为同步智能体，只有当该子智能体执行完毕后，才能雇佣新的子智能体；当前存在非`completed`状态且同步的子智能体时，雇佣会失败；一般用于关键分析任务，或后续依赖任务结果的场景。
                注意：此雇佣操作仅在当前智能体的层次内有效。
                新创建的智能体将属于当前Manager管理。
            """

            print(f"hire_agent tool called with: task_key_target={task_key_target}, enable_hiring={enable_hiring}, force_order={force_order}")
            
            # 检查当前是否有未完成的同步智能体
            uncompleted_agents = [agent for agent in self.agents.values() if agent["agent"].status != "completed" and agent["agent"].force_order]
            if uncompleted_agents:
                    return {
                        "status": "error",
                        "error": f"当前存在未完成的智能体:{[agent['agent'].agent_id for agent in uncompleted_agents]}，无法雇佣新的智能体。",
                        "suggestion": "请等待所有同步子任务完成后再尝试雇佣新的智能体。"
                    }
            return await self.hire(task_key_target, task_description, enable_hiring, force_order=force_order)
        return hire_agent
    
    def _create_list_tool(self):
        @tool
        def list_agents() -> Dict[str, Any]:
            """
            查看当前层次内的所有智能体状态。
            
            返回的信息仅包含当前Manager管理的智能体。
            每个智能体可能有自己的子Manager和子智能体。
            """
            return self.list_agents()
        
        return list_agents
    
    def _create_get_output_tool(self):
        @tool
        async def get_agent_output(agent_id: str, timeout_seconds: int = 30) -> Dict[str, Any]:
            """
            获取当前层次内指定智能体的产出结果。
            
            只能获取当前Manager管理的智能体的结果。
            如果需要获取更深层次的结果，需要调用对应层次的管理工具。
            """
            return await self.get_output(agent_id, timeout_seconds)
        
        return get_agent_output
    
    def _create_wait_tool(self):
        @tool
        async def wait_for_agents(wait_seconds: int = 5) -> Dict[str, Any]:
            """
            等待当前层次内的智能体任务完成。
            
            此操作只影响当前Manager管理的智能体。
            更深层次的智能体由其自己的Manager管理。
            """
            return await self.wait(wait_seconds)
        
        return wait_for_agents
    
    def _create_continue_tool(self):

        @tool
        async def continue_agent(agent_id: str, add_infos:str = None) -> Dict[str, Any]:
            """
                此工具有如下三种情形，可用；但请注意，如该子智能体正在working状态中，则无法调用此工具：
                    1. 如子智能体由于输出窗口等原因，输出内容未完成，可以调用此工具让其继续执行当前层次内指定智能体的未完成任务;
                    2. 子智能体提到由于信息不足，而需要提供更多信息，待信息补充后，继续执行任务；
                    3. 子智能体基础任务完成，形成方法论或者同类问题处理思路，可在其上下文窗口范围内，委托其进行同类问题的处理。
                参数：add_infos: 额外提供给子智能体的信息内容，可选。
                返回：继续执行指定智能体的未完成任务，直到完成或再次中断
            """
            if agent_id not in self.agents:
                return {"error": f"未找到智能体: {agent_id}"}
            agent = self.agents[agent_id]["agent"]
            if agent.status != "completed" and agent.status != "failed":
                return {"error": f"子智能体 {agent_id} 当前任务正在执行，待其子任务完成后，再调用此工具继续执行。"}
            return await agent.continuer(add_infos)
        
        return continue_agent

    def _create_signal_tool(self):
        
        @tool
        async def signal_agent(agent_id: str, signal: str, recursion = False) -> Dict[str, Any]:
            """
                向指定直接下属智能体发送信号。
                参数：
                    - agent_id: 目标智能体的ID 或者 ALL。
                    - signal: 信号内容。
                    - recursion: 是否将信号传递给孙智能体，默认为False。仅ALL模式下有效。
                返回：信号发送结果。
            """
            print(f"signal_agent tool called with agent_id={agent_id}, signal={signal}, recursion={recursion}")
            if len(self.agents) == 0:
                return {"error": "你还没有雇佣过子智能体呢。"}

            if agent_id == "ALL":
                results = {}
                for aid, item in self.agents.items():
                    agent = item["agent"]
                    agent.add_parent_signal(signal, recursion=recursion)
                    results[aid] = "Signal sent."
                return results

            if agent_id != "ALL" and agent_id not in self.agents:
                return {"error": f"未找到智能体: {agent_id}"}
            agent = self.agents[agent_id]["agent"]
            # 只有ALL模式下，才递归传递信号
            agent.add_parent_signal(signal, False)
            return "Signal sent."
        
        return signal_agent

    def _create_fire_tool(self):

        @tool
        def fire_agent(agent_id: str) -> Dict[str, Any]:
            """ 
                解雇当前层次内的指定智能体。
                注意：其下属智能体均会被解雇。
            """
            if agent_id not in self.agents:
                return {"error": f"未找到智能体: {agent_id}"}
            self.fire(agent_id)
            return {"status": "fired", "agent_id": agent_id, "message": "智能体已解雇"}
        
        return fire_agent

    def recursion_signals(self, signal: str) -> Dict[str, Any]:
        """向所有子智能体递归发送信号。"""
        if len(self.agents) == 0:
            return {"error": "你还没有雇佣过子智能体呢。"}

        results = {}
        for aid, item in self.agents.items():
            agent = item["agent"]
            agent.add_parent_signal(signal, recursion=True)
        return results

    def default_prompt(self) -> str:
        
        prompt = """
            雇佣动作拥有四个核心工具来管理子任务：
                1. hire_agent - 雇佣专门的智能体处理子任务
                - 使用时机：当遇到需要专业知识（如Python代码分析、复杂数据处理）时，
                  当你预期子任务占用的上下文太多，或者会分散你的注意力时，再或者你剩下的上下文已经不多了，
                  就雇佣一个小弟吧。
                - 你需要说清楚：要干什么，要得到什么，越明确越好；如果太笼统，你的小弟可能会再雇佣其他小弟，导致混乱。
                - 结果：你会得到雇员ID，用于后续查询
                
                2. list_agents - 查看所有已雇佣的智能体状态
                - 使用时机：你看着来，当你不忙，而且记得还有未交作业的小弟时，记得来看下
                - 结果：返回雇员列表和状态（也包括了你小弟的小弟）
                
                3. get_agent_output - 获取智能体的产出结果
                - 使用时机：你觉需要收作业的时候
                - 你需要提供：雇员ID
                - 结果：返回智能体的分析结果
                
                4. wait_for_agents - 等待子任务完成
                - 使用时机：当你没有其他事情可做，但子任务还没有完成时
                - 你可以指定等待时间（默认5秒），预计对任务负载的的判断，等待一段时间
                - 结果：返回当前子任务的完成情况和后续建议

                5. continue_agent - 继续执行任务
                - 使用时机：当智能体状态为”completed“或”failed“时，可以调用
                - 你需要提供：雇员ID，也可以提供额外的信息内容
                - 结果：指定智能体继续任务，直到完成或再次中断
                - tip： 如果子任务由于信息不足而中断，可以提供更多信息让其继续；
                        如果子任务由于输出窗口等原因未完成，可以调用此工具让其继续执行；
                        如果子任务基础任务完成，形成方法论或者同类问题处理思路，可在其上下文窗口范围内，委托其进行同类问题的处理。
                
                智能工作流建议：
                1. 遇到复杂子任务 → 调用 hire_agent
                2. 继续处理其他简单任务
                3. 间歇性检查状态 → 调用 list_agents  
                4. 如果无事可做但子任务未完成 → 调用 wait_for_agents
                5. 获取已完成的结果 → 调用 get_agent_output
                6. 将结果智能整合到最终描述中
                7. 如有未完成的任务或者想复用子任务成果 → 调用 continue_agent
                
                记住：保持主线任务的高效执行，合理利用等待时间。
                     你的生命周期，是整个上下文窗口，你保证在上下文窗口耗尽前完成整个任务。
                     当剩余上下文不足10%的时候，就要考虑收尾了，不要再雇佣新的小弟了。
                     你要权衡：雇佣的成本函数，取决于雇佣小弟和自己执行对上下文的消耗；当然，优先依照指令执行。
        """
        return prompt

    def yellow_pages(self) -> Dict[str, Any]:
        """提供黄页逻辑函数，返回树形结构的智能体信息，包含每个智能体的基本信息和层级关系，以及任务信息"""
        def build_tree(manager: 'AgentManager') -> Dict[str, Any]:
            tree = {
                "manager_depth": manager.depth,
                "parent_id": manager.parent_id,
                "agents": []
            }
            for item in manager.agents.values():
                agent = item["agent"]
                advisor = item["advisor"]
                agent_info = {
                    "agent_id": agent.agent_id,
                    "task_key_target": agent.task_key_target,
                    "task_description": agent.task_description,
                    "status": agent.status,
                    "hired_at": agent.hired_at,
                    "depth": agent.depth,
                    "has_child_manager": agent.child_manager is not None,
                    "advisor_info": {
                        "has_advisor": advisor is not None
                    }
                }
                if agent.child_manager:
                    agent_info["child_manager"] = build_tree(agent.child_manager)
                tree["agents"].append(agent_info)
            return tree
        
        return build_tree(self.root_manager)

    def lookup_agent(self, agent_id: str) -> SubAgent:
        """从黄页中查找指定ID的智能体实例"""
        def search_tree(manager: 'AgentManager', agent_id: str) -> SubAgent:
            for item in manager.agents.values():
                agent = item["agent"]
                if agent.agent_id == agent_id:
                    return agent
                if agent.child_manager:
                    found_agent = search_tree(agent.child_manager, agent_id)
                    if found_agent:
                        return found_agent
            return None
        
        return search_tree(self.root_manager, agent_id)

    def cancel_all_agents(self):
        """取消当前Manager及其所有子Manager的智能体任务"""
        for item in self.agents.values():
            agent = item["agent"]
            agent.cancel()
            
    # ============ 树形结构打印 ============
    
    def satisfy_sub_agents(self) -> List[SubAgent]:
        """递归获取所有下属智能体。"""
        all_agents = []
        for agent in self.agents.values():
            all_agents.append(agent["agent"])
            if agent["agent"].child_manager:
                all_agents.extend(agent["agent"].child_manager.satisfy_sub_agents())
        return all_agents

    def print_sub_tree(self):
        """打印当前Manager的智能体及其子Manager树"""
        for item in self.agents.values():
            agent = item["agent"]
            print(f"\t"*self.depth + f" - Agent ID: {agent.agent_id} : status: {agent.status}")
            agent.child_manager.print_sub_tree() if agent.child_manager else None

    def check_all_completed(self) -> bool:
        """检查当前Manager及其所有子Manager的智能体是否全部完成"""
        for item in self.agents.values():
            agent = item["agent"]
            if agent.status != "completed" and agent.status != "failed":
                return False
            if agent.child_manager and not agent.child_manager.check_all_completed():
                return False
        return True

    def _extract_hire_cost(self, response) -> int:
        tool_names = ["hire_agent", "list_agents", "get_agent_output", "wait_for_agents", "continue_agent"]
        if hasattr(response, 'get') and isinstance(response, dict):
            messages = response.get("messages", [])
            total_cost = 0

            # 计算全部工具调用的参数长度之和作为成本
            # 如msg中，存在tool_call类型的content_block，且name在tool_names中，则计算该msg的所有block之和
            # 如判断当条msg是tool_call类型，则需累加下一条ToolMessage的内容长度
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage):
                    for block in msg.content_blocks:
                        if block['type'] == 'tool_call' and block['name'] in tool_names:
                            # 累加该msg的所有block之和
                            block_length = sum(len(b['text']) for b in msg.content_blocks if b['type'] == 'text')
                            block_length += sum(len(str(b["args"])) for b in msg.content_blocks if b['type'] == 'tool_call' )
                            total_cost += block_length
                            
                            # 检查下一条消息是否为ToolMessage
                            if i + 1 < len(messages):
                                next_msg = messages[i + 1]
                                if isinstance(next_msg, ToolMessage) and next_msg.name == block['name']:
                                    total_cost += len(next_msg.text)
            return total_cost

    def _extract_advisor_cost(self, response) -> int:

        tool_names = ["ask_boss"]

        if hasattr(response, 'get') and isinstance(response, dict):
            messages = response.get("messages", [])
            total_cost = 0

            # 计算全部工具调用的参数长度之和作为成本
            # 如msg中，存在tool_call类型的content_block，且name在tool_names中，则计算该msg的所有block之和
            # 如判断当条msg是tool_call类型，则需累加下一条ToolMessage的内容长度
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage):
                    for block in msg.content_blocks:
                        if block['type'] == 'tool_call' and block['name'] in tool_names:
                            # 累加该msg的所有block之和
                            block_length = sum(len(b['text']) for b in msg.content_blocks if b['type'] == 'text')
                            block_length += sum(len(str(b["args"])) for b in msg.content_blocks if b['type'] == 'tool_call' )
                            total_cost += block_length
                            
                            # 检查下一条消息是否为ToolMessage
                            if i + 1 < len(messages):
                                next_msg = messages[i + 1]
                                if isinstance(next_msg, ToolMessage) and next_msg.name == block['name']:
                                    total_cost += len(next_msg.text)
            return total_cost

    def _extract_whole_cost(self, response) -> int:
        # 计算所有长度之和，包括用户信息、工具调用、工具响应、Ai响应等
        total_cost = 0
        if hasattr(response, 'get') and isinstance(response, dict):
            messages = response.get("messages", [])
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    total_cost += len(msg.content)
                elif isinstance(msg, AIMessage):
                    for block in msg.content_blocks:
                        if block['type'] == 'text':
                            total_cost += len(block['text'])
                        elif block['type'] == 'tool_call':
                            total_cost += len(str(block['args']))
                elif isinstance(msg, ToolMessage):
                    total_cost += len(msg.content)
        return total_cost

    def export_tree_json(self) -> Dict[str, Any]:
        """导出当前Manager及其子Manager的智能体树为JSON格式"""
        tree = {
            "manager_depth": self.depth,
            "parent_id": self.parent_id,
            "agents": []
        }
        for item in self.agents.values():
            agent = item["agent"]
            advisor = item["advisor"]

            sizeof_agent_system_promt = len(agent.system_prompt) if agent.system_prompt else 0
            sizeof_advisor_system_promt = len(advisor.system_prompt) if advisor and advisor.system_prompt else 0

            agent_last_response = agent.last_response
            advisor_last_response = advisor.last_response if advisor else None

            agent_hire_cost = self._extract_hire_cost(agent_last_response) if agent_last_response else 0
            agent_whole_cost = self._extract_whole_cost(agent_last_response) if agent_last_response else 0
            agent_advise_cost = self._extract_advisor_cost(agent_last_response) if agent_last_response else 0
            advisor_cost = self._extract_whole_cost(advisor_last_response) if advisor_last_response else 0

            agent_info = {
                "agent_id": agent.agent_id,
                "advisor_id": advisor.advisor_id if advisor else None,
                "task_key_target": agent.task_key_target,
                # "task_description": agent.task_description,
                "status": agent.status,
                "depth": agent.depth,
                "hired_at": agent.hired_at,
                "hirable": agent.enable_hiring,
                "force_order": agent.force_order,
                # "results": agent.results, 
                "costs": {
                    "agent" :{
                        "system_prompt_size": sizeof_agent_system_promt,
                        "hire_cost": agent_hire_cost,
                        "advise_cost": agent_advise_cost,
                        "whole_cost": agent_whole_cost
                    },
                    "advisor": {
                        "system_prompt_size": sizeof_advisor_system_promt,
                        "whole_cost": advisor_cost
                    } if advisor else {}
                },
                "child_manager": agent.child_manager.export_tree_json() if agent.child_manager else None
            }
            tree["agents"].append(agent_info)
        return tree

    def print_agent_tree(self, show_details: bool = False):
        """打印从当前Manager开始的智能体树"""
        print(f"\n{'='*60}")
        print(f"🌳 智能体系结构树 (Manager深度: {self.depth})")
        print(f"{'='*60}")
        
        # 打印当前Manager的信息
        manager_info = f"Manager [深度{self.depth}]"
        if self.parent_id:
            manager_info += f" ← 父智能体: {self.parent_id}"
        print(manager_info)
        
        # 打印当前Manager的所有智能体
        for agent_id, item in self.agents.items():
            agent = item["agent"]
            self._print_agent_recursive(agent, 1, show_details)
        
        # 打印统计
        status = self.list_agents()
        print(f"\n📊 当前Manager统计:")
        print(f"  智能体总数: {status['count']}")
        for stat, count in status["statistics"].items():
            print(f"  {stat}: {count}")
    
    def _print_agent_recursive(self, agent: SubAgent, level: int, show_details: bool):
        """递归打印智能体及其子Manager"""
        indent = "  " * level
        
        # 确定图标
        icon = "🔵"
        if agent.status == "completed":
            icon = "✅"
        elif agent.status == "working":
            icon = "⏳"
        elif agent.status == "failed":
            icon = "❌"
        
        # 基本信息
        task_preview = agent.task_description[:40] + "..." if len(agent.task_description) > 40 else agent.task_description
        base_info = f"{indent}{icon} {agent.agent_id} (深度:{agent.depth}, 状态:{agent.status}): {task_preview}"
        
        if show_details:
            details = f"\n{indent}   ├─ 任务: {agent.task_description}"
            details += f"\n{indent}   ├─ 雇佣时间: {agent.hired_at}"
            details += f"\n{indent}   ├─ 父智能体: {agent.parent_id or '根'}"
            details += f"\n{indent}   └─ 有子Manager: {agent.child_manager is not None}"
            
            if agent.results:
                details += f"\n{indent}       ├─ 耗时: {agent.results['execution_time']:.2f}秒"
                details += f"\n{indent}       └─ 结果: {agent.results['response'][:80]}..."
            
            print(base_info + details)
        else:
            print(base_info)
        
        # 递归打印子Manager的智能体
        if agent.child_manager:
            child_manager = agent.child_manager
            print(f"{indent}   └─ 子Manager [深度{child_manager.depth}]: {len(child_manager.agents)}个智能体")
            
            for child_agent_id, child_agent in child_manager.agents.items():
                self._print_agent_recursive(child_agent, level + 1, show_details)