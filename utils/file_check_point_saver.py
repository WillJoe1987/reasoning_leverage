from email import message
import json
import asyncio
import base64
import os,sys
from tabnanny import check
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple
from langchain.messages import AIMessage, HumanMessage, ToolMessage
# 引入aimessage对象


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # 官方 serializer

class FileByteSaver(BaseCheckpointSaver):
    
    """
    最小可用的文件后端：把对象用 serde.dumps_typed 序列化为 bytes 写入文件，
    读取时用 serde.loads_typed 恢复。
    文件布局（per thread_id dir）：
      - <checkpoint_id>.checkpoint.bin    # 序列化的 checkpoint bytes
      - <checkpoint_id>.metadata.bin      # 序列化的 metadata bytes
      - <checkpoint_id>.writes.bin        # （可选）序列化的 pending writes bytes
      - latest.txt                         # 存最新 checkpoint_id（文本）
    """

    def __init__(self, base_dir: str = "./checkpoints_bytes"):
        super().__init__(serde=JsonPlusSerializer(pickle_fallback=True))
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._system_prompt = None

    def _thread_dir(self, thread_id: str) -> Path:
        td = self.base_dir / thread_id
        td.mkdir(parents=True, exist_ok=True)
        return td

    def _checkpoint_path(self, thread_id: str, checkpoint_id: str) -> Path:
        td = self._thread_dir(thread_id)
        return td / f"{checkpoint_id}.json"

    def write_to_json(self, thread_id: str, checkpoint_id: str, data: dict) -> None:
        path = self._checkpoint_path(thread_id, checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有数据，若损坏则备份并从空字典开始
        existing_data = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                corrupt_path = path.with_suffix(".corrupt")
                try:
                    os.replace(str(path), str(corrupt_path))
                except Exception:
                    pass
                existing_data = {}

        existing_data.update(data)

        # 原子写入：先写临时文件，再替换
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def read_from_json(self, thread_id: str, checkpoint_id: str) -> dict:
        path = self._checkpoint_path(thread_id, checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # 尝试多次读取，以应对并发写入导致的短暂不完整文件
        last_exc = None
        for attempt in range(3):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError as e:
                last_exc = e
                # 等短暂时间后重试
                import time
                time.sleep(0.1 * (attempt + 1))
                continue
            except Exception as e:
                raise

        # 如果多次重试仍失败，抛出更明确的错误并把损坏文件重命名备份
        corrupt_path = path.with_suffix(".corrupt")
        try:
            os.replace(str(path), str(corrupt_path))
        except Exception:
            pass
        raise ValueError(f"Failed to parse checkpoint JSON after retries. Corrupted file moved to {corrupt_path}. Error: {last_exc}")

    def set_system_prompt(self, prompt: str) -> None:
        if prompt:
            self._system_prompt = prompt

    def _encode_bytes(self, b: bytes) -> str:
        return base64.b64encode(b).decode('ascii')
    
    def _decode_bytes(self, s: str) -> bytes:
        return base64.b64decode(s.encode('ascii'))

    def _atomic_write_text(self, path: Path, s: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(s)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # 同步 put：保存 checkpoint + metadata（都序列化为 bytes）
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        td = self._thread_dir(thread_id)

        if self._system_prompt:
            metadata["system_prompt"] = self._system_prompt

        # 用官方 serde 序列化为 bytes（支持复杂对象 / pandas / deque）
        tag_cp, cp_bytes = self.serde.dumps_typed(checkpoint)
        tag_md, md_bytes = self.serde.dumps_typed(metadata)

        cp_dict = { "tag": tag_cp, "data": self._encode_bytes(cp_bytes) }
        md_dict = { "tag": tag_md, "data": self._encode_bytes(md_bytes) }

        # 准备存储的数据
        data = {
            "checkpoint": cp_dict,
            "metadata": md_dict,
        }

        self.write_to_json(thread_id, checkpoint_id, data)

        # 更新 latest pointer
        self._atomic_write_text(td / "latest.txt", checkpoint_id)

        # 返回新的 config（只包含 thread/checkpoint）
        return {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    # put_writes 支持：把 writes（Sequence[tuple[channel, value]]）序列化并写文件
    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        # 把 writes 序列化为 bytes（框架期望的是 sequence of (channel, value)）
        tag, b = self.serde.dumps_typed(list(writes))

        data = {
            "writers": { "tag": tag, "data": self._encode_bytes(b)}
        }

        self.write_to_json(thread_id, checkpoint_id or "latest.write.json", data)

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    # 读取 checkpoint（优雅地从文件恢复）
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id is None:
            
            # 读取 latest.txt 获取最新 checkpoint_id
            latest_path = self._thread_dir(thread_id) / "latest.txt"
            if not latest_path.exists():
                return None
            with open(latest_path, "r", encoding="utf-8") as f:
                checkpoint_id = f.read().strip()

        json_data = self.read_from_json(thread_id, checkpoint_id)

        checkpoint = json_data.get("checkpoint", {})
        tag_cp = checkpoint.get("tag", "")
        cp_bytes = checkpoint.get("data", "")
        checkpoint = self.serde.loads_typed((tag_cp, self._decode_bytes(cp_bytes)))

        metadata = json_data.get("metadata", {})
        tag_md = metadata.get("tag", "")
        md_bytes = metadata.get("data", "")
        metadata = self.serde.loads_typed((tag_md, self._decode_bytes(md_bytes))) if md_bytes else {}

        pending_writes = json_data.get("writers", {})
        tag_pw = pending_writes.get("tag", "")
        pw_bytes = pending_writes.get("data", "")
        if pw_bytes:
            pending_writes = self.serde.loads_typed((tag_pw, self._decode_bytes(pw_bytes))) if pw_bytes else []
        else:
            pending_writes = []

        # 返回最小的 CheckpointTuple 结构（parent_config 可按需填）
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}},
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=None,
        )

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self.get_tuple, config)

def check_point_reader(check_point_name:str):

    if not check_point_name:
        raise ValueError("check_point_name is empty")
    
    custom_path = Path(project_root) / "agent_checkpoints" / check_point_name
    
    if not custom_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {custom_path}")
    
    saver = FileByteSaver(str(custom_path.parent))

    last_cfg = {
        "configurable": {"thread_id": check_point_name},
    }
    tup = saver.get_tuple(last_cfg)
    return tup

def sub_agent_check_origin_messages_reader(agent_name:str):

    if not agent_name:
        raise ValueError("agent_name is empty")
    thread_id = f"subagent_{agent_name}"

    tup = check_point_reader(thread_id)

    messages = tup[1]['channel_values']['messages']

    return messages

def sub_agent_check_point_reader(angent_name:str):

    if not angent_name:
        raise ValueError("angent_name is empty")
    thread_id = f"subagent_{angent_name}"

    tup = check_point_reader(thread_id)

    system_prompt = tup.metadata.get("system_prompt", None)
    
    chat_records = [{
        "role" : "system",
        "content": system_prompt
    }]

    messages = tup[1]['channel_values']['messages']
    
    for m in messages:
        if isinstance(m, HumanMessage):
            chat_records.append({
                "role": "user",
                "content": m.content
            })
        elif isinstance(m, AIMessage):

            conent_blocks = m.content_blocks
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

            chat_records.append({
                "role": "assistant",
                "content": contents

            })
        elif isinstance(m, ToolMessage):
            chat_records.append({
                "role": "tool",
                "name": m.name,
                "content": m.text
            })

    return chat_records

def pretty_print_chat_records(chat_records):
    messages = []
    for record in chat_records:
        # 根据record的类型，包括HumanMessage, AIMessage, ToolMessage等，进行不同的打印处理
        if isinstance(record, HumanMessage):
            print(f"User: {record.content}")
            messages.append("User:")
            messages.append(record.content)
        elif isinstance(record, AIMessage):
            print("AI: ")
            messages.append("AI:")
            for block in record.content_blocks:
                if block['type'] == 'text':
                    print(f"  {block['text']}")
                    messages.append(block['text'])
                elif block['type'] == 'tool_call':
                    print(f"  [Tool Call] Name: {block['name']}, Args: {block['args']}")
                    messages.append(f"[Tool Call] Name: {block['name']}, Args: {block['args']}")
        elif isinstance(record, ToolMessage):
            print(f"Tool ({record.name}): {record.text}")
            messages.append(f"Tool ({record.name}):")
            messages.append(f"{record.text}")
    
    return messages


if __name__ == "__main__":

    from utils.talent_training_market import AgentManager
    from data_descriptor.tool_model import get_default_model

    manager = AgentManager(get_default_model())

    agentname = "subagent_emp_1151acab"
    threadid = "subagent_emp_db855ede"

    # rec = sub_agent_check_origin_messages_reader(agentname)

    rec = check_point_reader(threadid)
    rec = rec[1]['channel_values']['messages']
    results = {
        "messages": rec
    }
    messages = pretty_print_chat_records(rec)
    filepath = Path(project_root) / "logs" / threadid / "chat_history.txt"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg + "\n\n")

    # print("Original messages:")
    # print(manager._extract_advisor_cost(results))
    # print(manager._extract_whole_cost(results))
    # print(manager._extract_hire_cost(results))
