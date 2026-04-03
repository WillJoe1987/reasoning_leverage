from datetime import datetime
from sqlite3 import paramstyle
import os,sys
from pathlib import Path
import json
import asyncio
import os
from dotenv import load_dotenv
import requests
load_dotenv()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.talent_training_market import AgentManager,SubAgent
from utils.file_check_point_saver import FileByteSaver
from data_descriptor.tool_model import get_default_model, get_qwen_model
import vendor.vendor_patch  # noqa: E402
from langchain.tools import tool
from agent_tools.local_tools.caculator import get_calculator_tool
from tools.general_tools import extract_tool_messages
# Custom path
custom_path = Path(project_root) / "agent_checkpoints"
custom_path.mkdir(exist_ok=True)
import pandas as pd
import csv
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
model_type = "deepseek"  # or "qwen""deepseek"


checkpointer = FileByteSaver(str(custom_path))

class resolver_agent(SubAgent):
    
    DEFAULT_DESCRIPTION_FILE = "description.txt"
    PROJECT_ROOT = project_root

    def __init__(self):

        self.child_manager = None
        model = get_default_model()
        # base_tools=[add, subtract, multiply, divide, power, distance, floor, ceil, modulus]
        base_tools = [save_result, get_result_with_bank_name, search_news]
        manager = AgentManager(llm_client=model)
        manager.open_shared_blackboard()
        usser_input = "加油加油加油！开干！"
        super().__init__(manager=manager, task_key_target="", task_description=f"{usser_input}", enable_hiring=True, depth=0, base_tools=base_tools)

        thread_id = "search_"+ datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {"configurable": {"thread_id": f"{thread_id}"},"recursion_limit": 1000}
        self.initial_tools = []
        self.agent_id = "root"   
        # TODO register a root advisor
        manager.register_agent(self)

    def final_handle(self, final_content: str):
        """"""
        return True

    def _generate_system_prompt(self) -> str:
        
        prompt = f"""

            你的核心任务是：搜索相关的网络信息，近一年2025年，大模型在银行信贷业务流程落地的具体建设场景；，并保存；
                以公开招投标的信息为主，
                侧重信贷业务各个流程环节场景应用

            ### 一、 任务目标
            1、用保存工具把你搜索到的信息保存下来；
            2、确保保存的信息准确无误，且与你的搜索目标高度相关，目标：2025年，大模型在银行领域的具体建设场景；
            3、入库前，先用获取工具看看之前有没有保存过类似的信息，避免重复保存；

            ### 二、 描述内容要素
            看着来吧，你是最棒的；

            ### 三、 工具使用与工作指引

            **核心工具**
                1.  search_news(Query: str, Site: str = "", FromDate: str = None, ToTime: str = None) : 使用搜索工具进行网络搜索。
                    （该工具可搜索返回指定时间内的相关新闻文章，如果不传入时间参数，则默认为‘2025-01-01’到‘2025-12-31’的范围）

            **辅助工具**
                1.  save_result(bank_name: str, scene: str, description: str, type: str = "", time_cycle: str = "", vendor: str = "") : 保存一条搜索结果。
                2.  get_result_with_bank_name(bank_name: str) : 根据银行名称获取保存的结果，避免重复保存。

            **基础工作规则**
            1. 联网搜索工具每次返回的内容大概在10-20条左右，规划好你的搜索策略，尽量全面而不重复；
            2. 关于共享黑板，聪明滴使用，也聪明地交代你的下属们如何使用；它每次都会返回所有代理保存的结果，不要让它太大，占据过多上下文，所以，存什么很关键；
            3. 你自己估摸估摸怎么规划任务吧，反正我也没啥好建议的；
            4. 你的小弟也都能使用这些工具，你可以让他们帮你分担任务；
            5. 如果你预估子模型的任务比较长，其实你可以等久一些，让他们多跑一会儿再看情况；
            6. 尽量预估好任务要消耗的上下文，提前布局要雇佣的小弟，避免中途因为上下文不够用而卡壳；

            {self.child_manager.default_prompt() if self.child_manager else ""}

            当你觉得任务完成时，输出
                {SubAgent.STOP_SIGNAL}
        """
        return prompt




@tool
def save_result(bank_name:str, title: str, summary: str, url: str = "") -> None:
    """
        保存一条搜索结果。
        Args:
            bank_name: 银行名称
            title: 文章标题
            summary: 文章摘要（300字以内）
            url: 文章链接
        Returns:
            None
    """
    
    result_file = Path(project_root) / "logs"/"search_res" / "search_items.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not result_file.exists():
        with open(result_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["BankName", "Title", "Summary", "URL"])
            writer.writerow([bank_name, title, summary, url])
    else:
        with open(result_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([bank_name, title, summary, url])

    print(f"保存结果到路径: {result_file}")
    
@tool
def save_result(bank_name:str, scene: str, description: str, type: str = "", time_cycle: str = "", vendor: str = "") -> None:
    """
        保存一条搜索结果。
        Args:
            bank_name: 银行名称
            scene: 场景
            description: 说明
            type: 项目形式
            time_cycle: 项目开展时间与周期
            vendor: 厂商
        Returns:
            None
    """
    
    result_file = Path(project_root) / "logs"/"search_res" / "search_items.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not result_file.exists():
        with open(result_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["BankName", "Scene", "Description", "Type", "TimeCycle", "Vendor"])
            writer.writerow([bank_name, scene, description, type, time_cycle, vendor])
    else:
        with open(result_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([bank_name, scene, description, type, time_cycle, vendor])

    print(f"保存结果到路径: {result_file}")

@tool
def get_result_with_bank_name(bank_name: str) -> dict:
    """
        根据银行名称获取保存的结果；
        反正要保存一条记录前，你最好先调用这个接口，看看之前有没有保存过类似的结果，避免重复保存。
    """
    try:
        result_file = Path(project_root) / "logs"/"search_res" / "search_items.csv"
        if not result_file.exists():
            print(f"结果文件不存在: {result_file}")
            return {"info": "尚无保存结果"}

        df = pd.read_csv(result_file)
        filtered_df = df[df['BankName'] == bank_name]

        result = ""

        for _, row in filtered_df.iterrows():
            result += f"Scene: {row['Scene']}\nDescription: {row['Description']}\nType: {row['Type']}\nTimeCycle: {row['TimeCycle']}\nVendor: {row['Vendor']}\n\n"

        return {"results": result}
    except Exception as e:
        print(f"获取结果失败: {str(e)}")
        return {"info": "获取结果失败"}

from data_sources.tencent.WSA_httprequest import query_wsa, get_timestamp_by_date
@tool
def search_news(Query:str, Site="", FromDate:str=None, ToTime:str=None) -> str:
    """
    使用网络搜索工具获取与指定公司相关的新闻文章。
    Args:
        Query: 搜索关键词
        Site: 限定搜索的网站(可选)
        FromDate: 起始日期，格式为 "YYYY-MM-DD"(可选)
        ToTime: 结束日期，格式为 "YYYY-MM-DD"(可选) 
    Returns:
        新闻文章的字符串表示
    """
    try:
        print("使用搜索工具进行网络搜索，关键词:", Query)
        response = query_wsa(
            Query=Query,
            Mode=0,
            Site=Site,
            FromTime=get_timestamp_by_date(FromDate) if FromDate else get_timestamp_by_date("2025-01-01"),
            ToTime=get_timestamp_by_date(ToTime) if ToTime else get_timestamp_by_date("2025-12-31")
        )
        print("搜索工具返回结果长度:", len(response))
        return response
    except Exception as e:
        return f"❌ 新闻搜索工具执行失败: {str(e)}"

@tool
def baidu(keywords: str) -> str:
    """
    使用百度搜索工具进行网络搜索。
    Args:
        keywords: 搜索关键词
    Returns:
        搜索结果的字符串表示
    """
    try:
        # browser_url = os.getenv("HEADLESSBROSWER_LOCATIONG") + f"browser/baidu"
        # params = {
        #     "wd": keywords
        # }
        # response = requests.post(browser_url, json=params, timeout=300)
        
        from urllib.parse import quote
        # 对关键词进行URL编码，确保中文等特殊字符能正确传输
        encoded_keyword = quote(keywords)

        # 2. 构造完整的请求URL
        url = f"https://www.baidu.com/s?wd={encoded_keyword}"

        # 3. 设置请求头，模拟一个普通浏览器的访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.baidu.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'max-age=0',
        }
        # 4. 发送HTTP GET请求
        response = requests.get(url, headers=headers)
        response.encoding = 'gbk'
        if response.status_code == 200:
            data = response.text
            print(data)
            if data:
                document = BeautifulSoup(data, 'html.parser')
                for s in document(["script", "style", "noscript", ""]):
                    s.decompose()
                for c in document.find_all(string=lambda text: isinstance(text, Comment)):
                    c.extract()
                body = document.body or document
                def render(node):
                    if isinstance(node, NavigableString):
                        return str(node)
                    if isinstance(node, Tag):
                        name = node.name.lower()
                        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                            level = int(name[1])
                            text = ''.join(render(c) for c in node.children).strip()
                            return f"\n{'#' * level} {text}\n\n"
                        if name in ("p", "div"):
                            text = ''.join(render(c) for c in node.children).strip()
                            return f"{text}\n\n" if text else ""
                        if name == "a":
                            href = node.get("href", "")
                            text = ''.join(render(c) for c in node.children).strip()
                            return f"[{text}]({href})" if href else text
                        if name in ("ul", "ol"):
                            items = []
                            for li in node.find_all("li", recursive=False):
                                items.append(render(li))
                            return '\n'.join(items) + '\n\n'
                        if name == "li":
                            text = ''.join(render(c) for c in node.children).strip()
                            return f"- {text}"
                        if name in ("pre", "code"):
                            text = node.get_text()
                            return f"\n```\n{text}\n```\n\n"
                        return ''.join(render(c) for c in node.children)
                    return ""
                content_id = "content_left"
                md = render(body.find(id=content_id)) if body.find(id=content_id) else render(body)
                # collapse and clean lines
                md = "\n".join(line.rstrip() for line in md.splitlines() if line.strip())
                
                return md
            else:
                return f"⚠️ 未找到与公司 '{keywords}' 相关的新闻文章。"
        else:
            return f"❌ 获取新闻失败，状态码: {response.status_code}"
    except Exception as e:
        print(f"获取新闻时发生错误: {str(e)}")
        return f"❌ 获取新闻时发生错误: {str(e)}"

@tool
def fetch_url(url: str) -> str:
    """
    使用网页内容抓取工具获取指定URL的内容。
    Args:
        url: 目标网页的URL
    Returns:
        网页内容的字符串表示
    """
    try:
        browser_url = os.getenv("HEADLESSBROSWER_LOCATIONG") + f"browser/fetch"

        params = {
            "url": url
        }
        response = requests.post(browser_url, json=params, timeout=300)
        if response.status_code == 200:
            content = response.text
        else:
            return f"❌ 网页内容抓取失败，状态码: {response.status_code}"

        return content

    except Exception as e:
        return f"❌ 网页内容抓取工具执行失败: {str(e)}"

async def main():
    agent = resolver_agent()
    response = await agent.execute()


    shares = agent.manager.shared_blackboard
    shared_file_path = Path(f"{project_root}/logs/search_res/search_items.json")
    shared_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shared_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(shares, ensure_ascii=False, indent=2))
    print(f"Shared blackboard saved to {shared_file_path}")
    print("最终结果:")
    print(len(response))







if __name__ == "__main__":
    # asyncio.run(run())
    # load_data()

    asyncio.run(main())
    # content= fetch_url.func("https://www.ccgp.gov.cn/cggg/dfgg/zbgg/202312/t20231228_890123.html") 
    # baidu_result = baidu.func("大模型 银行 投入 进展")
    # print("Fetched content length:", len(baidu_result))
    # print(baidu_result)  # Print first 500 characters