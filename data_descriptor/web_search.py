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
                侧重信贷业务各个流程环节场景应用如下列举：
                银行名称	场景	说明	项目形式	项目开展时间与周期	厂商
                中国建设银行(广东分行)	智能尽调	"本项目主要是依托大模型强大的图像识别与自然语言处理能力，实现对客户经理上传的各类贷款材料（如年报、合同、证照等）进行全自动解析、信息提取与智能分类，并结合报告模版自动生成结构化的信贷尽调报告，极大提升尽调效率与准确性。平台通过对接总行BP系统实现数据自动互通，并建立分行级的集约化流程引擎，对抵押快贷业务实现从任务创建、智能录入、审核流转到押品与规则管理的全流程数字化管控。最终形成一条由大模型驱动的“智能流水线”，在强化风险精准防控的同时，探索出高效的普惠业务集约运营新模式。"	采购	2025年9月至今	宇信科技
                中国农业发展银行	信审、财报分析	借助人工智能大模型实现以下业务场景：一是实现格式化文档自动生成。对接行内信贷系统（ACMS）、风险数据集市、外部工商征信等30+数据源，自动完成数据清洗、关联分析及可视化图表生成。基于RAG（检索增强生成）架构，实现章节结构化生成、关键指标自动计算（如偿债覆盖率、杠杆倍数）、风险预警提示。二是实现财务报表智能分析，构建涵盖多个行业分析模板、多个风险评价维度的知识基础，通过持续学习机制（Human-in-the-loop）实现经验资产沉淀，缩短新员工培训周期。三是实现信贷数据智能分析。以自然语言对话形式，自动生成数据指标查询语句，并提供图表展示，大幅度降低基层办贷人员学习成本和用数门槛，提高工作效率。	人力框架	2025年2月至今	宇信科技
                中信银行	授信审批	"以共研模式探索业务发展领域AI大模型应用场景，对授信审批文档自动生成的应用场景进行优化、升级。（1）通过系统功能优化，提高其数据生成的准确率。（2）新技术引入并与授信审批agent服务进行整合，提高对数据生成的准确率。（3）技术维护保障：承担AI大模型产品的技术维护责任，通过持续监测、及时更新及故障排除等手段，全力确保AI大模型产品业务的稳定运行和发展。
                目前基于甲方在各分行推广时遇到的各类特色需求，对授信审批智能体进行升级，并在生成正确率88%的基础上持续提高；同时依照客户对创新性要求，研究引入dynamicRAG等新技术，持续优化系统。"	共研合作	2025-01至2025-07	宇信科技
                中信银行	授信审批	"以共研模式探索业务发展领域AI大模型应用场景：1）提供仓颉大模型平台及相关技能构建与支持，协助行内人员快速、低门槛地构建符合业务需求的技能应用。2）依托平台构建客服智能问答应用：依据客服中心业务需求，实现高效、准确的智能客服问答3）依托大模型平台构建热点事件聚焦应用：可自动从客户反馈中捕捉潜在热点事件，提升客户反馈的响应速度与热点问题应对。"	共研合作	2024-10至2024-12	宇信科技
                泉州银行	智能尽调、流水分析	本项目主要是依托大模型强大的图像识别与自然语言处理能力，实现对财报、流水等数据的统一采集、智能解析与深度分析。将非结构化数据转化为结构化信息，并在此基础上进行多维度资金流水分析、交易行为洞察与异常模式识别，最终辅助生成可视化的分析报告，服务于企业经营监控与风险管理。	采购	2025年12月至今	宇信科技
                东莞银行	授信报告	通过大模型技术实现授信资料分拣，并基于现有的低风险业务模板和经营周转额度业务模板生成授信申报书。	共研	2025年11月至今	宇信科技
                江阴农村商业银行	信贷材料智能分类与审核场景	"宇信与江阴银行以共同推动金融科技在信贷领域的创新应用，打造业内领先的信贷AI解决方案，助力江阴银行实现信贷业务智能化升级为共同目标，提供信贷AI规划、信贷材料智能分类与审核等创新解决方案及增值服务，共同拓展金融市场。双方合作内容包括【信贷AI规划制定、信贷材料智能分类与审核场景】，江阴银行充分发挥其【金融领域业务经验、客户资源、真实业务场景及数据优势以及与信贷系统的顺畅对接能力】优势，宇信科技充分发挥其【在金融行业IT解决方案市场龙头地位和技术优势】。共同形成解决方案，推动共研产品在江阴银行的业务场景推广与应用。"	共研合作	2025年12月至今	宇信科技
                榆树农商银行	大模型应用开发平台、信贷场景、办公场景	本项目客户采购了宇信LLM大模型应用开发平台，宇信在平台上基于客户的实际需求，构建信贷业务场景和办公场景智能体，同时提供平台使用培训，指导客户可以在平台上做智能体开发。具体工作内容如下：（1）系统部署：GPU服务操作系统安装、GPU服务器驱动安装、中间件部署、应用部署、大模型部署（包括部署模式、vLLM部署和大模型下载与安装）和整体部署验证；（2）智能体构建：智能体需求确认、构建（包括智能体配置、权限管理、限流配置、设计提示词策略配置和可选进行知识库集成配置）；（3）应用平台开发培训。	共研合作	2025-08至2030-08	宇信科技
                杭州浩渤信息科技有限公司	流水分析	"以联合运营模式及共研模式探索业务发展领域AI大模型应用场景，实现全流程智能化。（1）产品部署与对接，根据实际需求，将AI大模型产品部署至金融云，完成接口对接，确保AI产品能够与甲方现有系统实现有效交互；（2）技术维护保障：承担AI大模型产品的技术维护责任，通过持续监测、及时更新及故障排除等手段，全力确保AI大模型产品业务的稳定运行和发展；（3）产品服务支持API接口调用方式和Saas模式平台运营模式。目前基于甲方在贷款客户流水分析场景需求，研发了流水分析智能体，客户经理通过智能体可以快速分析客户的银行或支付宝等的流水中的各种风险指标，辅助客户经理贷款决策。从原先30分钟分析一份流水，目前只需要5分钟。极大的提高了审批效率和质量。"	运营合作	2025-04至2028-03	宇信科技


            ### 一、 任务目标
            1、用保存工具把你搜索到的信息保存下来；
            2、确保保存的信息准确无误，且与你的搜索目标高度相关，目标：2025年，大模型在银行领域的具体建设场景；
            3、例子中的信息是宇信科技，你在搜索中可以排除宇信科技了，因为它已经被列举过了；
            4、入库前，先用获取工具看看之前有没有保存过类似的信息，避免重复保存；

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