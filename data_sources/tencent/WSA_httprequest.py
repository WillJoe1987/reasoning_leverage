# -*- coding: utf-8 -*-
# 腾讯云API签名v3实现示例
# 本代码基于腾讯云API签名v3文档实现: https://cloud.tencent.com/document/product/213/30654
# 请严格按照文档说明使用，不建议随意修改签名相关代码

import os
import hashlib
import hmac
import json
import sys
import time
from datetime import datetime

from httpx import get
if sys.version_info[0] <= 2:
    from httplib import HTTPSConnection
else:
    from http.client import HTTPSConnection
from dotenv import load_dotenv
load_dotenv()

def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

# 密钥信息从环境变量读取，需要提前在环境变量中设置 TC_WSAPI_SID 和 TC_WSAPI_SKEY
# 使用环境变量方式可以避免密钥硬编码在代码中，提高安全性
# 生产环境建议使用更安全的密钥管理方案，如密钥管理系统(KMS)、容器密钥注入等
# 请参见：https://cloud.tencent.com/document/product/1278/85305
# 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
secret_id = os.getenv("TC_WSAPI_SID")
secret_key = os.getenv("TC_WSAPI_SKEY")
token = ""

service = "wsa"
host = "wsa.tencentcloudapi.com"
region = ""
version = "2025-05-08"
action = "SearchPro"
endpoint = "https://wsa.tencentcloudapi.com"
algorithm = "TC3-HMAC-SHA256"


def get_timestamp_by_date(time_str=None):
    if time_str:
        dt = datetime.strptime(time_str, "%Y-%m-%d")
        return int(time.mktime(dt.timetuple()))
    else :  
        return int(time.time())

def query_wsa(**kwargs):

    Query ,Mode , Site, FromTime , ToTime = kwargs.values()

    if Query is None:
        return "查询关键词为空，无法执行请求。"

    payload = json.dumps({
        "Query": Query,
        "Mode": Mode,
        "Site": Site,
        "FromTime": FromTime,
        "ToTime": ToTime
    }, ensure_ascii=False)

    timestamp = get_timestamp_by_date()
    date = datetime.now().strftime("%Y-%m-%d")

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (ct, host, action.lower())
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                        canonical_uri + "\n" +
                        canonical_querystring + "\n" +
                        canonical_headers + "\n" +
                        signed_headers + "\n" +
                        hashed_request_payload)

    # ************* 步骤 2：拼接待签名字符串 *************
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                    str(timestamp) + "\n" +
                    credential_scope + "\n" +
                    hashed_canonical_request)

    # ************* 步骤 3：计算签名 *************
    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # ************* 步骤 4：拼接 Authorization *************
    authorization = (algorithm + " " +
                    "Credential=" + secret_id + "/" + credential_scope + ", " +
                    "SignedHeaders=" + signed_headers + ", " +
                    "Signature=" + signature)

    # ************* 步骤 5：构造并发起请求 *************
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version
    }
    if region:
        headers["X-TC-Region"] = region
    if token:
        headers["X-TC-Token"] = token

    try:
        # req = HTTPSConnection(host)
        # req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        # resp = req.getresponse()
        # print(resp.read())
        import httpx
        headers["X-TC-Timestamp"] = str(timestamp)  # 保证是字符串
        # 直接发送原始 bytes，避免 httpx 自动序列化或修改 body
        resp = httpx.post(endpoint, headers=headers, content=payload.encode("utf-8"), timeout=10.0)
        print(resp.text)
        return resp.text
    except Exception as err:
        print(err)
if __name__ == "__main__":
    print(get_timestamp_by_date("2026-01-01"))
    query_wsa(
        Query="半导体_AI_存储芯片_市场环境",
        Mode=0,
        Site="",
        FromTime=get_timestamp_by_date("2026-01-01"),
        ToTime=get_timestamp_by_date("2026-01-19")
    )
