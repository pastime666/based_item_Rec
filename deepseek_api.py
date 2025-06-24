import requests

def generate_item_metadata_with_deepseek(item_info: str, api_key="sk-fe41571d8d124dadb223bb643d1329e6") -> dict:
    """
    使用 DeepSeek API 根据商品描述生成 title、abstract、category 字段。

    参数:
        item_info (str): 商品描述文本
        api_key (str): DeepSeek 的 API Key

    返回:
        dict: 包含 title、abstract、category 的字典
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"""
你是一名商品数据分析师，请根据以下商品描述生成三个字段：标题（title）、摘要（abstract）、类别（category）。

商品描述：{item_info}

请返回标准JSON格式，如：
{{
  "title": "...",
  "abstract": "...",
  "category": "..."
}}
    """

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个高效的商品文本生成助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 解析为 JSON 格式（防止是字符串）
        import json
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            return {"error": "API 返回内容非标准 JSON", "raw": content}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
if __name__ == "__main__":
    # 示例商品文本
    #item_info = "这是一款适合夏季穿着的简约风格男士T恤，采用高透气纯棉面料，提供多种颜色选择，适合日常通勤或休闲搭配。"

    # 你的 DeepSeek API Key
    api_key = "sk-fe41571d8d124dadb223bb643d1329e6"

    #result = generate_item_metadata_with_deepseek(item_info, api_key)
    #print(result)
