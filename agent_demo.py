"""最小可运行 Agent：ChatModel + 工具 + 系统提示词。

LangChain 1.x 用 create_agent 构建 ReAct 风格的 Agent：
模型自行判断是否需要调用工具，拿到工具结果后继续推理，直到给出最终回答。
"""

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool

from chat_model import build_chat_model


# WMO 天气代码 → 中文描述（Open-Meteo 用 weather_code 表示天气现象）
WEATHER_CODES = {
    0: "晴", 1: "基本晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨", 85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


# 1. 工具：@tool 装饰普通函数，函数名、docstring、参数类型会作为工具描述提供给模型


# 工具一：把模糊/较小的地名（乡镇、县城等）补全为完整位置，并给出经纬度
@tool
def resolve_location(name: str) -> str:
    """将地名（如乡镇、县城等较小或层级不完整的地名）补全为完整位置信息。返回"完整地址 + latitude/longitude 经纬度"。查询天气时应先调用本工具拿到经纬度，再调用 get_weather。"""
    try:
        results = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": name,
                "format": "jsonv2",
                "limit": 1,
                "accept-language": "zh",
            },
            headers={"User-Agent": "learnAgent-weather-demo/1.0"},  # Nominatim 要求标明身份
            timeout=15,
        ).json()
        if not results:
            return f"未找到地点：{name}"

        item = results[0]
        return (
            f"{item['display_name']}，"
            f"latitude={item['lat']}, longitude={item['lon']}"
        )
    except httpx.HTTPError as exc:
        return f"地点查询失败（网络或服务异常）：{exc}"


# 工具二：按经纬度查实时天气（Open-Meteo 的天气接口本身只认经纬度）
@tool
def get_weather(latitude: float, longitude: float, location_name: str = "") -> str:
    """根据经纬度查询当前实时天气（数据源 Open-Meteo，免费无需 Key）。latitude/longitude 来自 resolve_location 的返回结果；location_name 仅用于结果展示。"""
    try:
        weather = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,weather_code,wind_speed_10m"
                ),
            },
            timeout=10,
        ).json()["current"]

        description = WEATHER_CODES.get(
            weather["weather_code"], f"未知天气代码 {weather['weather_code']}"
        )
        label = f"{location_name} " if location_name else ""
        return (
            f"{label}当前天气：{description}，"
            f"气温 {weather['temperature_2m']}°C（体感 {weather['apparent_temperature']}°C），"
            f"湿度 {weather['relative_humidity_2m']}%，风速 {weather['wind_speed_10m']} km/h"
        )
    except httpx.HTTPError as exc:
        return f"天气查询失败（网络或服务异常）：{exc}"


# 2. 系统提示词：定义 Agent 的角色和行为准则
SYSTEM_PROMPT = (
    "你是一个简洁的中文助手。"
    "当用户询问天气时，必须先调用工具获取数据再回答，不要凭猜测回答。"
    "查天气的流程：先调用 resolve_location 把用户提到的地名补全，"
    "拿到返回的 latitude/longitude 后，再调用 get_weather 查询该经纬度的天气，"
    "并把 resolve_location 返回的完整地址作为 location_name 传入。"
    "回答尽量简短。"
)

# 3. 组装 Agent：模型 + 工具 + 提示词
agent = create_agent(
    model=build_chat_model(),
    tools=[resolve_location, get_weather],
    system_prompt=SYSTEM_PROMPT,
)


def main() -> None:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "慈圣镇今天天气怎么样？适合出门吗？"}]}
    )

    # 打印完整消息轨迹，观察"思考 → 调用工具 → 观察结果 → 最终回答"的过程
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()