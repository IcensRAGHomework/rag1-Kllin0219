import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
import requests


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


def generate_hw01(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    message = HumanMessage(
        content=f"請回答以下問題並以 JSON 格式輸出，格式如下: {{\"Result\": [{{\"date\": \"YYYY-MM-DD\", \"name\": \"紀念日名稱\"}}]}}: {question}"
    )

    response = llm.invoke([message])
    parser = JsonOutputParser()
    try:
        parsed_result = parser.parse(response.content)
    except Exception:
        return {"json parse error"}

    if "Result" in parsed_result and isinstance(parsed_result["Result"], list):
        return json.dumps(parsed_result, ensure_ascii=False)
    else:
        return {"json not contain Result"}


@tool
def fetch_holidays_from_api(conutry, year, month, language) -> str:
    """ fetch_holidays_from_api

    Args:
        conutry: TThe country parameter must be in the iso-3166 format as specified in the document here. To view a list of countries and regions we support, visit our list of supported countries.
        year: The year you want to return the holidays. We currently support both historical and future years until 2049. The year must be specified as a number eg, 2019
        month: Limits the number of holidays to a particular month. Must be passed as the numeric value of the month [1..12].
        language: Returns the name of the holiday in the official language of the country if available. This defaults to english. This must be passed as the 2-letter ISO639 Language Code. An example is to return all the names of france holidays in french you can just add the parameter like this: fr
    """
    try:
        print(f"conutry: {conutry}, year: {year}, month: {month}, language: {language}")
        params = {
            "api_key": "wjr6QZn9jxNz3GlIdgYqM5yVzFawlIkn",
            "country": conutry,
            "year": year,
            "month": month,
            "language": language,
        }

        response = requests.get("https://calendarific.com/api/v2/holidays", params=params)
        if response.status_code == 200:
            data = response.json()

            holidays = data.get("response", {}).get("holidays", [])
            print(holidays)
            return {
                "Result": [
                    {"date": holiday["date"]["iso"], "name": holiday["name"]}
                    for holiday in holidays
                ]
            }
        else:
            return {"error": f"API request state: {response.status_code}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def generate_hw02(question):
    tools = [fetch_holidays_from_api]
    llm_with_tools = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    ).bind_tools(tools)

    messages = [HumanMessage(question)]

    ai_msg = llm_with_tools.invoke(messages)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"fetch_holidays_from_api": fetch_holidays_from_api}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
    
    return tool_output


def generate_hw03(question2, question3):
    pass


def generate_hw04(question):
    pass


def demo(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke("2024年台灣10月紀念日有哪些?")

    return response


if __name__ == '__main__':
    result = generate_hw02("2024年台灣10月紀念日有哪些?")
    print(result)
