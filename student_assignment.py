import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
import requests

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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
        params = {
            "api_key": "wjr6QZn9jxNz3GlIdgYqM5yVzFawlIkn",
            "country": conutry,
            "year": year,
            "month": month,
            "language": language,
        }

        response = requests.get(
            "https://calendarific.com/api/v2/holidays", params=params)
        if response.status_code == 200:
            data = response.json()

            holidays = data.get("response", {}).get("holidays", [])
            result = {
                "Result": [
                    {"date": holiday["date"]["iso"], "name": holiday["name"]}
                    for holiday in holidays
                ]
            }
            return json.dumps(result)
        else:
            return {"error": f"API request state: {response.status_code}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def fetch_holidays_if_valid(ai_msg):
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if (tool_call["name"].lower() == "fetch_holidays_from_api"):
                return fetch_holidays_from_api.invoke(tool_call["args"])

    return ai_msg


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

    chain = llm_with_tools | fetch_holidays_if_valid
    result = chain.invoke(messages)

    return result


def generate_hw03(question2, question3):
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    tools = [fetch_holidays_from_api]
    llm_with_tools = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    ).bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name='history'),
            ('human', '{question}'),
        ]
    )

    chain = (prompt | llm_with_tools | fetch_holidays_if_valid)

    llm_with_chat_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    response1 = llm_with_chat_history.invoke(
        {"question": HumanMessage(question2)},
        config={"configurable": {"session_id": "abc123"}},
    )

    response2 = llm_with_chat_history.invoke(
        {"question": HumanMessage(f'"請回答以下問題並以 JSON 格式輸出，格式如下: {{\"Result\": {{\"add\": \"這是一個整數，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 1；否則為 0。\", \"reason\": \"描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。\"}}}} : {question3}')},
        config={"configurable": {"session_id": "abc123"}},
    )

    return json.dumps(response2.content, ensure_ascii=False)


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
    # reuslt = generate_hw01("2024年台灣10月紀念日有哪些?")
    # print(reuslt)
    # result = generate_hw02("2024年台灣10月紀念日有哪些?")
    # print(result)
    result = generate_hw03("2024年台灣10月紀念日有哪些?", "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單?")
    print(result)
