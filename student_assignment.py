import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

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


def generate_hw02(question):
    pass


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
    response = llm.invoke([message])

    return response

if __name__ == '__main__':
    result = generate_hw01("2024年台灣10月紀念日有哪些?")
    print(result)