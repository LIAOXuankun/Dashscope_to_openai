# coding=utf-8
# Implements API for Qwen in OpenAI's format.
# Usage: python dashscope_to_openai.py
# Visit http://localhost:8000/docs for documents.

from http import HTTPStatus
import json
import time
from argparse import ArgumentParser
from typing import Dict, List, Literal, Optional, Union

import dashscope
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from dashscope.api_entities.dashscope_response import Message

dashscope.api_key = "sk-..."  # Your Qwen API Key

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 基本数据模型定义
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


# 简单的消息转换函数
def convert_chatmessage_to_message(messages: List[ChatMessage]) -> List[Message]:
    result = []
    for message in messages:
        if message.role in ["user", "assistant", "system"]:
            result.append(Message(role=message.role, content=message.content))
    return result


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # 设置生成参数
    gen_kwargs = {}
    if request.temperature is not None:
        gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    # 使用请求中指定的模型
    model_name = request.model


    # 非流式调用
    try:
        # 将OpenAI消息格式转换为DashScope格式
        messages = convert_chatmessage_to_message(request.messages)

        # 调用DashScope API
        response = dashscope.Generation.call(
            model=model_name,
            messages=messages,
            result_format="message",
            stop=request.stop,
            **gen_kwargs
        )
        # 构建响应
        content = ""
        
        # 提取响应内容
        response_dict = dict(response)
        if 'output' in response_dict:
            output_dict = dict(response_dict['output'])
            if 'choices' in output_dict:
                content = output_dict['choices'][0]['message']['content']
        
        # 设置有效的finish_reason
        finish_reason = "stop"
        
        # 创建响应
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish_reason
            )
        ]
        
        return ChatCompletionResponse(
            model=request.model, choices=choices, object="chat.completion"
        )
        # return response
    except Exception as e:
        print(f"错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--server-port", type=int, default=8020, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
