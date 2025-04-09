# Dashscope_to_openai
将阿里巴巴开源模型的Dashscope api 转换成Openai格式进行调用

注：这是一段魔改代码，改自[Dashscope_to_openai](https://github.com/lemon-little/Dashscope_to_openai/tree/main)，现已支持最新版的openai格式调用，为测试阿里云开源模型，只能进行最基本的非流式输出进行对话，如有精力会继续更新

开始前请先安装必要的代码库

```
pip install fastapi uvicorn openai pydantic sse_starlette dashscope
```

在终端运行代码
```
python dashscope_to_openai.py
```

使用api
```
import openai

openai.api_key = '...'

openai.base_url = ""http://localhost:8020/v1""
openai.default_headers = {"x-foo": "true"}

completion = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.choices[0].message.content)
```
