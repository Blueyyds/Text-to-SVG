from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json


def prepare_model(model_path, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model_path, tokenizer=model_path, max_model_len=max_model_len, trust_remote_code=True)
    return llm


def get_completion(prompts, llm, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids
    )
    # 初始化 vLLM 推理引擎
    outputs = llm.generate(prompts, sampling_params)
    return outputs


# 初始化 vLLM 推理引擎
model_path = "/home/models/Qwen2.5-Coder-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = prepare_model(model_path)

prompt = "Please Generate a SVG code of The Transformer model Architecture"
messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

# 应用template中的chat模板
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = get_completion(text, llm, max_tokens=1024, temperature=1, top_p=1, max_model_len=2048)
print(outputs[0].outputs[0].text)
