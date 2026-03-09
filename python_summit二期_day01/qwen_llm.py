"""
Qwen2.5 本地大模型封装类
基于 HuggingFace Transformers 和 LangChain
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun


class Qwen2_5_LLM(LLM):
    """基于本地 Qwen2.5 模型的自定义 LLM 类"""
    
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 3
    
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        print("完成本地模型的加载")
        
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    @property
    def _llm_type(self) -> str:
        return "Qwen2_5_LLM"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len
        }


class Qwen(LLM, ABC):
    """另一种 Qwen 模型封装，支持更多参数配置"""
    
    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 3
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        
        print("正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("模型加载完成")
    
    @property
    def _llm_type(self) -> str:
        return "Qwen"
    
    @property
    def _history_len(self) -> int:
        return self.history_len
    
    def set_history_len(self, history_len: int = 10) -> None:
        """设置历史对话长度"""
        self.history_len = history_len
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len
        }


# 使用示例
if __name__ == "__main__":
    # 方式1: 使用 Qwen2_5_LLM
    # llm = Qwen2_5_LLM("你的模型路径")
    # response = llm("你好，请介绍一下自己")
    # print(response)
    
    # 方式2: 使用 Qwen
    # llm = Qwen()
    # response = llm("Python 是什么?")
    # print(response)
    
    print("请根据实际情况修改模型路径后使用")
