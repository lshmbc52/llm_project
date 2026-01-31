# common_utils.py 파일로 저장
import os
import gc
import torch  # Moved torch import to top of file
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import List, Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


@contextmanager
def memory_cleanup():
    """GPU 및 시스템 메모리를 정리하는 컨텍스트 매니저"""
    try:
        yield
    finally:
        # GPU 캐시 정리 (torch가 설치되어 있는 경우)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU 캐시 비우기 완료")
        
        gc.collect()
        print("시스템 메모리 정리 완료")


def get_model(model_name="gpt-4o-mini", temperature=0.7, streaming=True, **kwargs):
    """설정된 모델 객체를 반환하는 함수"""
    return ChatOpenAI(
        model=model_name, temperature=temperature, streaming=streaming, **kwargs
    )


if __name__ == "__main__":
    print("--모델로드 테스트 시작__")
    model = get_model()  # Fixed: call the function to get model instance
    print(f"로드된 모델:{model.model}")
