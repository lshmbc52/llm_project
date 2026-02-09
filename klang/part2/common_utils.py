import os
import gc
from contextlib import contextmanager
from typing import List, Dict, Any

# torch가 없어도 에러나지 않게 보호막 생성!
try:
    import torch
except ImportError:
    torch = None

from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI

@contextmanager
def memory_cleanup():
    """GPU 및 시스템 메모리를 정리하는 컨텍스트 매니저"""
    try:
        yield
    finally:
        # torch가 설치되어 있고 CUDA(GPU)를 쓸 수 있을 때만 실행
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU 캐시 비우기 완료")
        gc.collect()
        print("✨ 시스템 메모리 정리 완료")

def get_gpt_model(model_name="gpt-4o-mini", temperature=0.5, **kwargs):
    """OpenAI GPT 모델 호출"""
    return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)

def get_solar_model(model_name="solar-pro", temperature=0.5, **kwargs):
    """Upstage Solar 모델 호출"""
    api_key = os.getenv("UPSTAGE_API_KEY")
    return ChatUpstage(model=model_name, upstage_api_key=api_key, temperature=temperature, **kwargs)

if __name__ == "__main__":
    print("\n-- 🚀 모델 로드 테스트 시작 --")

    try:
        with memory_cleanup():
            gpt_model = get_gpt_model()
            solar_model = get_solar_model()

            # 💡 [핵심 수정] 속성 이름이 무엇이든 상관없이 모델명을 가져오는 마법!
            def get_name(m):
                # model_name 먼저 찾아보고, 없으면 model 찾아보고, 그것도 없으면 'Unknown'
                return getattr(m, 'model_name', None) or getattr(m, 'model', 'Unknown')

            print(f"✅ GPT 모델 로드 성공: {get_name(gpt_model)}")
            print(f"✅ Solar 모델 로드 성공: {get_name(solar_model)}")
            print("\n🎉 모든 모델이 정상적으로 준비되었습니다!")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")