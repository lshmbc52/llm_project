import os
import gc
import torch
from contextlib import contextmanager
from langchain_upstage import ChatUpstage  # Upstage 전용 라이브러리 사용
from contextlib import contextmanager
from typing import List, Dict, Any


@contextmanager
def memory_cleanup():
    """GPU 및 시스템 메모리를 정리하는 컨텍스트 매니저"""
    try:
        yield
    finally:
        # GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU 캐시 비우기 완료 (Upstage 모드)")

        gc.collect()
        print("시스템 메모리 정리 완료")


def get_solar_model(model_name="solar-pro", temperature=0.5, **kwargs):
    """
    Upstage Solar 모델 객체를 반환하는 함수.
    환경변수 UPSTAGE_API_KEY를 자동으로 참조합니다.
    """
    # .bashrc에 등록한 키를 안전하게 가져옵니다.
    api_key = os.getenv("UPSTAGE_API_KEY")

    return ChatUpstage(
        model=model_name, upstage_api_key=api_key, temperature=temperature, **kwargs
    )


if __name__ == "__main__":
    print("-- Upstage Solar 모델 로드 테스트 시작 --")

    # 환경변수 체크
    if not os.getenv("UPSTAGE_API_KEY"):
        print("🚨 경고: UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.")

    try:
        model = get_solar_model()
        print(f"✅ 로드된 모델: {model.model}")

        # 간단한 테스트 호출
        # response = model.invoke("안녕, 너는 누구니?")
        # print(f"테스트 응답: {response.content}")

    except Exception as e:
        print(f"❌ 모델 로드 중 에러 발생: {e}")
