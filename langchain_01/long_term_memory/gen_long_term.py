import common_utils as utils
from langgraph.store.memory import InMemoryStore
from langchain_upstage import ChatUpstage

# Upstage Solar Pro 모델 초기화
solar_model = ChatUpstage(
    model="solar-pro", upstage_api_key="up_tvzv1tg1ranghIp3Q7tWpzWkEErcS"
)

# LangGraph의 InMemoryStore 생성 - 메모리 기반 저장소
# 이 저장소는 사용자별 장기 기억을 저장하고 검색하는 데 사용됨
store = InMemoryStore()  # Create an instance of InMemoryStore

# 사용자 및 애플리케이션 컨텍스트 정의
user_id = "user_001"
application_context = "personal_assistant"

# 네임스페이스 생성 - (user_id, application_context) 튜플로 고유 식별자 생성
namespace = (user_id, application_context)

# 메모리 001 저장 - 사용자의 기본 정보
store.put(
    namespace,
    "memory_001",
    {
        "facts": [
            "사용자는 커피보다 차를 선호함",
            "사용자는 매일 아침 6시에 일어남",
        ],
        "language": "Korean",
    },
)

# 메모리 001 조회 (주석 처리됨)
item = store.get(namespace, "memory_001")
# print(f"저장된 메모리: {item.value}")
# print(item)

# 메모리 검색 (주석 처리됨)
# items = store.search(namespace)
# print(items)

# 메모리 002 저장 - 사용자의 취미 정보
store.put(
    namespace,
    "memory_002",
    {
        "facts": [
            "사용자는 그림 회화 작품을 좋아함",
            "빈센트 반 고흐의 작품을 특히 좋아함",
        ],
    },
)

# 메모리 002 조회 (주석 처리됨)
# item = store.get(namespace, "memory_002")
# print(f"저장된 메모리:, {item.value}")

# 메모리 검색 (주석 처리됨)
# items = store.search(namespace)
# print("-----------")
# print(items)

# Context 클래스 정의 - 사용자 및 애플리케이션 정보를 담는 데이터 클래스
from dataclasses import dataclass


@dataclass
class Context:
    user_id: str
    app_name: str


# LangGraph 미들웨어 - 시스템 프롬프트에 사용자 메모리 주입
# wrap_model_call 데코레이터를 사용하여 모델 호출 전후에 로직 추가
from langchain.agents.middleware import wrap_model_call
from langchain.messages import HumanMessage, SystemMessage


@wrap_model_call
def inject_memory(request, handler):
    """Inject user-specific long-term memory into the system prompt."""
    
    # 컨텍스트 검증 - request.runtime에 context 속성이 있는지 확인
    if not hasattr(request.runtime, "context"):
        raise ValueError("Missing context in request.runtime")
    
    # 현재 사용자 및 애플리케이션 정보 추출
    current_user = request.runtime.context.user_id
    current_app = request.runtime.context.app_name
    
    # 저장소 검증 - request.runtime에 store 속성이 있는지 확인
    if not hasattr(request.runtime, "store"):
        raise ValueError("Missing store in request.runtime")
    
    # 사용자별 메모리 검색 - (current_user, current_app) 네임스페이스로 검색
    memories = request.runtime.store.search((current_user, current_app))
    
    # 사실 정보 추출 - 각 메모리 항목에서 "facts" 키의 값만 안전하게 추출
    extracted_facts = []
    for item in memories:
        if item and "facts" in item.value:
            extracted_facts.extend(item.value["facts"])
    
    # 메모리 내용 생성 - 추출된 사실들을 줄바꿈으로 연결
    memory_content = "\n-".join(extracted_facts) if extracted_facts else "기록된 정보 없음"
    
    # 시스템 메시지 생성 - LangGraph의 시스템 프롬프트에 사용자 메모리 포함
    system_message = f"사용자 관련 장기메모리: {memory_content}"
    
    # 시스템 프롬프트 오버라이드 - 기존 시스템 프롬프트를 사용자 메모리로 대체
    request = request.override(system_prompt=system_message)
    
    return handler(request)


# LangGraph 에이전트 생성 - solar_model과 store를 연결
from langchain.agents import create_agent

# agent = create_agent(
#     model="gpt-5-nano", store=store, context_schema=Context, middleware=[inject_memory]
# )

agent = create_agent(
    model=solar_model, store=store, context_schema=Context, middleware=[inject_memory]
)

if __name__ == "__main__":
    # 사용자 001에 대한 메모리 기반 응답 테스트
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "나에 대해 알고 있는 정보 알려줘"}]},
        context=Context(user_id="user_001", app_name="personal_assistant"),
    )
    # print(response)

    # 사용자 002에 대한 메모리 기반 응답 테스트
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "나에 대해서 알고있는 정보를 알려줘"}
            ]
        },
        context=Context(user_id="user_002", app_name="personal_assistant"),
    )

    print(response)
