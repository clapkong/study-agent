import asyncio
from pathlib import Path
from typing import Optional
import json

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

MODEL = "claude"
MATERIALS = "./test_files/biology.txt"


MODELS_MAP = {
    "claude": {
        "simple": "claude-haiku-4-5-20251001",
        "advanced": "claude-sonnet-4-6",
    },
    "openai": {
        "simple": "openai:gpt-4o-mini",
        "advanced": "openai:gpt-4o",
    },
}

def get_model(model_type):
    model_map = MODELS_MAP(MODEL, MODELS_MAP["claude"])
    model_name = model_map.get(model_type, model_map["advanced"]) # e.g., "openai:gpt-4o"
    return model_name

# ---- Subagent 생성 ----

def load_prompt(name, prompt_dir = "./subagents"):
    file_path = Path(prompt_dir) / f"{name}.txt"
    return file_path.read_text(encoding="utf-8")

def create_subagent(subagent_name, model_type):
    model_name = get_model(model_type)
    
    prompt = load_prompt(subagent_name) # ./subagents/

    return create_deep_agent(
        model=model_name,
        system_prompt=prompt,
        name = subagent_name
    )

analyzer_subagent = create_subagent("document-analyzer", "simple")
generator_subagent = create_subagent("quiz-generator", "advanced")
validator_subagent = create_subagent("quiz-validator", "advanced")
evaluator_subagent = create_subagent("feedback-evaluator", "simple")

# ---- 서브에이전트를 도구로 사용 ----
async def _run_subagent(subagent, prompt) -> str:
    result = await subagent.ainvoke(
        {"messages": [HumanMessage(prompt)]}
    )
    content = result["messages"][-1].content
    return content 
    

@tool
async def analyze_document(material: str) -> str:
    """학습 자료를 분석하여 concepts, topic, difficulty 등을 추출"""

    analyzer_prompt = (
        "다음 학습 자료를 분석한 결과를 반환하세요.\n"
        f"<material>\n{material}\n</material>\n\n"
    )

    analysis = await _run_subagent(analyzer_subagent, analyzer_prompt)

    return analysis


@tool
async def generate_quiz(material: str, analysis: str, validator_feedback: Optional[str]) -> str:
    """학습 자료 및 분석 결과(들)를 이용하여 적절한 Quiz 문항 및 답안 생성"""

    generator_prompt= (
        "다음 학습 자료와 분석 결과를 참고하여 JSON 형식의 퀴즈를 생성하세요.\n"
        f"<material>\n{material}\n</material>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n\n"
        (f"<validator_feedback>\n{validator_feedback}\n</validator_feedback>\n\n" if validator_feedback else "")
    )

    quiz_and_answer = await _run_subagent(generator_subagent, generator_prompt)

    return quiz_and_answer


@tool
async def validate_quiz(quiz: str, analysis:str) -> str:
    """퀴즈 품질 검증 및 PASS/FAIL 반환"""

    validator_prompt = (
            "분석 결과를 참고하여 아래 퀴즈의 품질을 검증하고, pass/fail 여부를 반환하세요.\n"
            f"<quiz>\n{quiz}\n</quiz>\n\n"
            f"<analysis>\n{analysis}\n</analysis>\n\n"
        )
    
    generation_validation = await _run_subagent(validator_subagent, validator_prompt)

    return generation_validation

@tool
async def evaluate_feedback(quiz: str, user_answers:str, material:str, analysis:str) -> str:
    """사용자 답변 채점 및 학습 피드백 생성"""

    grading_prompt = (
        "사용자의 답안을 채점하고 학습 자료 및 분석 결과 기반으로 피드백하시오.\n"
        f"<quiz>\n{json.dumps(quiz.get('quiz', {}), ensure_ascii=False, indent=2)}\n</quiz>\n\n"
        f"<answers>\n{json.dumps(quiz.get('answers', {}), ensure_ascii=False, indent=2)}\n</answers>\n\n"
        f"<user_answers>\n{json.dumps(user_answers, ensure_ascii=False, indent=2)}\n</user_answers>\n\n"
        f"<material>\n{material}\n</material>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n\n"
    )
    
    return await _run_subagent(evaluator_subagent, grading_prompt)


# ---- Orchestrator Agent 생성----
main_agent = create_deep_agent(
    model=get_model("advanced"),
    tools=[analyze_document, generate_quiz, validate_quiz, evaluate_feedback],
    system_prompt="""당신은 퀴즈 생성 오케스트레이터입니다.

학습 자료를 기반으로 퀴즈 생성, 검증, 채점 및 피드백을 수행합니다.

작업 절차:
1. document_analyzer: 개념, 주제, 난이도, 언어 분석
2. quiz_generator: 퀴즈 생성
3. quiz_validator: 퀴즈 품질 검증 (PASS/FAIL)
   - FAIL 시 최대 2회 재시도 (quiz_generator → quiz_validator 반복)
4. feedback_evaluator: 사용자 답변 채점 및 피드백 생성
5. build_result: 최종 결과 정리 및 이력 저장

규칙:
- 각 단계는 순차적으로 실행
- quiz_validator가 FAIL이면 quiz_generator부터 다시 수행
- 반드시 모든 단계를 거쳐야 함
- 최종 출력은 build_result 결과만 반환
""", 
    name="main-orchestrator",
)

