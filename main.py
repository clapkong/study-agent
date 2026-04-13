import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# .env 파일에서 API 키 등 환경변수 로드
load_dotenv()

def _parse_args():
    parser = argparse.ArgumentParser(description="Study Agent — AI 기반 학습 퀴즈 생성기")
    parser.add_argument("material_file", help="학습 자료 파일 경로")
    parser.add_argument(
        "--provider", choices=["claude", "openai"], default="claude",
        help="AI 제공자 (기본값: claude)",
    )
    parser.add_argument(
        "--output-dir", default="./outputs", metavar="PATH",
        help="결과 저장 디렉터리 (기본값: ./outputs)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=5, metavar="N",
        help="생성할 문항 수 (기본값: 5)",
    )
    parser.add_argument(
        "--difficulty", choices=["auto", "easy", "medium", "hard"], default="auto",
        help="난이도 — auto면 자료에서 자동 판단 (기본값: auto)",
    )
    parser.add_argument(
        "--question-types", default="multiple_choice,true_false,short_answer", metavar="TYPES",
        help="문항 유형, 콤마 구분 (기본값: multiple_choice,true_false,short_answer)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=1, metavar="N",
        help="퀴즈 검증 실패 시 최대 재생성 횟수 (기본값: 1)",
    )
    return parser.parse_args()


# 모듈 로드 시 바로 파싱 → 서브에이전트 생성 전에 MODEL/OUTPUT_DIR 확정
_args = _parse_args()

# 사용할 AI 제공자 및 출력 경로
MODEL = _args.provider
_OUTPUT_DIR = _args.output_dir

# 세션 전체 누적 토큰 사용량 추적
_total_input_tokens: int = 0
_total_output_tokens: int = 0

# AI 제공자별 모델 매핑 (simple | advanced)
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

def get_model(model_type: str) -> str:
    """MODEL 전역변수에 따라 실제 모델 ID를 반환."""
    model_map = MODELS_MAP.get(MODEL, MODELS_MAP["claude"])
    return model_map.get(model_type, model_map["advanced"])

def _parse_json(text: str) -> dict:
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    obj, _ = json.JSONDecoder().raw_decode(text, text.find('{')) # 에러 시 에러 전파
    return obj


# ── 세션 관리 ─────────────────────────────────────────────────────────────────
# 각 단계(분석, 퀴즈 생성, 검증 등)의 결과를 세션 디렉터리에 JSON으로 보관. 
# 중단된 세션이 있으면 API 호출 비용을 아끼기 위해 중간 결과 재활용.

class Session:
    def __init__(self, path: Path):
        self.path = path

    def save(self, name: str, content: str) -> None:
        """중간 결과를 {name}.json으로 저장"""
        p = self.path / f"{name}.json"
        p.write_text(content, encoding="utf-8")
        print(f"  저장됨 → {p}\n")

    def load(self, name: str) -> Optional[str]:
        """저장된 아티팩트를 읽어옴. 없으면 None 반환."""
        p = self.path / f"{name}.json"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def mark_complete(self) -> None:
        """session.json에 completed=True를 기록"""
        meta_path = self.path / "session.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["completed"] = True
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def new(cls, material_path: str, output_dir: str) -> "Session":
        """타임스탬프 디렉터리를 생성하고 새 세션을 반환"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(output_dir) / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {"material": material_path, "completed": False, "started_at": ts}
        (session_dir / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  세션 시작 → {session_dir}\n")
        return cls(session_dir)

    @classmethod
    def find_resumable(cls, material_path: str, output_dir: str) -> Optional["Session"]:
        """같은 자료에 대해 미완료된 가장 최근 세션을 탐색"""
        outputs = Path(output_dir)
        if not outputs.exists():
            return None
        for s in sorted(outputs.iterdir(), reverse=True):
            meta_path = s / "session.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("material") == material_path and not meta.get("completed"):
                print(f"  세션 재개 → {s}\n")
                return cls(s)
        return None

    @classmethod
    def resolve(cls, material_path: str, output_dir: str) -> tuple["Session", str]:
        """세션 탐색 → 재개 여부 확인 → 아티팩트 로드를 한 번에 처리.
        반환: (session, resume_context)"""
        prior = cls.find_resumable(material_path, output_dir)

        if prior:
            saved_files = ', '.join(p.name for p in prior.path.glob('*.json') if p.name != 'session.json')
            print(f"미완료 세션이 있습니다: {prior.path}")
            print(f"저장된 파일: {saved_files}\n")

            ans = input("  이어서 진행할까요? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                resume_context = ""
                saved_analysis = prior.load("analysis")
                saved_quiz     = prior.load("quiz")
                if saved_analysis:
                    resume_context += f"\n<saved_analysis>\n{saved_analysis}\n</saved_analysis>\n"
                if saved_quiz:
                    resume_context += f"\n<saved_quiz>\n{saved_quiz}\n</saved_quiz>\n"
                return prior, resume_context

        return cls.new(material_path, output_dir), ""


_session: Optional[Session] = None  # 현재 활성 세션


# ── 서브에이전트 헬퍼 ─────────────────────────────────────────────────────────
# 각 서브에이전트는 독립된 역할(분석, 생성, 검증, 평가)을 수행하며,
# 시스템 프롬프트는 ./subagents/{name}.txt에서 로드한다.

def load_prompt(name: str, prompt_dir: str = "./subagents") -> str:
    """./subagents/{name}.txt에서 시스템 프롬프트를 읽어옴"""
    return (Path(prompt_dir) / f"{name}.txt").read_text(encoding="utf-8")


def create_subagent(name: str, model_type: str):
    """서브에이전트 인스턴스 생성.
    name: 프롬프트 파일명 및 에이전트 식별자
    model_type: "simple" 또는 "advanced" — 작업 복잡도에 맞춰 선택"""
    return create_deep_agent(
        model=get_model(model_type),
        system_prompt=load_prompt(name),
        name=name,
    )

# 4개의 서브에이전트 생성
analyzer_subagent  = create_subagent("document-analyzer",  "simple")
generator_subagent = create_subagent("quiz-generator",     "advanced")
validator_subagent = create_subagent("quiz-validator",     "simple")
evaluator_subagent = create_subagent("feedback-evaluator", "advanced")


async def _run_subagent(subagent, prompt: str) -> str:
    """서브에이전트를 비동기 실행하고 응답 텍스트를 반환.

    동작 흐름:
    1. astream_events로 스트리밍 실행을 시도하여 실시간 출력
    2. 스트리밍 실패 시 ainvoke로 fallback (일반 호출)
    3. 토큰 사용량을 전역 카운터에 누적
    4. 응답을 JSON 파싱 시도 → 성공하면 정규화된 JSON 문자열, 실패하면 원문 반환
    """
    global _total_input_tokens, _total_output_tokens
    input_tokens = 0
    output_tokens = 0


    print("  생성 중...")
    result = await subagent.ainvoke({"messages": [HumanMessage(prompt)]})
    full_content = result["messages"][-1].content
    print(f"  {full_content[:120]}{'...' if len(full_content) > 120 else ''}") # 생성 미리보기

    # 토큰 관리
    if hasattr(result["messages"][-1], "usage_metadata") and result["messages"][-1].usage_metadata:
        input_tokens  = result["messages"][-1].usage_metadata.get("input_tokens", 0)
        output_tokens = result["messages"][-1].usage_metadata.get("output_tokens", 0)

    _total_input_tokens  += input_tokens
    _total_output_tokens += output_tokens

    if input_tokens or output_tokens:
        print(f"  input: {input_tokens:,} / output: {output_tokens:,} 토큰")
    print()

    # JSON 정규화 시도 — LLM 응답이 유효한 JSON이면 정규화된 문자열로,
    # 아니면 원문 그대로 반환 (후속 단계에서 재파싱 시도 가능)
    try:
        return json.dumps(_parse_json(full_content), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return full_content


# ── 도구(Tool) 정의 ───────────────────────────────────────────────────────────
# LangChain의 @tool 데코레이터로 정의된 함수들은
# 메인 오케스트레이터 에이전트가 필요에 따라 호출할 수 있는 도구가 된다.

@tool
async def analyze_document(material: str) -> str:
    """학습 자료를 분석하여 주제(topic), 핵심 개념(concepts),
    난이도(difficulty), 언어 등의 메타데이터를 추출.
    파이프라인의 첫 번째 단계."""

    print("\n[문서 분석] 실행 중...")

    result = await _run_subagent(
        analyzer_subagent,
        f"다음 학습 자료를 분석한 결과를 반환하세요.\n<material>\n{material}\n</material>\n",
    )

    print("[문서 분석] 완료")

    if _session:
        _session.save("analysis", result)

    # 분석 결과 요약 출력
    try:
        d = json.loads(result)
        print(f"  주제: {d.get('topic', '?')}  난이도: {d.get('difficulty', '?')}  개념 수: {len(d.get('concepts', []))}")
        print()
    except Exception:
        pass

    return result


@tool
async def generate_quiz(
    material: str,
    analysis: str,
    validator_feedback: Optional[str] = None,
) -> str:
    """학습 자료 + 분석 결과를 기반으로 퀴즈(문항 + 정답)를 JSON으로 생성.
    validator_feedback가 주어지면 이전 검증에서 지적된 문제를 반영하여 재생성."""

    print("\n[퀴즈 생성] 실행 중...")

    # 검증 실패 피드백이 있으면 프롬프트에 포함하여 품질 개선 유도
    feedback_block = (
        f"\n<validator_feedback>\n{validator_feedback}\n</validator_feedback>\n"
        if validator_feedback else ""
    )
    prompt = (
        "다음 학습 자료와 분석 결과를 참고하여 JSON 형식의 퀴즈를 생성하세요.\n"
        f"<material>\n{material}\n</material>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n"
        + feedback_block
    )
    result = await _run_subagent(generator_subagent, prompt)

    print("[퀴즈 생성] 완료")

    if _session:
        _session.save("quiz", result)

    try:
        d = json.loads(result)
        print(f"  생성된 문항 수: {len(d.get('quiz', []))}")
        print()
    except Exception:
        pass

    return result


@tool
async def validate_quiz(quiz: str, analysis: str) -> str:
    """생성된 퀴즈의 품질을 0~100 점수로 검증하고 PASS/FAIL 판정.
    FAIL이면 regeneration_instruction(재생성 지침)을 함께 반환하여
    오케스트레이터가 generate_quiz를 재호출할 수 있게 한다."""

    print("\n[품질 검증] 실행 중...")

    result = await _run_subagent(
        validator_subagent,
        "분석 결과를 참고하여 아래 퀴즈의 품질을 검증하고, pass/fail 여부를 반환하세요.\n"
        f"<quiz>\n{quiz}\n</quiz>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n",
    )

    try:
        d = json.loads(result)
        passed = d.get("pass", False)
        scores = d.get("scores", {})
        avg = sum(scores.values()) / len(scores) if scores else 0

        if passed:
            print(f"[품질 검증] 완료  평균 점수: {avg:.0f}  PASS")
            if _session:
                _session.save("validation", result)
        else:
            # FAIL인 경우 아티팩트를 저장하지 않음 — 재생성 후 새 퀴즈로 대체 예정
            print(f"[품질 검증] FAIL  평균 점수: {avg:.0f} → 재생성")
        print()
    except Exception:
        print("[품질 검증] 검증을 실패하였습니다.")
        if _session:
            _session.save("validation", result)

    return result


@tool
def collect_user_answers(quiz_json: str) -> str:
    """퀴즈를 터미널에 표시하고 사용자로부터 답변을 입력받아 수집.
    동기 함수 — input()으로 사용자 입력을 대기한다.
    반환값: {"문항번호": "사용자답변", ...} 형태의 JSON 문자열."""

    try:
        data = json.loads(quiz_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "퀴즈 파싱 실패"}, ensure_ascii=False)

    questions = data.get("quiz", [])
    answers: dict[str, str] = {}

    print()
    print("=" * 40)
    print("퀴즈")
    print("=" * 40)
    print()

    for q in questions:
        no      = q.get("no", "?")
        q_type  = q.get("type", "short_answer")  # multiple_choice | true_false | short_answer
        question = q.get("question", "")
        options  = q.get("options", [])

        print(f"Q{no}. {question}  [{q_type}]")

        # 문항 유형에 따라 입력 방식 분기
        if q_type == "multiple_choice" and options:
            for opt in options:
                print(f"   {opt}")
            print()
            ans = input("   답 (A/B/C/D): ").strip().upper()
        elif q_type == "true_false":
            ans = input("   답 (O/X): ").strip().upper()
        else:  # short_answer 등
            ans = input("   답: ").strip()

        answers[str(no)] = ans
        print()

    print("=" * 40)
    result = json.dumps(answers, ensure_ascii=False)
    if _session:
        _session.save("user_answers", result)
    return result


@tool
async def evaluate_feedback(
    quiz_json: str,
    user_answers: str,
    material: str,
    analysis: str,
) -> str:
    """사용자 답변을 정답과 대조하여 채점하고,
    학습 자료·분석 결과를 참고하여 개념별 맞춤 피드백을 생성.
    출력: item_results(문항별 정오), weak_areas, strong_areas, next_study 등."""

    print("\n[피드백 생성] 실행 중...")

    # 퀴즈 JSON에서 문항(quiz)과 정답(answers)을 분리하여 전달
    try:
        quiz_data = _parse_json(quiz_json)
    except (json.JSONDecodeError, ValueError):
        quiz_data = {}

    try:
        answers_data = _parse_json(user_answers)
    except (json.JSONDecodeError, ValueError):
        answers_data = {}

    prompt = (
        "사용자의 답안을 채점하고 학습 자료 및 분석 결과 기반으로 피드백하시오.\n"
        f"<quiz>\n{json.dumps(quiz_data.get('quiz', {}), ensure_ascii=False, indent=2)}\n</quiz>\n\n"
        f"<answers>\n{json.dumps(quiz_data.get('answers', {}), ensure_ascii=False, indent=2)}\n</answers>\n\n"
        f"<user_answers>\n{json.dumps(answers_data, ensure_ascii=False, indent=2)}\n</user_answers>\n\n"
        f"<material>\n{material}\n</material>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n"
    )
    result = await _run_subagent(evaluator_subagent, prompt)

    print("[피드백 생성] 완료")
    if _session:
        _session.save("feedback", result)
    return result


@tool
def display_result(feedback_json: str) -> str:
    """채점 결과와 학습 피드백을 사람이 읽기 좋은 형태로 터미널에 출력.
    파이프라인의 마지막 단계 — 호출 후 세션을 완료(completed=True)로 표시한다."""
    try:
        data = _parse_json(feedback_json)
    except (json.JSONDecodeError, ValueError):
        # JSON 파싱 실패 시 원문 그대로 출력
        print(feedback_json)
        return "출력 완료 (JSON 파싱 실패 — 원문 출력)"

    print()
    print("=" * 40)
    print("결과")
    print("=" * 40)
    print()

    # ── 총점 계산 ──
    score = data.get("score")
    if not score:
        items = data.get("item_results", [])
        if items:
            correct_count = sum(1 for item in items if item.get("correct", False))
            score = f"{correct_count}/{len(items)}"
        else:
            score = "?"
    print(f"점수: {score}")
    print()

    # ── 문항별 정답표 출력 ──
    items = data.get("item_results", [])
    if items:
        print("문항별 결과:")
        print(f"  {'번호':^4}  {'정답 여부':^4}  {'정답':<20}  {'내 답':<20}")
        print(f"  {'-'*4}  {'-'*4}  {'-'*20}  {'-'*20}")
        for item in items:
            correct = item.get("correct", False)
            icon = "O" if correct else "X"
            correct_ans = str(item.get("correct_answer", ""))
            user_ans    = str(item.get("user_answer", "")) if not correct else "-"
            print(f"  {str(item.get('no', '?')):^4}  {icon:^4}  {correct_ans:<20}  {user_ans:<20}")
        print()

    # ── 강점/약점 개념 요약 ──
    weak   = data.get("weak_areas", [])
    strong = data.get("strong_areas", [])

    if weak:
        tags = ", ".join(w['concept_name'] for w in weak)
        print(f"취약 개념: {tags}")
    if strong:
        tags = ", ".join(s['concept_name'] for s in strong)
        print(f"강한 개념: {tags}")

    # ── 후속 학습 추천 ──
    next_study = data.get("next_study", "")
    if next_study:
        print()
        print("학습 추천:")
        print(next_study)

    print()
    if _session:
        _session.mark_complete()  # 세션 완료 표시 — 이후 재개 대상에서 제외됨
    return "출력 완료"


# ── 메인 오케스트레이터 에이전트 ──────────────────────────────────────────────
# 위에서 정의한 6개의 도구를 사용하여 전체 파이프라인을 자율적으로 조율.
# 흐름: analyze → generate → validate → (FAIL이면 재생성) → collect → evaluate → display

main_agent = create_deep_agent(
    model=get_model("advanced"),
    tools=[
        analyze_document,
        generate_quiz,
        validate_quiz,
        collect_user_answers,
        evaluate_feedback,
        display_result,
    ],
    system_prompt="""당신은 학습 퀴즈 세션 오케스트레이터입니다.
사용자에게 학습 자료 기반 퀴즈를 출제하고, 답변을 채점하며, 맞춤형 피드백을 제공하는 것이 목표입니다.

## 사용 가능한 도구

| 도구 | 역할 |
|------|------|
| analyze_document | 학습 자료에서 주제·개념·난이도·언어를 추출 |
| generate_quiz | 분석 결과를 기반으로 퀴즈(문항+정답) 생성 |
| validate_quiz | 퀴즈 품질을 0~100 점수로 검증, PASS/FAIL 반환 |
| collect_user_answers | 퀴즈를 화면에 표시하고 사용자 답변을 수집 |
| evaluate_feedback | 사용자 답변을 채점하고 개념별 피드백 생성 |
| display_result | 채점 결과와 피드백을 화면에 출력 |

## 판단 기준

**퀴즈 품질 관리**
- validate_quiz 결과가 FAIL이면 regeneration_instruction을 validator_feedback으로 삼아 generate_quiz를 재호출한다.
- 재시도는 사용자 메시지에 명시된 횟수까지 허용한다. 횟수 초과 후에도 FAIL이면 가장 최근 퀴즈를 사용한다.
- 재시도 시 이전 피드백을 반드시 반영해야 한다.

**오류 처리**
- 어떤 도구든 오류(빈 응답, JSON 파싱 실패 등)가 발생하면 1회 재시도한다.
- 재시도 후에도 실패하면 사용자에게 상황을 알리고 중단한다.

**흐름 제어**
- 각 단계는 이전 단계의 결과를 충분히 활용해야 한다.
- collect_user_answers → evaluate_feedback 순서는 반드시 지킨다.
- 모든 처리가 끝난 뒤 반드시 display_result를 호출한다.

**재개 세션**
- 메시지에 <saved_analysis>, <saved_quiz> 태그가 있으면 해당 단계는 이미 완료된 것이다.
- 저장된 결과를 그대로 사용하고 그 다음 단계부터 진행한다.
- 저장된 퀴즈가 있으면 validate_quiz와 generate_quiz는 건너뛰고 collect_user_answers부터 시작한다.
""",
    name="main-orchestrator",
)


# ── 프로그램 진입점 ───────────────────────────────────────────────────────────

async def main():
    # 자료 불러오기
    material_path = _args.material_file
    material = Path(material_path).read_text(encoding="utf-8")
    
    # 출력
    print("\nStudy Agent  AI 기반 학습 퀴즈 생성기")
    print(f"자료: {material_path}\n")

    # ── 세션 탐색 · 재개 여부 확인 · 아티팩트 로드 ──
    global _session
    _session, resume_context = Session.resolve(material_path, _OUTPUT_DIR)

    # ── 오케스트레이터에게 전달할 최종 메시지 구성 ──
    difficulty_str = "자료에서 자동 판단" if _args.difficulty == "auto" else _args.difficulty
    question_types = [t.strip() for t in _args.question_types.split(",")]
    quiz_settings = (
        f"문항 수: {_args.num_questions}개 / "
        f"난이도: {difficulty_str} / "
        f"문항 유형: {', '.join(question_types)} / "
        f"검증 실패 시 최대 재생성 횟수: {_args.max_retries}회"
    )
    base_message = (
        f"다음 학습 자료로 퀴즈를 생성하고, 사용자 답변을 채점해 주세요.\n"
        f"퀴즈 설정: {quiz_settings}\n"
        f"<material>\n{material}\n</material>"
    )
    if resume_context:
        base_message += (
            "\n\n이전 세션에서 저장된 중간 결과가 있습니다. "
            "저장된 데이터를 그대로 활용하고 남은 단계부터 진행하세요."
            + resume_context
        )

    # 오케스트레이터 실행
    await main_agent.ainvoke({"messages": [HumanMessage(base_message)]})

    # 세션 종료 후 총 토큰 사용량 출력
    if _total_input_tokens or _total_output_tokens:
        print(f"\n  총 사용 토큰  input: {_total_input_tokens:,} / output: {_total_output_tokens:,}\n")


if __name__ == "__main__":
    asyncio.run(main())