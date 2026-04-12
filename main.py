import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from rich import box # TUI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

load_dotenv()

console = Console() # Rich 콘솔 (컬러 출력용)

MODEL = "claude"  # "claude" | "openai"

# ── 토큰 사용량 누적 ──────────────────────────────────────────────────────────
_total_input_tokens: int = 0
_total_output_tokens: int = 0

# AI 별 모델 매핑 — simple(가볍고 빠른)과 advanced(고품질)
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
    """MODEL 전역변수에 따라 실제 모델 ID를 반환"""
    model_map = MODELS_MAP.get(MODEL, MODELS_MAP["claude"])
    return model_map.get(model_type, model_map["advanced"])


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON을 추출. 코드펜스(```json```) 감싸져 있어도 처리.
    세 가지 전략을 순차 시도: ①```json 블록 ②``` 블록 ③가장 바깥쪽 {} 매칭"""
    import re

    # 1: ```json ... ``` 블록에서 추출
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 2: 레이블 없는 ``` ... ``` 블록
    match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3: 중괄호 깊이 추적으로 가장 바깥쪽 JSON 객체 추출
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return json.loads(text)  # 원본 그대로 시도 (JSONDecodeError 전파)


# ── 중간 결과 저장 ───────────────────────────────────────────
# 각 단계의 중간 결과(분석, 퀴즈, 답안 등)를 파일로 저장해서 중간에 에러가 나도 재개(resume)할 수 있게 하는 구조 (API 호출 비용 아끼기)

_SESSION_DIR: Optional[Path] = None

def _save_artifact(name: str, content: str) -> None:
    """중간 결과를 세션 디렉터리에 JSON 파일로 저장"""
    if _SESSION_DIR is None:
        return
    path = _SESSION_DIR / f"{name}.json"
    path.write_text(content, encoding="utf-8")
    console.print(f"  [dim]저장됨 → {path}[/dim]")
    console.print()


def _load_artifact(session_dir: Path, name: str) -> Optional[str]:
    """세션 디렉터리에서 이전에 저장된 아티팩트를 읽어옴"""
    path = session_dir / f"{name}.json"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _find_resumable_session(material_path: str) -> Optional[Path]:
    """같은 자료에 대해 미완료 세션이 있는지 ./outputs에서 탐색.
    가장 최근 세션부터 확인하며, completed=False인 세션을 반환."""
    outputs = Path("./outputs")
    if not outputs.exists():
        return None
    sessions = sorted(
        (s for s in outputs.iterdir() if s.is_dir()),
        reverse=True,
    )
    for s in sessions:
        meta_path = s / "session.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("material") == material_path and not meta.get("completed"):
            return s
    return None


def _init_session(material_path: str) -> None:
    """새 세션 디렉터리 생성 + 메타데이터(session.json) 기록"""
    global _SESSION_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _SESSION_DIR = Path("./outputs") / ts
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "material": material_path,
        "completed": False,
        "started_at": ts,
    }
    (_SESSION_DIR / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"  [dim]세션 시작 → {_SESSION_DIR}[/dim]")
    console.print()


def _resume_session(session_dir: Path) -> None:
    """기존 세션 디렉터리를 현재 세션으로 설정 (재개)"""
    global _SESSION_DIR
    _SESSION_DIR = session_dir
    console.print(f"  [dim]세션 재개 → {_SESSION_DIR}[/dim]")
    console.print()


def _mark_session_complete() -> None:
    """세션 메타데이터에 completed=True 기록 (모든 단계 완료 시)"""
    if _SESSION_DIR is None:
        return
    meta_path = _SESSION_DIR / "session.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["completed"] = True
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ── TUI 표시용 파이프라인 상태 추적 ─────────────────────────────────────────

_pipeline: dict[str, str] = {
    "document-analyzer":  "pending",
    "quiz-generator":     "pending",
    "quiz-validator":     "pending",
    "feedback-evaluator": "pending",
}
_retry_count: int = 0 # 검증 재시도 횟수

# 파이프라인 단계 정의 (키, 한글 라벨)
_STEPS = [
    ("document-analyzer",  "문서 분석"),
    ("quiz-generator",     "퀴즈 생성"),
    ("quiz-validator",     "품질 검증"),
    ("feedback-evaluator", "피드백 생성"),
]

# # 상태별 아이콘·텍스트 매핑
_STATUS_FMT = {
    "pending": ("[dim]○[/dim]",               "[dim]대기[/dim]"),
    "running": ("[yellow bold]●[/yellow bold]", "[yellow]실행 중...[/yellow]"),
    "done":    ("[green bold]✓[/green bold]",   "[green]완료[/green]"),
    "failed":  ("[red bold]✗[/red bold]",       "[red]실패[/red]"),
    "retry":   ("[yellow bold]↻[/yellow bold]", "[yellow]재시도[/yellow]"),
    "loaded":  ("[cyan bold]↑[/cyan bold]",     "[cyan]불러옴[/cyan]"),
}


def _print_pipeline():
    """파이프라인 전체 상태를 Rich 테이블로 출력"""
    t = Table(
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
        expand=False,
        border_style="bright_black",
        title="[bold]파이프라인[/bold]",
        title_justify="left",
    )
    t.add_column(width=3, no_wrap=True)
    t.add_column(min_width=14)
    t.add_column(min_width=16)

    for key, label in _STEPS:
        st = _pipeline.get(key, "pending")
        icon, stat_text = _STATUS_FMT.get(st, _STATUS_FMT["pending"])
        # 검증 단계에서 재시도 중이면 횟수 표시
        if key == "quiz-validator" and _retry_count > 0 and st in ("running", "retry"):
            stat_text += f" [dim]({_retry_count}회)[/dim]"
        t.add_row(icon, label, stat_text)

    console.print(t)


def _set_status(agent: str, status: str):
    """특정 단계의 상태를 갱신하고 파이프라인 테이블을 다시 출력"""
    _pipeline[agent] = status
    _print_pipeline()
    console.print()


# ── Subagent helpers ──────────────────────────────────────────────────────────
# 각 단계를 담당하는 LLM 에이전트 생성·실행 유틸리티

def load_prompt(name: str, prompt_dir: str = "./subagents") -> str:
    """./subagents/{name}.txt에서 시스템 프롬프트를 읽어옴"""
    return (Path(prompt_dir) / f"{name}.txt").read_text(encoding="utf-8")


def create_subagent(name: str, model_type: str):
    """이름과 모델 등급(simple|advanced)으로 서브에이전트 인스턴스 생성"""
    return create_deep_agent(
        model=get_model(model_type),
        system_prompt=load_prompt(name),
        name=name,
    )

# 서브에이전트 생성
analyzer_subagent  = create_subagent("document-analyzer",  "simple")
generator_subagent = create_subagent("quiz-generator",     "advanced")
validator_subagent = create_subagent("quiz-validator",     "simple")
evaluator_subagent = create_subagent("feedback-evaluator", "advanced")


async def _run_subagent(subagent, prompt: str) -> str:
    """서브에이전트를 실행하고 응답을 스트리밍 출력 (실패 시 일반 호출(ainvoke)).
    응답을 JSON으로 파싱 시도 후 반환."""
    global _total_input_tokens, _total_output_tokens

    console.print("  [dim]생성 중...[/dim]")
    full_content = ""
    in_stream = False
    input_tokens = 0
    output_tokens = 0

    try:
        # 스트리밍 모드: 토큰 단위로 실시간 출력 <- 대기 시 진행 중인 이벤트가 있다는 것을 보여주기
        async for event in subagent.astream_events(
            {"messages": [HumanMessage(prompt)]}, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    if not in_stream:
                        console.print("  ", end="", highlight=False)
                        in_stream = True
                    console.print(chunk, end="", highlight=False)
                    full_content += chunk
            elif event["event"] == "on_chat_model_end":
                usage = event.get("data", {}).get("output", {})
                if hasattr(usage, "usage_metadata") and usage.usage_metadata:
                    input_tokens  += usage.usage_metadata.get("input_tokens", 0)
                    output_tokens += usage.usage_metadata.get("output_tokens", 0)
        if in_stream:
            console.print()
    except Exception:
        # 스트리밍 실패 시 일반 호출로 폴백
        result = await subagent.ainvoke({"messages": [HumanMessage(prompt)]})
        full_content = result["messages"][-1].content
        if hasattr(result["messages"][-1], "usage_metadata") and result["messages"][-1].usage_metadata:
            input_tokens  = result["messages"][-1].usage_metadata.get("input_tokens", 0)
            output_tokens = result["messages"][-1].usage_metadata.get("output_tokens", 0)
        console.print(
            f"  [dim]{full_content[:120]}{'...' if len(full_content) > 120 else ''}[/dim]"
        )

    _total_input_tokens  += input_tokens
    _total_output_tokens += output_tokens
    if input_tokens or output_tokens:
        console.print(
            f"  [dim]input: {input_tokens:,} / output: {output_tokens:,} 토큰[/dim]"
        )

    console.print()

    # 응답을 JSON으로 정규화 시도
    try:
        return json.dumps(_parse_json(full_content), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return full_content # JSON 아니면 원문 그대로 반환 -> (저장 시 포맷 오류)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
async def analyze_document(material: str) -> str:
    """학습 자료를 분석하여 concepts, topic, difficulty 등을 추출"""
    _set_status("document-analyzer", "running") # TUI 상태 변환

    result = await _run_subagent(
        analyzer_subagent,
        f"다음 학습 자료를 분석한 결과를 반환하세요.\n<material>\n{material}\n</material>\n",
    ) # 

    _set_status("document-analyzer", "done") # TUI 상태 변환
    _save_artifact("analysis", result) # 결과 저장

    # TUI 출력
    try:
        d = json.loads(result)
        console.print(
            f"  [dim]주제:[/dim] {d.get('topic', '?')}  "
            f"[dim]난이도:[/dim] {d.get('difficulty', '?')}  "
            f"[dim]개념 수:[/dim] {len(d.get('concepts', []))}",
            highlight=False,
        )
        console.print()
    except Exception:
        pass # TUI에 출력하지 않고 넘어가기

    return result


@tool
async def generate_quiz(
    material: str,
    analysis: str,
    validator_feedback: Optional[str] = None,
) -> str:
    """학습 자료 및 분석 결과를 이용하여 Quiz 문항 및 답안 생성"""
    _set_status("quiz-generator", "running")

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
    _set_status("quiz-generator", "done")
    _save_artifact("quiz", result)

    try:
        d = json.loads(result)
        console.print(
            f"  [dim]생성된 문항 수:[/dim] {len(d.get('quiz', []))}",
            highlight=False,
        )
        console.print()
    except Exception:
        pass

    return result


@tool
async def validate_quiz(quiz: str, analysis: str) -> str:
    """퀴즈 품질 검증 및 PASS/FAIL 반환"""
    global _retry_count
    _retry_count += 1
    _set_status("quiz-validator", "running")

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
            _set_status("quiz-validator", "done")
            _save_artifact("validation", result)
            console.print(
                f"  [dim]평균 점수:[/dim] {avg:.0f}  [green bold]PASS[/green bold]",
                highlight=False,
            )
        else:
            _pipeline["quiz-validator"] = "retry"
            _print_pipeline()
            console.print()
            console.print(
                f"  [dim]평균 점수:[/dim] {avg:.0f}  [red bold]FAIL[/red bold] → 재생성",
                highlight=False,
            )
        console.print()
    except Exception:
        _set_status("quiz-validator", "done")
        _save_artifact("validation", result)

    return result


@tool
def collect_user_answers(quiz_json: str) -> str:
    """퀴즈를 화면에 표시하고 사용자 답변을 수집"""
    try:
        data = json.loads(quiz_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "퀴즈 파싱 실패"}, ensure_ascii=False)

    questions = data.get("quiz", [])
    answers: dict[str, str] = {}

    console.print()
    console.rule("[bold cyan]퀴즈[/bold cyan]")
    console.print()

    for q in questions:
        no      = q.get("no", "?")
        q_type  = q.get("type", "short_answer")
        question = q.get("question", "")
        options  = q.get("options", [])

        console.print(Panel(
            f"[bold]Q{no}.[/bold]  {question}",
            subtitle=f"[dim]{q_type}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        ))

        if q_type == "multiple_choice" and options:
            for opt in options:
                console.print(f"   {opt}")
            console.print()
            ans = Prompt.ask("   [cyan]답 (A/B/C/D)[/cyan]").strip().upper()
        elif q_type == "true_false":
            ans = Prompt.ask("   [cyan]답 (O/X)[/cyan]").strip().upper()
        else:
            ans = Prompt.ask("   [cyan]답[/cyan]").strip()

        answers[str(no)] = ans
        console.print()

    console.rule()
    result = json.dumps(answers, ensure_ascii=False)
    _save_artifact("user_answers", result)
    return result


@tool
async def evaluate_feedback(
    quiz_json: str,
    user_answers: str,
    material: str,
    analysis: str,
) -> str:
    """사용자 답변 채점 및 학습 피드백 생성"""
    _set_status("feedback-evaluator", "running")

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
    _set_status("feedback-evaluator", "done")
    _save_artifact("feedback", result)
    return result


@tool
def display_result(feedback_json: str) -> str:
    """채점 결과와 학습 피드백을 TUI로 출력. 모든 분석이 끝난 후 반드시 호출."""
    try:
        data = _parse_json(feedback_json)
    except (json.JSONDecodeError, ValueError):
        console.print(feedback_json)
        return "출력 완료 (JSON 파싱 실패 — 원문 출력)"

    console.print()
    console.rule("[bold cyan]결과[/bold cyan]")
    console.print()

    score = data.get("score")
    if not score:
        items = data.get("item_results", [])
        if items:
            correct_count = sum(1 for item in items if item.get("correct", False))
            score = f"{correct_count}/{len(items)}"
        else:
            score = "?"
    console.print(Panel(
        f"[bold cyan]점수  {score}[/bold cyan]",
        border_style="cyan",
        expand=False,
        padding=(0, 3),
    ))
    console.print()

    items = data.get("item_results", [])
    if items:
        t = Table(
            title="[bold]문항별 결과[/bold]",
            box=box.SIMPLE_HEAD,
            border_style="bright_black",
            padding=(0, 1),
        )
        t.add_column("번호", width=5, justify="center")
        t.add_column("정오", width=4, justify="center")
        t.add_column("정답", style="green", min_width=20)
        t.add_column("내 답", style="dim", min_width=20)

        for item in items:
            correct = item.get("correct", False)
            icon = "[green]✓[/green]" if correct else "[red]✗[/red]"
            correct_ans = str(item.get("correct_answer", ""))
            user_ans    = str(item.get("user_answer", "")) if not correct else "[dim]-[/dim]"
            t.add_row(str(item.get("no", "?")), icon, correct_ans, user_ans)

        console.print(t)
        console.print()

    weak   = data.get("weak_areas", [])
    strong = data.get("strong_areas", [])

    if weak:
        tags = "  ".join(f"[red]{w['concept_name']}[/red]" for w in weak)
        console.print(f"[bold]취약 개념:[/bold]  {tags}")
    if strong:
        tags = "  ".join(f"[green]{s['concept_name']}[/green]" for s in strong)
        console.print(f"[bold]강한 개념:[/bold]  {tags}")

    next_study = data.get("next_study", "")
    if next_study:
        console.print()
        console.print(Panel(
            next_study,
            title="[bold yellow]학습 추천[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        ))

    console.print()
    _mark_session_complete()
    return "출력 완료"


# ── Orchestrator ──────────────────────────────────────────────────────────────

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
- 재시도는 최대 1회까지 허용한다. 2회 후에도 FAIL이면 가장 최근 퀴즈를 사용한다.
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


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Study Agent")
    parser.add_argument("material_file", help="학습 자료 파일 경로")
    args = parser.parse_args()

    material_path = args.material_file
    material = Path(material_path).read_text(encoding="utf-8")

    # 타이틀 배너 출력
    console.print()
    console.print(Panel(
        "[bold cyan]Study Agent[/bold cyan]  [dim]AI 기반 학습 퀴즈 생성기[/dim]\n"
        f"[dim]자료: {material_path}[/dim]",
        border_style="cyan",
        padding=(0, 2),
        expand=False,
    ))
    console.print()

    # ── 이전 미완료 세션 탐색 및 재개 여부 확인 ──────────────────────────────
    # API 비용 절약을 위해, 같은 자료에 대한 미완료 세션이 있으면 중간 결과를 재활용
    resume_dir = _find_resumable_session(material_path)
    resume_context = ""

    if resume_dir:
        console.print(Panel(
            f"[yellow]미완료 세션이 있습니다:[/yellow] [dim]{resume_dir}[/dim]\n"
            f"저장된 파일: {', '.join(p.name for p in resume_dir.glob('*.json') if p.name != 'session.json')}",
            border_style="yellow",
            padding=(0, 1),
            expand=False,
        ))
        console.print()

        if Confirm.ask("  이어서 진행할까요?", default=True):
            _resume_session(resume_dir)

            # 저장된 중간 결과 로드
            saved_analysis = _load_artifact(resume_dir, "analysis")
            saved_quiz     = _load_artifact(resume_dir, "quiz")

            if saved_analysis:
                _pipeline["document-analyzer"] = "loaded"
                resume_context += f"\n<saved_analysis>\n{saved_analysis}\n</saved_analysis>\n"
            if saved_quiz:
                _pipeline["quiz-generator"]  = "loaded"
                _pipeline["quiz-validator"] = "loaded"
                resume_context += f"\n<saved_quiz>\n{saved_quiz}\n</saved_quiz>\n"
        else:
            _init_session(material_path) # 결과 저장 directory & 메타데이터만 생성. 실행 단계는 orchestrator가 자동으로 실행
    else:
        _init_session(material_path)

    _print_pipeline()
    console.print()

    # ── 오케스트레이터에게 전달할 메시지 구성 ─────────────────────────────────
    base_message = (
        "다음 학습 자료로 퀴즈를 생성하고, 사용자 답변을 채점해 주세요.\n"
        f"<material>\n{material}\n</material>"
    )
    if resume_context:
        # 재개 시: 저장된 중간 결과를 메시지에 포함하여 오케스트레이터가 건너뛸 수 있게 함
        base_message += (
            "\n\n이전 세션에서 저장된 중간 결과가 있습니다. "
            "저장된 데이터를 그대로 활용하고 남은 단계부터 진행하세요."
            + resume_context
        )

    # 오케스트레이터 실행 — 내부적으로 도구들을 자율 호출
    await main_agent.ainvoke({"messages": [HumanMessage(base_message)]})

    if _total_input_tokens or _total_output_tokens:
        console.print()
        console.print(
            f"  [dim]총 사용 토큰  input: {_total_input_tokens:,} / output: {_total_output_tokens:,}[/dim]"
        )
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
