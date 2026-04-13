# Study Agent

학습 자료를 분석해 퀴즈를 생성하고, 사용자 답변을 채점하여 맞춤형 피드백을 제공하는 AI 기반 CLI 학습 도구입니다.

## Architecture

```
                    ┌────────────────────────┐
                    │      오케스트레이터      │
                    │   (main-orchestrator)   │
                    └───────────┬────────────┘
                                │
          ┌──────────┬──────────┼──────────┐
          ▼          ▼          ▼          ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │ document ││  quiz-   ││  quiz-   ││ feedback │
    │-analyzer ││generator ││validator ││evaluator │
    └──────────┘└──────────┘└──────────┘└──────────┘
         ↑           ↑           ↑
         └───────────┴───────────┘
           순차 실행 (검증 실패 시 재생성)
```

## Agents

| 에이전트 | 역할 |
|---------|------|
| **document-analyzer** | 학습 자료에서 주제·개념·난이도·대상 추출 |
| **quiz-generator** | 분석 결과 기반 퀴즈(문항+정답+해설) 생성 |
| **quiz-validator** | 퀴즈 품질 검증 (목표 적합성·난이도·커버리지 채점 후 PASS/FAIL) |
| **feedback-evaluator** | 사용자 답변 채점 및 개념별 취약점·학습 추천 생성 |

## 구조

```
main.py                  # 오케스트레이터 + 서브에이전트 툴 + TUI
subagents/
  document-analyzer.txt  # 학습 자료 분석 프롬프트
  quiz-generator.txt     # 퀴즈 생성 프롬프트
  quiz-validator.txt     # 퀴즈 품질 검증 프롬프트
  feedback-evaluator.txt # 채점 및 피드백 프롬프트
test_files/              # 테스트용 materials 파일
  biology.txt
  economics.txt
  computer_science.txt
outputs/                 # 세션별 중간 결과 저장 디렉터리 (자동 생성)
  <timestamp>/
    session.json         # 세션 메타데이터
    analysis.json        # 문서 분석 결과
    quiz.json            # 생성된 퀴즈
    validation.json      # 검증 결과
    user_answers.json    # 사용자 답변
    feedback.json        # 채점 및 피드백
```

## 실행 흐름

```
main()
  └─ main_agent (Orchestrator)
        ├─ analyze_document     → document-analyzer 서브에이전트
        ├─ generate_quiz        → quiz-generator 서브에이전트
        ├─ validate_quiz        → quiz-validator 서브에이전트 (FAIL 시 최대 1회 재시도)
        ├─ collect_user_answers → 터미널 입력 (유일한 사용자 개입)
        ├─ evaluate_feedback    → feedback-evaluator 서브에이전트
        └─ display_result       → TUI 결과 출력
```

오케스트레이터(claude-sonnet)가 전체 흐름과 재시도 여부를 직접 판단하며,
문서 분석·퀴즈 생성·검증·채점은 각각 독립된 서브에이전트(claude-haiku / claude-sonnet)가 담당합니다.

각 단계의 결과는 `outputs/<timestamp>/`에 저장되며, 중간에 중단되더라도 같은 자료에 대해 재실행하면
저장된 결과를 불러와 이어서 진행할 수 있습니다 (API 호출 비용 절약).

## 설치

```bash
pip install deepagents langchain-core rich python-dotenv
```

## 환경 변수

`.env.example`을 복사해 `.env`를 만든 뒤 API 키를 입력하세요.

```bash
cp .env.example .env
```

| 변수 | 필수 | 설명 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✓ | Anthropic API 키 |
| `OPENAI_API_KEY` | | OpenAI 모델 사용 시 |

모델 제공자 변경은 `main.py` 상단의 `MODEL = "claude"` 를 직접 수정하세요.

## 실행

```bash
python main.py <학습자료파일>

# 예시
python main.py test_files/biology.txt
```
## 실행 결과
[![Video Label](http://img.youtube.com/vi/jGcoyYbuI9U/0.jpg)](https://youtu.be/jGcoyYbuI9U)

## 모델 구성

| 역할 | Claude | OpenAI |
|------|--------|--------|
| 오케스트레이터, 퀴즈 생성, 피드백 생성 | claude-sonnet-4-6 | gpt-4o |
| 문서 분석, 퀴즈 품질 검증 | claude-haiku-4-5 | gpt-4o-mini |


