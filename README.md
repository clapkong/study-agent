# Study Agent

학습 자료(현재 .txt 파일만 지원)를 넣으면 Multi-Agent AI가 퀴즈를 생성하고, 답변을 채점하여 개념별 맞춤 피드백을 제공하는 CLI 학습 도구

퀴즈 생성 → 검증 → 채점의 각 단계를 독립된 서브 에이전트가 맡고, 오케스트레이터가 순서를 제어한다.

## 아키텍처
오케스트레이터 1개 + 서브에이전트 4개. 단순 작업(분석, 검증)은 haiku, 복잡한 작업(생성, 피드백)은 sonnet을 사용한다. 복잡도는 모델이 수행하는 작업 자체의 논리적 복잡도 및 입력 컨텍스트의 양과 복잡도를 고려하였다. 

```
                    ┌────────────────────────┐
                    │      오케스트레이터      │
                    │   (main-orchestrator)   │
                    │   claude-sonnet-4-6     │
                    └───────────┬────────────┘
                                │
          ┌──────────┬──────────┼──────────┐
          ▼          ▼          ▼          ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │ document ││  quiz-   ││  quiz-   ││ feedback │
    │-analyzer ││generator ││validator ││evaluator │
    │  haiku   ││  sonnet  ││  haiku   ││  sonnet  │
    └──────────┘└──────────┘└──────────┘└──────────┘
                      ↑           │
                      └───────────┘
                   검증 FAIL 시 최대 N회 재생성
```

## 에이전트

| 에이전트 | 역할 | 모델 |
|---------|------|------|
| document-analyzer | 자료에서 주제·핵심 개념·난이도 추출 | haiku |
| quiz-generator | 분석 결과 기반으로 퀴즈 JSON 생성 | sonnet |
| quiz-validator | 퀴즈 품질 채점 (0~100), PASS/FAIL 판정 | haiku |
| feedback-evaluator | 사용자 답변 채점 + 취약 개념 분석 + 학습 추천 | sonnet |

## 실행 흐름

```
main()
  └─ main_agent (Orchestrator)
        ├─ analyze_document     → document-analyzer 서브에이전트
        ├─ generate_quiz        → quiz-generator 서브에이전트
        ├─ validate_quiz        → quiz-validator 서브에이전트
        │      └─ FAIL → generate_quiz 재호출 (최대 max_retries회)
        ├─ collect_user_answers → 터미널 입력 (사용자 개입. 답안 작성)
        ├─ evaluate_feedback    → feedback-evaluator 서브에이전트
        └─ display_result       → 결과 출력 후 세션 완료 표시
```

각 단계 결과는 `outputs/<timestamp>/`에 JSON으로 저장된다.  
실행 중간 중단할 경우, 같은 자료로 재실행하면 저장된 결과를 재활용하여 중간 단계부터 이어서 진행하여 불필요한 API 호출을 줄일 수 있도록 설계하였다.

## 동작 원리

오케스트레이터는 LangChain 에이전트로, 6개 도구와 호출 규칙만 시스템 프롬프트로 받는다. 오케스트레이터는 주어진 정보를 기반으로 다음에 어떤 tool을 호출할지 직접 판단한다. 각 도구 안에서 서브에이전트를 `ainvoke`로 호출하고, 응답을 JSON 파싱해서 다음 도구의 입력으로 전달한다.

분석 단계에서 추출한 개념마다 ID를 부여하고, 이 ID가 퀴즈 문항 태깅 → 채점 → 피드백까지 파이프라인 전체를 관통한다. `validate_quiz`가 FAIL을 반환하면 재생성 지침을 `generate_quiz`에 넘겨서 다시 호출하고, `max_retries` 초과 시 마지막 퀴즈를 그대로 사용한다.

각 단계 결과는 `outputs/<timestamp>/`에 JSON으로 저장된다. 중간에 끊겨도 같은 자료로 재실행하면 저장된 데까지 건너뛰고 이어서 진행한다.

## 실행 결과

[🔗 YouTube 실행 영상](https://youtu.be/jGcoyYbuI9U)

데모 영상에서는 `test_files/biology.txt`(세포의 구조와 기능)를 기본 설정으로 실행해보았다.

오케스트레이터가 분석 → 생성 → 검증 → 답변 수집 → 채점 → 출력 순서를 실행하였다. 검증은 처음 생성한 퀴즈가 평균 92점으로 PASS해서 재생성 루프를 진행하지는 않았고, 채점 결과 5/10에서 틀린 문항의 개념 ID를 역추적해 취약 개념 6개를 식별하였다. 세션 완료 후 각 단계별 JSON이 `outputs/`에 저장되고 `session.json`이 `completed: true`로 갱신되었다. 

본 데모에서 저장된 결과물 JSON 파일은 GitHub에 예시로 포함되어 있으며, `outputs/20260413_151917`에서 확인할 수 있다. 
---

## 디렉터리 구조

```
main.py                  # 기본 CLI (표준 출력)
main_tui.py              # Rich TUI (파이프라인 상태 테이블, 컬러 출력)
subagents/
  document-analyzer.txt  # 문서 분석 시스템 프롬프트
  quiz-generator.txt     # 퀴즈 생성 시스템 프롬프트
  quiz-validator.txt     # 품질 검증 시스템 프롬프트
  feedback-evaluator.txt # 채점·피드백 시스템 프롬프트
test_files/              # 샘플 학습 자료
  biology.txt
  economics.txt
  computer_science.txt
outputs/                 # 세션별 결과 저장 (자동 생성)
  <timestamp>/
    session.json         # 세션 메타데이터 (material, completed, started_at)
    analysis.json        # 문서 분석 결과
    quiz.json            # 생성된 퀴즈
    validation.json      # 검증 결과
    user_answers.json    # 사용자 답변
    feedback.json        # 채점 및 피드백
```

## 설치

```bash
pip install deepagents langchain-core rich python-dotenv
```

## 환경 변수

`.env.example`을 복사해 `.env`를 만든 뒤 API 키를 입력한다.

```bash
cp .env.example .env
```

## 실행

```bash
# 기본 (표준 CLI)
python main.py <학습자료파일>

# Rich TUI (파이프라인 상태 시각화)
python main_tui.py <학습자료파일>

# 예시
python main_tui.py test_files/biology.txt
```

### 옵션

```
--provider {claude,openai}          AI 제공자 (기본값: claude)
--num-questions N                   생성할 문항 수 (기본값: 10)
--difficulty {auto,easy,medium,hard} 난이도 — auto면 자료에서 자동 판단 (기본값: auto)
--question-types TYPES              문항 유형, 콤마 구분 (기본값: multiple_choice,true_false,short_answer)
--max-retries N                     퀴즈 검증 실패 시 최대 재생성 횟수 (기본값: 1)
--output-dir PATH                   결과 저장 디렉터리 (기본값: ./outputs)
```

```bash
# 예시: 20문항, 어려운 난이도, OpenAI 사용
python main_tui.py test_files/computer_science.txt \
  --provider openai \
  --num-questions 20 \
  --difficulty hard
```

## 모델 구성

| 역할 | Claude | OpenAI |
|------|--------|--------|
| 오케스트레이터, 퀴즈 생성, 피드백 생성 | claude-sonnet-4-6 | gpt-4o |
| 문서 분석, 퀴즈 품질 검증 | claude-haiku-4-5 | gpt-4o-mini |

⚠️ **주의:** 코드 상으로는 OpenAI도 구현되어 있으나, 현재 기준 대부분의 디버깅이 현재 Claude 기준으로 진행되었기 때문에 OpenAI로 실행할 경우 제대로 작동하지 않을 수 있습니다.