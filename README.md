# DL 문의 대응 에이전트

사내 메일링그룹(DL) 문의를 **10개 시나리오로 분류**하고 시나리오별 액션
(고정 응답 / 기안 결재 / 메일 전달 / DL API 조회)을 수행하는 에이전트.

- **분류 모델**: Qwen3-4B Q4_K_M (GGUF) — `llama-server` 로컬 서빙
- **오케스트레이션**: LangGraph
- **인터페이스**: FastAPI 웹 채팅 (응답 + 판단 근거 동시 표시)
- **도구**: 전부 스텁(모의 호출). 실제 부작용 없음

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 그래프 도식, 설계 근거, 모듈 배치 |
| [docs/evaluation.md](docs/evaluation.md) | 벤치마크 설계, 지표 정의, 실험 결과와 해석 |
| [docs/slides.html](docs/slides.html) | 발표용 3장 (요약 / 구성 / 평가) — 브라우저로 열면 됨 |
| [guideline.md](guideline.md) | 원본 업무 가이드 (시나리오 정의의 출처) |

## 결과 요약

직접 만든 200문항 벤치마크를 시나리오×난이도로 층화 분할 (train 60 / val 40 / test 100).
val에서만 설계를 고르고 **test는 마지막에 한 번** 돌렸다.

| 테스트셋 100문항 | 분류 정확도 | macro-F1 | End-to-End | 출력 파싱 실패 | 분류 p50 |
|---|---|---|---|---|---|
| **Qwen3-4B v2, τ=0.7** | **0.740** | 0.760 | 0.740 | **0.000** | 371ms |
| 규칙 베이스라인 (LLM 없음) | 0.690 | 0.724 | 0.690 | — | 0ms |

- 기권(τ=0.7) 도입으로 **위험 오류율**(틀린 안내를 자동 발송한 비율) **0.270 → 0.180**
- 프롬프트에 판별 규칙 한 문단을 넣은 것이 **+15%p** — few-shot 예시보다 훨씬 큰 효과
- few-shot 예시는 늘릴수록 **나빠졌다** (k=0 → 10 → 20에서 0.900 → 0.850 → 0.800)
- 베이스라인 대비 +5%p는 n=100에서 신뢰구간이 겹쳐 **우열을 단정할 수 없다**

자세한 근거와 실패 원인 분석은 [docs/evaluation.md](docs/evaluation.md).

---

## 빠른 시작

### 0. 사전 준비

```bash
brew install llama.cpp          # llama-server
python3 --version               # 3.11 이상
```

### 1. 모델 내려받기 (약 2.3 GB)

```bash
mkdir -p models
curl -L -o models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

### 2. 가상환경 + 설치

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

### 3. LLM 서버 기동 (터미널 1)

```bash
./scripts/serve_llm.sh          # 127.0.0.1:8080, OpenAI 호환 API
```

### 4. 웹 채팅 실행 (터미널 2)

```bash
./scripts/run_app.sh            # http://127.0.0.1:8000
```

브라우저에서 좌측에 문의를 입력하면, 우측 패널에 **레이블 확률분포 · 추출 슬롯 ·
도구 호출 인자와 결과 · 실행 경로**가 함께 표시된다.

---

## 벤치마크 재현

```bash
./scripts/run_eval.sh           # 스플릿 생성 → 검증셋 탐색 → 최종 테스트 1회
```

개별 실행:

```bash
.venv/bin/python benchmark/split.py                       # pool.jsonl → train/val/test
.venv/bin/python benchmark/run_eval.py --split val --few-shot 10 --prompt v2
.venv/bin/python benchmark/run_eval.py --split val --system keyword    # 규칙 베이스라인
.venv/bin/python benchmark/compare.py val                 # 결과 비교표
```

### 테스트

llama-server 없이 도는 스모크 테스트(규칙 기반 분류기를 그래프에 주입한다).
라우팅·도구 호출·응답 조립이 시나리오 정의와 어긋나지 않는지 확인한다.

```bash
.venv/bin/python tests/test_agent.py
```

### 데이터 분리

손으로 작성한 200문항(`benchmark/data/pool.jsonl`)을 시나리오(10) × 난이도(3)로
이중 층화 분할한다. 시드가 고정되어 있어 재현된다.

| 스플릿 | 문항 | 용도 |
|---|---|---|
| train | 60 | few-shot 예시 풀, 규칙 베이스라인 작성 |
| val | 40 | 프롬프트 버전 · few-shot 개수 · 기권 임계값 선택 |
| test | 100 | **최종 측정 1회.** 그 외 목적으로 열람하지 않음 |

few-shot 로더는 `train.jsonl` 만 읽는다(`src/dl_agent/prompts.py`).

---

## 설정

`.env` 또는 환경변수로 덮어쓴다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI 호환 엔드포인트 |
| `LLM_MODEL` | `qwen3-4b-q4km` | llama-server `--alias` 와 일치 |
| `GGUF_PATH` | `models/Qwen3-4B-Q4_K_M.gguf` | 모델 파일 |
| `CLS_FEW_SHOT_K` | `0` | 프롬프트에 넣을 예시 수. 검증셋에서 0이 가장 좋았다 |
| `CLS_ABSTAIN_THRESHOLD` | `0.7` | 이보다 확신이 낮으면 시나리오 10으로 기권 |
| `CLS_TOP_LOGPROBS` | `10` | 확률분포를 받을 상위 토큰 수 |
| `APP_PORT` | `8000` | 웹 서버 포트 |

---

## 프로젝트 구조

```
src/dl_agent/     에이전트 본체 (LangGraph 노드, LLM 클라이언트, 도구, 웹 서버)
benchmark/        데이터셋 · 스플릿 · 지표 · 실행기 · 비교표 · 심층 분석
tests/            LLM 없이 도는 스모크 테스트
docs/             구성 도식과 평가 리포트
scripts/          serve_llm.sh · run_app.sh · run_eval.sh
results/          실험 결과 JSON, 도구 감사 로그
models/           GGUF (git에 포함하지 않음)
```

## 주의

모든 도구는 스텁이다. 기안 서식이 실제로 열리거나 메일이 실제로 발송되지 않으며,
`dl_api_get` 은 DL 코드를 시드로 만든 모의 응답을 돌려준다. 호출 이력은
`results/tool_audit.log` 에 남는다.
