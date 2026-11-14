# Surrogate Alpha Search

같은 백테스트 예산 안에서, 트리 기반 대리 모델이 수식 알파 후보의 성과를 미리 예측해 고르면 무작위로 고르는 것보다 통과 시그널을 더 많이 찾는지 검증한 연구입니다.

- 미국(S&P 500)에서 A/B 테스트를 했습니다.
- 미국에서 학습한 모델을 재학습 없이 일본(닛케이 225)에 그대로 적용해 전이 여부를 확인했습니다.
- OOS(2024–2025)는 사전에 고정한 계획에 따라 한 번만 열었습니다.

| 문서 | 내용 |
|---|---|
| [`surrogate_alpha_search_design.md`](surrogate_alpha_search_design.md) | 실험 설계서 (단일 기준, 모든 변경 이력 포함) |
| [`research_results.md`](research_results.md) | 연구 결과 (H1·H3·사후 탐색·H2, 그림 목록) |
| [`CLAUDE.md`](CLAUDE.md) | 구현 규칙 (누수 방지, 고정값, 백테스트 사양) |

## 핵심 결과

| 질문 | 결과 |
|---|---|
| **H1** 같은 예산(B = 2,000)에서 대리 모델이 무작위보다 고유 통과 시그널을 더 찾는가 | **아니다.** 미국·일본 모두 모든 모델이 무작위보다 유의하게 적게 찾음 (무작위 24.1 vs 모델 9.5–15.6, 미국) |
| 실패 원인 (사후 진단) | 예측 상위 선택이 수익 공간의 소수 군집에 몰림. 선택 쌍의 약 61%가 IS 수익 상관 0.7 이상(무작위 10.5%)이라 중복 제거에서 99%가 사라짐 |
| **H3** 모델 간 차이 | 부스팅 3종(LightGBM·XGBoost·CatBoost) 간 차이 없음, Random Forest만 열세 |
| 피처 ablation (사전 등록) | 결합 피처가 고유 통과 수를 유의하게 올렸지만(+3.5개) 무작위에는 못 미침 |
| 사후 탐색: 잔차 타깃 + 확장 피처 + MMR | 미국에서 무작위를 유의하게 넘음 (35.6 vs 24.1). 일본은 같은 방향이지만 보정 후 유의하지 않음 |
| **H2** OOS 품질 (사전 등록 칸) | 사전 고정 규칙상 **판정 불가** (판정 불가 2칸, 열등 0칸). 트리 모델이 고른 시그널의 OOS Sharpe 중앙값이 무작위보다 0.2–0.5 높음 |
| H2 탐색 칸 (잔차 타깃) | 일본에서 OOS 품질 열등. 통과 시그널은 더 많이 찾지만 OOS 품질은 무작위보다 낮음 |

자세한 수치, 신뢰구간, 한계는 [`research_results.md`](research_results.md)에 있습니다.

## 실험 설정 요약

- **수식 풀:** 문법으로 생성한 4만 개. 필드 6개, 연산자 16개, 윈도우 {5, 10, 20, 60, 120}, 깊이 2–5
- **기간:** IS 2016–2021, Test 2022–2023, OOS 2024–2025. 다운로드는 2015년부터 받고, 2015년은 lookback 버퍼로만 씀
- **백테스트:** t일 종가 시그널 → 횡단면 디민 → t+1일 종가 체결 → t+2일 수익 반영. 편도 비용 5bp
- **통과 기준:** 방향 적용 IS 순 Sharpe ≥ 임계값(미국 0.35, 일본 0.40), 같은 방향 Test 순 Sharpe ≥ 0.3(일본 0.0), IS 회전율 ≤ 50%, 기존 통과 시그널과 IS 일별 수익 상관 < 0.7
- **실험:** warm-up W = 1,000, 예산 B = 2,000, 시드 30회. Optuna 50 trials로 튜닝하고, 같은 구조의 수식은 같은 그룹에 묶는 GroupKFold로 교차검증
- **데이터:** yfinance 무료 가격·거래량. 구성 종목은 위키백과 고정 편집판에서 가져옴

## 설치

Python 3.11 이상이 필요하고, 프로젝트 전용 가상환경을 씁니다.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]" matplotlib   # Windows
# .venv/bin/python -m pip install -e ".[dev]" matplotlib     # macOS / Linux
```

계산은 CPU 전제이고 GPU는 쓰지 않습니다.

## 실행 순서

```bash
python -m src.data.universe --market us          # 구성 종목 수집 (download가 없으면 자동 호출)
python -m src.data.download --market us          # 가격·거래량 다운로드, 정제, 캐시
python -m src.data.download --market jp
python -m src.generator.generate                 # 수식 풀 4만 개 → artifacts/pool_v1.parquet
python -m src.features.ast_features              # 기본·확장 피처 추출
python -m src.backtest.engine --market us        # 풀 전수 백테스트 (IS/Test와 OOS를 분리 저장)
python -m src.backtest.engine --market jp
python -m src.experiment.calibrate --market us   # 무작위 arm 시뮬레이션으로 IS 임계값 조정
python -m src.experiment.calibrate --market jp
python -m src.experiment.replay --market us --schema base     # H1·H3 (30시드 × 6 arm)
python -m src.experiment.replay --market us --schema ext      # 피처 ablation
python -m src.experiment.replay --market jp --schema base     # 일본 전이
python -m src.experiment.replay_mmr --market us               # 사후 탐색: MMR
python -m src.experiment.replay --market us --schema ext --target resid   # 사후 탐색: 잔차 타깃
python -m src.analysis.stats                     # 핵심 지표와 검정
python -m src.analysis.equal_budget              # 같은 총 예산 강건성 분석
python -m src.analysis.cluster_diagnostics       # 군집 집중도 사후 진단
python -m src.analysis.benchmark                 # 통일 조건 시간 벤치마크 (단독 실행)
python -m src.analysis.plots                     # 그림
pytest
```

- 위 명령은 모두 `.venv`의 파이썬으로 실행합니다 (Windows: `.venv/Scripts/python`).
- 리플레이와 MMR은 `--schema base|ext`, `--target raw|resid`, `--market us|jp` 조합으로 각 변형을 돌립니다.
- 사용한 조합 전체는 설계서 7.4–7.6절에 있습니다.

### OOS (H2)

```bash
python -m src.analysis.oos_report --dry-run-test-period        # Test 기간으로 시험 실행 (OOS를 읽지 않음)
python -m src.analysis.oos_report --confirm-single-oos-run     # 실제 OOS 1회
```

OOS는 이 연구에서 **이미 한 번 실행되어 잠겨 있습니다.** 잠금은 아래 장치들로 지켜집니다.

- **읽기 제한:** OOS 결과 파일은 `oos_report.py`만 읽으며, 테스트가 이를 확인합니다.
- **실행 조건:** 계획 문서와 코드에 커밋되지 않은 변경이 있으면 실행을 거부하고, 실행한 커밋 해시를 기록합니다.
- **재실행 차단:** 실행이 끝나면 `artifacts/oos_locked/RUN_COMPLETE.json`이 생기고, 이 파일이 있으면 다시 실행되지 않습니다.

## 레포 구조

```
config/          grammar.yaml (문법), experiment.yaml (고정값·시드·기준)
universe/        구성 종목 스냅샷 (universe_{us,jp}_v1.csv)
src/common/      경로·설정·로깅, 시드 파생
src/data/        구성 종목 수집, 다운로드·정제, 패널
src/generator/   문법, 정규형·해시, 신호 필터, 풀 생성
src/features/    AST 피처 (수식 문자열만 사용)
src/backtest/    numba 커널, 연산자, 지표, 전수 백테스트 엔진
src/surrogate/   모델 정의, 튜닝, 휴리스틱, 잔차 타깃
src/experiment/  통과 기준, 임계값 조정, 접근자, arm, 리플레이, MMR
src/analysis/    통계, 강건성 분석, 진단, 벤치마크, H2 리허설, OOS 리포트, 그림
tests/           누수·정규형·결정성·문법·접근자·OOS 격리 테스트
```

`data/`와 `artifacts/`는 git에서 제외되어 있습니다.

## 재현성

- 모든 난수는 `config/experiment.yaml`의 마스터 시드(20261101)에서 파생합니다. 같은 시드의 arm들은 같은 warm-up을 공유합니다.
- 태그
  - `h2-plan-frozen`: OOS를 열기 전에 H2 계획을 고정한 커밋. 실제 OOS도 이 커밋에서 실행했습니다
  - `v1.0-results`: 최종 결과와 그림
- 코드가 같아도 결과가 똑같이 나온다는 보장은 없습니다. yfinance 데이터는 받는 시점에 따라 수정주가가 달라질 수 있기 때문입니다. 연구에 쓴 데이터와 산출물은 레포에 포함하지 않았습니다.

## 한계

- **데이터:** 무료 데이터를 썼고, 현재 구성 종목으로 고정해 생존 편향이 있으며, 수정주가 오류가 있을 수 있습니다.
- **비용:** 편도 5bp로 단순하게 가정했습니다.
- **범위:** 결과는 이 문법·생성기가 만든 풀 하나에 한정됩니다. OOS는 2년이고, 30시드는 같은 풀과 같은 기간을 공유해 서로 독립이 아닙니다.
- **사후 분석:** MMR, 잔차 타깃, 같은 총 예산 분석, 군집 진단은 H1 결과를 본 뒤 추가한 사후 분석입니다. 각각 실행 전에 설정을 고정했고, 사전 등록 결과와 구분해 보고합니다.
