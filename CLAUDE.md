# CLAUDE.md

## 프로젝트 개요

대리 모델(트리 기반 회귀)로 수식 알파 후보의 백테스트 성과를 미리 예측해, **같은 백테스트 예산에서 무작위 선택보다 통과 시그널을 더 많이 찾는지** 검증하는 연구 프로젝트다. 미국(S&P 500)에서 A/B 테스트를 하고, 학습된 모델을 일본(닛케이225)에 재학습 없이 전이해 본다.

**설계의 단일 기준은 `surrogate_alpha_search_design.md`(프로젝트 루트)다.** 구현이 설계 문서와 다르면 설계 문서를 따르고, 설계 자체를 바꿔야 할 것 같으면 코드를 고치기 전에 사용자에게 먼저 묻는다.

## 디렉터리 구조

```
surrogate-alpha-search/
├── CLAUDE.md
├── surrogate_alpha_search_design.md   # 실험 설계서 (단일 기준)
├── config/
│   ├── grammar.yaml               # 필드, 연산자, 윈도우, 단위 규칙, 크기 제약
│   └── experiment.yaml            # N, W, B, k, 시드, 통과 기준, 비용, 기간
├── universe/
│   ├── universe_us_v1.csv
│   └── universe_jp_v1.csv
├── data/                          # gitignore: 원시 다운로드와 parquet 캐시
├── artifacts/                     # gitignore: 생성 결과물
│   ├── pool_v1.parquet            # 정규형 수식 풀
│   ├── feature_schema_v1.json
│   ├── features_v1.parquet
│   ├── results_{us,jp}_is_test.parquet
│   └── results_{us,jp}_oos.parquet   # 잠금 파일 (아래 규칙 참고)
├── src/
│   ├── common/        # config.py (경로·설정·로깅), seeds.py (마스터 시드 파생)
│   ├── data/          # download.py, universe.py, panel.py
│   ├── generator/     # grammar.py, generate.py, canonical.py, filters.py
│   ├── features/      # ast_features.py
│   ├── backtest/      # operators.py, kernels.py (numba), engine.py, metrics.py
│   ├── surrogate/     # models.py, tune.py, heuristic.py
│   ├── experiment/    # arms.py, replay.py, accessor.py
│   └── analysis/      # stats.py, plots.py, oos_report.py
└── tests/
```

## 실행 순서

프로젝트 전용 가상환경 `.venv`의 파이썬으로 실행한다 (Windows: `.venv/Scripts/python`). 전역 파이썬 환경은 패키지 상태가 섞여 있어 쓰지 않는다.

```bash
python -m src.data.universe --market us        # 위키백과 고정 편집판에서 구성 종목 수집 (download가 없으면 자동 호출)
python -m src.data.download --market us        # 가격/거래량 다운로드 및 캐싱
python -m src.generator.generate               # 풀 4만 개 생성 → pool_v1.parquet
python -m src.features.ast_features            # 피처 추출 → features_v1.parquet
python -m src.backtest.engine --market us      # 풀 전수 백테스트 (IS/Test와 OOS를 분리 저장)
python -m src.experiment.replay --market us    # 30시드 × 6 arm 리플레이
python -m src.experiment.replay --market jp    # 일본 전이 실험
python -m src.analysis.stats                   # 핵심 지표, 검정, 그래프
python -m src.analysis.oos_report              # OOS 최종 1회 (사용자 승인 후에만)
pytest
```

## 절대 규칙: 데이터 누수 방지

이 규칙들은 실험 결과의 타당성을 결정한다. 어기는 코드는 작성하지 않는다.

1. **대리 모델 학습 입력은 `(warm-up 피처, warm-up IS 순 Sharpe)` 쌍뿐이다.** Test·OOS 성과, 후보 풀의 성과는 어떤 형태로도 학습·튜닝에 들어가면 안 된다.
2. **피처는 수식 문자열만으로 계산한다.** 가격 데이터를 읽는 피처(시그널 분산, NaN 비율 등)는 추가하지 않는다.
3. **OOS 결과 파일(`results_*_oos.parquet`)은 `src/analysis/oos_report.py`에서만 읽는다.** 다른 모듈에서 import하거나 경로를 참조하지 않는다. `oos_report.py`는 사용자가 명시적으로 요청할 때만 실행한다.
4. **arm은 `experiment/accessor.py`의 접근자를 통해서만 결과를 조회한다.** 접근자는 해당 arm이 선택한 수식의 결과만 반환하고 조회 횟수를 예산으로 센다. 결과 테이블을 arm에 직접 넘기지 않는다.
5. **하이퍼파라미터 튜닝은 warm-up 내부 CV로만 한다.** GroupKFold(구조가 같은 수식끼리 같은 그룹), 모델당 Optuna 50 trials로 동일하게 준다.
6. **일본 실험에서는 일본 데이터로 재학습하거나 튜닝하지 않는다.** 미국 warm-up으로 학습한 모델을 그대로 적용한다.

## 고정값 (변경 금지)

아래 값은 `config/`에 있고 결과를 보기 전에 확정한 값이다. 사용자가 명시적으로 지시하지 않으면 바꾸지 않는다. 바꿔야 하면 파일 버전(`_v1` → `_v2`)을 올리고 이전 결과와 섞지 않는다.

- 문법: 필드 6, 연산자 16, 윈도우 {5, 10, 20, 60, 120}, 깊이 2~5, 노드 ≤ 15, 누적 lookback ≤ 250
- 풀 구성: 깊이 2는 전수 열거(유효 363개 중 필터 통과분 전부), 나머지를 깊이 3·4·5에 균등 배분
- 풀 N = 40,000 / warm-up W = 1,000 / 예산 B = 2,000 / k = 200 / 시드 30회
- 기간: IS 2016–2021, Test 2022–2023, OOS 2024–2025 (다운로드는 2015-01-01부터, 2015년은 lookback 버퍼로만 사용)
- 유니버스: 2026-11-01 스냅샷 (`universe_*_v1.csv`)
- 피처 스키마: `feature_schema_v1.json`
- 통과 기준: |IS 순 Sharpe| ≥ 0.7 (시작값, base rate에 맞춰 0.3~1.5에서 조정), 같은 방향 Test 순 Sharpe ≥ 0.3, IS 평균 회전율 ≤ 50%, 기존 통과 시그널과 IS 일별 순수익 상관 < 0.7
- base rate 목표 3% (허용 1~5%). 임계값을 조정할 때는 생성 규칙이 아니라 임계값을 바꾼다
- 합격선은 의도적으로 느슨하다 (통과 = 추가 검토 후보, 무료 데이터 한계). 근거는 설계 문서 5.3절 "합격선 설정 원칙"

## 백테스트 사양

- 전처리: inf·-inf를 결측 처리 → 날짜별 횡단면 1%/99% winsorize
- t일 종가 시그널 → 횡단면 디민 → `Σ|w| = 1` 정규화 (결측 종목은 비중 0, 유효 종목만으로 디민·정규화)
- 체결: t+1일 종가 MOC 체결 (지연 1일). 수익은 `r(t+2) = Σ w(t) × ret(t+2)`, `ret(t+2) = close(t+2) / close(t+1) − 1`
- 순수익: `r(t+2) − cost(t)` (비용은 t+1일 체결 시 발생)
- 비용: `5bp × Σ|w(t) − w(t−1)|`, 회전율: `0.5 × Σ|w(t) − w(t−1)|`
- Sharpe: 일별 순수익 `평균 / 표준편차 × √252`
- IS Sharpe가 음수면 부호를 뒤집고, 그 방향을 Test·OOS에 그대로 적용
- 10년을 하나의 연속 백테스트로 돌리고(구간 경계에서 비중 초기화 없음), 일별 순수익은 수익 발생일(t+2) 기준으로 구간에 귀속
- 모든 지표의 "통과"는 기준 1~4 전부 적용, 상관 중복 제거는 평가 대상 집합마다 따로 (설계 문서 5.3절)
- 날짜별 유니버스는 그날 유효한 데이터(종가 있음, 거래량 > 0)가 있는 종목. 휴장일 행(가격 있는 종목의 50% 이상이 거래량 0)은 날짜째 삭제, 거래량 0 종목-일은 결측

## 구현 원칙

- **환경은 CPU 전제다.** GPU를 가정하지 않는다. XGBoost는 `tree_method="hist"`, LightGBM·CatBoost도 CPU 설정을 쓴다.
- **백테스트는 numpy로 벡터화한다.** 날짜×종목 2차원 배열 연산으로 처리하고, 종목별·날짜별 파이썬 루프를 쓰지 않는다. 롤링·순위 연산자는 `backtest/kernels.py`의 numba 커널을 쓰며, pandas 기준 구현과 결과가 같은지 테스트로 확인한다. 대량 평가는 프로세스 병렬(워커당 numba 스레드 1개)로 한다.
- **재현성:** 모든 난수는 `experiment.yaml`의 시드에서 파생한다. 같은 시드의 모든 arm은 같은 warm-up을 공유한다.
- **정규형:** 피처 추출, 중복 제거, 해시는 모두 정규형 변환을 거친 수식에 대해 한다.
- **기록:** 생성 단계별 탈락률, 유니버스 제외 종목과 사유, 날짜별 유효 종목 수, 단계별 소요 시간을 로그와 artifacts에 남긴다.
- 대용량 결과는 parquet로 저장하고, 데이터와 artifacts는 커밋하지 않는다.

## 필수 테스트

새 연산자나 백테스트 로직을 건드리면 아래 테스트가 통과해야 한다.

- **미래 정보 누수 테스트:** t일 이후 데이터를 임의로 바꿔도 t일 시그널이 변하지 않는다.
- **정규형 테스트:** `add(a, b)`와 `add(b, a)`가 같은 정규형·해시를 갖는다.
- **피처 결정성 테스트:** 같은 수식은 항상 같은 피처 벡터를 낸다.
- **문법 테스트:** 금지 패턴과 단위 규칙 위반 수식이 생성되지 않는다.
- **접근자 테스트:** arm이 선택하지 않은 수식의 결과를 조회할 수 없고, 조회 횟수가 예산에 정확히 반영된다.

## 코드 스타일

- Python 3.11, 타입 힌트 사용, `ruff`로 포맷·린트, `pytest`로 테스트
- 식별자와 코드 주석은 영어, 설계 문서와 README는 한국어
- 설정값을 코드에 하드코딩하지 않고 `config/`에서 읽는다

## 하지 말 것

- 설계 문서에 없는 비교군, 지표, 확장 실험을 임의로 추가하지 않는다 (제안은 가능, 구현은 승인 후)
- 결과를 보고 고정값을 바꾸지 않는다
- OOS 결과를 미리 출력하거나 요약하지 않는다
- 피처 선택을 하지 않는다 (스키마의 모든 열을 사용)
