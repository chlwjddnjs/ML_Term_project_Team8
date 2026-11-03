# 팀 작업 가이드 (내부용)

## 0. 시작하기 (각자 컴퓨터에서, 처음 한 번)
```bash
git clone https://github.com/chlwjddnjs/ML_Term_project_Team8.git
cd ML_Term_project_Team8
pip install -r requirements.txt
python run.py --data sample     # 샘플 데이터로 전체 파이프라인 확인
```
- 데이터 파일 3개는 git에 올라가지 않으니 각자 받아서 `data/raw/`에 넣기 (`data/raw/README.md` 참고).
- 모든 명령어는 `run.py`가 있는 폴더에서 실행.
- 2026-11-03 기준 전체 데이터에서 5개 모델 모두 `[ok]`, 교차 데이터셋 검증 및 결과 저장까지 확인했습니다.
- 아래 체크는 구현/실행 확인과 실험 해석/제출물 작성을 구분합니다. 미체크는 미완료이거나 결과를 아직 확인하지 못한 항목입니다.
- 처음엔 `--data sample`(와인 100개)로 개발하고, 동작하면 `data/raw/`에 Slim을 넣고 `python run.py`.

## 1. 작업할 때마다 (git)
```bash
git pull                          # 시작 전에 최신 코드 받기
# ... 자기 파일 수정 ...
python run.py --data sample       # 돌아가는지 확인
git add recsys/models/mf.py       # 자기 파일만 add (git add . 보다 안전)
git commit -m "MF 학습 구현"
git push                          # 거절되면 git pull 후 다시 push
```

## 2. 규칙 (충돌 방지)
1. **자기 담당 파일만 수정.** 공통 파일(`base.py`, `run.py`, `config.yaml`, `split.py`, `evaluator.py`)은 팀장만 수정하고, 바꿀 일이 생기면 먼저 공유.
2. **공통 스키마 컬럼 이름은 바꾸지 않기.** (`adapter.py` 맨 위에 정의)
   - ITEMS: `item_id, name, winery, country, region, type, grapes(list), harmonize(list), abv, body, acidity`
   - RATINGS: `user_id, item_id, rating(1~5), date`
3. **모델은 `_fit`과 `score` 두 개만 구현.** `recommend`, `predict`는 `base.py`가 알아서 처리.
4. **하이퍼파라미터는 코드에 박지 말고 `config.yaml`에.**
5. **split은 `split_by_user` + config의 seed만 사용.** 각자 따로 나누면 결과 비교 불가.
6. 인터넷 코드를 참고했으면 바로 `docs/CITATIONS.md`에 출처 추가 (안 하면 표절 처리).
7. 수업에서 안 배운 함수/클래스를 쓰면 바로 `docs/LIBRARIES.md` 자기 칸에 파라미터와 함께 설명 추가 (스펙 필수 항목).

## 3. 역할별 할 일

### ① 김도원 — 데이터 · DatasetAdapter (`data/adapter.py`, `data/cleaner.py`)
- [x] EDA 자료 생성: `cleaner.quality_report` 출력 + `outputs/eda_rating_distribution.png` 생성 확인
- [ ] EDA 해석을 PPT "Data exploration"에 반영: 결측치·중복·분포와 정제 이유 설명 필요
- [x] `parse_winemag_title`: 제목에서 와이너리·연도·(지역) 제거 → 와인 이름
- [x] `normalize_text`: 악센트·구두점·흔한 단어(winery, domaine…) 제거
- [x] `match_items(method="fuzzy")`: 전체 실행에서 169개 와인과 415건 전문가 리뷰 연결 확인
- [x] `scale_points(method="quantile")`: 분위수 기반 척도 변환 구현 (현재 `run.py` 기본 실행은 linear 사용)
- [x] `clean_items / clean_ratings / clean_expert`: 정제 구현 및 전체 실행 확인
- [ ] ablation 결과: `experiments/data_analysis.py`는 추가됨. exact vs fuzzy 매칭 수, linear vs quantile 변환의 실제 결과를 확인·비교하고 해석 정리 필요
- [ ] 매칭 품질 검토: 연결된 와인 일부를 직접 확인하고 오매칭 가능성과 비교 범위(169개)의 한계 정리
- [x] (기존 보조 기능) `parse_list / apply_column_map`: 리스트 문자열 파싱 및 설정 기반 공통 컬럼 변환. 원본 컬럼이 없으면 경고 후 결측값으로 채움 (공통 베이스 포함)
- [x] (추가 구현) `iqr_bounds`: 가격의 로그 스케일 IQR 범위를 계산해 이상치 보정에 활용 (전체 정제 실행 확인)

### ② 김한결 — Matrix Factorization (`models/mf.py`)
- [x] `_fit`: SGD 학습, epoch마다 `self.history`에 train_rmse 기록 → 전체 실행 및 `outputs/learning_curve_MF.png` 저장 확인
- [x] `score`: 벡터 연산으로 여러 아이템 한 번에 예측, 모르는 사용자는 `mu + b_i` (전체 평점·순위 평가 완료)
- [ ] 하이퍼파라미터 실험: `experiments/tune_mf.py`는 추가됨. `n_factors`, `reg`, `learning_rate`별 실제 결과·선정 근거 및 train/validation 과적합 그래프 확인·정리 필요 (튜닝에는 test가 아닌 별도 validation 사용)
- [x] (추가 구현) `reg_bias / bias_damping`: 편향 정규화를 분리하고 평가 수가 적은 와인의 `b_i`를 상대적으로 강하게 억제. 현재 `config.yaml`의 `bias_damping=10`으로 전체 실행 확인
- [x] (추가 구현) `set_validation / _rmse`: 별도 validation을 설정하면 epoch별 검증 RMSE도 `history`에 기록 (`run.py` 기본 실행은 train RMSE만 기록)
- [x] (추가 구현) `experiments/check_mf_top.py`: 추천 빈도, 추천 와인의 평가 수, 상위 아이템 편향을 표로 저장하는 분석 스크립트. 팀원 제공 MF 실험 문서에 관련 분석 결과가 있음 (원본 출력 파일은 별도 확인 필요)
- 참고: 순수 파이썬 SGD는 Slim에서 epoch당 몇 초 걸림. 느리면 epoch 수를 줄여 먼저 확인.

### ③ 최정원 — 콘텐츠 기반 · 하이브리드 (`models/content.py`, `models/hybrid.py`)
- [x] `build_item_features`: one-hot / multi-hot / ABV 스케일링 (+ 블록별 정규화로 균형 조정; 별도 가중치 튜닝은 미진행)
- [x] `ContentBasedRecommender._fit / score / similar_items`
- [x] `HybridRecommender.user_weight`: 평점 적은 사용자일수록 콘텐츠 비중↑ (가중치 검증 및 MF 연동 전체 실행·교차 데이터셋 평가 완료)
- [x] (추가 분석) 고정 vs 사용자별 가중치 통제 비교 완료: 기본 2,000명 및 전체 평가 가능 8,904명 비교. 전체 차이는 미미하고, 평점 5개 미만 79명에서 NDCG 소폭 상승·Recall 하락으로 효과가 혼재함. `docs/HYBRID_WEIGHT_EXPERIMENT.md` 참조 (단일 분할 결과, 우월성 확정 불가)
- [ ] (선택) 와인 임베딩 K-means + PCA/t-SNE 시각화: 미구현, 필수 작업 아님
- [x] (추가 구현) `ItemFeatureEncoder`: 학습한 속성 사전을 보관하고 새 와인도 같은 특징 공간으로 변환. 미등록 속성은 무시하고 결측 속성은 0으로 처리 (기능 검증 완료)
- [x] (선택 기능 구현) `use_text=True`일 때 설명 텍스트의 TF-IDF 특징 결합. 작은 예제로 검증했으나 기본 설정은 `false`이고 X-Wines 공통 스키마에는 설명 텍스트가 없어 최종 실험에는 사용하지 않음
- [x] (예외 처리 구현) 단일/동일 평점 이력은 동일 가중치 취향 벡터 사용, 미등록 사용자·아이템은 0점 반환 (인기순 대체 추천 기능은 아님)

### ④ 신우엽 — 평가 · 분석 (`models/baseline.py`, `eval/*`)
- [x] `ItemKNNRecommender` (메모리 기반 CF 베이스라인)
- [x] `cross_dataset.model_quality_per_item` (교차 데이터셋 비교용 와인별 점수)
- [x] `analysis.worst_predictions`: 실패 사례 추출 함수 구현
- [x] 실패 사례 분석: `experiments/failure_analysis.py`로 ItemKNN·MF의 worst 20개를 사용자/와인 정보와 함께 `outputs/failure_worst_*.csv`로 저장, 실패 패턴·실제 점수별 오차·평가 수별 오차 표와 그래프 생성 (Slim 실행 확인). 결론: 큰 오차는 거의 전부 평소 후한 사용자가 인기 와인에 준 1~2점 평점
- [ ] 실패 사례 PPT 반영: 대표 사례 표 + `failure_error_by_rating.png`
- [x] 전체 5개 모델 비교표·그래프, 콜드스타트 분석 및 교차 데이터셋 결과 생성 (`outputs/`)
- [ ] 최종 비교표·그래프 정리: 추가 실험 후 최종 결과를 확정하고 발표용 표·그림 선정 및 해석 필요. 현재 Hybrid는 MF·Content보다 NDCG가 높지만 Popularity보다 낮음
- [x] (추가 분석) 순위 실패 해석: `experiments/ranking_analysis.py`로 Popularity 우위 원인 분석 (Slim 실행 확인). 상위 50개(5%) 와인이 테스트 정답의 37%를 차지하는 인기 편향 때문이며, 이 50개를 정답에서 빼면 Popularity NDCG는 0이고 Content > Hybrid > MF 순. MF는 추천 와인 23개로 개인화가 약하고 ItemKNN 순위는 Random 수준. `outputs/ranking_*` 생성
- [x] (기존 베이스 기능) `PopularityRecommender`: 평점 수 중심의 비개인화 추천 기준 모델 (전체 실행 확인)
- [x] (추가 구현) `predictions_with_errors`: 실제 평점·예측·오차·사용자/와인 학습 이력 수를 합친 상세 표 생성. 함수 구현 완료이며 기본 `run.py`에는 자동 호출·저장이 없음
- [x] (추가 구현) `error_by_group`: 사용자 또는 와인의 학습 이력 수별 MAE·평균 오차·평가 건수 집계. 함수 구현 완료이며 실제 결과 생성·해석은 실패 사례 분석 작업에 포함
- [x] (평가 기능) `rank_agreement / expert_score_per_item`: 매칭된 와인의 전문가 평점 집계, Spearman·p-value·Top-K 겹침 계산 (169개 와인으로 전체 실행 확인)
- [x] (평가 기능) `cold_start_analysis`: 사용자 학습 평점 수별 순위 지표와 사용자 수 집계 (`outputs/cold_start_analysis.csv` 생성 확인)

### 공통 베이스 기능 — 공동 관리 (개별 신규 기여와 구분)
- [x] `loader`: CSV 로딩, X-Wines 파일명 차이 자동 탐색, 전문가 파일 부재 시 교차 검증 생략
- [x] `config.py / run.py`: YAML 경로·파라미터 로딩, `--data / --models / --config` 명령행 옵션
- [x] `split_by_user`: 사용자별 학습/테스트 분리, seed 고정, 테스트 사용자에게 최소 1개 학습 평점 유지 (전체 실행 확인)
- [x] `Recommender`: 모델 공통 인터페이스, 사용자·아이템 인덱스, 이미 평가한 와인 제외, Top-K 추천, 평점 모델의 예측값 1~5 제한
- [x] `metrics / evaluator`: 평점·순위·Coverage의 공통 평가 및 seed 기반 평가 사용자 표본 추출 (전체 실행 확인)
- [x] `plots / run.py`: 분포·모델 비교·학습곡선 PNG, 결과 CSV 및 `run_info.json` 설정 저장 (전체 실행 확인)
- 위 체크는 코드에 기능이 존재한다는 의미이며, 모든 예외 경로를 각각 실행했다는 의미는 아님. 공통 파일 변경은 기존 팀 규칙에 따라 담당자와 공유

## 4. 진행 순서
```
① fuzzy 매칭·척도 변환 ─────────────→ 교차 데이터셋 검증
② MF ───────┐
            ├→ 하이브리드 → 최종 비교 실험
③ 콘텐츠 ────┘
④ KNN·분석 함수는 처음부터 병렬로 진행
```

## 5. 최종 제출물 체크리스트 (스펙 기준, 11/9 마감)
**PPT**
- [ ] Business objective / Data exploration / Preprocessing / Modeling / Evaluation & analysis (성공·실패 모두 설명)
- [ ] Learning experience: 어려웠던 점과 해결, 다시 한다면, 개인별로 배운 점
- [ ] Teamwork: 역할 분담 + **기여도 %**
- [ ] Open-Source SW: 아키텍처 설명(README의 다이어그램 활용) + **GitHub URL**

**소스코드 & Word**
- [x] 주요 구현 코드에 설명 주석 작성 (전체 모델 실행 확인)
- [ ] 소스코드·Word 최종 편집: 실행 방법, 모델 설명, 결과 및 제한사항을 제출 형식에 맞게 정리
- [ ] 인용 출처 최종 점검: `docs/CITATIONS.md`는 작성됨. 모든 참고 코드·데이터·도구 사용이 빠짐없이 기록됐는지 확인
- [ ] 수업 외 라이브러리·함수 설명 최종 점검: `docs/LIBRARIES.md`는 작성됨. 실제 사용 함수·파라미터와 수업 범위 대조 필요
- [x] 출력물 생성: 그래프, 실행 결과 (`outputs/`, 2026-11-03 전체 실행 확인)
- [ ] 최종 출력물 제출 준비: 추가 실험 결과까지 포함해 보관하고 PPT·Word에 반영. `outputs/`는 재실행 시 덮어써지므로 최종 실행 조건과 함께 보존
