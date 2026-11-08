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
- 발표의 내부 모델 비교는 `python experiments/cross_validate.py --data slim`으로 재현합니다. Sample은 `--data sample`로 확인할 수 있습니다.
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
5. **같은 비교표의 모델은 동일한 분할과 config의 seed를 사용.** 기본 실험은 `split_by_user`로 나누고, 10-fold 비교는 이 분할의 학습 부분을 `experiments/cross_validate.py`에서 다시 나눔.
6. 인터넷 코드를 참고했으면 바로 `docs/CITATIONS.md`에 출처 추가 (안 하면 표절 처리).
7. 수업에서 안 배운 함수/클래스를 쓰면 바로 `docs/LIBRARIES.md` 자기 칸에 파라미터와 함께 설명 추가 (스펙 필수 항목).

## 3. 역할별 할 일

추가 실험은 팀의 담당 영역에 따라 분류합니다. 실제 변경 작성자는 Git 커밋 기록에서 확인할 수 있습니다.

### ① 김도원 — 데이터 · DatasetAdapter (`data/adapter.py`, `data/cleaner.py`)
- [x] EDA 자료 생성: `cleaner.quality_report` 출력 + `outputs/eda_rating_distribution.png` 생성 확인
- [x] 데이터 탐색·정제 PPT 반영 (3–4장, 10장): 두 데이터셋의 역할·규모·점수 범위, 결측 범주 처리·중복 제거·도수 보정, 와인별 평점 수의 편중을 설명
- [x] `parse_winemag_title`: 제목에서 와이너리·연도·(지역) 제거 → 와인 이름
- [x] `normalize_text`: 악센트·구두점·흔한 단어(winery, domaine…) 제거
- [x] `match_items(method="fuzzy")`: 전체 실행에서 169개 와인과 415건 전문가 리뷰 연결 확인
- [x] `scale_points(method="quantile")`: 분위수 기반 척도 변환 구현 (현재 `run.py` 기본 실행은 linear 사용)
- [x] `clean_items / clean_ratings / clean_expert`: 정제 구현 및 전체 실행 확인
- [x] ablation 분석 구현: `experiments/data_analysis.py`에 원본/exact/fuzzy(0.80·0.85·0.90) 매칭 수·순위 상관 비교, linear/quantile 분포·사용자 평균과의 오차 비교 구현. 결과는 `outputs/ablation_*`로 저장 (모델 재학습 성능 비교가 아닌 데이터 정렬 방법 비교)
- [x] ablation Slim 실행·결과 확인 (2026-11-04): 원본 0개 → 정규화 exact 134개 → fuzzy(0.85) 169개 와인. linear/quantile의 사용자 평균 대비 MAE는 1.052/0.340, 순위 상관은 0.700/0.693 (척도 정렬 효과이며 quantile 모델 성능 개선을 뜻하지 않음)
- [x] 매칭 품질 검토 자료: `experiments/fuzzy_match_labels.csv`에 fuzzy-only 와인 쌍의 same/different/unsure 라벨·판정 이유 추가. `data_analysis.py`가 기준값별 품질과 오매칭 표 생성. 라벨은 팀원 판정이며 독립 정답은 아님; unsure 제외·exact 매칭 정답 가정을 사용한 추정 정확도라는 한계 포함
- [x] 오매칭 제외 비교 평가 완료 (2026-11-05): 별도 `experiments/compare_clean_matches.py`에서 different·unsure 15쌍/리뷰 연결 19건 제외. 와인 169→163개, 리뷰 연결 415→396건. Hybrid Spearman 0.709090→0.690305, Top-10 겹침 0.5 유지. `docs/CLEAN_MATCH_COMPARISON.md` 참조; 기존 결과·기본 run.py 매칭은 보존
- [x] 정제 후 교차 데이터셋 평가를 PPT에 반영 (4장, 12장): 오매칭·불확실 매칭을 제외한 와인 163개를 사용. `python experiments/compare_clean_matches.py --include-blend`로 재현하고, 결과는 `outputs/clean_matches_with_blend/`에 저장
- [x] (기존 보조 기능) `parse_list / apply_column_map`: 리스트 문자열 파싱 및 설정 기반 공통 컬럼 변환. 원본 컬럼이 없으면 경고 후 결측값으로 채움 (공통 베이스 포함)
- [x] (추가 구현) `iqr_bounds`: 가격의 로그 스케일 IQR 범위를 계산해 이상치 보정에 활용 (전체 정제 실행 확인)

### ② 김한결 — Matrix Factorization (`models/mf.py`)
- [x] `_fit`: SGD 학습, epoch마다 `self.history`에 train_rmse 기록 → 전체 실행 및 `outputs/learning_curve_MF.png` 저장 확인
- [x] `score`: 벡터 연산으로 여러 아이템 한 번에 예측, 모르는 사용자는 `mu + b_i` (전체 평점·순위 평가 완료)
- [x] 하이퍼파라미터 실험: 팀원 제공 MF 실험 문서에서 validation 기반 후보 비교·설정 선정 근거 확인 (`experiments/tune_mf.py`). 원본 튜닝 출력은 별도 보관 필요
- [x] train/validation 과적합 확인 (2026-11-06): `experiments/curve_mf.py` Slim 40 epoch 실행, train 107,538건/valid 12,576건. validation RMSE 최저 0.4842는 19 epoch, 20 epoch 0.4847, 40 epoch 0.4869. 후반 train 오차 감소·valid 오차 소폭 상승으로 과적합 징후 관찰. 현재 20 epoch는 최소점 부근이며 40 epoch까지 늘릴 이점은 이 실험에서 없음. `outputs/learning_curve_MF_valid.png / .csv` 생성 확인
- [x] (추가 구현) `reg_bias / bias_damping`: 편향 정규화를 분리하고 평가 수가 적은 와인의 `b_i`를 상대적으로 강하게 억제. 현재 `config.yaml`의 `bias_damping=10`으로 전체 실행 확인
- [x] (추가 구현) `set_validation / _rmse`: 별도 validation을 설정하면 epoch별 검증 RMSE도 `history`에 기록 (`run.py` 기본 실행은 train RMSE만 기록)
- [x] (추가 구현) `experiments/check_mf_top.py`: 추천 빈도, 추천 와인의 평가 수, 상위 아이템 편향을 표로 저장하는 분석 스크립트. 팀원 제공 MF 실험 문서에 관련 분석 결과가 있음 (원본 출력 파일은 별도 확인 필요)
- 참고: 순수 파이썬 SGD는 Slim에서 epoch당 몇 초 걸림. 느리면 epoch 수를 줄여 먼저 확인.

### ③ 최정원 — 콘텐츠 기반 · 하이브리드 (`models/content.py`, `models/hybrid.py`)
- [x] `build_item_features`: one-hot / multi-hot / ABV 스케일링 (+ 블록별 정규화로 균형 조정; 별도 가중치 튜닝은 미진행)
- [x] `ContentBasedRecommender._fit / score / similar_items`
- [x] `HybridRecommender.user_weight`: 평점 적은 사용자일수록 콘텐츠 비중↑ (가중치 검증 및 MF 연동 전체 실행·교차 데이터셋 평가 완료)
- [x] (추가 분석) 고정 vs 사용자별 가중치 통제 비교 완료: 기본 2,000명 및 전체 평가 가능 8,904명 비교. 전체 차이는 미미하고, 평점 5개 미만 79명에서 NDCG 소폭 상승·Recall 하락으로 효과가 혼재함. `docs/HYBRID_WEIGHT_EXPERIMENT.md` 참조 (단일 분할 결과, 우월성 확정 불가)
- [x] (추가 실험) Popularity + Hybrid 50:50 혼합 구현·평가 완료. `experiments/compare_popularity_hybrid.py`에서 사용자별 추천 후보의 두 점수를 각각 정규화해 절반씩 합산. 기존 검증에서 고른 비율을 고정해 Slim 10-fold 평균 NDCG@10 0.0716, Recall@10 14.03%, Coverage 6.79% 확인
- [ ] (선택) 와인 임베딩 K-means + PCA/t-SNE 시각화: 미구현, 필수 작업 아님
- [x] (추가 구현) `ItemFeatureEncoder`: 학습한 속성 사전을 보관하고 새 와인도 같은 특징 공간으로 변환. 미등록 속성은 무시하고 결측 속성은 0으로 처리 (기능 검증 완료)
- [x] (선택 기능 구현) `use_text=True`일 때 설명 텍스트의 TF-IDF 특징 결합. 작은 예제로 검증했으나 기본 설정은 `false`이고 X-Wines 공통 스키마에는 설명 텍스트가 없어 최종 실험에는 사용하지 않음
- [x] (예외 처리 구현) 단일/동일 평점 이력은 동일 가중치 취향 벡터 사용, 미등록 사용자·아이템은 0점 반환 (인기순 대체 추천 기능은 아님)

### ④ 신우엽 — 평가 · 분석 (`models/baseline.py`, `eval/*`)
- [x] `ItemKNNRecommender` (메모리 기반 CF 베이스라인)
- [x] `cross_dataset.model_quality_per_item` (교차 데이터셋 비교용 와인별 점수)
- [x] `analysis.worst_predictions`: 실패 사례 추출 함수 구현
- [x] 실패 사례 분석: `experiments/failure_analysis.py`로 ItemKNN·MF의 worst 20개를 사용자/와인 정보와 함께 `outputs/failure_worst_*.csv`로 저장, 실패 패턴·실제 점수별 오차·평가 수별 오차 표와 그래프 생성 (Slim 실행 확인). 결론: 큰 오차는 거의 전부 평소 후한 사용자가 인기 와인에 준 1~2점 평점
- [x] 실패 사례 PPT 반영 (10장): 실제 평점별 MF 평균 예측 그래프, 최대 오차 20건의 실제 평점이 모두 2점 이하라는 관찰, 대표 사례 실제 1.00점 → 예측 4.58점
- [x] 전체 5개 모델 비교표·그래프, 콜드스타트 분석 및 교차 데이터셋 결과 생성 (`outputs/`)
- [x] (추가 평가) `experiments/cross_validate.py`로 사용자별 10-fold 교차 검증 완료. 기존 학습 데이터 안에서 기본 모델 5개와 50:50 혼합을 같은 폴드에서 비교하고, 폴드별 지표와 평균·표준편차를 `outputs/cross_validation/<profile>/`에 저장
- [x] 발표용 내부 비교표·그래프 정리 완료. Slim 평균 NDCG@10은 기본 Hybrid 0.0440, Popularity 0.0553, 50:50 혼합 0.0716. 혼합의 추천 범위는 인기순보다 넓고 기본 Hybrid보다 좁음. 전체 6개 모델 결과는 [README](../README.md#10-fold-evaluation) 참조
- [x] 기본 Hybrid와 50:50 혼합의 외부 Top-10 평가 완료 및 PPT 반영 (12장). Wine Enthusiast 전문가 평균 평점 상위 10개를 정답으로 비교한 결과, 기본 Hybrid NDCG@10 0.6236·Recall@10 50%, 혼합 0.0851·10%. `outputs/clean_matches_with_blend/expert_topk_metrics.csv` 및 `expert_topk_rankings.csv` 참조
- [x] (추가 분석) 순위 실패 해석: `experiments/ranking_analysis.py` Slim 재실행 확인 (2026-11-04). 상위 50개(5%) 와인에 테스트 정답의 36.8%가 집중돼 Popularity에 유리한 평가 특성을 보여줌 (우위의 유일한 원인으로 단정 불가). 기존 추천 목록을 유지한 채 이 50개를 정답에서 제외한 tail 평가에서는 Content > Hybrid > MF, Popularity NDCG는 0. MF는 23개 와인에 추천이 집중하고 ItemKNN NDCG는 단일 Random 실행보다 낮음. `outputs/ranking_*` 생성; Random 비교의 통계적 유의성은 검증하지 않음
- [x] (기존 베이스 기능) `PopularityRecommender`: 평점 수 중심의 비개인화 추천 기준 모델 (전체 실행 확인)
- [x] (추가 구현) `predictions_with_errors`: 실제 평점·예측·오차·사용자/와인 학습 이력 수를 합친 상세 표 생성. `failure_analysis.py`에서 사용·실행 확인 (기본 `run.py`에는 자동 호출·저장이 없음)
- [x] (추가 구현) `error_by_group`: 사용자 또는 와인의 학습 이력 수별 MAE·평균 오차·평가 건수 집계. Slim 실행으로 `failure_error_by_user_group.csv / failure_error_by_item_group.csv` 및 그래프 생성 확인
- [x] (추가 구현) `error_by_true_rating / failure_patterns`: 실제 평점별 오차 및 worst 사례의 저평점·적은 학습 이력 등 패턴 비율 집계. `failure_analysis.py` Slim 실행 및 결과 저장 확인
- [x] (평가 기능) `rank_agreement / expert_score_per_item`: 매칭된 와인의 전문가 평점 집계, Spearman·p-value·Top-K 겹침 계산 (169개 와인으로 전체 실행 확인)
- [x] (평가 기능) `cold_start_analysis`: 사용자 학습 평점 수별 순위 지표와 사용자 수 집계 (`outputs/cold_start_analysis.csv` 생성 확인)

### 공통 베이스 기능 — 공동 관리 (개별 신규 기여와 구분)
- [x] `loader`: CSV 로딩, X-Wines 파일명 차이 자동 탐색, 전문가 파일 부재 시 교차 데이터셋 평가 생략
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
- [x] Project Objectives (2장): 개인화 와인 추천과 Wine Enthusiast 전문가 리뷰를 활용한 외부 평가를 두 목표로 나란히 제시
- [x] Data exploration / Preprocessing / Modeling / Evaluation & analysis (3–5장, 7–12장): 두 데이터셋, 정제, 6개 모델, 내부 10-fold와 외부 Top-10, 개선과 실패 사례 설명
- [x] Learning experience (13–14장): 어려웠던 점과 해결 과정, 다시 한다면 검증할 항목 정리
- [x] 개인별로 배운 점 (16장): 최정원이 팀원별 내용을 최종 PPT에 직접 입력
- [x] Teamwork (15장): 역할 분담 및 기여도 기재 (4명 각 25%, 합계 100%)
- [x] Open-Source SW (6장, 17장): 아키텍처 설명 및 GitHub URL 기재

**소스코드 & Word**
- [x] 주요 구현 코드에 설명 주석 작성 (전체 모델 실행 확인)
- [ ] 소스코드·Word 최종 편집: 실행 방법, 모델 설명, 결과 및 제한사항을 제출 형식에 맞게 정리
- [x] 저장소 기준 인용 출처 점검: `docs/CITATIONS.md`에 데이터셋, 모델·평가 참고자료, 외부 API 문서 및 도구 사용을 확인·보완
- [x] 저장소 기준 라이브러리·함수 설명 점검: `docs/LIBRARIES.md`를 현재 코드와 대조하고 10-fold 분할, 50:50 혼합, 외부 Top-10 평가의 주요 함수와 인자를 보완
- [x] 출력물 생성: 그래프, 실행 결과 (`outputs/`, 2026-11-03 전체 실행 확인)
- [ ] 최종 출력물 제출 준비: 추가 실험 결과까지 포함해 보관하고 PPT·Word에 반영. `outputs/`는 재실행 시 덮어써지므로 최종 실행 조건과 함께 보존
