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

### ② 김한결 — Matrix Factorization (`models/mf.py`)
- [x] `_fit`: SGD 학습, epoch마다 `self.history`에 train_rmse 기록 → 전체 실행 및 `outputs/learning_curve_MF.png` 저장 확인
- [x] `score`: 벡터 연산으로 여러 아이템 한 번에 예측, 모르는 사용자는 `mu + b_i` (전체 평점·순위 평가 완료)
- [ ] 하이퍼파라미터 실험: `experiments/tune_mf.py`는 추가됨. `n_factors`, `reg`, `learning_rate`별 실제 결과·선정 근거 및 train/validation 과적합 그래프 확인·정리 필요 (튜닝에는 test가 아닌 별도 validation 사용)
- 참고: 순수 파이썬 SGD는 Slim에서 epoch당 몇 초 걸림. 느리면 epoch 수를 줄여 먼저 확인.

### ③ 최정원 — 콘텐츠 기반 · 하이브리드 (`models/content.py`, `models/hybrid.py`)
- [x] `build_item_features`: one-hot / multi-hot / ABV 스케일링 (+ 블록별 정규화로 균형 조정; 별도 가중치 튜닝은 미진행)
- [x] `ContentBasedRecommender._fit / score / similar_items`
- [x] `HybridRecommender.user_weight`: 평점 적은 사용자일수록 콘텐츠 비중↑ (가중치 검증 및 MF 연동 전체 실행·교차 데이터셋 평가 완료)
- [ ] (추가 분석) 평점 수별 Hybrid 성능을 `outputs/cold_start_analysis.csv`로 해석. 적응형 가중치 자체의 효과를 주장하려면 고정 가중치 Hybrid와 비교 필요
- [ ] (선택) 와인 임베딩 K-means + PCA/t-SNE 시각화: 미구현, 필수 작업 아님

### ④ 신우엽 — 평가 · 분석 (`models/baseline.py`, `eval/*`)
- [x] `ItemKNNRecommender` (메모리 기반 CF 베이스라인)
- [x] `cross_dataset.model_quality_per_item` (교차 데이터셋 비교용 와인별 점수)
- [x] `analysis.worst_predictions`: 실패 사례 추출 함수 구현
- [ ] 실패 사례 분석 → PPT: 함수 호출로 실제 사례를 추출·저장하고, 사용자/와인 정보와 함께 오류 원인 해석 필요 (`run.py`에는 자동 호출·저장 연결이 없음)
- [x] 전체 5개 모델 비교표·그래프, 콜드스타트 분석 및 교차 데이터셋 결과 생성 (`outputs/`)
- [ ] 최종 비교표·그래프 정리: 추가 실험 후 최종 결과를 확정하고 발표용 표·그림 선정 및 해석 필요. 현재 Hybrid는 MF·Content보다 NDCG가 높지만 Popularity보다 낮음

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
