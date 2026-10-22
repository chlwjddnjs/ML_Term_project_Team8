# 팀 작업 가이드 (내부용)

## 0. 시작하기
```bash
pip install -r requirements.txt
python run.py --data sample     # 지금 바로 끝까지 돌아감 (Popularity만 결과, 나머지는 TODO로 표시)
```
- 자기 담당 TODO를 채우면 `run.py`를 다시 돌렸을 때 그 모델 줄이 `[TODO]`에서 `[ok]`로 바뀝니다.
- 코드에서 `TODO(1)`~`TODO(4)`로 검색하면 각자 할 일이 나옵니다.
- 처음엔 `--data sample`(와인 100개)로 개발하고, 동작하면 `data/raw/`에 Slim을 넣고 `python run.py`.

## 1. 규칙 (충돌 방지)
1. **자기 담당 파일만 수정.** 공통 파일(`base.py`, `run.py`, `config.yaml`, `split.py`, `evaluator.py`)은 팀장만 수정하고, 바꿀 일이 생기면 먼저 공유.
2. **공통 스키마 컬럼 이름은 바꾸지 않기.** (`adapter.py` 맨 위에 정의)
   - ITEMS: `item_id, name, winery, country, region, type, grapes(list), harmonize(list), abv, body, acidity`
   - RATINGS: `user_id, item_id, rating(1~5), date`
3. **모델은 `_fit`과 `score` 두 개만 구현.** `recommend`, `predict`는 `base.py`가 알아서 처리.
4. **하이퍼파라미터는 코드에 박지 말고 `config.yaml`에.**
5. **split은 `split_by_user` + config의 seed만 사용.** 각자 따로 나누면 결과 비교 불가.
6. 인터넷 코드를 참고했으면 바로 `docs/CITATIONS.md`에 출처 추가 (안 하면 표절 처리).
7. 수업에서 안 배운 함수/클래스를 쓰면 바로 `docs/LIBRARIES.md` 자기 칸에 파라미터와 함께 설명 추가 (스펙 필수 항목).

## 2. 역할별 할 일

### ① 김도원 — 데이터 · DatasetAdapter (`data/adapter.py`, `data/cleaner.py`)
- [ ] EDA: `cleaner.quality_report` 결과 + 분포 그래프 (PPT "Data exploration")
- [ ] `parse_winemag_title`: 제목에서 와이너리·연도·(지역) 제거 → 와인 이름
- [ ] `normalize_text`: 악센트·구두점·흔한 단어(winery, domaine…) 제거
- [ ] `match_items(method="fuzzy")`: 문자열 유사도 매칭 ← **가장 먼저** (겹치는 와인 수 확인)
- [ ] `scale_points(method="quantile")`: 분위수 기반 척도 변환
- [ ] `clean_items / clean_ratings / clean_expert`의 TODO
- [ ] ablation 결과: exact vs fuzzy 매칭 수, linear vs quantile 변환 결과 비교

### ② 김한결 — Matrix Factorization (`models/mf.py`)
- [ ] `_fit`: SGD 학습, epoch마다 `self.history`에 train_rmse 기록 → 학습곡선 자동 저장
- [ ] `score`: 벡터 연산으로 여러 아이템 한 번에 예측, 모르는 사용자는 `mu + b_i`
- [ ] 하이퍼파라미터 실험: `n_factors`, `reg`, `learning_rate` (과적합 그래프)
- 참고: 순수 파이썬 SGD는 Slim에서 epoch당 몇 초 걸림. 느리면 epoch 수를 줄여 먼저 확인.

### ③ 최정원 — 콘텐츠 기반 · 하이브리드 (`models/content.py`, `models/hybrid.py`)
- [ ] `build_item_features`: one-hot / multi-hot / ABV 스케일링 (+ 블록 가중치)
- [ ] `ContentBasedRecommender._fit / score / similar_items`
- [ ] `HybridRecommender.user_weight`: 평점 적은 사용자일수록 콘텐츠 비중↑
- [ ] (선택) 와인 임베딩 K-means + PCA/t-SNE 시각화

### ④ 신우엽 — 평가 · 분석 (`models/baseline.py`, `eval/*`)
- [ ] `ItemKNNRecommender` (메모리 기반 CF 베이스라인)
- [ ] `cross_dataset.model_quality_per_item` (교차 데이터셋 비교용 와인별 점수)
- [ ] `analysis.worst_predictions` (실패 사례 분석 → PPT)
- [ ] 최종 비교표·그래프 정리 (`outputs/` 결과 사용)

## 3. 진행 순서
```
① fuzzy 매칭·척도 변환 ─────────────→ 교차 데이터셋 검증
② MF ───────┐
            ├→ 하이브리드 → 최종 비교 실험
③ 콘텐츠 ────┘
④ KNN·분석 함수는 처음부터 병렬로 진행
```

## 4. 최종 제출물 체크리스트 (스펙 기준, 11/9 마감)
**PPT**
- [ ] Business objective / Data exploration / Preprocessing / Modeling / Evaluation & analysis (성공·실패 모두 설명)
- [ ] Learning experience: 어려웠던 점과 해결, 다시 한다면, 개인별로 배운 점
- [ ] Teamwork: 역할 분담 + **기여도 %**
- [ ] Open-Source SW: 아키텍처 설명(README의 다이어그램 활용) + **GitHub URL**

**소스코드 & Word**
- [ ] 주석이 잘 달린 코드
- [ ] 인용 출처 (`docs/CITATIONS.md`)
- [ ] 수업 외 라이브러리·함수 설명 (`docs/LIBRARIES.md`)
- [ ] 출력물: 그래프, 실행 결과 (`outputs/`)
