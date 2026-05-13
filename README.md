# GNN-based Fraud Review Detection

> ITDA 연합학술제 | SKK DScover

## Overview

YelpZip 데이터셋을 활용하여 **GNN(Graph Neural Network) 기반 조직적 어뷰징 네트워크 탐지 모델**을 구축하는 프로젝트입니다.  
리뷰 간 관계를 그래프로 모델링하고, 10가지 엣지 조합 × 5가지 GNN 모델을 전수 비교하여 최적 구성을 탐색합니다.

## Task

- **목표**: 사기 리뷰(Fraudulent Review) 탐지 (Binary Classification)
- **데이터**: [YelpZip Dataset](https://www.kaggle.com/datasets/vaibhavsonkar/yelpzip) — 608,458개 리뷰
- **핵심 모델**: GCN / GraphSAGE / GAT / MultiHeadGAT / RGCN
- **평가 지표**: Macro F1, PR-AUC

## Project Structure

```
fraud_review_detection-dscover-project/
├── data/                        # 원본 및 전처리 데이터 (git 미추적)
│   ├── yelpzip.csv
│   └── yelpzip_label_processed.csv
│
├── notebooks/                   # 탐색적 분석 노트북
│   ├── 01_EDA.ipynb
│   ├── 02_subgraph_sampling.ipynb
│   ├── 03_graph_construction.ipynb
│   └── 04_bert_embedding.ipynb
│
├── pipeline_1/                  # 메인 파이프라인 (실험 재현용)
│   ├── 01.Preprocessing_0512.ipynb   # 전처리 및 피처 엔지니어링
│   ├── 02.Graph_Construction.ipynb   # 10종 그래프 생성
│   ├── 03.GNN_Training.ipynb         # 10 × 5 Grid Search 학습
│   ├── Preprocessing_0512.csv        # 전처리 결과 CSV
│   ├── graphs/                       # 그래프 파일 (.pt)
│   ├── models/                       # 학습된 모델 체크포인트 (.pt)
│   └── grid_results.csv              # 실험 결과 (자동 저장)
│
├── 기타자료/                    # 참고 논문 및 PPT
├── .gitattributes
└── README.md
```

## Pipeline

```
[01. Preprocessing]
  YelpZip CSV → 피처 엔지니어링 (user_int, prod_int, year_month 등)
        ↓
[02. Graph Construction]
  리뷰 노드 + 4종 엣지 관계 → 10가지 엣지 조합 그래프 생성 (.pt 저장)
        ↓
[03. GNN Training]
  10 Graphs × 5 Models → Grid Search → Macro F1 / PR-AUC 비교 → 히트맵 시각화
```

## Graph Design

리뷰를 **노드**, 리뷰 간 관계를 **엣지**로 표현한 동종(Homogeneous) 그래프입니다.

| 관계 | 연결 기준 | 탐지 패턴 |
|------|-----------|-----------|
| **R-U-R** | 동일 유저가 작성한 리뷰끼리 | 반복 리뷰어, 조직적 계정 패턴 |
| **R-T-R** | 동일 식당 + 동일 연월에 작성된 리뷰끼리 | 시간적 burst 패턴, 캠페인형 공격 |
| **R-X-R** | 동일 식당 + 극단 별점(1점 또는 5점) 리뷰끼리 | 별점 조작 클러스터 |
| **R-S-R** | 동일 식당 + 동일 별점(1~5점) 리뷰끼리 | 평점 동조 이상 패턴 |

### 10가지 엣지 조합

| 조합 | 관계 수 | 설명 |
|------|---------|------|
| RUR | 1 | 유저 행동 단독 |
| RTR | 1 | 시간 군집 단독 |
| RXR | 1 | 극단 별점 단독 |
| RSR | 1 | 동일 평점 단독 |
| RUR\_RTR | 2 | 유저 + 시간 |
| RUR\_RXR | 2 | 유저 + 극단 별점 |
| RTR\_RXR | 2 | 시간 + 극단 별점 |
| RUR\_RSR | 2 | 유저 + 동일 평점 |
| RUR\_RTR\_RXR | 3 | 유저 + 시간 + 극단 별점 |
| RUR\_RTR\_RSR | 3 | 유저 + 시간 + 동일 평점 |

## Models

| 모델 | 설명 |
|------|------|
| **GCN** | Graph Convolutional Network — 이웃 노드 평균 집계 |
| **GraphSAGE** | Sample & Aggregate — 이웃 샘플링 후 Mean Aggregation |
| **GAT** | Graph Attention Network — Attention 가중치로 이웃 선별 (1 Head) |
| **MultiHeadGAT** | Multi-Head GAT — 4개 Attention Head 병렬 사용 |
| **RGCN** | Relational GCN — 엣지 종류별 독립 가중치 행렬 학습 (복합 엣지 특화) |

## Evaluation Metrics

- **Macro F1**: 클래스 불균형 환경에서 정상/사기 클래스 평균 F1
- **PR-AUC**: Precision-Recall 곡선 아래 면적 — 사기 탐지 핵심 지표

> ROC-AUC는 클래스 불균형 데이터셋에서 과낙관적 결과를 내므로 사용하지 않음

## How to Run

```bash
# 1. 환경 설정
conda create -n gnn python=3.11
conda activate gnn
pip install -r requirements.txt

# 2. 데이터 준비
# data/ 폴더에 yelpzip.csv 배치

# 3. 파이프라인 순서대로 실행
pipeline_1/01.Preprocessing_0512.ipynb   # 전처리
pipeline_1/02.Graph_Construction.ipynb   # 그래프 생성
pipeline_1/03.GNN_Training.ipynb         # 학습 및 평가
```

> **체크포인트 기능**: `03.GNN_Training.ipynb`은 실험 결과를 `grid_results.csv`에 실시간 저장합니다.  
> 중단 후 재실행 시 완료된 실험은 자동으로 SKIP하고 미완료 실험부터 이어서 진행합니다.

## Schedule

| 단계 | 기간 | 제출일 | 제출물 |
|------|------|--------|--------|
| **예선** | 5/2 ~ 5/15 | 5/15 오후 1시 | 분석 보고서 + 코드 |
| **본선** | 5/16 ~ 5/23 | 5/22 오전 1시 | PPT + 보고서 + 대시보드 + 코드 |

## Team

SKK DScover
