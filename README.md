# GNN-based Fraud Review Detection

> ITDA 연합학술제 | SKK DScover

## Overview

YelpZip 데이터셋을 활용하여 **GNN(Graph Neural Network) 기반 조직적 어뷰징 네트워크 탐지 모델**을 구축하고, 결과를 대시보드로 시각화하는 프로젝트입니다.

## Task

- **목표**: 사기 리뷰(Fraudulent Review) 탐지 (Binary Classification)
- **데이터**: [YelpZip Dataset](https://www.kaggle.com/datasets/vaibhavsonkar/yelpzip)
- **핵심 모델**: GNN 계열 (GCN / GAT / GraphSAGE 등)
- **평가 지표**: PR-AUC, Macro F1

## Graph Design

| 구성 요소 | 설명 |
|---|---|
| **Node** | 리뷰 단위 |
| **기본 Relation** | R-U-R / R-T-R / R-S-R 중 1개 이상 |
| **커스텀 Relation** | 팀 자체 설계 (창의성 평가 요소) |

## Project Structure

```
fraud_review_detection-dscover-project/
├── data/               # 원본 및 전처리 데이터 (git 미추적)
├── notebooks/          # EDA 및 실험 노트북
├── src/                # 모델 및 전처리 소스 코드
├── dashboard/          # Streamlit / Dash 시각화
├── reports/            # 분석 보고서
└── README.md
```

## Schedule

| 단계 | 기간 | 제출일 | 제출물 |
|---|---|---|---|
| **예선** | 5/2 ~ 5/15 | 5/15 오후 1시 | 분석 보고서 + 코드 |
| **본선** | 5/16 ~ 5/23 | 5/22 오전 1시 | PPT + 보고서 + 대시보드 + 코드 |

## Requirements

```bash
pip install -r requirements.txt
```

## Team

SKK DScover
