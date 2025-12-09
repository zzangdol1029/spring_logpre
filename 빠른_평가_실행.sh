#!/bin/bash

# 빠른 평가 모델 생성 스크립트
# 실행 시간: 약 15-20분

cd /Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog

echo "======================================================================"
echo "빠른 평가 모델 생성 (약 15-20분 소요)"
echo "======================================================================"
echo ""
echo "설정:"
echo "  - 학습 데이터: 10만개"
echo "  - Epochs: 5"
echo "  - 평가 데이터: 2만개"
echo "  - 최적 임계값 자동 탐색"
echo ""

python log_specific_model_comparison.py \
    --optimize-threshold \
    --sample-size 100000 \
    --epochs 5 \
    --batch-size 64 \
    --eval-sample-size 20000

echo ""
echo "======================================================================"
echo "완료!"
echo "======================================================================"




