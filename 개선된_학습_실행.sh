#!/bin/bash

# 개선된 학습 및 평가 실행 스크립트
# --optimize-threshold 옵션과 더 많은 학습 데이터 사용

cd /Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog

echo "======================================================================"
echo "개선된 모델 학습 및 평가 시작"
echo "======================================================================"
echo ""
echo "설정:"
echo "  - 최적 임계값 탐색: 활성화"
echo "  - 학습 데이터: 500,000개"
echo "  - Epochs: 20"
echo "  - Batch Size: 64"
echo "  - 목표 성능:"
echo "    * 정확도: 70%"
echo "    * 정밀도: 50%"
echo "    * 재현율: 60%"
echo "    * F1 점수: 55%"
echo "    * 특이도: 80%"
echo ""
echo "예상 소요 시간: 약 2-3시간"
echo ""
echo "======================================================================"
echo ""

python log_specific_model_comparison.py \
    --load-split split_data \
    --optimize-threshold \
    --sample-size 500000 \
    --epochs 20 \
    --batch-size 64 \
    --target-accuracy 0.70 \
    --target-precision 0.50 \
    --target-recall 0.60 \
    --target-f1 0.55 \
    --target-specificity 0.80

echo ""
echo "=" | tr -d '\n' | head -c 70
echo ""
echo "✅ 학습 및 평가 완료!"
echo "=" | tr -d '\n' | head -c 70
echo ""

