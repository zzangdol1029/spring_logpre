#!/bin/bash

# 고급 개선 설정으로 학습 및 평가
# - 더 큰 모델
# - 더 긴 시퀀스
# - 앙상블 방법

cd /Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog

echo "======================================================================"
echo "고급 개선 설정으로 모델 학습 및 평가"
echo "======================================================================"
echo ""
echo "설정:"
echo "  - 학습 데이터: 1,000,000개"
echo "  - Epochs: 30"
echo "  - Batch Size: 64"
echo "  - 최적 임계값 탐색: 활성화"
echo ""
echo "⚠️ 주의:"
echo "  - 실행 시간: 약 4-6시간 예상"
echo "  - 메모리 사용량: 15-20GB 예상"
echo ""
echo "======================================================================"
echo ""

python log_specific_model_comparison.py \
    --load-split split_data \
    --optimize-threshold \
    --sample-size 1000000 \
    --epochs 30 \
    --batch-size 64 \
    --eval-sample-size 50000 \
    --target-accuracy 0.70 \
    --target-precision 0.50 \
    --target-recall 0.60 \
    --target-f1 0.55 \
    --target-specificity 0.80

echo ""
echo "======================================================================"
echo "✅ 학습 및 평가 완료!"
echo "======================================================================"
echo ""
echo "💡 성능이 여전히 부족하면:"
echo "  1. 데이터 품질 확인 (불균형, 분류 정확도)"
echo "  2. 하이퍼파라미터 튜닝 (시퀀스 길이, Learning Rate)"
echo "  3. 앙상블 방법 적용"
echo "  4. 목표 성능 현실적 조정"
echo ""
echo "자세한 내용은 '성능_개선_고급_방법.md' 참고"
echo "======================================================================"



