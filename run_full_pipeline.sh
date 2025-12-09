#!/bin/bash
# 청크 데이터셋 생성부터 학습까지 전체 파이프라인 실행 스크립트
# log_specific_model_comparison.py를 사용하여 전체 프로세스 실행

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "로그 특화 모델 전체 파이프라인 실행"
echo "======================================================================"
echo ""

# 기본 설정
LOG_DIR="${LOG_DIR:-logs/backup}"
CHUNK_DIR="${CHUNK_DIR:-chunks}"
SPLIT_DIR="${SPLIT_DIR:-split_data}"
PARSED_DATA="${PARSED_DATA:-parsed_data.parquet}"

# 옵션 파싱
SKIP_PARSING=false
SKIP_SPLIT=false
KEEP_CHUNKS=true
CHUNK_SIZE=10000
CHUNK_READ_SIZE=100000

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-parsing)
            SKIP_PARSING=true
            shift
            ;;
        --skip-split)
            SKIP_SPLIT=true
            shift
            ;;
        --no-keep-chunks)
            KEEP_CHUNKS=false
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-read-size)
            CHUNK_READ_SIZE="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: $0 [--skip-parsing] [--skip-split] [--no-keep-chunks] [--chunk-size SIZE] [--chunk-read-size SIZE]"
            exit 1
            ;;
    esac
done

echo "설정:"
echo "  - 로그 디렉토리: $LOG_DIR"
echo "  - 청크 디렉토리: $CHUNK_DIR"
echo "  - 분할 데이터 디렉토리: $SPLIT_DIR"
echo "  - 파싱 데이터 파일: $PARSED_DATA"
echo "  - 청크 크기: $CHUNK_SIZE"
echo "  - 청크 읽기 크기: $CHUNK_READ_SIZE"
echo "  - 청크 파일 유지: $KEEP_CHUNKS"
echo ""

# 1단계: 로그 파싱 및 청크 생성
if [ "$SKIP_PARSING" = false ]; then
    echo "======================================================================"
    echo "1단계: 로그 파싱 및 청크 파일 생성"
    echo "======================================================================"
    
    KEEP_CHUNKS_FLAG=""
    if [ "$KEEP_CHUNKS" = true ]; then
        KEEP_CHUNKS_FLAG="--keep-chunks"
    fi
    
    python log_specific_model_comparison.py \
        --save-parsed "$PARSED_DATA" \
        --chunk-size "$CHUNK_SIZE" \
        $KEEP_CHUNKS_FLAG
    
    if [ $? -ne 0 ]; then
        echo "❌ 파싱 실패"
        exit 1
    fi
    
    echo ""
    echo "✅ 파싱 완료"
    echo ""
else
    echo "⏭️  파싱 건너뛰기 (--skip-parsing)"
    echo ""
fi

# 2단계: 데이터 분할 (스트리밍 방식)
if [ "$SKIP_SPLIT" = false ]; then
    echo "======================================================================"
    echo "2단계: 데이터 분할 (스트리밍 방식)"
    echo "======================================================================"
    
    if [ ! -f "$PARSED_DATA" ]; then
        echo "❌ 파싱 데이터 파일이 없습니다: $PARSED_DATA"
        echo "   --skip-parsing 옵션을 제거하거나 파싱을 먼저 수행하세요."
        exit 1
    fi
    
    python log_specific_model_comparison.py \
        --load-parsed "$PARSED_DATA" \
        --streaming-split \
        --split-output-dir "$SPLIT_DIR" \
        --chunk-read-size "$CHUNK_READ_SIZE"
    
    if [ $? -ne 0 ]; then
        echo "❌ 데이터 분할 실패"
        exit 1
    fi
    
    echo ""
    echo "✅ 데이터 분할 완료"
    echo ""
else
    echo "⏭️  데이터 분할 건너뛰기 (--skip-split)"
    echo ""
fi

# 3단계: 모델 학습 및 평가
echo "======================================================================"
echo "3단계: 모델 학습 및 평가"
echo "======================================================================"

if [ ! -d "$SPLIT_DIR" ]; then
    echo "❌ 분할 데이터 디렉토리가 없습니다: $SPLIT_DIR"
    echo "   --skip-split 옵션을 제거하거나 데이터 분할을 먼저 수행하세요."
    exit 1
fi

python log_specific_model_comparison.py \
    --load-split "$SPLIT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 모델 학습/평가 실패"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ 전체 파이프라인 완료!"
echo "======================================================================"
echo ""
echo "생성된 파일/디렉토리:"
echo "  - 파싱 데이터: $PARSED_DATA"
if [ "$KEEP_CHUNKS" = true ]; then
    echo "  - 청크 파일: $CHUNK_DIR/"
fi
echo "  - 분할 데이터: $SPLIT_DIR/"
echo "  - 결과: results/log_specific_comparison/"
echo ""




