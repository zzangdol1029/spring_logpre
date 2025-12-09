#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
청크 데이터셋 생성부터 학습까지 전체 워크플로우 실행 스크립트

사용법:
    python train_from_chunks.py
    
옵션:
    --epochs: 학습 epoch 수 (기본: 10, 빠른 테스트용)
    --batch-size: 배치 크기 (기본: 64, 메모리 효율적)
    --chunk-size: 청크 크기 (기본: 10000)
    --keep-chunks: 청크 파일 유지 (기본: True, 재사용 가능)
"""

import os
import sys
import argparse
from datetime import datetime
from log_anomaly_detector import SpringBootLogParser
from log_specific_model_comparison import LogSpecificModelComparator

def main():
    """청크 생성부터 학습까지 전체 프로세스"""
    parser = argparse.ArgumentParser(description='청크 데이터셋 생성부터 학습까지')
    parser.add_argument('--epochs', type=int, default=10,
                       help='학습 epoch 수 (기본: 10, 빠른 테스트용)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='배치 크기 (기본: 64, 메모리 효율적)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='파싱 청크 크기 (기본: 10000)')
    parser.add_argument('--keep-chunks', action='store_true', default=True,
                       help='청크 파일 유지 (기본: True)')
    parser.add_argument('--skip-parsing', action='store_true',
                       help='파싱 건너뛰기 (이미 청크 파일이 있는 경우)')
    parser.add_argument('--skip-split', action='store_true',
                       help='데이터 분할 건너뛰기 (이미 분할된 데이터가 있는 경우)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='로그 디렉토리 경로 (기본: pattern/prelog/logs/backup)')
    parser.add_argument('--chunk-dir', type=str, default=None,
                       help='청크 파일 저장 디렉토리 (기본: pattern/prelog/chunks)')
    parser.add_argument('--split-dir', type=str, default=None,
                       help='분할 데이터 저장 디렉토리 (기본: pattern/prelog/split_data)')
    parser.add_argument('--models', type=str, nargs='+', default=['deeplog', 'loganomaly'],
                       help='학습할 모델 목록 (기본: deeplog loganomaly)')
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_directory = args.log_dir or os.path.join(script_dir, 'logs', 'backup')
    chunk_dir = args.chunk_dir or os.path.join(script_dir, 'chunks')
    split_dir = args.split_dir or os.path.join(script_dir, 'split_data')
    
    print("=" * 70)
    print("청크 데이터셋 생성부터 학습까지 전체 워크플로우")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n설정:")
    print(f"  - 로그 디렉토리: {log_directory}")
    print(f"  - 청크 디렉토리: {chunk_dir}")
    print(f"  - 분할 데이터 디렉토리: {split_dir}")
    print(f"  - Epoch 수: {args.epochs}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - 청크 크기: {args.chunk_size:,}")
    print(f"  - 학습 모델: {', '.join(args.models)}")
    print("=" * 70)
    
    # 1단계: 로그 파싱 및 청크 생성
    print("\n" + "=" * 70)
    print("1단계: 로그 파싱 및 청크 파일 생성")
    print("=" * 70)
    
    parser_obj = SpringBootLogParser()
    
    if args.skip_parsing:
        print("⏭️  파싱 건너뛰기 (기존 청크 파일 사용)")
        if not os.path.exists(chunk_dir) or len(os.listdir(chunk_dir)) == 0:
            print("⚠️  청크 디렉토리가 비어있습니다. --skip-parsing 옵션을 제거하세요.")
            return
    else:
        if not os.path.exists(log_directory):
            print(f"❌ 로그 디렉토리가 없습니다: {log_directory}")
            return
        
        # 청크 디렉토리 생성
        os.makedirs(chunk_dir, exist_ok=True)
        print(f"📁 청크 파일 저장 위치: {chunk_dir}")
        
        # 파싱 수행
        print(f"📝 로그 파일 파싱 중...")
        logs_df = parser_obj.parse_directory(
            log_directory,
            max_files=None,
            sample_lines=None,
            chunk_size=args.chunk_size,
            max_total_lines=None,
            save_chunks_to_disk=True,
            chunk_dir=chunk_dir,
            keep_chunks=args.keep_chunks
        )
        
        if logs_df.empty:
            print("⚠️  파싱된 로그가 없습니다.")
            return
        
        print(f"✅ 파싱 완료: {len(logs_df):,}개 로그")
        
        # 파싱 데이터를 단일 파일로 저장 (선택적)
        parsed_data_path = os.path.join(script_dir, 'parsed_data.parquet')
        print(f"💾 파싱 데이터 저장 중: {parsed_data_path}")
        parser_obj.save_parsed_data(logs_df, parsed_data_path)
        print(f"✅ 파싱 데이터 저장 완료")
        
        # 메모리 해제
        del logs_df
        import gc
        gc.collect()
    
    # 2단계: 데이터 분할 (스트리밍 방식)
    print("\n" + "=" * 70)
    print("2단계: 데이터 분할 (스트리밍 방식)")
    print("=" * 70)
    
    if args.skip_split:
        print("⏭️  데이터 분할 건너뛰기 (기존 분할 데이터 사용)")
        if not os.path.exists(split_dir):
            print("⚠️  분할 데이터 디렉토리가 없습니다. --skip-split 옵션을 제거하세요.")
            return
    else:
        # 파싱 데이터 로드 또는 청크에서 로드
        parsed_data_path = os.path.join(script_dir, 'parsed_data.parquet')
        
        if os.path.exists(parsed_data_path):
            print(f"📂 파싱 데이터 로드: {parsed_data_path}")
            input_path = parsed_data_path
        else:
            print(f"📂 청크 파일에서 데이터 사용: {chunk_dir}")
            input_path = chunk_dir
        
        # 분할 디렉토리 생성
        os.makedirs(split_dir, exist_ok=True)
        print(f"📁 분할 데이터 저장 위치: {split_dir}")
        
        # 스트리밍 분할 수행
        print(f"💡 스트리밍 분할 모드: 메모리 효율적으로 분할 중...")
        if os.path.isdir(input_path):
            # 청크 디렉토리인 경우
            print("   청크 파일에서 직접 분할...")
            # 청크 파일을 하나씩 읽어서 분할
            chunk_files = parser_obj.get_chunk_files(input_path)
            print(f"   총 {len(chunk_files)}개 청크 파일 발견")
            
            # 간단한 방법: 청크를 로드해서 분할
            # 더 효율적인 방법은 prepare_data_streaming을 수정해야 함
            # 여기서는 parsed_data.parquet가 있다고 가정
            if os.path.exists(parsed_data_path):
                split_files = parser_obj.prepare_data_streaming(
                    parsed_data_path,
                    split_dir,
                    train_ratio=0.8,
                    valid_ratio=0.2,
                    chunk_size=100000
                )
            else:
                print("⚠️  parsed_data.parquet 파일이 없습니다. 먼저 파싱을 수행하세요.")
                return
        else:
            # 단일 파일인 경우
            split_files = parser_obj.prepare_data_streaming(
                input_path,
                split_dir,
                train_ratio=0.8,
                valid_ratio=0.2,
                chunk_size=100000
            )
        
        print(f"✅ 데이터 분할 완료")
        print(f"   - Train: {split_files.get('train', 'N/A')}")
        print(f"   - Valid: {split_files.get('valid', 'N/A')}")
        print(f"   - Test: {split_files.get('test', 'N/A')}")
    
    # 3단계: 모델 학습
    print("\n" + "=" * 70)
    print("3단계: 모델 학습")
    print("=" * 70)
    
    comparator = LogSpecificModelComparator()
    
    # 분할된 데이터 로드 (학습용만)
    print("📂 학습 데이터 로드 중...")
    data = comparator.prepare_data_from_files(split_dir, load_only_train=True)
    
    if not data or data.get('train_normal', None) is None or data['train_normal'].empty:
        print("⚠️  학습 데이터가 없습니다.")
        return
    
    print(f"✅ 학습 데이터 로드 완료:")
    print(f"   - Train Normal: {len(data['train_normal']):,}개")
    
    # 사용 가능한 모델 확인
    available_models = []
    for model in args.models:
        if model == 'logrobust':
            try:
                import torch
                available_models.append(model)
                print(f"   ✅ {model.upper()} 사용 가능")
            except ImportError:
                print(f"   ⚠️  {model.upper()} 제외 (PyTorch 미설치)")
        else:
            available_models.append(model)
            print(f"   ✅ {model.upper()} 사용 가능")
    
    if not available_models:
        print("⚠️  학습 가능한 모델이 없습니다.")
        return
    
    # 학습 로그 디렉토리 설정
    log_dir = os.path.join(script_dir, 'logs', 'training')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n🚀 모델 학습 시작")
    print(f"   - Epoch: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   💡 학습 시간 단축을 위해 Epoch 수를 줄였습니다.")
    print(f"   💡 메모리 효율을 위해 배치 크기를 늘렸습니다.")
    
    # 모델별 학습 파라미터 설정
    # DeepLog의 경우 train 메서드에서 epochs와 batch_size를 받을 수 있도록 수정 필요
    # 여기서는 기본값으로 진행 (실제 적용은 모델 학습 함수 내부에서 처리)
    comparator.train_models(
        data['train_normal'],
        valid_normal_logs=data.get('valid_normal'),
        model_types=available_models,
        log_dir=log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not comparator.trained_systems:
        print("⚠️  학습된 모델이 없습니다.")
        return
    
    # 학습 데이터 메모리 해제
    del data['train_normal']
    if 'train_error' in data:
        del data['train_error']
    if 'valid_normal' in data:
        del data['valid_normal']
    import gc
    gc.collect()
    
    # 4단계: 모델 평가
    print("\n" + "=" * 70)
    print("4단계: 모델 성능 평가")
    print("=" * 70)
    
    # 평가용 데이터 로드
    print("📂 평가 데이터 로드 중...")
    test_data = comparator.load_test_data(split_dir)
    
    if test_data['test_logs'].empty:
        print("⚠️  테스트 데이터가 없습니다.")
        return
    
    print(f"✅ 평가 데이터 로드 완료:")
    print(f"   - Test Logs: {len(test_data['test_logs']):,}개")
    
    results = comparator.evaluate_models(test_data['test_logs'], test_data['y_test'])
    
    if not results:
        print("⚠️  평가 결과가 없습니다.")
        return
    
    # 5단계: 결과 리포트
    print("\n" + "=" * 70)
    print("5단계: 결과 리포트 생성")
    print("=" * 70)
    
    output_dir = os.path.join(script_dir, 'results', 'log_specific_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df, best_model = comparator.generate_comparison_report(output_dir=output_dir)
    
    # 최적 모델 선정
    best_model_name, best_system = comparator.get_best_model()
    
    print(f"\n{'='*70}")
    print(f"🏆 최종 선정된 모델: {best_model_name.upper()}")
    print(f"{'='*70}")
    
    if best_model_name in results:
        best_metrics = results[best_model_name]
        print(f"\n성능 지표:")
        print(f"   정확도: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"   정밀도: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
        print(f"   재현율: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
        print(f"   F1 점수: {best_metrics['f1_score']:.4f}")
        if best_metrics.get('roc_auc'):
            print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    print(f"\n결과 저장 위치: {output_dir}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("✅ 전체 프로세스 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

