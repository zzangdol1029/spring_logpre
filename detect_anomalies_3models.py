#!/usr/bin/env python3
"""
3개 모델(DeepLog, LogAnomaly, LogRobust)을 사용한 이상 탐지 스크립트

사용법:
    python detect_anomalies_3models.py --log-file <로그파일경로>
    python detect_anomalies_3models.py --log-dir <로그디렉토리경로>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from log_anomaly_detector import SpringBootLogParser
from log_specific_anomaly_detectors import LogSpecificAnomalySystem


def load_trained_models(results_dir=None):
    """
    학습된 모델 로드
    
    Args:
        results_dir: 결과 디렉토리 경로 (None이면 자동으로 찾기)
    
    Returns:
        dict: 모델 타입별 시스템 딕셔너리
    """
    if results_dir is None:
        # 자동으로 최신 결과 디렉토리 찾기
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_base = os.path.join(base_dir, 'results', 'log_specific_comparison')
        
        # 번호가 가장 큰 디렉토리 찾기
        max_num = -1
        latest_dir = None
        
        if os.path.exists(results_base):
            latest_dir = results_base
            max_num = 0
        
        for i in range(1, 100):
            check_dir = f"{results_base}_{i}"
            if os.path.exists(check_dir):
                latest_dir = check_dir
                max_num = i
            else:
                break
        
        if latest_dir:
            results_dir = latest_dir
            print(f"📂 최신 학습 결과 디렉토리 사용: {results_dir}")
        else:
            print("⚠️ 학습된 모델을 찾을 수 없습니다.")
            print("   먼저 log_specific_model_comparison.py로 모델을 학습하세요.")
            return None
    
    # 모델은 학습 시점에 저장되지 않으므로, 
    # LogSpecificAnomalySystem을 새로 생성하고 학습된 상태를 가정
    # 실제로는 모델을 저장/로드하는 기능이 필요함
    
    print("⚠️ 현재는 모델 저장/로드 기능이 없습니다.")
    print("   이상 탐지를 하려면 먼저 모델을 학습해야 합니다.")
    print("   학습된 모델은 메모리에만 존재합니다.")
    
    return None


def detect_with_3models(logs_df, trained_systems=None):
    """
    3개 모델로 이상 탐지 수행
    
    Args:
        logs_df: 로그 DataFrame
        trained_systems: 학습된 시스템 딕셔너리 (None이면 새로 학습)
    
    Returns:
        dict: 모델별 탐지 결과
    """
    print("=" * 70)
    print("3개 모델 이상 탐지 시작")
    print("=" * 70)
    print(f"📊 분석할 로그: {len(logs_df):,}개")
    print()
    
    results = {}
    
    # 3개 모델 타입
    model_types = ['deeplog', 'loganomaly', 'logrobust']
    
    # PyTorch 확인
    try:
        import torch
    except ImportError:
        print("⚠️ PyTorch가 없어 LogRobust를 사용할 수 없습니다.")
        model_types = ['deeplog', 'loganomaly']
    
    if trained_systems is None:
        print("⚠️ 학습된 모델이 없습니다. 먼저 모델을 학습해야 합니다.")
        print("   log_specific_model_comparison.py를 실행하여 모델을 학습하세요.")
        return None
    
    # 각 모델로 이상 탐지
    for model_type in model_types:
        if model_type not in trained_systems:
            print(f"⚠️ {model_type.upper()} 모델이 학습되지 않았습니다. 건너뜁니다.")
            continue
        
        print(f"\n[{model_type.upper()}] 이상 탐지 중...")
        system = trained_systems[model_type]
        
        try:
            # 이상 탐지
            detection_results = system.detect_anomalies(logs_df)
            
            if detection_results and not detection_results.get('anomalies', pd.DataFrame()).empty:
                anomalies_df = detection_results['anomalies']
                summary = detection_results.get('summary', {})
                
                print(f"   ✅ {len(anomalies_df)}개 이상 시퀀스 탐지")
                
                if 'by_severity' in summary:
                    print(f"   심각도 분포:")
                    for level, count in summary['by_severity'].items():
                        print(f"      {level}: {count}개")
                
                results[model_type] = {
                    'anomalies': anomalies_df,
                    'summary': summary,
                    'total_detected': len(anomalies_df)
                }
            else:
                print(f"   ✅ 이상치가 탐지되지 않았습니다.")
                results[model_type] = {
                    'anomalies': pd.DataFrame(),
                    'summary': {},
                    'total_detected': 0
                }
        except Exception as e:
            print(f"   ❌ 이상 탐지 실패: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {
                'anomalies': pd.DataFrame(),
                'summary': {},
                'total_detected': 0,
                'error': str(e)
            }
    
    return results


def compare_results(results):
    """
    3개 모델의 탐지 결과 비교
    
    Args:
        results: 모델별 탐지 결과 딕셔너리
    """
    print("\n" + "=" * 70)
    print("3개 모델 탐지 결과 비교")
    print("=" * 70)
    
    comparison_data = []
    for model_type, result in results.items():
        total = result.get('total_detected', 0)
        summary = result.get('summary', {})
        by_severity = summary.get('by_severity', {})
        
        comparison_data.append({
            '모델': model_type.upper(),
            '탐지된 이상': f"{total}개",
            'CRITICAL': by_severity.get('CRITICAL', 0),
            'HIGH': by_severity.get('HIGH', 0),
            'MEDIUM': by_severity.get('MEDIUM', 0),
            'LOW': by_severity.get('LOW', 0),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 탐지 결과 비교:")
    print(comparison_df.to_string(index=False))
    
    # 공통으로 탐지된 이상 찾기
    print("\n🔍 공통 탐지 분석:")
    all_anomalies = {}
    for model_type, result in results.items():
        anomalies_df = result.get('anomalies', pd.DataFrame())
        if not anomalies_df.empty and 'sequence_index' in anomalies_df.columns:
            detected_indices = set(anomalies_df['sequence_index'].values)
            all_anomalies[model_type] = detected_indices
    
    if len(all_anomalies) >= 2:
        # 2개 이상 모델이 공통으로 탐지한 시퀀스
        common_indices = set.intersection(*all_anomalies.values())
        print(f"   공통 탐지: {len(common_indices)}개 시퀀스")
        
        # 각 모델만 탐지한 시퀀스
        for model_type, indices in all_anomalies.items():
            unique = indices - set.union(*[v for k, v in all_anomalies.items() if k != model_type])
            print(f"   {model_type.upper()}만 탐지: {len(unique)}개 시퀀스")


def save_results(results, output_dir):
    """
    탐지 결과 저장
    
    Args:
        results: 모델별 탐지 결과
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 결과 저장 중: {output_dir}")
    
    for model_type, result in results.items():
        anomalies_df = result.get('anomalies', pd.DataFrame())
        
        if not anomalies_df.empty:
            output_path = os.path.join(output_dir, f"{model_type}_anomalies_{timestamp}.csv")
            anomalies_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"   ✅ {model_type.upper()}: {output_path}")
    
    # 비교 결과 저장
    comparison_data = []
    for model_type, result in results.items():
        total = result.get('total_detected', 0)
        summary = result.get('summary', {})
        by_severity = summary.get('by_severity', {})
        
        comparison_data.append({
            '모델': model_type.upper(),
            '탐지된_이상': total,
            'CRITICAL': by_severity.get('CRITICAL', 0),
            'HIGH': by_severity.get('HIGH', 0),
            'MEDIUM': by_severity.get('MEDIUM', 0),
            'LOW': by_severity.get('LOW', 0),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 비교 결과: {comparison_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='3개 모델로 이상 탐지')
    parser.add_argument('--log-file', type=str, default=None,
                       help='분석할 로그 파일 경로')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='분석할 로그 디렉토리 경로')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='결과 저장 디렉토리 (기본: results/detection_YYYYMMDD_HHMMSS)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='학습된 모델이 있는 디렉토리 (기본: 최신 결과 디렉토리)')
    args = parser.parse_args()
    
    # 로그 파일/디렉토리 확인
    if not args.log_file and not args.log_dir:
        print("❌ --log-file 또는 --log-dir 옵션이 필요합니다.")
        parser.print_help()
        return
    
    # 로그 파싱
    print("=" * 70)
    print("로그 파일 파싱")
    print("=" * 70)
    
    parser_obj = SpringBootLogParser()
    
    if args.log_file:
        if not os.path.exists(args.log_file):
            print(f"❌ 로그 파일을 찾을 수 없습니다: {args.log_file}")
            return
        print(f"📄 로그 파일: {args.log_file}")
        logs_df = parser_obj.parse_log_file(args.log_file)
    else:
        if not os.path.exists(args.log_dir):
            print(f"❌ 로그 디렉토리를 찾을 수 없습니다: {args.log_dir}")
            return
        print(f"📁 로그 디렉토리: {args.log_dir}")
        logs_df = parser_obj.parse_directory(args.log_dir, max_files=None, sample_lines=None)
    
    if logs_df.empty:
        print("⚠️ 파싱된 로그가 없습니다.")
        return
    
    print(f"✅ {len(logs_df):,}개 로그 라인 파싱 완료")
    
    # 학습된 모델 로드 (현재는 학습 기능이 없으므로 경고만)
    print("\n" + "=" * 70)
    print("⚠️ 중요: 현재는 모델 저장/로드 기능이 없습니다.")
    print("=" * 70)
    print("3개 모델로 이상 탐지를 하려면:")
    print("  1. log_specific_model_comparison.py를 실행하여 모델 학습")
    print("  2. 학습된 모델을 메모리에 유지한 상태에서")
    print("  3. 이 스크립트를 같은 프로세스에서 실행")
    print()
    print("또는 log_specific_model_comparison.py의 evaluate_models 결과를 사용하세요.")
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, 'results', f'detection_{timestamp}')
    
    print(f"\n📁 결과 저장 디렉토리: {output_dir}")
    
    # 실제 사용 예시 출력
    print("\n" + "=" * 70)
    print("사용 방법")
    print("=" * 70)
    print("""
현재는 모델 저장/로드 기능이 없으므로, 다음과 같이 사용하세요:

방법 1: log_specific_model_comparison.py 사용 (권장)
  - 이 스크립트는 학습과 평가를 함께 수행합니다
  - 평가 단계에서 test 데이터에 대해 3개 모델로 이상 탐지를 수행합니다
  - 결과는 results/log_specific_comparison_*/ 폴더에 저장됩니다

방법 2: 직접 코드 작성
  from log_specific_anomaly_detectors import LogSpecificAnomalySystem
  from log_anomaly_detector import SpringBootLogParser
  
  # 1. 로그 파싱
  parser = SpringBootLogParser()
  logs_df = parser.parse_directory("logs/backup")
  
  # 2. 각 모델 학습
  systems = {}
  for model_type in ['deeplog', 'loganomaly', 'logrobust']:
      system = LogSpecificAnomalySystem(model_type=model_type)
      system.load_logs(logs_df)
      system.train()
      systems[model_type] = system
  
  # 3. 새로운 로그에 대해 이상 탐지
  new_logs_df = parser.parse_log_file("new_log.log")
  for model_type, system in systems.items():
      results = system.detect_anomalies(new_logs_df)
      print(f"{model_type}: {len(results['anomalies'])}개 탐지")
""")


if __name__ == "__main__":
    main()

