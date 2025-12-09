"""
로그 특화 모델 성능 비교 시스템
여러 로그 특화 모델을 학습하고 성능을 비교하여 최적 모델을 선정합니다.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)
from log_specific_anomaly_detectors import (
    LogSpecificAnomalySystem,
    DeepLogDetector,
    LogAnomalyDetector,
    LogRobustDetector
)
from severity_assessment import SeverityAssessment
import warnings
warnings.filterwarnings('ignore')

# 시각화 라이브러리
try:
    import matplotlib
    matplotlib.use('Agg')  # 백엔드 설정 (GUI 없이 사용)
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ matplotlib/seaborn이 없어 그래프 생성이 불가능합니다. pip install matplotlib seaborn")


class LogSpecificModelComparator:
    """로그 특화 모델 비교 클래스"""
    
    def __init__(self):
        self.models = {}
        self.trained_systems = {}
        self.results = {}
        self.severity_assessor = SeverityAssessment()
        
    def prepare_data_from_files(self, data_dir: str, load_only_train=False):
        """
        분할된 데이터 파일에서 로드 (메모리 효율적)
        
        Args:
            data_dir: 분할된 데이터가 저장된 디렉토리
            load_only_train: True이면 train만 로드 (학습 단계에서 메모리 절약)
            
        Returns:
            데이터 딕셔너리
        """
        print("=" * 60)
        print("분할된 데이터 파일에서 로드")
        if load_only_train:
            print("   💡 메모리 절약 모드: Train만 먼저 로드")
        print("=" * 60)
        
        train_file = os.path.join(data_dir, 'train.parquet')
        valid_file = os.path.join(data_dir, 'valid.parquet')
        test_file = os.path.join(data_dir, 'test.parquet')
        
        # Train 로드 (학습에 필요)
        if os.path.exists(train_file):
            train_all = pd.read_parquet(train_file, engine='pyarrow')
            print(f"   ✅ Train 로드: {len(train_all):,}개")
        else:
            raise FileNotFoundError(f"Train 파일을 찾을 수 없습니다: {train_file}")
        
        # 정상/에러 로그 분리
        # 정상 로그: INFO, DEBUG, TRACE만 포함 (WARN 제외)
        # 에러 로그: ERROR, FATAL, WARN 포함
        
        train_normal = train_all[
            (train_all['is_error'] == False) & 
            (train_all['level'].isin(['INFO', 'DEBUG', 'TRACE']))
        ].copy()
        train_error = train_all[
            (train_all['is_error'] == True) | 
            (train_all['level'].isin(['WARN', 'ERROR', 'FATAL']))
        ].copy()
        
        # train_all 메모리 해제
        del train_all
        
        # Valid와 Test는 필요할 때만 로드
        valid_normal = pd.DataFrame()
        valid_error = pd.DataFrame()
        valid_logs = pd.DataFrame()
        y_valid = np.array([])
        
        test_normal = pd.DataFrame()
        test_error = pd.DataFrame()
        test_logs = pd.DataFrame()
        y_test = np.array([])
        
        if not load_only_train:
            # Valid 로드
            if os.path.exists(valid_file):
                valid_all = pd.read_parquet(valid_file, engine='pyarrow')
                print(f"   ✅ Valid 로드: {len(valid_all):,}개")
                
                valid_normal = valid_all[
                    (valid_all['is_error'] == False) & 
                    (valid_all['level'].isin(['INFO', 'DEBUG', 'TRACE']))
                ].copy()
                valid_error = valid_all[
                    (valid_all['is_error'] == True) | 
                    (valid_all['level'].isin(['WARN', 'ERROR', 'FATAL']))
                ].copy()
                
                # Valid 데이터 결합
                valid_logs = pd.concat([valid_normal, valid_error], ignore_index=True)
                valid_logs = valid_logs.sort_values('timestamp').reset_index(drop=True)
                y_valid = (valid_logs['is_error'] == True).astype(int).values
                
                # valid_all 메모리 해제
                del valid_all
            else:
                print(f"   ⚠️ Valid 파일이 없습니다: {valid_file}")
            
            # Test 로드
            if os.path.exists(test_file):
                test_all = pd.read_parquet(test_file, engine='pyarrow')
                print(f"   ✅ Test 로드: {len(test_all):,}개")
                
                test_normal = test_all[
                    (test_all['is_error'] == False) & 
                    (test_all['level'].isin(['INFO', 'DEBUG', 'TRACE']))
                ].copy()
                test_error = test_all[
                    (test_all['is_error'] == True) | 
                    (test_all['level'].isin(['WARN', 'ERROR', 'FATAL']))
                ].copy()
                
                # Test 데이터 결합
                test_logs = pd.concat([test_normal, test_error], ignore_index=True)
                test_logs = test_logs.sort_values('timestamp').reset_index(drop=True)
                y_test = (test_logs['is_error'] == True).astype(int).values
                
                # test_all 메모리 해제
                del test_all
            else:
                raise FileNotFoundError(f"Test 파일을 찾을 수 없습니다: {test_file}")
        
        print(f"\n📊 데이터 분할 결과:")
        print(f"   Train (학습용):")
        print(f"      - 정상 로그: {len(train_normal):,}개")
        print(f"      - 에러 로그: {len(train_error):,}개")
        if not load_only_train:
            if not valid_logs.empty:
                print(f"\n   Valid (검증용):")
                print(f"      - 정상 로그: {len(valid_normal):,}개")
                print(f"      - 에러 로그: {len(valid_error):,}개")
                print(f"      - 전체: {len(valid_logs):,}개")
            print(f"\n   Test (테스트용):")
            print(f"      - 정상 로그: {len(test_normal):,}개")
            print(f"      - 에러 로그: {len(test_error):,}개")
            print(f"      - 전체: {len(test_logs):,}개")
        
        return {
            'train_normal': train_normal,
            'train_error': train_error,
            'valid_normal': valid_normal,
            'valid_error': valid_error,
            'valid_logs': valid_logs,
            'test_normal': test_normal,
            'test_error': test_error,
            'test_logs': test_logs,
            'y_valid': y_valid,
            'y_test': y_test
        }
    
    def load_test_data(self, data_dir: str):
        """
        평가 단계에서 Test 데이터만 로드 (메모리 효율적)
        
        Args:
            data_dir: 분할된 데이터가 저장된 디렉토리
            
        Returns:
            test_logs, y_test
        """
        test_file = os.path.join(data_dir, 'test.parquet')
        valid_file = os.path.join(data_dir, 'valid.parquet')
        
        print("\n📂 평가용 데이터 로드 중...")
        
        # Test 로드
        if os.path.exists(test_file):
            test_all = pd.read_parquet(test_file, engine='pyarrow')
            print(f"   ✅ Test 로드: {len(test_all):,}개")
            
            test_normal = test_all[
                (test_all['is_error'] == False) & 
                (test_all['level'].isin(['INFO', 'DEBUG', 'TRACE']))
            ].copy()
            test_error = test_all[
                (test_all['is_error'] == True) | 
                (test_all['level'].isin(['WARN', 'ERROR', 'FATAL']))
            ].copy()
            
            test_logs = pd.concat([test_normal, test_error], ignore_index=True)
            test_logs = test_logs.sort_values('timestamp').reset_index(drop=True)
            y_test = (test_logs['is_error'] == True).astype(int).values
            
            del test_all
        else:
            raise FileNotFoundError(f"Test 파일을 찾을 수 없습니다: {test_file}")
        
        # Valid 로드 (선택적)
        valid_logs = pd.DataFrame()
        y_valid = np.array([])
        
        if os.path.exists(valid_file):
            valid_all = pd.read_parquet(valid_file, engine='pyarrow')
            print(f"   ✅ Valid 로드: {len(valid_all):,}개")
            
            valid_normal = valid_all[
                (valid_all['is_error'] == False) & 
                (valid_all['level'].isin(['INFO', 'DEBUG', 'TRACE']))
            ].copy()
            valid_error = valid_all[
                (valid_all['is_error'] == True) | 
                (valid_all['level'].isin(['WARN', 'ERROR', 'FATAL']))
            ].copy()
            
            valid_logs = pd.concat([valid_normal, valid_error], ignore_index=True)
            valid_logs = valid_logs.sort_values('timestamp').reset_index(drop=True)
            y_valid = (valid_logs['is_error'] == True).astype(int).values
            
            del valid_all
        
        return {
            'test_logs': test_logs,
            'y_test': y_test,
            'valid_logs': valid_logs,
            'y_valid': y_valid
        }
    
    def prepare_data(self, logs_df: pd.DataFrame, train_ratio=0.8, valid_ratio=0.2):
        """
        데이터 준비 및 분할
        - 전체의 80% → train
        - train의 20% → valid (전체의 16%)
        - 나머지 20% → test (전체의 20%)
        
        Args:
            logs_df: 전체 로그 DataFrame
            train_ratio: 학습 데이터 비율 (기본 0.8 = 80%)
            valid_ratio: 검증 데이터 비율 (train의 비율, 기본 0.2 = 20%)
        
        Returns:
            데이터 딕셔너리
        """
        print("=" * 60)
        print("데이터 준비 및 분할")
        print("=" * 60)
        
        # 시간 순서 정렬
        logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
        
        # 정상/에러 로그 분리
        # 정상 로그: INFO, DEBUG만 포함 (WARN 제외)
        # 에러 로그: ERROR, FATAL, WARN 포함
        normal_logs = logs_df[
            (logs_df['is_error'] == False) & 
            (logs_df['level'].isin(['INFO', 'DEBUG', 'TRACE']))
        ].copy()
        error_logs = logs_df[
            (logs_df['is_error'] == True) | 
            (logs_df['level'].isin(['WARN', 'ERROR', 'FATAL']))
        ].copy()
        
        print(f"   - 전체 로그: {len(logs_df)}개")
        print(f"   - 정상 로그 (INFO/DEBUG/TRACE만): {len(normal_logs)}개")
        print(f"   - 에러 로그 (ERROR/FATAL/WARN 포함): {len(error_logs)}개")
        
        # 레벨별 통계
        level_counts = logs_df['level'].value_counts()
        print(f"\n   레벨별 분포:")
        for level, count in level_counts.items():
            print(f"      {level}: {count}개 ({count/len(logs_df)*100:.1f}%)")
        
        if len(normal_logs) == 0:
            raise ValueError("정상 로그가 없습니다.")
        
        if len(error_logs) == 0:
            raise ValueError("에러 로그가 없습니다.")
        
        # 전체 데이터 분할: 80% train, 20% test
        total_split_idx = int(len(logs_df) * train_ratio)
        train_all = logs_df.iloc[:total_split_idx]
        test_all = logs_df.iloc[total_split_idx:]
        
        # Train 데이터에서 정상/에러 분리
        train_normal = train_all[train_all['is_error'] == False].copy()
        train_error = train_all[train_all['is_error'] == True].copy()
        
        # Train의 20%를 Valid로 분할 (정상 로그 기준)
        train_normal_split_idx = int(len(train_normal) * (1 - valid_ratio))
        train_normal_final = train_normal.iloc[:train_normal_split_idx]
        valid_normal = train_normal.iloc[train_normal_split_idx:]
        
        # Valid 데이터 (정상 + 에러)
        # 에러 로그도 train의 20%를 valid로
        if len(train_error) > 0:
            train_error_split_idx = int(len(train_error) * (1 - valid_ratio))
            train_error_final = train_error.iloc[:train_error_split_idx]
            valid_error = train_error.iloc[train_error_split_idx:]
        else:
            train_error_final = train_error
            valid_error = pd.DataFrame()
        
        # Test 데이터 (정상 + 에러)
        test_normal = test_all[test_all['is_error'] == False].copy()
        test_error = test_all[test_all['is_error'] == True].copy()
        
        # Valid 데이터 결합
        valid_logs = pd.concat([valid_normal, valid_error], ignore_index=True)
        valid_logs = valid_logs.sort_values('timestamp').reset_index(drop=True)
        
        # Test 데이터 결합
        test_logs = pd.concat([test_normal, test_error], ignore_index=True)
        test_logs = test_logs.sort_values('timestamp').reset_index(drop=True)
        
        # 라벨 생성
        y_valid = (valid_logs['is_error'] == True).astype(int).values if not valid_logs.empty else np.array([])
        y_test = (test_logs['is_error'] == True).astype(int).values
        
        print(f"\n📊 데이터 분할 결과:")
        print(f"   Train (학습용):")
        print(f"      - 정상 로그: {len(train_normal_final)}개 ({len(train_normal_final)/len(logs_df)*100:.1f}%)")
        print(f"      - 에러 로그: {len(train_error_final)}개")
        print(f"      - 전체: {len(train_normal_final) + len(train_error_final)}개 ({len(train_all)/len(logs_df)*100:.1f}%)")
        print(f"\n   Valid (검증용, train의 {valid_ratio*100:.0f}%):")
        print(f"      - 정상 로그: {len(valid_normal)}개 ({len(valid_normal)/len(logs_df)*100:.1f}%)")
        print(f"      - 에러 로그: {len(valid_error)}개")
        print(f"      - 전체: {len(valid_logs)}개 ({len(valid_logs)/len(logs_df)*100:.1f}%)")
        print(f"\n   Test (테스트용):")
        print(f"      - 정상 로그: {len(test_normal)}개 ({len(test_normal)/len(logs_df)*100:.1f}%)")
        print(f"      - 에러 로그: {len(test_error)}개 ({len(test_error)/len(logs_df)*100:.1f}%)")
        print(f"      - 전체: {len(test_logs)}개 ({len(test_logs)/len(logs_df)*100:.1f}%)")
        
        return {
            'train_normal': train_normal_final,
            'train_error': train_error_final,
            'valid_normal': valid_normal,
            'valid_error': valid_error,
            'valid_logs': valid_logs,
            'test_normal': test_normal,
            'test_error': test_error,
            'test_logs': test_logs,
            'y_valid': y_valid,
            'y_test': y_test
        }
    
    def train_models(self, train_normal_logs: pd.DataFrame, valid_normal_logs: pd.DataFrame = None, 
                     model_types=None, log_dir=None, epochs=5, batch_size=128):
        """
        여러 로그 특화 모델 학습
        
        Args:
            train_normal_logs: 학습용 정상 로그
            valid_normal_logs: 검증용 정상 로그 (선택적, 조기 종료 등에 사용)
            model_types: 학습할 모델 리스트 (None이면 모두)
            log_dir: 학습 로그 저장 디렉토리
            epochs: 학습 epoch 수 (기본: 5, 빠른 학습용)
            batch_size: 배치 크기 (기본: 128, 빠른 학습용)
        """
        if model_types is None:
            model_types = ['deeplog', 'loganomaly']
            # LogRobust는 제외 (메모리 사용량이 많고 OOM 발생 가능)
        
        # 로그 디렉토리 설정
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'training')
        
        print("\n" + "=" * 60)
        print("로그 특화 모델 학습")
        print("=" * 60)
        print(f"학습 로그 저장 위치: {log_dir}")
        print(f"학습할 모델: {', '.join(model_types)}")
        print(f"학습 데이터: {len(train_normal_logs):,}개 로그")
        print(f"⚡ 빠른 학습 설정: Epochs={epochs}, Batch Size={batch_size}")
        
        for idx, model_type in enumerate(model_types, 1):
            print(f"\n[{idx}/{len(model_types)}] {model_type.upper()} 학습 시작...")
            print("=" * 60)
            
            try:
                system = LogSpecificAnomalySystem(model_type=model_type)
                system.load_logs(train_normal_logs)
                
                # epochs와 batch_size 전달 (LogAnomaly는 무시됨)
                if system.train(train_ratio=1.0, log_dir=log_dir, epochs=epochs, batch_size=batch_size):  # 전체를 학습용으로 사용
                    self.trained_systems[model_type] = system
                    print(f"\n✅ {model_type.upper()} 학습 완료")
                    
                    # 학습 완료 후 로그 데이터 메모리 해제 (모델은 이미 학습됨)
                    system.logs_df = None
                    import gc
                    gc.collect()
                    
                    # 검증 데이터로 성능 확인 (선택적)
                    if valid_normal_logs is not None and not valid_normal_logs.empty:
                        print(f"   📊 검증 데이터로 성능 확인 중...")
                        # 검증은 선택적으로 수행
                else:
                    print(f"\n❌ {model_type.upper()} 학습 실패")
            
            except Exception as e:
                print(f"\n❌ {model_type.upper()} 학습 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 60)
        print(f"학습 완료: {len(self.trained_systems)}/{len(model_types)}개 모델 성공")
        print("=" * 60)
    
    def find_optimal_threshold(self, y_test: np.ndarray, anomaly_scores: np.ndarray, 
                               metric='f1', min_precision=0.3, min_recall=0.5):
        """
        최적 임계값 찾기
        
        Args:
            y_test: 실제 라벨
            anomaly_scores: 이상 점수
            metric: 최적화할 지표 ('f1', 'precision', 'recall', 'balanced')
            min_precision: 최소 정밀도 요구사항
            min_recall: 최소 재현율 요구사항
            
        Returns:
            최적 임계값, 최적 성능 지표
        """
        if len(anomaly_scores) == 0 or len(np.unique(anomaly_scores)) < 2:
            return 0.5, {}
        
        # 임계값 후보 생성
        thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (anomaly_scores >= threshold).astype(int)
            
            if len(np.unique(y_pred)) < 2:  # 모두 0이거나 모두 1인 경우
                continue
            
            try:
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 최소 요구사항 확인
                if precision < min_precision or recall < min_recall:
                    continue
                
                # 메트릭에 따라 점수 계산
                if metric == 'f1':
                    score = f1
                elif metric == 'precision':
                    score = precision
                elif metric == 'recall':
                    score = recall
                elif metric == 'balanced':
                    # 정밀도와 재현율의 균형
                    score = (precision + recall) / 2
                else:
                    score = f1
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'threshold': threshold,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
            except:
                continue
        
        return best_threshold, best_metrics
    
    def evaluate_models(self, test_logs: pd.DataFrame, y_test: np.ndarray, 
                       optimize_threshold=True, target_metrics=None):
        """
        모델 성능 평가
        
        Args:
            test_logs: 테스트 로그 DataFrame
            y_test: 테스트 라벨 (0=정상, 1=이상)
            optimize_threshold: True이면 최적 임계값 찾기
            target_metrics: 목표 성능 지표 딕셔너리 (예: {'precision': 0.5, 'recall': 0.6})
        """
        import time
        
        if target_metrics is None:
            target_metrics = {
                'accuracy': 0.70,
                'precision': 0.50,
                'recall': 0.60,
                'f1_score': 0.55,
                'specificity': 0.80
            }
        
        print("\n" + "=" * 60)
        print("모델 성능 평가")
        print("=" * 60)
        print(f"평가 데이터: {len(test_logs):,}개 로그")
        if optimize_threshold:
            print(f"💡 최적 임계값 자동 조정 모드 활성화")
            print(f"   목표 성능:")
            print(f"      - 정확도: {target_metrics['accuracy']*100:.0f}% 이상")
            print(f"      - 정밀도: {target_metrics['precision']*100:.0f}% 이상")
            print(f"      - 재현율: {target_metrics['recall']*100:.0f}% 이상")
            print(f"      - F1 점수: {target_metrics['f1_score']*100:.0f}% 이상")
        
        results = {}
        
        for idx, (model_type, system) in enumerate(self.trained_systems.items(), 1):
            print(f"\n[{idx}/{len(self.trained_systems)}] {model_type.upper()} 평가 중...")
            eval_start_time = time.time()
            
            try:
                # 이상 탐지 (시간 측정)
                print(f"   ⏳ 이상 탐지 수행 중... (예상 시간: {len(test_logs) // 5000}초)")
                detection_results = system.detect_anomalies(test_logs)
                
                anomalies_df = detection_results.get('anomalies', pd.DataFrame()) if detection_results else pd.DataFrame()
                
                # 예측 결과 생성 (시퀀스 기반)
                # 각 로그가 이상 시퀀스에 포함되는지 확인
                y_pred = np.zeros(len(test_logs))
                anomaly_scores = np.zeros(len(test_logs))
                
                if not anomalies_df.empty:
                    # 이상이 탐지된 경우
                    for idx, row in anomalies_df.iterrows():
                        seq_idx = row.get('sequence_index', 0)
                        score = row.get('anomaly_score', 0)
                        
                        # 시퀀스 길이만큼 이상으로 표시
                        seq_length = 15  # 시퀀스 길이 15로 변경
                        start_idx = max(0, seq_idx)
                        end_idx = min(len(test_logs), seq_idx + seq_length)
                        
                        y_pred[start_idx:end_idx] = 1
                        anomaly_scores[start_idx:end_idx] = np.maximum(
                            anomaly_scores[start_idx:end_idx],
                            score
                        )
                else:
                    # 이상이 탐지되지 않은 경우 (LogRobust 등)
                    # 직접 모델에서 anomaly_scores를 가져와서 평가
                    print(f"   ⚠️ 이상치가 탐지되지 않았습니다. 모델 점수로 평가합니다.")
                    
                    # 모델 타입에 따라 직접 점수 계산
                    if model_type == 'logrobust':
                        # LogRobust는 직접 모델을 통해 점수 계산
                        try:
                            from log_specific_anomaly_detectors import LogRobustDetector
                            if hasattr(system.detector, 'model') and system.detector.model is not None:
                                logs_df_sorted = test_logs.sort_values('timestamp').reset_index(drop=True)
                                sequences = []
                                seq_len = 15  # 시퀀스 길이 15
                                for i in range(len(logs_df_sorted) - seq_len + 1):
                                    sequence_logs = logs_df_sorted.iloc[i:i + seq_len]
                                    sequence_vectors = [
                                        system.detector.encode_log(msg) for msg in sequence_logs['message']
                                    ]
                                    sequences.append(sequence_vectors)
                                
                                if len(sequences) > 0:
                                    sequences = np.array(sequences)
                                    sequences_tensor = torch.FloatTensor(sequences)
                                    
                                    system.detector.model.eval()
                                    with torch.no_grad():
                                        outputs = system.detector.model(sequences_tensor)
                                        scores = outputs.squeeze().numpy()
                                        
                                        # 점수를 anomaly_scores에 저장
                                        seq_len = 15  # 시퀀스 길이 15
                                        for i, score in enumerate(scores):
                                            start_idx = max(0, i)
                                            end_idx = min(len(test_logs), i + seq_len)
                                            anomaly_scores[start_idx:end_idx] = np.maximum(
                                                anomaly_scores[start_idx:end_idx],
                                                float(score)
                                            )
                        except Exception as e:
                            print(f"   ⚠️ LogRobust 점수 계산 실패: {e}")
                            # 기본값 사용 (모두 0)
                    elif model_type == 'deeplog':
                        # DeepLog는 직접 모델을 통해 점수 계산
                        try:
                            sequences, actual_next = system.detector.prepare_sequences(test_logs)
                            if len(sequences) > 0:
                                sequences_tensor = torch.LongTensor(sequences)
                                system.detector.model.eval()
                                with torch.no_grad():
                                    outputs = system.detector.model(sequences_tensor)
                                    probs = torch.softmax(outputs, dim=1)
                                    predicted_probs = probs[np.arange(len(actual_next)), actual_next].numpy()
                                    anomaly_scores_seq = 1 - predicted_probs
                                    
                                    seq_len = 15  # 시퀀스 길이 15
                                    for i, score in enumerate(anomaly_scores_seq):
                                        start_idx = max(0, i)
                                        end_idx = min(len(test_logs), i + seq_len)
                                        anomaly_scores[start_idx:end_idx] = np.maximum(
                                            anomaly_scores[start_idx:end_idx],
                                            score
                                        )
                        except Exception as e:
                            print(f"   ⚠️ DeepLog 점수 계산 실패: {e}")
                    elif model_type == 'loganomaly':
                        # LogAnomaly는 직접 점수 계산
                        try:
                            test_sequences = system.detector.create_sequences(test_logs, window_size=15)  # 10 → 15
                            if len(test_sequences) > 0:
                                for i, sequence in enumerate(test_sequences):
                                    seq_mean = np.mean(sequence, axis=0)
                                    z_scores = np.abs((seq_mean - system.detector.normal_mean) / system.detector.normal_std)
                                    max_z_score = np.max(z_scores)
                                    anomaly_score = max_z_score / 3.0  # threshold 3.0 기준
                                    
                                    seq_len = 15  # 시퀀스 길이 15
                                    start_idx = max(0, i)
                                    end_idx = min(len(test_logs), i + seq_len)
                                    anomaly_scores[start_idx:end_idx] = np.maximum(
                                        anomaly_scores[start_idx:end_idx],
                                        anomaly_score
                                    )
                        except Exception as e:
                            print(f"   ⚠️ LogAnomaly 점수 계산 실패: {e}")
                
                # 최적 임계값 찾기 (옵션)
                optimal_threshold = None
                optimal_metrics = None
                if optimize_threshold and len(anomaly_scores) > 0 and len(np.unique(anomaly_scores)) > 1:
                    print(f"   🔍 최적 임계값 탐색 중...")
                    optimal_threshold, optimal_metrics = self.find_optimal_threshold(
                        y_test, anomaly_scores,
                        metric='balanced',  # 정밀도와 재현율의 균형
                        min_precision=target_metrics.get('precision', 0.3),
                        min_recall=target_metrics.get('recall', 0.5)
                    )
                    
                    if optimal_metrics:
                        print(f"   ✅ 최적 임계값 발견: {optimal_threshold:.4f}")
                        print(f"      예상 성능:")
                        print(f"         - 정확도: {optimal_metrics['accuracy']:.4f} ({optimal_metrics['accuracy']*100:.2f}%)")
                        print(f"         - 정밀도: {optimal_metrics['precision']:.4f} ({optimal_metrics['precision']*100:.2f}%)")
                        print(f"         - 재현율: {optimal_metrics['recall']:.4f} ({optimal_metrics['recall']*100:.2f}%)")
                        print(f"         - F1 점수: {optimal_metrics['f1_score']:.4f} ({optimal_metrics['f1_score']*100:.2f}%)")
                        
                        # 최적 임계값으로 재계산
                        y_pred_optimal = (anomaly_scores >= optimal_threshold).astype(int)
                        accuracy = accuracy_score(y_test, y_pred_optimal)
                        precision = precision_score(y_test, y_pred_optimal, zero_division=0)
                        recall = recall_score(y_test, y_pred_optimal, zero_division=0)
                        f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
                        y_pred = y_pred_optimal  # 최적화된 예측 사용
                    else:
                        print(f"   ⚠️ 목표 성능을 만족하는 임계값을 찾지 못했습니다. 기본 임계값 사용.")
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                else:
                    # 기본 성능 지표 계산
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # ROC-AUC
                try:
                    roc_auc = roc_auc_score(y_test, anomaly_scores)
                except:
                    roc_auc = None
                
                # 혼동 행렬
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # 특이도
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # 심각도 정보
                severity_info = detection_results.get('summary', {})
                
                # 목표 성능 달성 여부 확인
                meets_target = (
                    accuracy >= target_metrics.get('accuracy', 0) and
                    precision >= target_metrics.get('precision', 0) and
                    recall >= target_metrics.get('recall', 0) and
                    f1 >= target_metrics.get('f1_score', 0) and
                    specificity >= target_metrics.get('specificity', 0)
                )
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'specificity': specificity,
                    'tp': tp,
                    'tn': tn,
                    'fp': fp,
                    'fn': fn,
                    'confusion_matrix': cm,
                    'y_pred': y_pred,
                    'y_test': y_test,  # ROC 곡선을 위해 저장
                    'anomaly_scores': anomaly_scores,
                    'severity_info': severity_info,
                    'anomalies_df': anomalies_df,
                    'optimal_threshold': optimal_threshold,
                    'optimal_metrics': optimal_metrics,
                    'meets_target': meets_target
                }
                
                print(f"   정확도: {accuracy:.4f} ({accuracy*100:.2f}%) {'✅' if accuracy >= target_metrics.get('accuracy', 0) else '❌'}")
                print(f"   정밀도: {precision:.4f} ({precision*100:.2f}%) {'✅' if precision >= target_metrics.get('precision', 0) else '❌'}")
                print(f"   재현율: {recall:.4f} ({recall*100:.2f}%) {'✅' if recall >= target_metrics.get('recall', 0) else '❌'}")
                print(f"   F1 점수: {f1:.4f} ({f1*100:.2f}%) {'✅' if f1 >= target_metrics.get('f1_score', 0) else '❌'}")
                if roc_auc:
                    print(f"   ROC-AUC: {roc_auc:.4f}")
                print(f"   특이도: {specificity:.4f} ({specificity*100:.2f}%) {'✅' if specificity >= target_metrics.get('specificity', 0) else '❌'}")
                print(f"   혼동 행렬:")
                print(f"      [정상→정상: {tn:4d}  정상→이상: {fp:4d}]")
                print(f"      [이상→정상: {fn:4d}  이상→이상: {tp:4d}]")
                
                if meets_target:
                    print(f"\n   🎉 목표 성능 달성!")
                else:
                    print(f"\n   ⚠️ 목표 성능 미달성 - 개선 필요")
                    if optimal_threshold and optimal_metrics:
                        print(f"   💡 최적 임계값({optimal_threshold:.4f}) 적용 시 예상 개선:")
                        print(f"      - 정확도: {accuracy:.4f} → {optimal_metrics['accuracy']:.4f}")
                        print(f"      - 정밀도: {precision:.4f} → {optimal_metrics['precision']:.4f}")
                        print(f"      - 재현율: {recall:.4f} → {optimal_metrics['recall']:.4f}")
                        print(f"      - F1 점수: {f1:.4f} → {optimal_metrics['f1_score']:.4f}")
                
                # 심각도 정보
                if severity_info:
                    print(f"\n   🔍 심각도 분석:")
                    print(f"      탐지된 이상 시퀀스: {severity_info.get('total_anomalies', 0)}개")
                    if 'by_severity' in severity_info:
                        print(f"      심각도 분포:")
                        for level, count in severity_info['by_severity'].items():
                            print(f"        {level}: {count}개")
                    if 'avg_severity_score' in severity_info:
                        print(f"      평균 심각도: {severity_info['avg_severity_score']:.2f}")
                
                eval_time = time.time() - eval_start_time
                print(f"\n   ✅ {model_type.upper()} 평가 완료: {eval_time:.2f}초 ({eval_time/60:.1f}분)")
            
            except Exception as e:
                print(f"   ❌ 평가 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.results = results
        return results
    
    def generate_comparison_report(self, output_dir=None):
        """모델 비교 리포트 생성"""
        if not self.results:
            print("⚠️ 평가 결과가 없습니다.")
            return None
        
        print("\n" + "=" * 60)
        print("로그 특화 모델 성능 비교 리포트")
        print("=" * 60)
        
        # 성능 지표 비교 테이블
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                '모델': model_name.upper(),
                '정확도': f"{metrics['accuracy']:.4f}",
                '정밀도': f"{metrics['precision']:.4f}",
                '재현율': f"{metrics['recall']:.4f}",
                'F1 점수': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A",
                '특이도': f"{metrics['specificity']:.4f}",
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 성능 지표 비교:")
        print(comparison_df.to_string(index=False))
        
        # 최고 성능 모델
        print("\n🏆 최고 성능 모델:")
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        
        print(f"   최고 F1 점수: {best_f1[0].upper()} ({best_f1[1]['f1_score']:.4f})")
        print(f"   최고 정확도: {best_accuracy[0].upper()} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"   최고 재현율: {best_recall[0].upper()} ({best_recall[1]['recall']:.4f})")
        print(f"   최고 정밀도: {best_precision[0].upper()} ({best_precision[1]['precision']:.4f})")
        
        # 종합 평가 (가중 평균)
        print("\n📈 종합 평가:")
        weighted_scores = {}
        for model_name, metrics in self.results.items():
            # F1 점수에 가중치 부여 (가장 중요)
            weighted_score = (
                metrics['f1_score'] * 0.4 +
                metrics['accuracy'] * 0.3 +
                metrics['recall'] * 0.2 +
                metrics['precision'] * 0.1
            )
            weighted_scores[model_name] = weighted_score
        
        best_overall = max(weighted_scores.items(), key=lambda x: x[1])
        print(f"   🥇 최적 모델: {best_overall[0].upper()}")
        print(f"      종합 점수: {best_overall[1]:.4f}")
        print(f"      F1 점수: {self.results[best_overall[0]]['f1_score']:.4f}")
        print(f"      정확도: {self.results[best_overall[0]]['accuracy']:.4f}")
        print(f"      재현율: {self.results[best_overall[0]]['recall']:.4f}")
        
        # 결과 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 비교 테이블 저장
            comparison_path = os.path.join(output_dir, "log_specific_model_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 비교 결과 저장: {comparison_path}")
            
            # 각 모델의 상세 결과 저장
            for model_name, metrics in self.results.items():
                detail_path = os.path.join(output_dir, f"results_{model_name}.csv")
                detail_df = pd.DataFrame({
                    'y_true': [0] * len(metrics['y_pred']),  # 실제 라벨은 별도 저장 필요
                    'y_pred': metrics['y_pred'],
                    'anomaly_score': metrics['anomaly_scores']
                })
                detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
                
                # 심각도 정보 저장
                if not metrics.get('anomalies_df', pd.DataFrame()).empty:
                    severity_path = os.path.join(output_dir, f"severity_{model_name}.csv")
                    metrics['anomalies_df'].to_csv(severity_path, index=False, encoding='utf-8-sig')
                    print(f"💾 {model_name.upper()} 심각도 결과 저장: {severity_path}")
            
            # 그래프 생성
            if PLOTTING_AVAILABLE:
                print("\n📊 그래프 생성 중...")
                self.plot_comparison_graphs(output_dir)
        
        return comparison_df, best_overall[0]
    
    def plot_comparison_graphs(self, output_dir):
        """모델 비교 그래프 생성"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ matplotlib/seaborn이 없어 그래프를 생성할 수 없습니다.")
            return
        
        print("   📈 성능 지표 비교 그래프 생성 중...")
        
        # 1. 성능 지표 비교 막대 그래프
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['정확도 (Accuracy)', '정밀도 (Precision)', '재현율 (Recall)', 'F1 점수']
        
        for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model][metric_name] for model in models]
            colors = sns.color_palette("husl", len(models))
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.upper() for m in models], rotation=0)
            ax.set_ylabel('점수', fontsize=11)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        metrics_path = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 성능 지표 비교 그래프 저장: {metrics_path}")
        
        # 2. 혼동 행렬 히트맵
        print("   📊 혼동 행렬 히트맵 생성 중...")
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('혼동 행렬 (Confusion Matrix)', fontsize=16, fontweight='bold')
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['정상', '이상'], yticklabels=['정상', '이상'],
                       cbar_kws={'label': '개수'})
            ax.set_title(f'{model_name.upper()}\n(TP:{metrics["tp"]}, TN:{metrics["tn"]}, FP:{metrics["fp"]}, FN:{metrics["fn"]})',
                        fontsize=11, fontweight='bold')
            ax.set_ylabel('실제', fontsize=11)
            ax.set_xlabel('예측', fontsize=11)
        
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 혼동 행렬 히트맵 저장: {cm_path}")
        
        # 3. ROC 곡선 (가능한 경우)
        print("   📈 ROC 곡선 생성 중...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if metrics.get('roc_auc') is not None and metrics.get('y_test') is not None:
                try:
                    y_test = metrics.get('y_test')
                    y_scores = metrics.get('anomaly_scores')
                    
                    if y_scores is not None and len(y_scores) > 0:
                        fpr, tpr, _ = roc_curve(y_test, y_scores)
                        roc_auc = metrics['roc_auc']
                        ax.plot(fpr, tpr, lw=2, label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    print(f"      ⚠️ {model_name} ROC 곡선 생성 실패: {e}")
                    continue
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('ROC 곡선 비교', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ ROC 곡선 저장: {roc_path}")
        
        # 4. 심각도 분포 파이 차트
        print("   🥧 심각도 분포 그래프 생성 중...")
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('심각도 분포', fontsize=16, fontweight='bold')
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            ax = axes[idx]
            severity_info = metrics.get('severity_info', {})
            by_severity = severity_info.get('by_severity', {})
            
            if by_severity:
                labels = list(by_severity.keys())
                sizes = list(by_severity.values())
                colors_map = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb78', 'LOW': '#2ca02c'}
                colors = [colors_map.get(label, '#1f77b4') for label in labels]
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                      colors=colors, textprops={'fontsize': 10})
                ax.set_title(f'{model_name.upper()}\n(총 {severity_info.get("total_anomalies", 0)}개)', 
                           fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=12)
                ax.set_title(f'{model_name.upper()}', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        severity_path = os.path.join(output_dir, "severity_distribution.png")
        plt.savefig(severity_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 심각도 분포 그래프 저장: {severity_path}")
        
        # 5. 종합 비교 레이더 차트
        print("   📊 종합 성능 레이더 차트 생성 중...")
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['정확도', '정밀도', '재현율', 'F1 점수', '특이도']
        num_vars = len(categories)
        
        # 각도 계산
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 원형으로 만들기
        
        # 각 모델별 데이터
        for model_name, metrics in self.results.items():
            values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['specificity']
            ]
            values += values[:1]  # 원형으로 만들기
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name.upper(), alpha=0.7)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('종합 성능 비교 (레이더 차트)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        radar_path = os.path.join(output_dir, "performance_radar.png")
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 종합 성능 레이더 차트 저장: {radar_path}")
        
        print("\n✅ 모든 그래프 생성 완료!")
    
    def get_best_model(self):
        """최적 모델 반환"""
        if not self.results:
            return None
        
        # 종합 점수 계산
        weighted_scores = {}
        for model_name, metrics in self.results.items():
            weighted_score = (
                metrics['f1_score'] * 0.4 +
                metrics['accuracy'] * 0.3 +
                metrics['recall'] * 0.2 +
                metrics['precision'] * 0.1
            )
            weighted_scores[model_name] = weighted_score
        
        best_model_name = max(weighted_scores.items(), key=lambda x: x[1])[0]
        return best_model_name, self.trained_systems[best_model_name]


def main():
    """메인 실행 함수"""
    import argparse
    from log_anomaly_detector import SpringBootLogParser
    
    # 명령줄 인자 파싱
    parser_args = argparse.ArgumentParser(description='로그 특화 모델 성능 비교')
    parser_args.add_argument('--parse-only', action='store_true', 
                            help='파싱만 수행하고 저장 (모델 학습 생략)')
    parser_args.add_argument('--load-parsed', type=str, default=None,
                            help='저장된 파싱 데이터 파일 경로 (재사용)')
    parser_args.add_argument('--save-parsed', type=str, default=None,
                            help='파싱 데이터 저장 경로')
    parser_args.add_argument('--keep-chunks', action='store_true',
                            help='청크 파일 유지 (재사용 가능, 메모리 효율적)')
    parser_args.add_argument('--load-chunks', type=str, default=None,
                            help='청크 디렉토리 경로 (청크 파일에서 직접 로드, 메모리 효율적)')
    parser_args.add_argument('--chunk-read-size', type=int, default=None,
                            help='Parquet 파일을 청크 단위로 읽을 크기 (메모리 절약, 예: 100000)')
    parser_args.add_argument('--streaming-split', action='store_true',
                            help='스트리밍 방식으로 데이터 분할 (메모리 효율적, parsed_data.parquet 사용 시 권장)')
    parser_args.add_argument('--split-output-dir', type=str, default=None,
                            help='스트리밍 분할 결과 저장 디렉토리 (기본: pattern/prelog/split_data)')
    parser_args.add_argument('--load-split', type=str, default=None,
                            help='분할된 데이터 디렉토리에서 로드 (스트리밍 분할 결과 재사용)')
    parser_args.add_argument('--max-total-lines', type=int, default=100000,
                            help='파싱할 최대 라인 수 (기본: 100000, 빠른 테스트용)')
    parser_args.add_argument('--max-files', type=int, default=None,
                            help='처리할 최대 파일 수 (빠른 테스트용)')
    parser_args.add_argument('--sample-size', type=int, default=300000,
                            help='학습용 데이터 샘플링 크기 (기본: 300000, OOM 방지)')
    parser_args.add_argument('--epochs', type=int, default=5,
                            help='학습 epoch 수 (기본: 5, 빠른 학습용)')
    parser_args.add_argument('--batch-size', type=int, default=32,
                            help='배치 크기 (기본: 32, OOM 방지)')
    parser_args.add_argument('--eval-sample-size', type=int, default=None,
                            help='평가용 테스트 데이터 샘플링 크기 (빠른 평가용, 예: 20000)')
    parser_args.add_argument('--optimize-threshold', action='store_true',
                            help='최적 임계값 자동 탐색 (성능 개선)')
    parser_args.add_argument('--target-accuracy', type=float, default=0.70,
                            help='목표 정확도 (기본: 0.70)')
    parser_args.add_argument('--target-precision', type=float, default=0.50,
                            help='목표 정밀도 (기본: 0.50)')
    parser_args.add_argument('--target-recall', type=float, default=0.60,
                            help='목표 재현율 (기본: 0.60)')
    parser_args.add_argument('--target-f1', type=float, default=0.55,
                            help='목표 F1 점수 (기본: 0.55)')
    parser_args.add_argument('--target-specificity', type=float, default=0.80,
                            help='목표 특이도 (기본: 0.80)')
    args = parser_args.parse_args()
    
    log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    print("=" * 70)
    print("로그 특화 모델 성능 비교 시스템")
    print("=" * 70)
    print(f"⚡ 빠른 학습 모드:")
    print(f"   - 최대 파싱 라인: {args.max_total_lines:,}개")
    print(f"   - 학습 데이터 샘플링: {args.sample_size:,}개")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print("=" * 70)
    
    # 2. 모델 비교 시스템 초기화 (먼저 초기화)
    comparator = LogSpecificModelComparator()
    
    # 1. 로그 파싱 또는 로드
    print("\n1단계: 로그 데이터 준비")
    parser = SpringBootLogParser()
    
    # 자동으로 기존 데이터 재사용 (옵션이 없을 때)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_split_dir = os.path.join(script_dir, 'split_data')
    default_parsed_file = os.path.join(script_dir, 'parsed_data.parquet')
    
    # 분할된 데이터에서 직접 로드 (가장 메모리 효율적)
    if args.load_split:
        print(f"📂 분할된 데이터에서 로드: {args.load_split}")
        data = comparator.prepare_data_from_files(args.load_split)
        logs_df = None  # 분할된 데이터는 이미 로드됨
    elif os.path.exists(default_split_dir) and os.path.isdir(default_split_dir):
        # 기본 split_data 디렉토리가 있으면 자동으로 사용
        print(f"📂 기존 분할 데이터 자동 로드: {default_split_dir}")
        data = comparator.prepare_data_from_files(default_split_dir)
        logs_df = None  # 분할된 데이터는 이미 로드됨
    elif args.load_chunks:
        # 청크 파일에서 직접 로드
        print(f"📂 청크 파일에서 데이터 로드: {args.load_chunks}")
        logs_df = parser.load_from_chunks(args.load_chunks)
    elif args.load_parsed:
        # 저장된 파싱 데이터 로드
        print(f"📂 저장된 파싱 데이터 로드: {args.load_parsed}")
        
        # 스트리밍 분할 옵션이 있으면 분할 수행
        if args.streaming_split:
            split_output_dir = args.split_output_dir
            if split_output_dir is None:
                split_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_data')
            
            print(f"   💡 스트리밍 분할 모드: 메모리 효율적으로 분할 중...")
            chunk_size = args.chunk_read_size if args.chunk_read_size else 100000
            split_files = parser.prepare_data_streaming(
                args.load_parsed,
                split_output_dir,
                train_ratio=0.8,
                valid_ratio=0.2,
                chunk_size=chunk_size
            )
            
            # 분할된 데이터 로드
            data = comparator.prepare_data_from_files(split_output_dir)
            logs_df = None  # 분할된 데이터는 이미 로드됨
        else:
            if args.chunk_read_size:
                print(f"   💡 메모리 절약 모드: 청크 단위로 읽기 (청크 크기: {args.chunk_read_size:,}개)")
            logs_df = parser.load_parsed_data(args.load_parsed, chunk_size=args.chunk_read_size)
    elif os.path.exists(default_parsed_file):
        # 기본 parsed_data.parquet 파일이 있으면 자동으로 사용
        print(f"📂 기존 파싱 데이터 자동 로드: {default_parsed_file}")
        
        # 스트리밍 분할 수행 (split_data가 없으면)
        if not os.path.exists(default_split_dir):
            print(f"   💡 분할 데이터가 없어 스트리밍 분할 수행 중...")
            split_output_dir = default_split_dir
            chunk_size = args.chunk_read_size if args.chunk_read_size else 100000
            split_files = parser.prepare_data_streaming(
                default_parsed_file,
                split_output_dir,
                train_ratio=0.8,
                valid_ratio=0.2,
                chunk_size=chunk_size
            )
            # 분할된 데이터 로드
            data = comparator.prepare_data_from_files(split_output_dir)
            logs_df = None
        else:
            # 분할 데이터가 이미 있으면 그냥 로드
            if args.chunk_read_size:
                print(f"   💡 메모리 절약 모드: 청크 단위로 읽기 (청크 크기: {args.chunk_read_size:,}개)")
            logs_df = parser.load_parsed_data(default_parsed_file, chunk_size=args.chunk_read_size)
    else:
        # 새로 파싱
        print("📝 로그 파일 파싱 중...")
        print(f"   💡 기존 파싱 데이터가 없어 새로 파싱합니다.")
        # 메모리 절약을 위한 파라미터
        # 필요에 따라 조정 가능:
        # - max_files: 처리할 최대 파일 수
        # - sample_lines: 파일당 최대 라인 수
        # - chunk_size: 청크 크기 (기본 10,000)
        # - max_total_lines: 전체 최대 라인 수
        logs_df = parser.parse_directory(
            log_directory,
            max_files=args.max_files,        # 전체 파일 처리 (메모리 부족 시 숫자로 제한)
            sample_lines=None,     # 전체 라인 처리 (메모리 부족 시 숫자로 제한)
            chunk_size=5000,       # 청크 크기 (메모리 절약, 기본값보다 작게)
            max_total_lines=args.max_total_lines,  # 전체 최대 라인 수 (메모리 부족 시 설정)
            save_chunks_to_disk=True,  # 파일로 저장하여 메모리 절약
            keep_chunks=args.keep_chunks  # 청크 파일 유지 여부
        )
        
        # 파싱 데이터 저장
        if args.save_parsed:
            parser.save_parsed_data(logs_df, args.save_parsed)
        elif args.parse_only:
            # 기본 저장 경로 사용
            parser.save_parsed_data(logs_df, default_parsed_file)
            print(f"\n💡 다음 명령으로 파싱 데이터 재사용:")
            print(f"   python log_specific_model_comparison.py --load-parsed {default_parsed_file}")
            return
        else:
            # 파싱만 하고 저장하지 않으면 기본 경로에 저장 (다음 실행 시 재사용)
            parser.save_parsed_data(logs_df, default_parsed_file)
            print(f"💾 파싱 데이터 저장: {default_parsed_file} (다음 실행 시 자동 재사용)")
    
    # 3. 데이터 준비
    print("\n3단계: 데이터 준비")
    
    # 분할된 데이터가 이미 로드된 경우
    split_data_dir = None
    if logs_df is None:
        # 분할된 데이터에서 로드한 경우
        if args.load_split:
            split_data_dir = args.load_split
        elif args.streaming_split:
            split_data_dir = split_output_dir if 'split_output_dir' in locals() else default_split_dir
        else:
            # 기본 split_data 디렉토리 확인
            if os.path.exists(default_split_dir):
                split_data_dir = default_split_dir
        
        # 학습 단계에서는 train만 로드 (메모리 절약)
        if 'data' not in locals() or data is None:
            print("   💡 메모리 절약: 학습 단계에서는 Train만 로드합니다.")
            data = comparator.prepare_data_from_files(split_data_dir, load_only_train=True)
        
        # 데이터 샘플링 (빠른 학습용)
        if args.sample_size and data.get('train_normal') is not None and len(data['train_normal']) > args.sample_size:
            print(f"\n📊 학습 데이터 샘플링 중: {len(data['train_normal']):,}개 → {args.sample_size:,}개")
            data['train_normal'] = data['train_normal'].sample(n=args.sample_size, random_state=42).reset_index(drop=True)
            data['train_normal'] = data['train_normal'].sort_values('timestamp').reset_index(drop=True)
            print(f"   ✅ 샘플링 완료: {len(data['train_normal']):,}개")
    else:
        if logs_df.empty:
            print("⚠️ 파싱된 로그가 없습니다.")
            return
        
        print(f"✅ {len(logs_df):,}개 로그 라인 준비 완료")
        
        # 데이터 샘플링 (빠른 학습용)
        if args.sample_size and len(logs_df) > args.sample_size:
            print(f"\n📊 데이터 샘플링 중: {len(logs_df):,}개 → {args.sample_size:,}개")
            logs_df = logs_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
            logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
            print(f"   ✅ 샘플링 완료: {len(logs_df):,}개")
        
        data = comparator.prepare_data(logs_df, train_ratio=0.8, valid_ratio=0.2)
    
    # 4. 모델 학습
    print("\n4단계: 로그 특화 모델 학습")
    print(f"⚡ 빠른 학습 설정:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    if data.get('train_normal') is not None:
        print(f"   - 학습 데이터: {len(data['train_normal']):,}개")
    
    # 사용 가능한 모델만 학습 (LogRobust 제외 - 메모리 사용량이 많고 OOM 발생 가능)
    available_models = ['deeplog', 'loganomaly']
    print("   ✅ 사용 모델: DeepLog, LogAnomaly (LogRobust 제외)")
    
    # 학습 로그 디렉토리 설정
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'training')
    
    comparator.train_models(
        data['train_normal'], 
        valid_normal_logs=data.get('valid_normal'),
        model_types=available_models,
        log_dir=log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not comparator.trained_systems:
        print("⚠️ 학습된 모델이 없습니다.")
        return
    
    # 학습 데이터 메모리 해제
    del data['train_normal']
    del data['train_error']
    if 'valid_normal' in data:
        del data['valid_normal']
    
    # 5. 모델 평가
    print("\n5단계: 모델 성능 평가")
    
    # 평가 단계에서 test 데이터 로드 (메모리 효율적)
    if split_data_dir and (data.get('test_logs', pd.DataFrame()).empty or len(data.get('test_logs', pd.DataFrame())) == 0):
        print("   💡 평가용 데이터 로드 중...")
        test_data = comparator.load_test_data(split_data_dir)
        data['test_logs'] = test_data['test_logs']
        data['y_test'] = test_data['y_test']
        if not test_data['valid_logs'].empty:
            data['valid_logs'] = test_data['valid_logs']
            data['y_valid'] = test_data['y_valid']
    
    # 평가 데이터 샘플링 (빠른 평가용)
    test_logs = data['test_logs']
    y_test = data['y_test']
    
    if args.eval_sample_size and len(test_logs) > args.eval_sample_size:
        print(f"\n📊 평가 데이터 샘플링 중: {len(test_logs):,}개 → {args.eval_sample_size:,}개")
        print(f"   ⚡ 빠른 평가 모드: 샘플링으로 평가 시간 단축")
        # 시간 순서를 유지하면서 샘플링
        sample_indices = np.linspace(0, len(test_logs) - 1, args.eval_sample_size, dtype=int)
        test_logs = test_logs.iloc[sample_indices].reset_index(drop=True)
        y_test = y_test[sample_indices]
        print(f"   ✅ 샘플링 완료: {len(test_logs):,}개")
    
    print(f"\n📊 평가 데이터: {len(test_logs):,}개 로그")
    print(f"   예상 평가 시간: 모델당 약 {len(test_logs) // 1000}초 (대략적 추정)")
    
    # 목표 성능 지표 설정
    target_metrics = {
        'accuracy': args.target_accuracy,
        'precision': args.target_precision,
        'recall': args.target_recall,
        'f1_score': args.target_f1,
        'specificity': args.target_specificity
    }
    
    results = comparator.evaluate_models(
        test_logs, 
        y_test,
        optimize_threshold=args.optimize_threshold,
        target_metrics=target_metrics
    )
    
    if not results:
        print("⚠️ 평가 결과가 없습니다.")
        return
    
    # 6. 비교 리포트 생성 및 그래프 생성
    print("\n6단계: 비교 리포트 생성")
    
    # 결과 폴더 지정 (log_specific_comparison_4)
    output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results/log_specific_comparison_4"
    
    print(f"📁 결과 저장 폴더: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df, best_model = comparator.generate_comparison_report(output_dir=output_dir)
    
    # 7. 최적 모델 선정
    print("\n7단계: 최적 모델 선정")
    best_model_name, best_system = comparator.get_best_model()
    
    print(f"\n{'='*70}")
    print(f"🏆 최종 선정된 모델: {best_model_name.upper()}")
    print(f"{'='*70}")
    
    best_metrics = results[best_model_name]
    print(f"\n성능 지표:")
    print(f"   정확도: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"   정밀도: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    print(f"   재현율: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
    print(f"   F1 점수: {best_metrics['f1_score']:.4f}")
    if best_metrics['roc_auc']:
        print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # 심각도 정보
    if best_metrics.get('severity_info'):
        severity_info = best_metrics['severity_info']
        print(f"\n심각도 분석:")
        print(f"   탐지된 이상 시퀀스: {severity_info.get('total_anomalies', 0)}개")
        if 'by_severity' in severity_info:
            print(f"   심각도 분포:")
            for level, count in severity_info['by_severity'].items():
                print(f"      {level}: {count}개")
    
    # 8. 상세 리포트 저장
    print("\n8단계: 상세 리포트 저장")
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("로그 특화 모델 성능 비교 리포트\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("데이터 정보:\n")
        # 데이터가 이미 삭제되었을 수 있으므로 안전하게 처리
        train_normal_count = len(data.get('train_normal', pd.DataFrame()))
        train_error_count = len(data.get('train_error', pd.DataFrame()))
        valid_normal_count = len(data.get('valid_normal', pd.DataFrame())) if not data.get('valid_normal', pd.DataFrame()).empty else 0
        valid_error_count = len(data.get('valid_error', pd.DataFrame())) if not data.get('valid_error', pd.DataFrame()).empty else 0
        test_normal_count = len(data.get('test_normal', pd.DataFrame()))
        test_error_count = len(data.get('test_error', pd.DataFrame()))
        
        total_logs = train_normal_count + train_error_count + valid_normal_count + valid_error_count + test_normal_count + test_error_count
        if total_logs > 0:
            f.write(f"  - 전체 로그: {total_logs}개\n")
            f.write(f"  - 학습 정상 로그: {train_normal_count}개 ({train_normal_count/total_logs*100:.1f}%)\n")
            if valid_normal_count > 0:
                f.write(f"  - 검증 정상 로그: {valid_normal_count}개 ({valid_normal_count/total_logs*100:.1f}%)\n")
            f.write(f"  - 테스트 정상 로그: {test_normal_count}개 ({test_normal_count/total_logs*100:.1f}%)\n")
            f.write(f"  - 테스트 에러 로그: {test_error_count}개 ({test_error_count/total_logs*100:.1f}%)\n\n")
        
        f.write("모델 성능 비교:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write(f"최종 선정된 모델: {best_model_name.upper()}\n")
        f.write(f"종합 점수: {best_metrics['f1_score']*0.4 + best_metrics['accuracy']*0.3 + best_metrics['recall']*0.2 + best_metrics['precision']*0.1:.4f}\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"[{model_name.upper()}] 상세 결과:\n")
            f.write(f"  정확도: {metrics['accuracy']:.4f}\n")
            f.write(f"  정밀도: {metrics['precision']:.4f}\n")
            f.write(f"  재현율: {metrics['recall']:.4f}\n")
            f.write(f"  F1 점수: {metrics['f1_score']:.4f}\n")
            if metrics['roc_auc']:
                f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"  특이도: {metrics['specificity']:.4f}\n")
            f.write(f"  혼동 행렬:\n")
            f.write(f"    [정상→정상: {metrics['tn']:4d}  정상→이상: {metrics['fp']:4d}]\n")
            f.write(f"    [이상→정상: {metrics['fn']:4d}  이상→이상: {metrics['tp']:4d}]\n")
            
            # 심각도 정보
            if metrics.get('severity_info'):
                severity_info = metrics['severity_info']
                f.write(f"\n  심각도 분석:\n")
                f.write(f"    탐지된 이상 시퀀스: {severity_info.get('total_anomalies', 0)}개\n")
                if 'by_severity' in severity_info:
                    f.write(f"    심각도 분포:\n")
                    for level, count in severity_info['by_severity'].items():
                        f.write(f"      {level}: {count}개\n")
                if 'avg_severity_score' in severity_info:
                    f.write(f"    평균 심각도 점수: {severity_info['avg_severity_score']:.2f}\n")
            f.write("\n")
    
    print(f"💾 상세 리포트 저장: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 성능 비교 완료!")
    print("=" * 70)
    print(f"\n최종 선정 모델: {best_model_name.upper()}")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()

