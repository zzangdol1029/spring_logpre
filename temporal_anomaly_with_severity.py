"""
Temporal (Time-series) Anomaly Detection + 심각도 평가 통합 시스템
시계열 기반 이상 탐지 후 심각도 평가를 수행합니다.
"""

import re
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from severity_assessment import SeverityAssessment
import warnings
warnings.filterwarnings('ignore')


class TemporalLogAnomalyDetector:
    """시계열 기반 로그 이상치 탐지 클래스"""
    
    def __init__(self, window_size=10, step_size=1, model_type='autoencoder'):
        """
        Args:
            window_size: 시계열 윈도우 크기 (로그 라인 수 또는 시간 단위)
            step_size: 슬라이딩 윈도우 스텝 크기
            model_type: 사용할 모델 ('autoencoder', 'isolation_forest', 'lof', 'copod')
        """
        self.window_size = window_size
        self.step_size = step_size
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_sequences(self, features_df, time_col='timestamp'):
        """
        시계열 시퀀스 생성 (슬라이딩 윈도우)
        
        Args:
            features_df: 특징 DataFrame (시간 순서대로 정렬되어 있어야 함)
            time_col: 시간 컬럼명
        
        Returns:
            sequences: 시퀀스 배열
            sequence_indices: 각 시퀀스의 원본 인덱스
        """
        if features_df.empty:
            return np.array([]), []
        
        # 시간 순서 정렬
        if time_col in features_df.columns:
            features_df = features_df.sort_values(time_col).reset_index(drop=True)
        
        # 수치형 특징만 선택
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.array([]), []
        
        features = features_df[numeric_cols].values
        
        sequences = []
        sequence_indices = []
        
        for i in range(0, len(features) - self.window_size + 1, self.step_size):
            sequence = features[i:i + self.window_size]
            sequences.append(sequence)
            sequence_indices.append((i, i + self.window_size))
        
        return np.array(sequences), sequence_indices
    
    def train(self, sequences):
        """
        시계열 이상 탐지 모델 학습
        
        Args:
            sequences: 시계열 시퀀스 배열
        """
        if len(sequences) == 0:
            print("⚠️ 시퀀스가 없습니다.")
            return False
        
        print(f"📊 시계열 모델 학습 중...")
        print(f"   시퀀스 수: {len(sequences)}")
        print(f"   시퀀스 길이: {self.window_size}")
        print(f"   특징 차원: {sequences.shape[2] if len(sequences.shape) > 2 else sequences.shape[1]}")
        
        # 시퀀스를 2D로 변환 (모델 입력용)
        n_samples, seq_len, n_features = sequences.shape
        X = sequences.reshape(n_samples, seq_len * n_features)
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 선택 및 학습
        try:
            if self.model_type == 'autoencoder':
                # AutoEncoder는 시퀀스 형태로 학습 가능
                # 3D → 2D 변환 후 학습
                self.model = AutoEncoder(
                    contamination=0.1,
                    hidden_neurons=[128, 64, 32, 64, 128],
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    random_state=42
                )
            elif self.model_type == 'isolation_forest':
                self.model = IForest(contamination=0.1, random_state=42)
            elif self.model_type == 'lof':
                self.model = LOF(contamination=0.1, n_neighbors=20)
            elif self.model_type == 'copod':
                self.model = COPOD(contamination=0.1)
            else:
                raise ValueError(f"알 수 없는 모델 타입: {self.model_type}")
            
            self.model.fit(X_scaled)
            self.is_fitted = True
            print(f"   ✅ {self.model_type} 모델 학습 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ 모델 학습 실패: {e}")
            return False
    
    def predict(self, sequences):
        """
        시계열 이상 탐지
        
        Args:
            sequences: 시계열 시퀀스 배열
        
        Returns:
            predictions: 이상치 예측 (1=이상, 0=정상)
            scores: 이상 점수
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        if len(sequences) == 0:
            return np.array([]), np.array([])
        
        # 시퀀스를 2D로 변환
        n_samples, seq_len, n_features = sequences.shape
        X = sequences.reshape(n_samples, seq_len * n_features)
        
        # 정규화
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        return predictions, scores


class TemporalAnomalyWithSeverity:
    """시계열 이상 탐지 + 심각도 평가 통합 시스템"""
    
    def __init__(self, window_size=10, model_type='autoencoder'):
        """
        Args:
            window_size: 시계열 윈도우 크기
            model_type: 이상 탐지 모델 타입
        """
        self.window_size = window_size
        self.detector = TemporalLogAnomalyDetector(
            window_size=window_size,
            model_type=model_type
        )
        self.severity_assessor = SeverityAssessment()
        self.logs_df = None
        self.features_df = None
        
    def prepare_features(self, logs_df):
        """
        로그에서 특징 추출
        
        Args:
            logs_df: 파싱된 로그 DataFrame
        """
        from log_anomaly_detector import LogAnomalyDetector
        
        print("=" * 60)
        print("특징 추출")
        print("=" * 60)
        
        # 시간 윈도우별 특징 추출
        detector = LogAnomalyDetector()
        self.features_df = detector.extract_features(logs_df)
        
        if self.features_df.empty:
            print("⚠️ 특징 추출 실패")
            return False
        
        # 시간 순서 정렬
        if 'time_window' in self.features_df.columns:
            self.features_df = self.features_df.sort_values('time_window').reset_index(drop=True)
        
        print(f"✅ {len(self.features_df)}개 시간 윈도우 특징 추출 완료")
        self.logs_df = logs_df
        return True
    
    def train(self, train_ratio=0.8):
        """
        시계열 모델 학습
        
        Args:
            train_ratio: 학습 데이터 비율
        """
        if self.features_df is None or self.features_df.empty:
            print("⚠️ 특징 데이터가 없습니다.")
            return False
        
        print("\n" + "=" * 60)
        print("시계열 이상 탐지 모델 학습")
        print("=" * 60)
        
        # 학습/테스트 분할
        split_idx = int(len(self.features_df) * train_ratio)
        train_features = self.features_df.iloc[:split_idx]
        
        print(f"학습 데이터: {len(train_features)}개 윈도우")
        print(f"테스트 데이터: {len(self.features_df) - split_idx}개 윈도우")
        
        # 시계열 시퀀스 생성
        train_sequences, _ = self.detector.create_sequences(
            train_features,
            time_col='time_window' if 'time_window' in train_features.columns else None
        )
        
        if len(train_sequences) == 0:
            print("⚠️ 학습 시퀀스 생성 실패")
            return False
        
        # 모델 학습
        return self.detector.train(train_sequences)
    
    def detect_anomalies(self):
        """
        시계열 기반 이상 탐지 및 심각도 평가
        
        Returns:
            결과 딕셔너리
        """
        if not self.detector.is_fitted:
            print("⚠️ 모델이 학습되지 않았습니다.")
            return {}
        
        print("\n" + "=" * 60)
        print("시계열 기반 이상 탐지")
        print("=" * 60)
        
        # 전체 데이터에 대한 시퀀스 생성
        all_sequences, sequence_indices = self.detector.create_sequences(
            self.features_df,
            time_col='time_window' if 'time_window' in self.features_df.columns else None
        )
        
        if len(all_sequences) == 0:
            print("⚠️ 시퀀스 생성 실패")
            return {}
        
        # 이상 탐지
        predictions, scores = self.detector.predict(all_sequences)
        
        # 이상치로 탐지된 시퀀스 필터링
        anomaly_indices = np.where(predictions == 1)[0]
        
        print(f"✅ {len(anomaly_indices)}개 이상 시퀀스 탐지")
        
        # 각 이상 시퀀스에 해당하는 로그 추출
        anomaly_results = []
        
        for idx in anomaly_indices:
            start_idx, end_idx = sequence_indices[idx]
            
            # 해당 시퀀스의 시간 윈도우들
            sequence_windows = self.features_df.iloc[start_idx:end_idx]
            
            if 'time_window' in sequence_windows.columns:
                time_windows = sequence_windows['time_window'].unique()
                
                # 해당 시간 윈도우의 로그들 추출
                sequence_logs = self.logs_df[
                    self.logs_df['timestamp'].dt.floor('10T').isin(time_windows)
                ].copy()
                
                if not sequence_logs.empty:
                    # 심각도 평가
                    severity_info = self.severity_assessor.assess_time_window_severity(
                        sequence_logs
                    )
                    
                    anomaly_results.append({
                        'sequence_index': idx,
                        'time_windows': list(time_windows),
                        'start_time': time_windows[0] if len(time_windows) > 0 else None,
                        'end_time': time_windows[-1] if len(time_windows) > 0 else None,
                        'anomaly_score': scores[idx],
                        'log_count': len(sequence_logs),
                        'max_severity_score': severity_info['max_severity_score'],
                        'max_severity_level': severity_info['max_severity_level'],
                        'avg_severity_score': severity_info['avg_severity_score'],
                        'critical_count': severity_info['critical_count'],
                        'high_count': severity_info['high_count'],
                        'medium_count': severity_info['medium_count'],
                        'low_count': severity_info['low_count'],
                        'logs': sequence_logs  # 원본 로그 포함
                    })
        
        # 결과 정리
        results_df = pd.DataFrame([
            {
                k: v for k, v in result.items() 
                if k != 'logs'  # 로그는 별도 저장
            }
            for result in anomaly_results
        ])
        
        # 심각도 점수 기준 정렬
        if not results_df.empty and 'max_severity_score' in results_df.columns:
            results_df = results_df.sort_values(
                'max_severity_score',
                ascending=False
            )
            results_df['priority'] = range(1, len(results_df) + 1)
        
        return {
            'anomaly_sequences': results_df,
            'anomaly_logs': pd.concat([r['logs'] for r in anomaly_results], ignore_index=True) if anomaly_results else pd.DataFrame(),
            'total_anomalies': len(anomaly_results),
            'summary': self._generate_summary(results_df)
        }
    
    def _generate_summary(self, results_df):
        """요약 통계 생성"""
        if results_df.empty:
            return {}
        
        return {
            'total_anomaly_sequences': len(results_df),
            'by_severity': results_df['max_severity_level'].value_counts().to_dict() if 'max_severity_level' in results_df.columns else {},
            'avg_severity_score': results_df['max_severity_score'].mean() if 'max_severity_score' in results_df.columns else 0,
            'max_severity_score': results_df['max_severity_score'].max() if 'max_severity_score' in results_df.columns else 0,
            'avg_anomaly_score': results_df['anomaly_score'].mean() if 'anomaly_score' in results_df.columns else 0,
        }
    
    def generate_report(self, results):
        """결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("시계열 이상 탐지 + 심각도 평가 결과")
        print("=" * 60)
        
        if not results or results.get('total_anomalies', 0) == 0:
            print("✅ 이상치가 탐지되지 않았습니다.")
            return
        
        summary = results.get('summary', {})
        
        print(f"\n📊 탐지 결과:")
        print(f"   총 이상 시퀀스: {summary.get('total_anomaly_sequences', 0)}개")
        print(f"   평균 이상 점수: {summary.get('avg_anomaly_score', 0):.4f}")
        
        if 'by_severity' in summary:
            print(f"\n🔍 심각도 분포:")
            for level, count in summary['by_severity'].items():
                print(f"   {level}: {count}개")
        
        print(f"\n   평균 심각도 점수: {summary.get('avg_severity_score', 0):.2f}")
        print(f"   최고 심각도 점수: {summary.get('max_severity_score', 0):.2f}")
        
        # 상위 5개 이상 시퀀스
        anomaly_df = results.get('anomaly_sequences', pd.DataFrame())
        if not anomaly_df.empty:
            print(f"\n🚨 상위 5개 이상 시퀀스:")
            top_5 = anomaly_df.head(5)
            for idx, row in top_5.iterrows():
                print(f"\n   [{row.get('priority', 'N/A')}] 우선순위")
                print(f"   시간: {row.get('start_time')} ~ {row.get('end_time')}")
                print(f"   이상 점수: {row.get('anomaly_score', 0):.4f}")
                print(f"   심각도: {row.get('max_severity_level', 'N/A')} ({row.get('max_severity_score', 0):.2f})")
                print(f"   로그 수: {row.get('log_count', 0)}개")
                print(f"   심각도 분포: CRITICAL={row.get('critical_count', 0)}, HIGH={row.get('high_count', 0)}, MEDIUM={row.get('medium_count', 0)}")


def main():
    """메인 실행 함수"""
    from log_anomaly_detector import SpringBootLogParser
    
    log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    print("=" * 60)
    print("시계열 기반 이상 탐지 + 심각도 평가 시스템")
    print("=" * 60)
    
    # 1. 로그 파싱
    print("\n1단계: 로그 파일 파싱")
    parser = SpringBootLogParser()
    logs_df = parser.parse_directory(
        log_directory,
        max_files=None,
        sample_lines=None
    )
    
    if logs_df.empty:
        print("⚠️ 파싱된 로그가 없습니다.")
        return
    
    print(f"✅ {len(logs_df)}개 로그 라인 파싱 완료")
    
    # 2. 시스템 초기화
    print("\n2단계: 시스템 초기화")
    system = TemporalAnomalyWithSeverity(
        window_size=10,  # 10개 시간 윈도우 시퀀스
        model_type='autoencoder'  # 또는 'isolation_forest', 'lof', 'copod'
    )
    
    # 3. 특징 추출
    print("\n3단계: 특징 추출")
    if not system.prepare_features(logs_df):
        return
    
    # 4. 모델 학습
    print("\n4단계: 모델 학습")
    if not system.train(train_ratio=0.8):
        return
    
    # 5. 이상 탐지 및 심각도 평가
    print("\n5단계: 이상 탐지 및 심각도 평가")
    results = system.detect_anomalies()
    
    # 6. 리포트 생성
    print("\n6단계: 결과 리포트")
    system.generate_report(results)
    
    # 7. 결과 저장
    output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results/temporal"
    os.makedirs(output_dir, exist_ok=True)
    
    if results and not results.get('anomaly_sequences', pd.DataFrame()).empty:
        # 이상 시퀀스 결과 저장
        anomaly_path = os.path.join(output_dir, "temporal_anomalies.csv")
        results['anomaly_sequences'].to_csv(anomaly_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 이상 시퀀스 결과 저장: {anomaly_path}")
        
        # 이상 로그 저장
        if not results.get('anomaly_logs', pd.DataFrame()).empty:
            logs_path = os.path.join(output_dir, "temporal_anomaly_logs.csv")
            results['anomaly_logs'].to_csv(logs_path, index=False, encoding='utf-8-sig')
            print(f"💾 이상 로그 저장: {logs_path}")
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()














