"""
로그 특화 이상 탐지 모델 구현
- LogRobust: Attention + Bi-LSTM
- DeepLog: LSTM 기반 실행 경로 예측
- LogAnomaly: Template2Vec 기반 의미 벡터화
"""

import re
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
import sys
import time
warnings.filterwarnings('ignore')

# Deep Learning 라이브러리 (선택적)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    
    # 디바이스 설정 (MPS > CUDA > CPU 순으로 선택)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS) 사용 가능 - GPU 가속 활성화")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("✅ NVIDIA GPU (CUDA) 사용 가능 - GPU 가속 활성화")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ CPU 모드로 실행됩니다.")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️ PyTorch가 설치되지 않았습니다. Deep Learning 모델을 사용하려면 설치하세요: pip install torch")

# Transformers는 LogRobust에서만 필요하므로 지연 import
# 모듈 레벨에서 import하지 않음 (TensorFlow 의존성 문제 방지)
TRANSFORMERS_AVAILABLE = None  # None으로 초기화, 실제 사용 시 확인

from severity_assessment import SeverityAssessment


def setup_training_logger(model_name: str, log_dir: str = None) -> logging.Logger:
    """
    학습 로그를 위한 로거 설정
    
    Args:
        model_name: 모델 이름
        log_dir: 로그 파일 저장 디렉토리 (None이면 콘솔만)
        
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(f'training_{model_name}')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, 
            f'{model_name}_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"📝 학습 로그 파일: {log_file}")
    
    return logger


class LogTemplateExtractor:
    """로그 템플릿 추출 클래스"""
    
    def __init__(self):
        self.templates = {}
        self.template_patterns = {}
        
    def extract_template(self, log_message: str) -> str:
        """
        로그 메시지에서 템플릿 추출
        
        예시:
        "Connection to database failed: timeout after 30 seconds"
        → "Connection to database failed: timeout after * seconds"
        """
        # 숫자, IP 주소, 파일 경로 등을 *로 변환
        template = log_message
        
        # 숫자 변환
        template = re.sub(r'\d+', '*', template)
        
        # IP 주소 변환
        template = re.sub(r'\d+\.\d+\.\d+\.\d+', '*', template)
        
        # 파일 경로 변환
        template = re.sub(r'[/\\][\w/\\\.]+', '/*', template)
        
        # UUID 변환
        template = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '*', template, flags=re.IGNORECASE)
        
        # 타임스탬프 변환
        template = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', '*', template)
        
        return template.strip()
    
    def build_template_vocabulary(self, log_messages: List[str]) -> Dict[str, int]:
        """로그 템플릿 사전 구축"""
        templates = [self.extract_template(msg) for msg in log_messages]
        template_counts = Counter(templates)
        
        # 빈도순으로 정렬하여 인덱스 할당
        template_vocab = {template: idx for idx, (template, _) in enumerate(template_counts.most_common())}
        
        return template_vocab


class DeepLogDetector:
    """DeepLog: LSTM 기반 실행 경로 예측"""
    
    def __init__(self, embedding_dim=128, hidden_dim=64, num_layers=2, sequence_length=10):
        """
        Args:
            embedding_dim: 로그 임베딩 차원
            hidden_dim: LSTM hidden 차원
            num_layers: LSTM 레이어 수
            sequence_length: 시퀀스 길이
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.template_extractor = LogTemplateExtractor()
        self.template_vocab = {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다. pip install torch")
    
    def prepare_sequences(self, logs_df: pd.DataFrame, logger=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        로그 시퀀스 준비
        
        Args:
            logs_df: 로그 DataFrame (timestamp, message 컬럼 필요)
            logger: 로거 (진행 상황 출력용, 선택적)
        
        Returns:
            sequences: 시퀀스 배열
            next_logs: 다음 로그 (라벨)
        """
        # 시간 순서 정렬 (inplace=False이므로 새로운 DataFrame 생성, 원본은 유지)
        # 메모리 효율을 위해 정렬된 결과만 사용하고 원본 참조는 유지
        logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
        
        # 템플릿 추출 및 사전 구축
        if not self.template_vocab:
            if logger:
                logger.info("  템플릿 사전 구축 중...")
            log_messages = logs_df['message'].tolist()
            self.template_vocab = self.template_extractor.build_template_vocabulary(log_messages)
            if logger:
                logger.info(f"  ✅ 템플릿 사전 구축 완료: {len(self.template_vocab):,}개 템플릿")
        
        # 시퀀스 생성
        # 메모리 효율적: 미리 NumPy 배열 할당 (리스트 오버헤드 제거)
        total_sequences = len(logs_df) - self.sequence_length
        if logger:
            logger.info(f"  시퀀스 생성 중: 예상 {total_sequences:,}개 시퀀스...")
            # 예상 메모리 사용량 계산
            estimated_mb = (total_sequences * self.sequence_length * 4) / (1024 * 1024)  # int32 = 4 bytes
            logger.info(f"  예상 메모리 사용량: {estimated_mb:.1f} MB")
        
        # 미리 NumPy 배열 할당 (메모리 효율적)
        sequences = np.zeros((total_sequences, self.sequence_length), dtype=np.int32)
        next_logs = np.zeros(total_sequences, dtype=np.int32)
        
        start_time = time.time()
        for i in range(total_sequences):
            # 현재 시퀀스
            sequence_logs = logs_df.iloc[i:i + self.sequence_length]
            sequence_templates = [
                self.template_extractor.extract_template(msg)
                for msg in sequence_logs['message']
            ]
            sequence_indices = [
                self.template_vocab.get(template, 0)
                for template in sequence_templates
            ]
            
            # 다음 로그
            next_log = logs_df.iloc[i + self.sequence_length]
            next_template = self.template_extractor.extract_template(next_log['message'])
            next_index = self.template_vocab.get(next_template, 0)
            
            # NumPy 배열에 직접 할당 (리스트 append 대신)
            sequences[i] = sequence_indices
            next_logs[i] = next_index
            
            # 진행 상황 출력 (5%마다 또는 100만개마다)
            if logger and ((i + 1) % max(1, min(total_sequences // 20, 1000000)) == 0 or i == total_sequences - 1):
                progress = ((i + 1) / total_sequences) * 100
                elapsed = time.time() - start_time
                if i > 0:
                    rate = (i + 1) / elapsed  # 시퀀스/초
                    remaining = (total_sequences - (i + 1)) / rate if rate > 0 else 0
                    logger.info(f"  시퀀스 생성 진행: {i + 1:,}/{total_sequences:,} ({progress:.1f}%) - "
                              f"경과: {elapsed:.1f}초 - 예상 남은 시간: {remaining:.1f}초 ({remaining/60:.1f}분)")
                else:
                    logger.info(f"  시퀀스 생성 진행: {i + 1:,}/{total_sequences:,} ({progress:.1f}%)")
        
        return sequences, next_logs
    
    def build_model(self, vocab_size: int):
        """LSTM 모델 구축"""
        class LSTMPredictor(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                # 마지막 시퀀스 출력 사용
                last_output = lstm_out[:, -1, :]
                output = self.fc(last_output)
                return output
        
        self.model = LSTMPredictor(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # 모델을 MPS/CUDA/CPU 디바이스로 이동
        if TORCH_AVAILABLE and DEVICE is not None:
            self.model = self.model.to(DEVICE)
        
        return self.model
    
    def train(self, logs_df: pd.DataFrame, epochs=50, batch_size=32, learning_rate=0.001, log_dir=None):
        """모델 학습"""
        # 로거 설정
        log_dir_path = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'training')
        logger = setup_training_logger('deeplog', log_dir_path)
        
        logger.info("=" * 60)
        logger.info("DeepLog 모델 학습 시작")
        logger.info("=" * 60)
        logger.info(f"학습 파라미터:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - 학습 데이터: {len(logs_df):,}개 로그")
        
        start_time = time.time()
        
        # 시퀀스 준비
        logger.info("시퀀스 준비 중...")
        sequences, next_logs = self.prepare_sequences(logs_df, logger=logger)
        
        if len(sequences) == 0:
            logger.warning("⚠️ 시퀀스가 없습니다.")
            return False
        
        logger.info(f"✅ 시퀀스 준비 완료: {len(sequences):,}개")
        
        vocab_size = len(self.template_vocab) + 1  # +1 for unknown
        logger.info(f"  - 어휘 크기: {vocab_size:,}")
        
        # logs_df 메모리 해제 (시퀀스 생성 완료 후 더 이상 필요 없음)
        del logs_df
        import gc
        gc.collect()
        logger.info(f"   💡 원본 로그 데이터 메모리 해제 완료")
        
        # 모델 구축
        if self.model is None:
            logger.info("모델 구축 중...")
            self.model = self.build_model(vocab_size)
            device_str = str(DEVICE) if TORCH_AVAILABLE and DEVICE is not None else "CPU"
            logger.info(f"✅ 모델 구축 완료 (디바이스: {device_str})")
        
        # PyTorch 텐서 변환 및 디바이스 이동
        device_str = str(DEVICE) if TORCH_AVAILABLE and DEVICE is not None else "CPU"
        logger.info(f"데이터 텐서 변환 중... (디바이스: {device_str})")
        num_sequences = len(sequences)  # 삭제 전에 길이 저장
        sequences_tensor = torch.LongTensor(sequences)
        next_logs_tensor = torch.LongTensor(next_logs)
        
        # 텐서를 디바이스로 이동
        if TORCH_AVAILABLE and DEVICE is not None:
            sequences_tensor = sequences_tensor.to(DEVICE)
            next_logs_tensor = next_logs_tensor.to(DEVICE)
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        # NumPy 배열 메모리 해제 (텐서 변환 완료 후)
        del sequences
        del next_logs
        import gc
        gc.collect()  # 가비지 컬렉션 강제 실행
        
        logger.info(f"✅ 텐서 변환 완료: {num_batches:,}개 배치")
        logger.info(f"   💡 NumPy 배열 메모리 해제 완료 (텐서만 메모리에 유지)")
        
        # 옵티마이저 및 손실 함수
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 학습
        logger.info("\n" + "=" * 60)
        logger.info("학습 시작")
        logger.info("=" * 60)
        self.model.train()
        
        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            num_batches_processed = 0
            
            for i in range(0, num_sequences, batch_size):
                batch_sequences = sequences_tensor[i:i + batch_size]
                batch_next = next_logs_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_next)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches_processed += 1
                
                # 배치 진행 상황 (10%마다)
                if num_batches_processed % max(1, num_batches // 10) == 0:
                    progress = (num_batches_processed / num_batches) * 100
                    logger.info(f"  Epoch {epoch + 1}/{epochs} - 배치 {num_batches_processed}/{num_batches} ({progress:.1f}%) - 현재 Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches_processed
            epoch_time = time.time() - epoch_start
            
            # 매 epoch마다 로그 출력
            elapsed_time = time.time() - start_time
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs if epoch > 0 else 0
            
            logger.info(f"Epoch {epoch + 1}/{epochs} 완료:")
            logger.info(f"  - 평균 Loss: {avg_loss:.4f}")
            logger.info(f"  - Epoch 소요 시간: {epoch_time:.2f}초")
            logger.info(f"  - 총 경과 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.1f}분)")
            if estimated_remaining > 0:
                logger.info(f"  - 예상 남은 시간: {estimated_remaining:.2f}초 ({estimated_remaining/60:.1f}분)")
            
            # 최고 성능 기록
            if avg_loss < best_loss:
                best_loss = avg_loss
                logger.info(f"  ⭐ 최고 Loss 갱신: {best_loss:.4f}")
        
        total_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ DeepLog 모델 학습 완료")
        logger.info(f"  - 총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        logger.info(f"  - 최종 Loss: {avg_loss:.4f}")
        logger.info(f"  - 최고 Loss: {best_loss:.4f}")
        logger.info("=" * 60)
        
        return True
    
    def predict_anomaly(self, logs_df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
        """
        이상치 탐지
        
        Args:
            logs_df: 테스트 로그 DataFrame
            threshold: 이상치 임계값
        
        Returns:
            이상치 탐지 결과 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        sequences, actual_next = self.prepare_sequences(logs_df)
        
        if len(sequences) == 0:
            return pd.DataFrame()
        
        self.model.eval()
        anomalies = []
        
        with torch.no_grad():
            sequences_tensor = torch.LongTensor(sequences)
            # 텐서를 디바이스로 이동
            if TORCH_AVAILABLE and DEVICE is not None:
                sequences_tensor = sequences_tensor.to(DEVICE)
            
            outputs = self.model(sequences_tensor)
            
            # 확률 계산
            probs = torch.softmax(outputs, dim=1)
            # CPU로 이동 후 numpy 변환
            if TORCH_AVAILABLE and DEVICE is not None:
                predicted = torch.argmax(probs, dim=1).cpu().numpy()
                predicted_probs = probs[np.arange(len(actual_next)), actual_next].cpu().numpy()
            else:
                predicted = torch.argmax(probs, dim=1).numpy()
                predicted_probs = probs[np.arange(len(actual_next)), actual_next].numpy()
            
            # 이상치 판단 (예측 확률이 낮으면 이상)
            anomaly_scores = 1 - predicted_probs
            
            for i in range(len(sequences)):
                if anomaly_scores[i] > threshold:
                    anomalies.append({
                        'sequence_index': i,
                        'predicted_template': predicted[i],
                        'actual_template': actual_next[i],
                        'prediction_prob': predicted_probs[i],
                        'anomaly_score': anomaly_scores[i],
                        'is_anomaly': True
                    })
        
        return pd.DataFrame(anomalies)


class LogAnomalyDetector:
    """LogAnomaly: Template2Vec 기반 의미 벡터화"""
    
    def __init__(self, vector_dim=100):
        """
        Args:
            vector_dim: 템플릿 벡터 차원
        """
        self.vector_dim = vector_dim
        self.template_extractor = LogTemplateExtractor()
        self.template_vectors = {}
        self.template_vocab = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def build_template_vectors(self, log_messages: List[str]):
        """템플릿 벡터 구축 (간단한 TF-IDF 기반)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 템플릿 추출
        templates = [self.template_extractor.extract_template(msg) for msg in log_messages]
        unique_templates = list(set(templates))
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=self.vector_dim, ngram_range=(1, 2))
        template_vectors = vectorizer.fit_transform(unique_templates).toarray()
        
        # 템플릿별 벡터 저장
        self.template_vectors = {
            template: vector
            for template, vector in zip(unique_templates, template_vectors)
        }
        
        # 템플릿 사전
        self.template_vocab = {template: idx for idx, template in enumerate(unique_templates)}
        
        return self.template_vectors
    
    def create_sequences(self, logs_df: pd.DataFrame, window_size=10) -> np.ndarray:
        """로그 시퀀스를 벡터 시퀀스로 변환"""
        logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
        
        sequences = []
        for i in range(len(logs_df) - window_size + 1):
            sequence_logs = logs_df.iloc[i:i + window_size]
            sequence_templates = [
                self.template_extractor.extract_template(msg)
                for msg in sequence_logs['message']
            ]
            
            # 템플릿 벡터로 변환
            sequence_vectors = [
                self.template_vectors.get(template, np.zeros(self.vector_dim))
                for template in sequence_templates
            ]
            
            sequences.append(sequence_vectors)
        
        return np.array(sequences)
    
    def train(self, logs_df: pd.DataFrame, window_size=10, log_dir=None, epochs=None, batch_size=None):
        """
        정상 패턴 학습
        
        Args:
            logs_df: 학습용 로그 DataFrame
            window_size: 시퀀스 윈도우 크기
            log_dir: 로그 저장 디렉토리
            epochs: 무시됨 (통계 기반 모델이므로 불필요)
            batch_size: 무시됨 (통계 기반 모델이므로 불필요)
        """
        # 로거 설정
        log_dir_path = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'training')
        logger = setup_training_logger('loganomaly', log_dir_path)
        
        logger.info("=" * 60)
        logger.info("LogAnomaly 모델 학습 시작")
        logger.info("=" * 60)
        logger.info(f"학습 파라미터:")
        logger.info(f"  - Window Size: {window_size}")
        logger.info(f"  - 학습 데이터: {len(logs_df):,}개 로그")
        if epochs is not None:
            logger.info(f"  - Epochs: {epochs} (무시됨, 통계 기반 모델)")
        if batch_size is not None:
            logger.info(f"  - Batch Size: {batch_size} (무시됨, 통계 기반 모델)")
        
        start_time = time.time()
        
        # 템플릿 벡터 구축
        logger.info("템플릿 벡터 구축 중...")
        log_messages = logs_df['message'].tolist()
        logger.info(f"  - 로그 메시지 수: {len(log_messages):,}개")
        
        self.build_template_vectors(log_messages)
        logger.info(f"✅ 템플릿 벡터 구축 완료: {len(self.template_vectors):,}개 템플릿")
        
        # 정상 시퀀스 생성
        logger.info("정상 시퀀스 생성 중...")
        normal_sequences = self.create_sequences(logs_df, window_size)
        
        if len(normal_sequences) == 0:
            logger.warning("⚠️ 시퀀스가 없습니다.")
            return False
        
        logger.info(f"✅ 시퀀스 생성 완료: {len(normal_sequences):,}개")
        
        # 정상 패턴 통계 저장
        logger.info("정상 패턴 통계 계산 중...")
        self.normal_mean = np.mean(normal_sequences, axis=(0, 1))
        self.normal_std = np.std(normal_sequences, axis=(0, 1)) + 1e-8
        logger.info(f"  - 평균 벡터 차원: {len(self.normal_mean)}")
        logger.info(f"  - 표준편차 벡터 차원: {len(self.normal_std)}")
        
        total_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ LogAnomaly 모델 학습 완료")
        logger.info(f"  - 총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        logger.info(f"  - 템플릿 수: {len(self.template_vectors):,}개")
        logger.info(f"  - 시퀀스 수: {len(normal_sequences):,}개")
        logger.info("=" * 60)
        
        return True
    
    def predict_anomaly(self, logs_df: pd.DataFrame, window_size=10, threshold=3.0) -> pd.DataFrame:
        """이상치 탐지 (Z-score 기반)"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        test_sequences = self.create_sequences(logs_df, window_size)
        
        if len(test_sequences) == 0:
            return pd.DataFrame()
        
        anomalies = []
        
        for i, sequence in enumerate(test_sequences):
            # 시퀀스 평균
            seq_mean = np.mean(sequence, axis=0)
            
            # Z-score 계산
            z_scores = np.abs((seq_mean - self.normal_mean) / self.normal_std)
            max_z_score = np.max(z_scores)
            
            if max_z_score > threshold:
                anomalies.append({
                    'sequence_index': i,
                    'max_z_score': max_z_score,
                    'anomaly_score': max_z_score / threshold,
                    'is_anomaly': True
                })
        
        return pd.DataFrame(anomalies)


class LogRobustDetector:
    """LogRobust: Attention + Bi-LSTM (간소화 버전)"""
    
    def __init__(self, embedding_dim=128, hidden_dim=64, num_layers=2):
        """
        Args:
            embedding_dim: 임베딩 차원
            hidden_dim: LSTM hidden 차원
            num_layers: LSTM 레이어 수
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.template_extractor = LogTemplateExtractor()
        self.tokenizer = None
        self.model = None
        self.is_fitted = False
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다. pip install torch")
        
        # Transformers 지연 import (실제 사용 시에만)
        global TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE is None:
            try:
                from transformers import BertTokenizer
                TRANSFORMERS_AVAILABLE = True
                # BertTokenizer 로드 시도
                try:
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                except Exception as e:
                    print(f"⚠️ BertTokenizer 모델 로드 실패: {e}")
                    self.tokenizer = None
            except Exception as e:
                TRANSFORMERS_AVAILABLE = False
                print(f"⚠️ Transformers 라이브러리 로드 실패: {e}")
                print("   LogRobust는 간소화된 버전(해시 기반 인코딩)으로 동작합니다.")
                self.tokenizer = None
    
    def encode_log(self, log_message: str) -> np.ndarray:
        """로그 메시지를 벡터로 인코딩"""
        if self.tokenizer is not None:
            try:
                # BERT 기반 인코딩 (간소화)
                tokens = self.tokenizer(log_message, return_tensors='pt', truncation=True, max_length=128)
                # 실제로는 BERT 모델을 통과시켜야 하지만, 여기서는 간소화
                return np.random.randn(self.embedding_dim)  # 임시
            except Exception as e:
                # BERT 실패 시 대체 방법 사용
                pass
        
        # 간단한 해시 기반 인코딩 (기본 방법)
        return np.array([hash(log_message) % 1000] * self.embedding_dim) / 1000.0
    
    def build_model(self, input_dim: int):
        """Bi-LSTM + Attention 모델 구축"""
        class BiLSTMAttention(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    bidirectional=True
                )
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim * 2,
                    num_heads=4,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_dim * 2, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                # 평균 풀링
                pooled = torch.mean(attn_out, dim=1)
                output = self.sigmoid(self.fc(pooled))
                return output
        
        self.model = BiLSTMAttention(input_dim, self.hidden_dim, self.num_layers)
        
        # 모델을 MPS/CUDA/CPU 디바이스로 이동
        if TORCH_AVAILABLE and DEVICE is not None:
            self.model = self.model.to(DEVICE)
        
        return self.model
    
    def train(self, logs_df: pd.DataFrame, sequence_length=10, epochs=50, batch_size=32, log_dir=None):
        """모델 학습"""
        # 로거 설정
        log_dir_path = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'training')
        logger = setup_training_logger('logrobust', log_dir_path)
        
        logger.info("=" * 60)
        logger.info("LogRobust 모델 학습 시작")
        logger.info("=" * 60)
        logger.info(f"학습 파라미터:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Sequence Length: {sequence_length}")
        logger.info(f"  - Embedding Dim: {self.embedding_dim}")
        logger.info(f"  - 학습 데이터: {len(logs_df):,}개 로그")
        
        start_time = time.time()
        
        logger.info("데이터 정렬 중...")
        logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
        
        # 시퀀스 생성
        logger.info("시퀀스 생성 중...")
        sequences = []
        labels = []  # 정상=0, 이상=1 (여기서는 정상 데이터만 학습)
        
        total_sequences = len(logs_df) - sequence_length + 1
        logger.info(f"  - 예상 시퀀스 수: {total_sequences:,}개")
        
        for i in range(total_sequences):
            sequence_logs = logs_df.iloc[i:i + sequence_length]
            sequence_vectors = [
                self.encode_log(msg) for msg in sequence_logs['message']
            ]
            sequences.append(sequence_vectors)
            labels.append(0)  # 정상 데이터
            
            # 진행 상황 출력 (10%마다)
            if (i + 1) % max(1, total_sequences // 10) == 0:
                progress = ((i + 1) / total_sequences) * 100
                logger.info(f"  시퀀스 생성 진행: {i + 1:,}/{total_sequences:,} ({progress:.1f}%)")
        
        if len(sequences) == 0:
            logger.warning("⚠️ 시퀀스가 없습니다.")
            return False
        
        logger.info(f"✅ 시퀀스 생성 완료: {len(sequences):,}개")
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        logger.info(f"  - 시퀀스 형태: {sequences.shape}")
        
        # 모델 구축
        if self.model is None:
            logger.info("모델 구축 중...")
            self.model = self.build_model(self.embedding_dim)
            device_str = str(DEVICE) if TORCH_AVAILABLE and DEVICE is not None else "CPU"
            logger.info(f"✅ 모델 구축 완료 (디바이스: {device_str})")
        
        # 학습
        device_str = str(DEVICE) if TORCH_AVAILABLE and DEVICE is not None else "CPU"
        logger.info(f"데이터 텐서 변환 중... (디바이스: {device_str})")
        sequences_tensor = torch.FloatTensor(sequences)
        labels_tensor = torch.FloatTensor(labels).unsqueeze(1)
        
        # 텐서를 디바이스로 이동
        if TORCH_AVAILABLE and DEVICE is not None:
            sequences_tensor = sequences_tensor.to(DEVICE)
            labels_tensor = labels_tensor.to(DEVICE)
        
        logger.info("✅ 텐서 변환 완료")
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        logger.info("\n" + "=" * 60)
        logger.info("학습 시작")
        logger.info("=" * 60)
        
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            optimizer.zero_grad()
            outputs = self.model(sequences_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs if epoch > 0 else 0
            
            # 매 epoch마다 로그 출력
            logger.info(f"Epoch {epoch + 1}/{epochs} 완료:")
            logger.info(f"  - Loss: {loss.item():.4f}")
            logger.info(f"  - Epoch 소요 시간: {epoch_time:.2f}초")
            logger.info(f"  - 총 경과 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.1f}분)")
            if estimated_remaining > 0:
                logger.info(f"  - 예상 남은 시간: {estimated_remaining:.2f}초 ({estimated_remaining/60:.1f}분)")
            
            # 최고 성능 기록
            if loss.item() < best_loss:
                best_loss = loss.item()
                logger.info(f"  ⭐ 최고 Loss 갱신: {best_loss:.4f}")
        
        total_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ LogRobust 모델 학습 완료")
        logger.info(f"  - 총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        logger.info(f"  - 최종 Loss: {loss.item():.4f}")
        logger.info(f"  - 최고 Loss: {best_loss:.4f}")
        logger.info("=" * 60)
        
        return True
    
    def predict_anomaly(self, logs_df: pd.DataFrame, sequence_length=10, threshold=0.5) -> pd.DataFrame:
        """이상치 탐지"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        logs_df = logs_df.sort_values('timestamp').reset_index(drop=True)
        
        sequences = []
        for i in range(len(logs_df) - sequence_length + 1):
            sequence_logs = logs_df.iloc[i:i + sequence_length]
            sequence_vectors = [
                self.encode_log(msg) for msg in sequence_logs['message']
            ]
            sequences.append(sequence_vectors)
        
        if len(sequences) == 0:
            return pd.DataFrame()
        
        sequences = np.array(sequences)
        sequences_tensor = torch.FloatTensor(sequences)
        
        # 텐서를 디바이스로 이동
        if TORCH_AVAILABLE and DEVICE is not None:
            sequences_tensor = sequences_tensor.to(DEVICE)
        
        self.model.eval()
        anomalies = []
        
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            # CPU로 이동 후 numpy 변환
            if TORCH_AVAILABLE and DEVICE is not None:
                anomaly_scores = outputs.squeeze().cpu().numpy()
            else:
                anomaly_scores = outputs.squeeze().numpy()
            
            for i, score in enumerate(anomaly_scores):
                if score > threshold:
                    anomalies.append({
                        'sequence_index': i,
                        'anomaly_score': float(score),
                        'is_anomaly': True
                    })
        
        return pd.DataFrame(anomalies)


class LogSpecificAnomalySystem:
    """로그 특화 이상 탐지 통합 시스템"""
    
    def __init__(self, model_type='deeplog'):
        """
        Args:
            model_type: 'deeplog', 'loganomaly', 'logrobust'
        """
        self.model_type = model_type
        self.detector = None
        self.severity_assessor = SeverityAssessment()
        self.logs_df = None
        
        if model_type == 'deeplog':
            self.detector = DeepLogDetector(sequence_length=15)  # 10 → 15
        elif model_type == 'loganomaly':
            self.detector = LogAnomalyDetector()
        elif model_type == 'logrobust':
            self.detector = LogRobustDetector()
        else:
            raise ValueError(f"알 수 없는 모델 타입: {model_type}")
    
    def load_logs(self, logs_df: pd.DataFrame):
        """로그 데이터 로드"""
        self.logs_df = logs_df.copy()
        print(f"✅ {len(self.logs_df)}개 로그 로드 완료")
    
    def train(self, train_ratio=0.8, log_dir=None, epochs=5, batch_size=128):
        """모델 학습"""
        if self.logs_df is None or self.logs_df.empty:
            print("⚠️ 로그 데이터가 없습니다.")
            return False
        
        # 정상 로그만 학습
        normal_logs = self.logs_df[self.logs_df['is_error'] == False]
        
        if len(normal_logs) == 0:
            print("⚠️ 정상 로그가 없습니다.")
            return False
        
        split_idx = int(len(normal_logs) * train_ratio)
        train_logs = normal_logs.iloc[:split_idx]
        
        print(f"\n학습 데이터: {len(train_logs):,}개 로그 (정상)")
        print(f"⚡ 빠른 학습 설정: Epochs={epochs}, Batch Size={batch_size}")
        
        # 모델 타입에 따라 시퀀스 길이 전달
        if self.model_type == 'loganomaly':
            return self.detector.train(train_logs, window_size=15, log_dir=log_dir, epochs=epochs, batch_size=batch_size)  # 10 → 15
        elif self.model_type == 'logrobust':
            return self.detector.train(train_logs, sequence_length=15, epochs=epochs, batch_size=batch_size, log_dir=log_dir)  # 10 → 15
        else:
            return self.detector.train(train_logs, log_dir=log_dir, epochs=epochs, batch_size=batch_size)
    
    def detect_anomalies(self, test_logs_df=None):
        """이상치 탐지 및 심각도 평가"""
        if not self.detector.is_fitted:
            print("⚠️ 모델이 학습되지 않았습니다.")
            return {}
        
        if test_logs_df is None:
            test_logs_df = self.logs_df
        
        print("\n" + "=" * 60)
        print(f"{self.model_type.upper()} 이상 탐지")
        print("=" * 60)
        
        # 이상치 탐지 (시퀀스 길이 15로 설정)
        if self.model_type == 'loganomaly':
            anomalies = self.detector.predict_anomaly(test_logs_df, window_size=15)  # 10 → 15
        elif self.model_type == 'logrobust':
            anomalies = self.detector.predict_anomaly(test_logs_df, sequence_length=15)  # 10 → 15
        else:
            anomalies = self.detector.predict_anomaly(test_logs_df)
        
        if anomalies.empty:
            print("✅ 이상치가 탐지되지 않았습니다.")
            return {'anomalies': pd.DataFrame(), 'summary': {}}
        
        print(f"✅ {len(anomalies)}개 이상 시퀀스 탐지")
        
        # 탐지된 시퀀스의 로그 추출 및 심각도 평가
        anomaly_logs_list = []
        
        for idx, row in anomalies.iterrows():
            seq_idx = row['sequence_index']
            # 시퀀스에 해당하는 로그 추출 (간단화)
            if seq_idx < len(test_logs_df):
                # 시퀀스 길이에 맞게 조정 (DeepLog는 15, LogAnomaly는 15, LogRobust는 15)
                seq_len = getattr(self.detector, 'sequence_length', getattr(self.detector, 'window_size', 15))
                sequence_logs = test_logs_df.iloc[seq_idx:seq_idx + seq_len]  # 시퀀스 길이만큼
                
                # 심각도 평가
                severity_info = self.severity_assessor.assess_time_window_severity(sequence_logs)
                
                anomaly_logs_list.append({
                    'sequence_index': seq_idx,
                    'anomaly_score': row.get('anomaly_score', 0),
                    'max_severity_score': severity_info['max_severity_score'],
                    'max_severity_level': severity_info['max_severity_level'],
                    'avg_severity_score': severity_info['avg_severity_score'],
                    'critical_count': severity_info['critical_count'],
                    'high_count': severity_info['high_count'],
                    'medium_count': severity_info['medium_count'],
                    'low_count': severity_info['low_count'],
                })
        
        results_df = pd.DataFrame(anomaly_logs_list)
        
        # 심각도 기준 정렬
        if not results_df.empty and 'max_severity_score' in results_df.columns:
            results_df = results_df.sort_values('max_severity_score', ascending=False)
            results_df['priority'] = range(1, len(results_df) + 1)
        
        # 요약 통계
        summary = {
            'total_anomalies': len(results_df),
            'by_severity': results_df['max_severity_level'].value_counts().to_dict() if 'max_severity_level' in results_df.columns else {},
            'avg_severity_score': results_df['max_severity_score'].mean() if 'max_severity_score' in results_df.columns else 0,
            'max_severity_score': results_df['max_severity_score'].max() if 'max_severity_score' in results_df.columns else 0,
        }
        
        return {
            'anomalies': results_df,
            'summary': summary
        }
    
    def generate_report(self, results):
        """결과 리포트 생성"""
        print("\n" + "=" * 60)
        print(f"{self.model_type.upper()} 이상 탐지 + 심각도 평가 결과")
        print("=" * 60)
        
        if not results or results.get('total_anomalies', 0) == 0:
            print("✅ 이상치가 탐지되지 않았습니다.")
            return
        
        summary = results.get('summary', {})
        
        print(f"\n📊 탐지 결과:")
        print(f"   총 이상 시퀀스: {summary.get('total_anomalies', 0)}개")
        
        if 'by_severity' in summary:
            print(f"\n🔍 심각도 분포:")
            for level, count in summary['by_severity'].items():
                print(f"   {level}: {count}개")
        
        print(f"\n   평균 심각도 점수: {summary.get('avg_severity_score', 0):.2f}")
        print(f"   최고 심각도 점수: {summary.get('max_severity_score', 0):.2f}")


def analyze_risk_level(anomalies_df: pd.DataFrame, test_logs_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    이상 탐지 결과를 위험도별로 분석 (개선된 로직)
    
    Args:
        anomalies_df: 이상 탐지 결과 DataFrame
        test_logs_df: 테스트 로그 DataFrame (로그 메시지 확인용, 선택적)
    
    Returns:
        위험도 분석 결과 DataFrame
    """
    if anomalies_df.empty:
        return pd.DataFrame()
    
    anomalies_df = anomalies_df.copy()
    
    # 실제 위험 키워드 (예외, 오류 등)
    CRITICAL_KEYWORDS = [
        'exception', 'error', 'failed', 'timeout', 'nullpointer',
        'outofmemory', 'connection refused', 'unauthorized', 'forbidden',
        'sql injection', 'xss', 'csrf', 'stacktrace', 'traceback',
        'fatal', 'critical', 'panic', 'crash', 'hang', 'deadlock',
        'out of memory', 'memory leak', 'disk full', 'permission denied'
    ]
    
    # 정상 쿼리 패턴 (위험도 낮춤)
    NORMAL_QUERY_PATTERNS = [
        'binding parameter', '==> parameters', '==>  preparing',
        'committing jdbc', 'extracted value', '<==      total',
        'creating a new sqlsession', 'closing non transactional',
        'jdbc connection', 'hikariproxyconnection', 'will not be managed',
        'registered for synchronization', 'accept-language',
        'heartbeat status: 200', 'discoveryclient'
    ]
    
    # 위험도 점수 계산 함수 (개선된 로직)
    def calculate_risk_score(row):
        anomaly_score = row['anomaly_score']
        severity_score = row['max_severity_score']
        
        # 로그 메시지 확인 (가능한 경우)
        messages = ""
        if test_logs_df is not None and 'sequence_index' in row:
            try:
                seq_idx = int(row['sequence_index'])
                seq_len = 15  # 시퀀스 길이
                start_idx = max(0, seq_idx)
                end_idx = min(len(test_logs_df), seq_idx + seq_len)
                sequence_logs = test_logs_df.iloc[start_idx:end_idx]
                messages = ' '.join(sequence_logs['message'].astype(str).tolist()).lower()
            except:
                pass
        
        # sample_messages 컬럼이 있으면 사용
        if 'sample_messages' in row and pd.notna(row['sample_messages']):
            messages = str(row['sample_messages']).lower()
        
        # 정상 쿼리 패턴 확인
        is_normal_query = False
        if messages:
            is_normal_query = any(pattern in messages for pattern in NORMAL_QUERY_PATTERNS)
        
        # 실제 위험 키워드 확인
        has_real_exception = False
        if messages:
            has_real_exception = any(keyword in messages for keyword in CRITICAL_KEYWORDS)
        
        # 위험도 계산 (상황별 가중치 조정)
        if is_normal_query and not has_real_exception:
            # 정상 쿼리 로그: 위험도 대폭 낮춤 (CRITICAL 방지)
            # 정상 쿼리는 최대 79점으로 제한하여 CRITICAL이 되지 않도록 함
            risk_score = (
                anomaly_score * 15 +  # 이상 점수 가중치 더 낮춤
                (severity_score / 10) * 10  # 심각도 점수 가중치 더 낮춤
            )
            # 정상 쿼리는 최대 79점으로 제한
            risk_score = min(79, risk_score)
        elif has_real_exception:
            # 실제 예외/오류: 위험도 높임
            risk_score = (
                anomaly_score * 60 +  # 이상 점수 가중치 높임
                (severity_score / 10) * 70  # 심각도 점수 가중치 높임
            )
        else:
            # 기본 계산 (기존과 유사하지만 약간 조정)
            risk_score = (
                anomaly_score * 40 +  # 이상 점수 40% 가중치
                (severity_score / 10) * 40  # 심각도 점수 40% 가중치
            )
        
        # 점수 범위 제한 (0-150, 하지만 100 이상은 매우 드뭄)
        return min(150, max(0, risk_score))
    
    # 위험도 점수 계산
    anomalies_df['risk_score'] = anomalies_df.apply(calculate_risk_score, axis=1)
    
    # 위험도 레벨 분류
    def classify_risk_level(risk_score):
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'INFO'
    
    anomalies_df['risk_level'] = anomalies_df['risk_score'].apply(classify_risk_level)
    
    return anomalies_df


def generate_risk_report(anomalies_df: pd.DataFrame, test_logs_df: pd.DataFrame) -> Dict:
    """
    위험도 분석 리포트 생성
    
    Args:
        anomalies_df: 이상 탐지 결과 DataFrame
        test_logs_df: 테스트 로그 DataFrame
    
    Returns:
        위험도 분석 리포트 딕셔너리
    """
    if anomalies_df.empty:
        return {
            'total_anomalies': 0,
            'risk_distribution': {},
            'critical_anomalies': pd.DataFrame(),
            'high_anomalies': pd.DataFrame(),
            'medium_anomalies': pd.DataFrame(),
            'low_anomalies': pd.DataFrame(),
            'info_anomalies': pd.DataFrame()
        }
    
    # 위험도별 분류
    critical = anomalies_df[anomalies_df['risk_level'] == 'CRITICAL'].copy()
    high = anomalies_df[anomalies_df['risk_level'] == 'HIGH'].copy()
    medium = anomalies_df[anomalies_df['risk_level'] == 'MEDIUM'].copy()
    low = anomalies_df[anomalies_df['risk_level'] == 'LOW'].copy()
    info = anomalies_df[anomalies_df['risk_level'] == 'INFO'].copy()
    
    # 위험도 분포
    risk_distribution = anomalies_df['risk_level'].value_counts().to_dict()
    
    # 각 위험도별 상세 정보 추가 (로그 내용 포함)
    def add_log_details(risk_df, test_logs_df):
        if risk_df.empty:
            return risk_df
        
        log_details = []
        for idx, row in risk_df.iterrows():
            seq_idx = row['sequence_index']
            seq_len = 15  # 시퀀스 길이
            start_idx = max(0, seq_idx)
            end_idx = min(len(test_logs_df), seq_idx + seq_len)
            
            sequence_logs = test_logs_df.iloc[start_idx:end_idx]
            
            # 로그 메시지 요약
            log_messages = sequence_logs['message'].tolist()
            log_levels = sequence_logs['level'].tolist()
            
            log_details.append({
                'log_count': len(sequence_logs),
                'log_levels': ', '.join(set(log_levels)),
                'sample_messages': ' | '.join(log_messages[:3])  # 처음 3개만
            })
        
        details_df = pd.DataFrame(log_details)
        for col in details_df.columns:
            risk_df[col] = details_df[col].values
        
        return risk_df
    
    critical = add_log_details(critical, test_logs_df)
    high = add_log_details(high, test_logs_df)
    medium = add_log_details(medium, test_logs_df)
    low = add_log_details(low, test_logs_df)
    info = add_log_details(info, test_logs_df)
    
    return {
        'total_anomalies': len(anomalies_df),
        'risk_distribution': risk_distribution,
        'critical_anomalies': critical,
        'high_anomalies': high,
        'medium_anomalies': medium,
        'low_anomalies': low,
        'info_anomalies': info,
        'avg_risk_score': anomalies_df['risk_score'].mean(),
        'max_risk_score': anomalies_df['risk_score'].max()
    }


def print_risk_report(risk_report: Dict):
    """위험도 분석 리포트 출력"""
    print("\n" + "=" * 70)
    print("위험도 분석 리포트")
    print("=" * 70)
    
    print(f"\n📊 전체 통계:")
    print(f"   총 이상 탐지: {risk_report['total_anomalies']:,}개")
    print(f"   평균 위험도 점수: {risk_report.get('avg_risk_score', 0):.2f}/100")
    print(f"   최고 위험도 점수: {risk_report.get('max_risk_score', 0):.2f}/100")
    
    print(f"\n🔍 위험도 분포:")
    risk_dist = risk_report.get('risk_distribution', {})
    risk_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
    for level in risk_order:
        count = risk_dist.get(level, 0)
        if count > 0:
            percentage = (count / risk_report['total_anomalies']) * 100
            print(f"   {level:10s}: {count:5d}개 ({percentage:5.1f}%)")
    
    # 위험도별 상세 정보
    for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM']:
        risk_df = risk_report.get(f'{risk_level.lower()}_anomalies', pd.DataFrame())
        if not risk_df.empty:
            print(f"\n⚠️ {risk_level} 위험 이상 ({len(risk_df)}개):")
            print("   " + "-" * 66)
            
            # 상위 10개만 출력
            top_risks = risk_df.head(10)
            for idx, row in top_risks.iterrows():
                print(f"   [{row.get('priority', idx+1)}] 위험도: {row.get('risk_score', 0):.1f}/100")
                print(f"       - 이상 점수: {row.get('anomaly_score', 0):.4f}")
                print(f"       - 심각도: {row.get('max_severity_level', 'N/A')} (점수: {row.get('max_severity_score', 0):.2f})")
                print(f"       - 시퀀스 인덱스: {row.get('sequence_index', 'N/A')}")
                if 'log_count' in row:
                    print(f"       - 로그 수: {row.get('log_count', 0)}개")
                    print(f"       - 로그 레벨: {row.get('log_levels', 'N/A')}")
                print()


def main():
    """메인 실행 함수 - LogAnomaly 모델을 사용한 이상 탐지 및 위험도 분석"""
    from log_anomaly_detector import SpringBootLogParser
    
    log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    print("=" * 70)
    print("LogAnomaly 기반 이상 탐지 및 위험도 분석 시스템")
    print("=" * 70)
    
    # 모델 선택: LogAnomaly (성능 측정으로 선정된 모델)
    model_type = 'loganomaly'
    
    print(f"\n✅ 사용 모델: {model_type.upper()} (성능 측정 선정 모델)")
    
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
    
    print(f"✅ {len(logs_df):,}개 로그 라인 파싱 완료")
    
    # 2. 시스템 초기화
    print(f"\n2단계: {model_type.upper()} 시스템 초기화")
    system = LogSpecificAnomalySystem(model_type=model_type)
    
    # 학습 데이터와 테스트 데이터 분리
    normal_logs = logs_df[logs_df['is_error'] == False]
    error_logs = logs_df[logs_df['is_error'] == True]
    
    # 메모리 최적화: 학습 데이터 샘플링 (최대 500K)
    max_train_samples = 500000
    
    if len(normal_logs) > max_train_samples:
        print(f"\n⚠️ 메모리 최적화: 정상 로그 {len(normal_logs):,}개 → {max_train_samples:,}개로 샘플링")
        # 균등하게 샘플링 (시간 순서 유지)
        step = len(normal_logs) // max_train_samples
        normal_logs = normal_logs.iloc[::step][:max_train_samples].copy()
        print(f"   샘플링 완료: {len(normal_logs):,}개")
    
    # 학습용: 정상 로그의 80%
    train_size = int(len(normal_logs) * 0.8)
    train_logs = normal_logs.iloc[:train_size].copy()
    
    # 테스트 데이터도 샘플링 (최대 100K)
    max_test_normal = 80000
    max_test_error = 20000
    
    test_normal = normal_logs.iloc[train_size:].copy()
    if len(test_normal) > max_test_normal:
        print(f"\n⚠️ 테스트 정상 로그 샘플링: {len(test_normal):,}개 → {max_test_normal:,}개")
        step = len(test_normal) // max_test_normal
        test_normal = test_normal.iloc[::step][:max_test_normal].copy()
    
    if len(error_logs) > max_test_error:
        print(f"⚠️ 테스트 에러 로그 샘플링: {len(error_logs):,}개 → {max_test_error:,}개")
        step = len(error_logs) // max_test_error
        error_logs = error_logs.iloc[::step][:max_test_error].copy()
    
    test_logs = pd.concat([
        test_normal,
        error_logs
    ], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n📊 데이터 준비 완료:")
    print(f"   학습 데이터: {len(train_logs):,}개 (정상 로그)")
    print(f"   테스트 데이터: {len(test_logs):,}개 (정상: {len(test_normal):,}개, 에러: {len(error_logs):,}개)")
    
    # 메모리 예상치 계산
    estimated_memory_gb = (len(train_logs) * 15 * 100 * 4) / (1024 ** 3)
    print(f"\n💾 예상 메모리 사용량: {estimated_memory_gb:.2f} GB")
    if estimated_memory_gb > 16:
        print(f"⚠️ 경고: 메모리 사용량이 높을 수 있습니다. 더 작은 샘플 사용을 권장합니다.")
        return
    
    # 3. 모델 학습
    print("\n3단계: 모델 학습")
    system.load_logs(train_logs)
    if not system.train(train_ratio=1.0, epochs=10, batch_size=32):
        print("❌ 모델 학습 실패")
        return
    
    print("✅ 모델 학습 완료")
    
    # 4. 이상 탐지 및 심각도 평가
    print("\n4단계: 이상 탐지 및 심각도 평가")
    print(f"   테스트 데이터 분석 중: {len(test_logs):,}개 로그...")
    results = system.detect_anomalies(test_logs)
    
    if not results or results.get('anomalies', pd.DataFrame()).empty:
        print("✅ 이상이 탐지되지 않았습니다.")
        return
    
    anomalies_df = results['anomalies']
    print(f"✅ {len(anomalies_df):,}개 이상 시퀀스 탐지")
    
    # 5. 위험도 분석 (개선된 로직 적용)
    print("\n5단계: 위험도 분석 (개선된 로직: 정상 쿼리 필터링, 실제 예외 감지)")
    anomalies_with_risk = analyze_risk_level(anomalies_df, test_logs)
    risk_report = generate_risk_report(anomalies_with_risk, test_logs)
    
    # 6. 위험도 리포트 출력
    print_risk_report(risk_report)
    
    # 7. 결과 저장 (동적 폴더 생성)
    base_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results"
    base_folder_name = "loganomaly_risk_analysis"
    
    # 폴더가 이미 존재하면 번호를 증가시켜 새 폴더 생성
    output_dir = os.path.join(base_dir, base_folder_name)
    folder_num = 0
    while os.path.exists(output_dir):
        folder_num += 1
        output_dir = os.path.join(base_dir, f"{base_folder_name}_{folder_num}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 결과 저장 폴더: {os.path.basename(output_dir)}")
    
    # 전체 이상 탐지 결과 저장
    if not anomalies_with_risk.empty:
        output_path = os.path.join(output_dir, "anomalies_with_risk.csv")
        anomalies_with_risk.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 전체 이상 탐지 결과 저장: {output_path}")
    
    # 위험도별 결과 저장
    for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        risk_df = risk_report.get(f'{risk_level.lower()}_anomalies', pd.DataFrame())
        if not risk_df.empty:
            risk_path = os.path.join(output_dir, f"risk_{risk_level.lower()}.csv")
            risk_df.to_csv(risk_path, index=False, encoding='utf-8-sig')
            print(f"💾 {risk_level} 위험 이상 저장: {risk_path} ({len(risk_df)}개)")
    
    # 위험도 요약 리포트 저장
    summary_path = os.path.join(output_dir, "risk_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LogAnomaly 기반 이상 탐지 및 위험도 분석 리포트\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"전체 통계:\n")
        f.write(f"  - 총 이상 탐지: {risk_report['total_anomalies']:,}개\n")
        f.write(f"  - 평균 위험도 점수: {risk_report.get('avg_risk_score', 0):.2f}/100\n")
        f.write(f"  - 최고 위험도 점수: {risk_report.get('max_risk_score', 0):.2f}/100\n\n")
        f.write(f"위험도 분포:\n")
        risk_dist = risk_report.get('risk_distribution', {})
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = risk_dist.get(level, 0)
            if count > 0:
                percentage = (count / risk_report['total_anomalies']) * 100
                f.write(f"  - {level:10s}: {count:5d}개 ({percentage:5.1f}%)\n")
    
    print(f"💾 위험도 요약 리포트 저장: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✅ 이상 탐지 및 위험도 분석 완료!")
    print("=" * 70)
    print(f"\n📁 결과 저장 위치: {output_dir}")
    print(f"   - 전체 이상 탐지: {len(anomalies_with_risk):,}개")
    print(f"   - CRITICAL 위험: {len(risk_report.get('critical_anomalies', pd.DataFrame()))}개")
    print(f"   - HIGH 위험: {len(risk_report.get('high_anomalies', pd.DataFrame()))}개")
    print(f"   - MEDIUM 위험: {len(risk_report.get('medium_anomalies', pd.DataFrame()))}개")


if __name__ == "__main__":
    main()

