"""
Spring Boot 로그 이상치 탐지 모델
backup 폴더의 로그 파일들을 분석하여 이상 패턴을 탐지합니다.
"""

import re
import os
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import warnings
warnings.filterwarnings('ignore')


class SpringBootLogParser:
    """Spring Boot 로그 파서"""
    
    # Spring Boot 로그 패턴: 2025-07-02 15:59:36.514  INFO 12185 --- [           main] k.r.b.f.c.Application : Starting...
    LOG_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+'
        r'(\w+)\s+'
        r'(\d+)\s+'
        r'---\s+'
        r'\[([^\]]+)\]\s+'
        r'([^\s:]+)\s*:?\s*'
        r'(.*)'
    )
    
    ERROR_KEYWORDS = [
        'Exception', 'Error', 'Failed', 'Fatal', 'Critical',
        'Timeout', 'Connection refused', 'OutOfMemoryError',
        'NullPointerException', 'StackOverflowError',
        'ClassNotFoundException', 'NoClassDefFoundError',
        'SQLException', 'IOException', 'SocketException'
    ]
    
    def __init__(self):
        self.parsed_logs = []
        
    def parse_log_line(self, line):
        """로그 라인 파싱"""
        match = self.LOG_PATTERN.match(line.strip())
        if match:
            timestamp_str, level, pid, thread, class_path, message = match.groups()
            try:
                timestamp = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S.%f')
            except:
                timestamp = pd.to_datetime(timestamp_str, errors='coerce')
            
            # 에러 키워드 확인
            is_error = level in ['ERROR', 'FATAL'] or any(
                keyword.lower() in message.lower() for keyword in self.ERROR_KEYWORDS
            )
            
            return {
                'timestamp': timestamp,
                'level': level,
                'pid': pid,
                'thread': thread.strip(),
                'class_path': class_path,
                'message': message,
                'is_error': is_error,
                'message_length': len(message),
                'has_exception': 'Exception' in message or 'Error' in message
            }
        return None
    
    def parse_log_file(self, file_path, max_lines=None):
        """
        로그 파일 파싱
        
        Args:
            file_path: 로그 파일 경로
            max_lines: 최대 파싱할 라인 수 (None이면 전체)
        """
        logs = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if max_lines and line_num > max_lines:
                        break
                    parsed = self.parse_log_line(line)
                    if parsed:
                        parsed['file_path'] = os.path.basename(file_path)
                        parsed['line_number'] = line_num
                        logs.append(parsed)
        except Exception as e:
            print(f"파일 읽기 오류 {file_path}: {e}")
        return logs
    
    def parse_directory(self, directory_path, max_files=None, sample_lines=None, 
                       chunk_size=10000, max_total_lines=None, 
                       save_chunks_to_disk=True, chunk_dir=None, keep_chunks=False):
        """
        디렉토리 내 모든 로그 파일 파싱 (메모리 효율적)
        
        Args:
            directory_path: 로그 디렉토리 경로
            max_files: 최대 파싱할 파일 수 (None이면 전체)
            sample_lines: 파일당 최대 파싱할 라인 수 (None이면 전체)
            chunk_size: 청크 단위로 처리할 로그 라인 수 (메모리 절약)
            max_total_lines: 전체 최대 파싱할 라인 수 (None이면 제한 없음)
            save_chunks_to_disk: 청크를 디스크에 저장할지 여부 (기본 True, 메모리 절약)
            chunk_dir: 청크 파일 저장 디렉토리 (None이면 프로젝트 폴더/chunks)
            keep_chunks: 청크 파일을 유지할지 여부 (기본 False, 병합 후 삭제)
        """
        import tempfile
        import shutil
        
        log_files = glob.glob(os.path.join(directory_path, '*.log'))
        
        if max_files:
            log_files = log_files[:max_files]
        
        print(f"총 {len(log_files)}개 로그 파일 발견")
        if max_files:
            print(f"  (최대 {max_files}개 파일만 처리)")
        if max_total_lines:
            print(f"  (전체 최대 {max_total_lines:,}개 라인만 처리)")
        print(f"  (청크 크기: {chunk_size:,}개 라인)")
        
        # 청크 저장 방식 결정
        if save_chunks_to_disk:
            if chunk_dir is None:
                # 프로젝트 폴더 밑에 chunks 디렉토리 생성
                # log_anomaly_detector.py가 있는 디렉토리 기준
                current_dir = os.path.dirname(os.path.abspath(__file__))
                chunk_dir = os.path.join(current_dir, 'chunks')
            os.makedirs(chunk_dir, exist_ok=True)
            print(f"  📁 청크 파일 저장 위치: {chunk_dir}")
            chunk_files = []  # 파일 경로 리스트
        else:
            chunk_dfs = []  # 메모리 리스트
        
        total_parsed = 0
        file_count = 0
        chunk_count = 0
        
        for file_path in log_files:
            if max_total_lines and total_parsed >= max_total_lines:
                print(f"\n⚠️ 전체 최대 라인 수({max_total_lines:,})에 도달하여 중단합니다.")
                break
                
            file_count += 1
            print(f"\n[{file_count}/{len(log_files)}] 파싱 중: {os.path.basename(file_path)}")
            
            # 파일을 청크 단위로 파싱
            file_logs = []
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if max_total_lines and total_parsed >= max_total_lines:
                            break
                        if sample_lines and line_num > sample_lines:
                            break
                            
                        parsed = self.parse_log_line(line)
                        if parsed:
                            parsed['file_path'] = os.path.basename(file_path)
                            parsed['line_number'] = line_num
                            file_logs.append(parsed)
                            total_parsed += 1
                            
                            # 청크 크기에 도달하면 처리
                            if len(file_logs) >= chunk_size:
                                chunk_df = pd.DataFrame(file_logs)
                                
                                if save_chunks_to_disk:
                                    # 파일로 저장 (메모리 절약)
                                    chunk_count += 1
                                    chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_count:06d}.parquet')
                                    try:
                                        chunk_df.to_parquet(chunk_file, compression='snappy', engine='pyarrow')
                                        chunk_files.append(chunk_file)
                                        print(f"  - 청크 파일 저장: {chunk_file} ({len(file_logs):,}개 라인, 누적: {total_parsed:,}개)")
                                    except ImportError:
                                        # pyarrow가 없으면 pickle 사용
                                        chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_count:06d}.pkl')
                                        chunk_df.to_pickle(chunk_file)
                                        chunk_files.append(chunk_file)
                                        print(f"  - 청크 파일 저장: {chunk_file} ({len(file_logs):,}개 라인, 누적: {total_parsed:,}개)")
                                    del chunk_df  # 즉시 메모리 해제
                                else:
                                    # 메모리에 저장
                                    chunk_dfs.append(chunk_df)
                                    print(f"  - 청크 저장: {len(file_logs):,}개 라인 (누적: {total_parsed:,}개)")
                                
                                file_logs = []  # 메모리 해제
                                
            except Exception as e:
                print(f"  ⚠️ 파일 읽기 오류 {file_path}: {e}")
                continue
            
            # 남은 로그 처리
            if file_logs:
                chunk_df = pd.DataFrame(file_logs)
                
                if save_chunks_to_disk:
                    chunk_count += 1
                    chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_count:06d}.parquet')
                    try:
                        chunk_df.to_parquet(chunk_file, compression='snappy', engine='pyarrow')
                        chunk_files.append(chunk_file)
                        print(f"  - 청크 파일 저장: {chunk_file} ({len(file_logs):,}개 라인, 누적: {total_parsed:,}개)")
                    except ImportError:
                        chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_count:06d}.pkl')
                        chunk_df.to_pickle(chunk_file)
                        chunk_files.append(chunk_file)
                        print(f"  - 청크 파일 저장: {chunk_file} ({len(file_logs):,}개 라인, 누적: {total_parsed:,}개)")
                    del chunk_df
                else:
                    chunk_dfs.append(chunk_df)
                    print(f"  - 청크 저장: {len(file_logs):,}개 라인 (누적: {total_parsed:,}개)")
            
            print(f"  ✅ 파일 완료: {total_parsed:,}개 라인 파싱")
            
            # 메모리 정리
            del file_logs
        
        # 모든 청크를 하나의 DataFrame으로 병합
        if save_chunks_to_disk:
            if not chunk_files:
                print("⚠️ 파싱된 로그가 없습니다.")
                if os.path.exists(chunk_dir) and chunk_dir.startswith(tempfile.gettempdir()):
                    os.rmdir(chunk_dir)
                return pd.DataFrame()
            
            print(f"\n📊 청크 파일 병합 중... (총 {len(chunk_files)}개 파일)")
            
            # 스트리밍 방식으로 병합 (메모리 절약)
            result_dfs = []
            for i, chunk_file in enumerate(chunk_files):
                try:
                    if chunk_file.endswith('.parquet'):
                        chunk_df = pd.read_parquet(chunk_file, engine='pyarrow')
                    else:
                        chunk_df = pd.read_pickle(chunk_file)
                    
                    result_dfs.append(chunk_df)
                    
                    # 일정 개수마다 병합하여 메모리 절약
                    if len(result_dfs) >= 10:  # 10개마다 병합
                        temp_df = pd.concat(result_dfs, ignore_index=True)
                        result_dfs = [temp_df]  # 병합된 결과만 유지
                    
                    # 파일 삭제는 나중에 keep_chunks 옵션에 따라 결정
                    # 여기서는 삭제하지 않음 (병합 후 일괄 처리)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  - {i + 1}/{len(chunk_files)}개 파일 처리 완료")
                        
                except Exception as e:
                    print(f"  ⚠️ 청크 파일 읽기 오류 {chunk_file}: {e}")
                    continue
            
            # 최종 병합
            if result_dfs:
                result_df = pd.concat(result_dfs, ignore_index=True)
            else:
                result_df = pd.DataFrame()
            
            # 청크 파일 정리
            if not keep_chunks:
                # 청크 파일 삭제
                print(f"\n🗑️  청크 파일 삭제 중...")
                try:
                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                    # 디렉토리가 비어있으면 삭제
                    if os.path.exists(chunk_dir) and not os.listdir(chunk_dir):
                        os.rmdir(chunk_dir)
                        print(f"  ✅ 청크 디렉토리 삭제 완료: {chunk_dir}")
                    else:
                        print(f"  ✅ 청크 파일 삭제 완료 ({len(chunk_files)}개)")
                except Exception as e:
                    print(f"  ⚠️ 청크 파일 삭제 중 오류: {e}")
            else:
                # 청크 파일 유지
                print(f"\n💾 청크 파일 저장 완료: {chunk_dir}")
                print(f"  - 총 {len(chunk_files)}개 청크 파일")
                print(f"  💡 청크 파일 삭제: shutil.rmtree('{chunk_dir}')")
            
        else:
            if not chunk_dfs:
                print("⚠️ 파싱된 로그가 없습니다.")
                return pd.DataFrame()
            
            print(f"\n📊 청크 병합 중... (총 {len(chunk_dfs)}개 청크)")
            result_df = pd.concat(chunk_dfs, ignore_index=True)
            del chunk_dfs
        
        print(f"✅ 총 {len(result_df):,}개 로그 라인 파싱 완료")
        return result_df
    
    def save_parsed_data(self, logs_df: pd.DataFrame, output_path: str):
        """
        파싱된 데이터를 Parquet 파일로 저장 (pyarrow 없으면 pickle 사용)
        
        Args:
            logs_df: 저장할 DataFrame
            output_path: 저장 경로 (파일명 포함)
        """
        # 디렉토리 경로 추출 (파일명만 있는 경우 처리)
        dir_path = os.path.dirname(output_path)
        if dir_path:  # 디렉토리 경로가 있는 경우
            os.makedirs(dir_path, exist_ok=True)
        # 디렉토리 경로가 없으면 현재 디렉토리에 저장 (별도 처리 불필요)
        
        # Parquet 저장 시도 (pyarrow 사용)
        try:
            logs_df.to_parquet(output_path, compression='snappy', engine='pyarrow')
            file_format = "Parquet"
        except ImportError:
            # pyarrow가 없으면 pickle로 저장
            if output_path.endswith('.parquet'):
                # 확장자를 .pkl로 변경
                output_path = output_path.replace('.parquet', '.pkl')
            logs_df.to_pickle(output_path)
            file_format = "Pickle"
            print("⚠️ pyarrow가 설치되지 않아 Pickle 형식으로 저장합니다.")
            print("   💡 Parquet 형식을 사용하려면: pip install pyarrow")
        
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
        print(f"✅ 파싱 데이터 저장 완료: {output_path} ({file_format}, {file_size:.2f} MB)")
    
    def load_parsed_data(self, input_path: str, chunk_size: int = None) -> pd.DataFrame:
        """
        저장된 파싱 데이터를 로드 (Parquet 또는 Pickle)
        
        Args:
            input_path: 로드할 파일 경로
            chunk_size: 청크 단위로 읽을 크기 (None이면 전체 로드, 메모리 절약 시 사용)
            
        Returns:
            로드된 DataFrame
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")
        
        # 청크 단위로 읽기 (메모리 효율적)
        if chunk_size and input_path.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(input_path)
                
                print(f"📂 Parquet 파일을 청크 단위로 읽는 중... (청크 크기: {chunk_size:,}개)")
                print(f"   총 행 수: {parquet_file.metadata.num_rows:,}개")
                
                result_dfs = []
                total_rows = 0
                
                for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                    batch_df = batch.to_pandas()
                    result_dfs.append(batch_df)
                    total_rows += len(batch_df)
                    
                    # 일정 개수마다 병합하여 메모리 절약
                    if len(result_dfs) >= 10:
                        temp_df = pd.concat(result_dfs, ignore_index=True)
                        result_dfs = [temp_df]
                    
                    if (i + 1) % 10 == 0:
                        print(f"   - {i + 1}개 배치 처리 완료 ({total_rows:,}개 행)")
                
                # 최종 병합
                if result_dfs:
                    logs_df = pd.concat(result_dfs, ignore_index=True)
                else:
                    logs_df = pd.DataFrame()
                
                print(f"✅ 파싱 데이터 로드 완료: {len(logs_df):,}개 로그 라인 (청크 방식)")
                return logs_df
            except ImportError:
                print("⚠️ pyarrow가 없어 청크 읽기를 사용할 수 없습니다. 전체 로드합니다.")
                chunk_size = None  # 전체 로드로 전환
        
        # 전체 로드 (기존 방식)
        if input_path.endswith('.parquet'):
            try:
                logs_df = pd.read_parquet(input_path, engine='pyarrow')
            except ImportError:
                raise ImportError(
                    "Parquet 파일을 읽으려면 pyarrow가 필요합니다.\n"
                    "설치 방법: pip install pyarrow\n"
                    "또는 Pickle 파일(.pkl)을 사용하세요."
                )
        elif input_path.endswith('.pkl'):
            logs_df = pd.read_pickle(input_path)
        else:
            # 확장자가 없으면 시도해봄
            try:
                logs_df = pd.read_parquet(input_path, engine='pyarrow')
            except (ImportError, Exception):
                try:
                    logs_df = pd.read_pickle(input_path)
                except Exception:
                    raise ValueError(f"지원하지 않는 파일 형식입니다: {input_path}")
        
        print(f"✅ 파싱 데이터 로드 완료: {len(logs_df):,}개 로그 라인")
        return logs_df
    
    def get_chunk_files(self, chunk_dir: str) -> list:
        """
        청크 디렉토리에서 청크 파일 목록 반환
        
        Args:
            chunk_dir: 청크 디렉토리 경로
            
        Returns:
            청크 파일 경로 리스트 (정렬됨)
        """
        if not os.path.exists(chunk_dir):
            return []
        
        chunk_files = []
        for ext in ['.parquet', '.pkl']:
            chunk_files.extend(glob.glob(os.path.join(chunk_dir, f'chunk_*{ext}')))
        
        # 파일명 기준 정렬
        chunk_files.sort()
        return chunk_files
    
    def load_from_chunks(self, chunk_dir: str, max_chunks: int = None) -> pd.DataFrame:
        """
        청크 파일에서 직접 데이터 로드 (메모리 효율적)
        
        Args:
            chunk_dir: 청크 디렉토리 경로
            max_chunks: 최대 로드할 청크 수 (None이면 전체)
            
        Returns:
            로드된 DataFrame
        """
        chunk_files = self.get_chunk_files(chunk_dir)
        
        if not chunk_files:
            print(f"⚠️ 청크 파일을 찾을 수 없습니다: {chunk_dir}")
            return pd.DataFrame()
        
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
        
        print(f"📂 청크 파일에서 데이터 로드 중... (총 {len(chunk_files)}개 파일)")
        
        # 스트리밍 방식으로 로드
        result_dfs = []
        for i, chunk_file in enumerate(chunk_files):
            try:
                if chunk_file.endswith('.parquet'):
                    chunk_df = pd.read_parquet(chunk_file, engine='pyarrow')
                else:
                    chunk_df = pd.read_pickle(chunk_file)
                
                result_dfs.append(chunk_df)
                
                # 일정 개수마다 병합하여 메모리 절약
                if len(result_dfs) >= 10:
                    temp_df = pd.concat(result_dfs, ignore_index=True)
                    result_dfs = [temp_df]
                
                if (i + 1) % 10 == 0:
                    print(f"  - {i + 1}/{len(chunk_files)}개 파일 로드 완료")
                    
            except Exception as e:
                print(f"  ⚠️ 청크 파일 읽기 오류 {chunk_file}: {e}")
                continue
        
        # 최종 병합
        if result_dfs:
            result_df = pd.concat(result_dfs, ignore_index=True)
            print(f"✅ 총 {len(result_df):,}개 로그 라인 로드 완료")
        else:
            result_df = pd.DataFrame()
        
        return result_df
    
    def prepare_data_streaming(self, input_path: str, output_dir: str, 
                               train_ratio=0.8, valid_ratio=0.2, 
                               chunk_size=100000):
        """
        Parquet 파일을 청크 단위로 읽으면서 train/valid/test로 스트리밍 분할
        메모리 효율적으로 처리하여 각 분할을 별도 파일로 저장
        
        Args:
            input_path: 입력 Parquet 파일 경로
            output_dir: 분할된 데이터 저장 디렉토리
            train_ratio: 학습 데이터 비율 (기본 0.8 = 80%)
            valid_ratio: 검증 데이터 비율 (train의 비율, 기본 0.2 = 20%)
            chunk_size: 청크 단위로 읽을 크기
            
        Returns:
            분할된 데이터 파일 경로 딕셔너리
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")
        
        if not input_path.endswith('.parquet'):
            raise ValueError("스트리밍 분할은 Parquet 파일만 지원합니다.")
        
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("스트리밍 분할을 위해 pyarrow가 필요합니다: pip install pyarrow")
        
        os.makedirs(output_dir, exist_ok=True)
        
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        
        print("=" * 60)
        print("스트리밍 방식 데이터 분할")
        print("=" * 60)
        print(f"   입력 파일: {input_path}")
        print(f"   총 행 수: {total_rows:,}개")
        print(f"   청크 크기: {chunk_size:,}개")
        print(f"   분할 비율: Train {train_ratio*100:.0f}% / Valid {valid_ratio*100:.0f}% (train의) / Test {100-train_ratio*100:.0f}%")
        print(f"   저장 디렉토리: {output_dir}")
        
        # 분할 경계 계산
        train_end_idx = int(total_rows * train_ratio)
        valid_start_idx = int(train_end_idx * (1 - valid_ratio))
        
        # 각 분할의 임시 파일 저장 디렉토리
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 각 분할의 임시 파일 리스트 (메모리 효율적)
        train_temp_files = []
        valid_temp_files = []
        test_temp_files = []
        
        # 청크를 일정 개수마다 파일로 저장 (메모리 절약)
        merge_chunk_count = 5  # 5개 청크마다 병합하여 파일로 저장
        
        # 각 분할의 현재 청크 리스트
        train_chunks = []
        valid_chunks = []
        test_chunks = []
        
        # 통계
        train_count = 0
        valid_count = 0
        test_count = 0
        current_idx = 0
        
        print(f"\n📊 청크 단위로 읽으면서 분할 중... (메모리 효율 모드)")
        print(f"   💡 {merge_chunk_count}개 청크마다 파일로 저장하여 메모리 절약")
        
        def save_chunks_to_file(chunks, temp_files, prefix):
            """청크들을 파일로 저장하고 메모리에서 해제"""
            if not chunks:
                return temp_files
            
            # 병합
            merged_df = pd.concat(chunks, ignore_index=True)
            chunks.clear()  # 메모리 해제
            
            # 임시 파일로 저장
            temp_file = os.path.join(temp_dir, f'{prefix}_{len(temp_files):06d}.parquet')
            merged_df.to_parquet(temp_file, compression='snappy', engine='pyarrow')
            temp_files.append(temp_file)
            del merged_df  # 메모리 해제
            
            return temp_files
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            batch_df = batch.to_pandas()
            batch_size = len(batch_df)
            batch_start = current_idx
            batch_end = current_idx + batch_size
            
            # 시간 순서 정렬 (이미 정렬되어 있어야 하지만 안전을 위해)
            batch_df = batch_df.sort_values('timestamp').reset_index(drop=True)
            
            # 분할
            if batch_end <= train_end_idx:
                # Train 영역
                if batch_start < valid_start_idx:
                    # Train (valid 제외)
                    train_chunks.append(batch_df)
                    train_count += batch_size
                    
                    # 일정 개수마다 파일로 저장
                    if len(train_chunks) >= merge_chunk_count:
                        train_temp_files = save_chunks_to_file(train_chunks, train_temp_files, 'train')
                else:
                    # Valid 영역
                    valid_chunks.append(batch_df)
                    valid_count += batch_size
                    
                    # 일정 개수마다 파일로 저장
                    if len(valid_chunks) >= merge_chunk_count:
                        valid_temp_files = save_chunks_to_file(valid_chunks, valid_temp_files, 'valid')
            else:
                # Test 영역
                if batch_start < train_end_idx:
                    # 일부는 train/valid, 일부는 test
                    split_in_batch = train_end_idx - batch_start
                    train_valid_part = batch_df.iloc[:split_in_batch]
                    test_part = batch_df.iloc[split_in_batch:]
                    
                    # Train/Valid 분할
                    if batch_start < valid_start_idx:
                        valid_split = valid_start_idx - batch_start
                        train_part = train_valid_part.iloc[:valid_split]
                        valid_part = train_valid_part.iloc[valid_split:]
                        train_chunks.append(train_part)
                        valid_chunks.append(valid_part)
                        train_count += len(train_part)
                        valid_count += len(valid_part)
                        
                        # 일정 개수마다 파일로 저장
                        if len(train_chunks) >= merge_chunk_count:
                            train_temp_files = save_chunks_to_file(train_chunks, train_temp_files, 'train')
                        if len(valid_chunks) >= merge_chunk_count:
                            valid_temp_files = save_chunks_to_file(valid_chunks, valid_temp_files, 'valid')
                    else:
                        valid_chunks.append(train_valid_part)
                        valid_count += len(train_valid_part)
                        
                        if len(valid_chunks) >= merge_chunk_count:
                            valid_temp_files = save_chunks_to_file(valid_chunks, valid_temp_files, 'valid')
                    
                    test_chunks.append(test_part)
                    test_count += len(test_part)
                    
                    if len(test_chunks) >= merge_chunk_count:
                        test_temp_files = save_chunks_to_file(test_chunks, test_temp_files, 'test')
                else:
                    # 전체 Test
                    test_chunks.append(batch_df)
                    test_count += batch_size
                    
                    if len(test_chunks) >= merge_chunk_count:
                        test_temp_files = save_chunks_to_file(test_chunks, test_temp_files, 'test')
            
            current_idx = batch_end
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0:
                print(f"   - {batch_idx + 1}개 배치 처리 완료 ({current_idx:,}/{total_rows:,}개 행, {current_idx/total_rows*100:.1f}%)")
                print(f"      메모리: Train 청크 {len(train_chunks)}개, Valid 청크 {len(valid_chunks)}개, Test 청크 {len(test_chunks)}개")
                print(f"      저장된 임시 파일: Train {len(train_temp_files)}개, Valid {len(valid_temp_files)}개, Test {len(test_temp_files)}개")
        
        print(f"\n📝 남은 청크 저장 및 최종 병합 중...")
        
        # 남은 청크들도 파일로 저장
        if train_chunks:
            train_temp_files = save_chunks_to_file(train_chunks, train_temp_files, 'train')
        if valid_chunks:
            valid_temp_files = save_chunks_to_file(valid_chunks, valid_temp_files, 'valid')
        if test_chunks:
            test_temp_files = save_chunks_to_file(test_chunks, test_temp_files, 'test')
        
        # 각 분할의 임시 파일들을 최종 파일로 병합 (스트리밍 방식)
        output_files = {}
        
        def merge_temp_files(temp_files, output_file, split_name):
            """임시 파일들을 스트리밍 방식으로 병합"""
            if not temp_files:
                return None
            
            print(f"   📦 {split_name} 병합 중... ({len(temp_files)}개 임시 파일)")
            
            # 스트리밍 방식으로 병합 (메모리 절약)
            result_dfs = []
            for i, temp_file in enumerate(temp_files):
                temp_df = pd.read_parquet(temp_file, engine='pyarrow')
                result_dfs.append(temp_df)
                
                # 일정 개수마다 병합하여 메모리 절약
                if len(result_dfs) >= 5:
                    merged = pd.concat(result_dfs, ignore_index=True)
                    result_dfs = [merged]
                
                # 임시 파일 삭제
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # 최종 병합
            if result_dfs:
                final_df = pd.concat(result_dfs, ignore_index=True)
                final_df = final_df.sort_values('timestamp').reset_index(drop=True)
                final_df.to_parquet(output_file, compression='snappy', engine='pyarrow')
                del final_df, result_dfs
                return output_file
            return None
        
        # Train 병합
        train_file = os.path.join(output_dir, 'train.parquet')
        if merge_temp_files(train_temp_files, train_file, 'Train'):
            output_files['train'] = train_file
            print(f"   ✅ Train 저장: {train_count:,}개 ({train_count/total_rows*100:.1f}%)")
        
        # Valid 병합
        valid_file = os.path.join(output_dir, 'valid.parquet')
        if merge_temp_files(valid_temp_files, valid_file, 'Valid'):
            output_files['valid'] = valid_file
            print(f"   ✅ Valid 저장: {valid_count:,}개 ({valid_count/total_rows*100:.1f}%)")
        
        # Test 병합
        test_file = os.path.join(output_dir, 'test.parquet')
        if merge_temp_files(test_temp_files, test_file, 'Test'):
            output_files['test'] = test_file
            print(f"   ✅ Test 저장: {test_count:,}개 ({test_count/total_rows*100:.1f}%)")
        
        # 임시 디렉토리 삭제
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
        print(f"\n✅ 스트리밍 분할 완료!")
        print(f"   저장 위치: {output_dir}")
        
        return output_files


class LogAnomalyDetector:
    """로그 이상치 탐지 클래스"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.scaler = StandardScaler()
        self.models = {}
        
    def extract_features(self, df):
        """로그 데이터에서 특징 추출"""
        if df.empty:
            return pd.DataFrame()
        
        # 시간대별 집계
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['date'] = df['timestamp'].dt.date
        
        # 시간 윈도우별 집계 (10분 단위)
        df['time_window'] = df['timestamp'].dt.floor('10T')
        
        # 집계
        features = []
        for window in df['time_window'].unique():
            window_df = df[df['time_window'] == window]
            
            feature = {
                'time_window': window,
                'total_logs': len(window_df),
                'error_count': window_df['is_error'].sum(),
                'warn_count': (window_df['level'] == 'WARN').sum(),
                'error_rate': window_df['is_error'].mean(),
                'warn_rate': (window_df['level'] == 'WARN').mean(),
                'unique_classes': window_df['class_path'].nunique(),
                'unique_threads': window_df['thread'].nunique(),
                'avg_message_length': window_df['message_length'].mean(),
                'exception_count': window_df['has_exception'].sum(),
                'exception_rate': window_df['has_exception'].mean(),
                'unique_files': window_df['file_path'].nunique(),
            }
            
            # 레벨별 카운트
            level_counts = window_df['level'].value_counts()
            for level in ['ERROR', 'WARN', 'INFO', 'DEBUG']:
                feature[f'{level.lower()}_count'] = level_counts.get(level, 0)
            
            # 가장 많이 나온 클래스
            top_class = window_df['class_path'].value_counts().head(1)
            if not top_class.empty:
                feature['top_class'] = top_class.index[0]
                feature['top_class_count'] = top_class.values[0]
            else:
                feature['top_class'] = ''
                feature['top_class_count'] = 0
            
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        
        # 클래스 경로를 숫자로 변환 (간단한 해시)
        if 'top_class' in features_df.columns:
            features_df['top_class_hash'] = features_df['top_class'].apply(
                lambda x: hash(x) % 1000 if x else 0
            )
        
        return features_df
    
    def calculate_baseline(self, features_df):
        """기준선 통계 계산"""
        if features_df.empty:
            return {}
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        baseline = {}
        
        for col in numeric_cols:
            baseline[f'{col}_mean'] = features_df[col].mean()
            baseline[f'{col}_std'] = features_df[col].std()
            baseline[f'{col}_median'] = features_df[col].median()
            baseline[f'{col}_q25'] = features_df[col].quantile(0.25)
            baseline[f'{col}_q75'] = features_df[col].quantile(0.75)
            baseline[f'{col}_q95'] = features_df[col].quantile(0.95)
        
        return baseline
    
    def detect_statistical_anomalies(self, features_df, threshold=3.0):
        """통계적 이상치 탐지 (Z-score 기반)"""
        if features_df.empty or not self.baseline_stats:
            return pd.DataFrame()
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        anomalies = []
        
        for idx, row in features_df.iterrows():
            anomaly_score = 0
            reasons = []
            
            for col in numeric_cols:
                mean_key = f'{col}_mean'
                std_key = f'{col}_std'
                
                if mean_key in self.baseline_stats and std_key in self.baseline_stats:
                    mean_val = self.baseline_stats[mean_key]
                    std_val = self.baseline_stats[std_key]
                    
                    if std_val > 0:
                        z_score = abs((row[col] - mean_val) / std_val)
                        if z_score > threshold:
                            anomaly_score += z_score
                            reasons.append(f"{col}: Z-score={z_score:.2f}")
            
            if anomaly_score > 0:
                anomalies.append({
                    'time_window': row['time_window'],
                    'anomaly_score': anomaly_score,
                    'reasons': '; '.join(reasons),
                    'features': row.to_dict()
                })
        
        return pd.DataFrame(anomalies)
    
    def detect_error_spikes(self, features_df, threshold_multiplier=5.0):
        """에러 급증 탐지"""
        if features_df.empty or not self.baseline_stats:
            return pd.DataFrame()
        
        baseline_error_rate = self.baseline_stats.get('error_rate_mean', 0)
        baseline_error_std = self.baseline_stats.get('error_rate_std', 0)
        
        if baseline_error_rate == 0:
            baseline_error_rate = 0.01  # 최소값
        
        spikes = []
        for idx, row in features_df.iterrows():
            current_error_rate = row['error_rate']
            
            if current_error_rate > baseline_error_rate * threshold_multiplier:
                spikes.append({
                    'time_window': row['time_window'],
                    'baseline_error_rate': baseline_error_rate,
                    'current_error_rate': current_error_rate,
                    'multiplier': current_error_rate / baseline_error_rate,
                    'error_count': row['error_count'],
                    'total_logs': row['total_logs']
                })
        
        return pd.DataFrame(spikes)
    
    def detect_unusual_patterns(self, df):
        """비정상적인 패턴 탐지"""
        anomalies = []
        
        # 1. 특정 클래스에서 에러 집중
        error_by_class = df[df['is_error']].groupby('class_path').size()
        if not error_by_class.empty:
            top_error_class = error_by_class.idxmax()
            error_count = error_by_class.max()
            total_errors = error_by_class.sum()
            
            if error_count > total_errors * 0.5:  # 전체 에러의 50% 이상이 한 클래스에서
                anomalies.append({
                    'type': 'error_concentration',
                    'class': top_error_class,
                    'error_count': error_count,
                    'total_errors': total_errors,
                    'percentage': (error_count / total_errors) * 100
                })
        
        # 2. 로그 빈도 이상 (너무 많거나 적음)
        if 'time_window' in df.columns:
            log_frequency = df.groupby('time_window').size()
            if not log_frequency.empty:
                mean_freq = log_frequency.mean()
                std_freq = log_frequency.std()
                
                for window, count in log_frequency.items():
                    if std_freq > 0:
                        z_score = abs((count - mean_freq) / std_freq)
                        if z_score > 3:
                            anomalies.append({
                                'type': 'frequency_anomaly',
                                'time_window': window,
                                'log_count': count,
                                'mean': mean_freq,
                                'z_score': z_score
                            })
        
        # 3. 새로운 예외 타입 탐지
        exception_patterns = df[df['has_exception']]['message'].apply(
            lambda x: re.search(r'(\w+Exception|\w+Error)', x)
        )
        exception_types = exception_patterns.dropna().apply(lambda x: x.group(1))
        
        if not exception_types.empty:
            exception_counts = exception_types.value_counts()
            # 전체의 1% 미만이면 새로운 예외로 간주
            total_exceptions = len(exception_types)
            for exc_type, count in exception_counts.items():
                if count < total_exceptions * 0.01 and count > 0:
                    anomalies.append({
                        'type': 'new_exception_type',
                        'exception_type': exc_type,
                        'count': count,
                        'percentage': (count / total_exceptions) * 100
                    })
        
        return pd.DataFrame(anomalies)
    
    def train_ml_model(self, features_df, model_type='isolation_forest'):
        """머신러닝 모델 학습"""
        if features_df.empty:
            return None
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].fillna(0)
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 선택 및 학습
        if model_type == 'isolation_forest':
            model = IForest(contamination=0.1, random_state=42)
        elif model_type == 'autoencoder':
            # AutoEncoder는 데이터 크기에 따라 파라미터 조정 필요
            n_samples, n_features = X_scaled.shape
            
            # 데이터가 너무 적으면 AutoEncoder 사용 불가
            if n_samples < 10:
                print(f"⚠️ AutoEncoder 학습 실패: 데이터가 너무 적습니다 ({n_samples}개 샘플)")
                print(f"   최소 10개 이상의 샘플이 필요합니다.")
                return None
            
            # 특징 수에 따라 hidden layer 크기 조정
            if n_features <= 5:
                hidden_neurons = [max(4, n_features), max(2, n_features//2), max(4, n_features)]
            elif n_features <= 10:
                hidden_neurons = [16, 8, 16]
            else:
                hidden_neurons = [64, 32, 16, 32, 64]
            
            # 샘플 수에 따라 epoch와 batch_size 조정
            if n_samples < 50:
                epoch_num = 20
                batch_size = min(8, n_samples)
            elif n_samples < 100:
                epoch_num = 30
                batch_size = 16
            else:
                epoch_num = 50
                batch_size = 32
            
            try:
                model = AutoEncoder(
                    contamination=0.1,
                    hidden_neurons=hidden_neurons,
                    epochs=epoch_num,
                    batch_size=batch_size,
                    dropout_rate=0.2,
                    verbose=0,  # 진행 상황 출력 비활성화
                    random_state=42
                )
            except TypeError:
                # 파라미터 이름이 다른 버전의 pyod일 수 있음
                try:
                    model = AutoEncoder(
                        contamination=0.1,
                        hidden_neuron_list=hidden_neurons,
                        epoch_num=epoch_num,
                        batch_size=batch_size,
                        dropout_rate=0.2,
                        verbose=0,
                        random_state=42
                    )
                except Exception as e:
                    print(f"⚠️ AutoEncoder 초기화 실패: {e}")
                    return None
        elif model_type == 'lof':
            model = LOF(contamination=0.1)
        else:
            model = IForest(contamination=0.1, random_state=42)
        
        try:
            model.fit(X_scaled)
            self.models[model_type] = model
            return model
        except Exception as e:
            print(f"⚠️ {model_type} 모델 학습 실패: {e}")
            print(f"   데이터 크기: {X_scaled.shape}")
            print(f"   모델을 건너뜁니다.")
            return None
    
    def predict_anomalies_ml(self, features_df, model_type='isolation_forest'):
        """머신러닝 모델로 이상치 예측"""
        if features_df.empty or model_type not in self.models:
            return pd.DataFrame()
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        model = self.models[model_type]
        predictions = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
        anomalies = features_df[predictions == 1].copy()
        anomalies['anomaly_score'] = -scores[predictions == 1]  # 음수 점수를 양수로
        
        return anomalies


class LogAnomalyDetectionSystem:
    """통합 로그 이상치 탐지 시스템"""
    
    def __init__(self, log_directory, max_files=None, sample_lines=None):
        self.log_directory = log_directory
        self.max_files = max_files
        self.sample_lines = sample_lines
        self.parser = SpringBootLogParser()
        self.detector = LogAnomalyDetector()
        self.logs_df = None
        self.features_df = None
        
    def load_logs(self):
        """로그 파일 로드 및 파싱"""
        print("=" * 60)
        print("로그 파일 로드 중...")
        print("=" * 60)
        
        self.logs_df = self.parser.parse_directory(
            self.log_directory, 
            max_files=self.max_files,
            sample_lines=self.sample_lines
        )
        
        if self.logs_df.empty:
            print("⚠️ 파싱된 로그가 없습니다.")
            return False
        
        print(f"\n✅ 총 {len(self.logs_df)}개 로그 라인 파싱 완료")
        print(f"   - 기간: {self.logs_df['timestamp'].min()} ~ {self.logs_df['timestamp'].max()}")
        print(f"   - 에러 로그: {self.logs_df['is_error'].sum()}개")
        print(f"   - 경고 로그: {(self.logs_df['level'] == 'WARN').sum()}개")
        
        return True
    
    def extract_features(self):
        """특징 추출"""
        print("\n" + "=" * 60)
        print("특징 추출 중...")
        print("=" * 60)
        
        self.features_df = self.detector.extract_features(self.logs_df)
        
        if self.features_df.empty:
            print("⚠️ 추출된 특징이 없습니다.")
            return False
        
        print(f"✅ {len(self.features_df)}개 시간 윈도우 특징 추출 완료")
        print(f"\n특징 통계:")
        print(self.features_df.describe())
        
        return True
    
    def train_baseline(self, train_ratio=0.8, validation_ratio=0.1):
        """
        기준선 학습 (80% 학습, 10% 검증, 10% 테스트)
        
        Args:
            train_ratio: 학습 데이터 비율 (기본 0.8 = 80%)
            validation_ratio: 검증 데이터 비율 (기본 0.1 = 10%, 나머지 10%는 테스트)
        """
        print("\n" + "=" * 60)
        print("기준선 학습 중...")
        print("=" * 60)
        
        if self.features_df.empty:
            print("⚠️ 특징 데이터가 없습니다.")
            return False
        
        # 시간순으로 정렬 후 분할
        self.features_df = self.features_df.sort_values('time_window')
        total_samples = len(self.features_df)
        
        # 80% 학습, 10% 검증, 10% 테스트로 분할
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * validation_ratio)
        
        train_df = self.features_df.iloc[:train_end]
        val_df = self.features_df.iloc[train_end:val_end]
        test_df = self.features_df.iloc[val_end:]
        
        # 기준선 통계 계산 (학습 데이터만 사용)
        self.detector.baseline_stats = self.detector.calculate_baseline(train_df)
        
        print("✅ 기준선 통계 계산 완료")
        print(f"   - 전체 데이터: {total_samples}개 윈도우")
        print(f"   - 학습 데이터: {len(train_df)}개 윈도우 ({len(train_df)/total_samples*100:.1f}%)")
        print(f"   - 검증 데이터: {len(val_df)}개 윈도우 ({len(val_df)/total_samples*100:.1f}%)")
        print(f"   - 테스트 데이터: {len(test_df)}개 윈도우 ({len(test_df)/total_samples*100:.1f}%)")
        
        # ML 모델 학습
        print("\n머신러닝 모델 학습 중...")
        models_trained = {}
        
        if self.detector.train_ml_model(train_df, model_type='isolation_forest'):
            print("   ✅ Isolation Forest 학습 완료")
            models_trained['isolation_forest'] = True
        else:
            print("   ⚠️ Isolation Forest 학습 실패")
            models_trained['isolation_forest'] = False
        
        if self.detector.train_ml_model(train_df, model_type='autoencoder'):
            print("   ✅ AutoEncoder 학습 완료")
            models_trained['autoencoder'] = True
        else:
            print("   ⚠️ AutoEncoder 학습 실패 (건너뜀)")
            models_trained['autoencoder'] = False
        
        print("✅ 모델 학습 완료")
        
        # 검증 데이터로 성능 평가
        print("\n" + "=" * 60)
        print("검증 데이터로 모델 성능 평가")
        print("=" * 60)
        
        validation_results = self._evaluate_models(val_df, models_trained)
        self._print_validation_results(validation_results)
        
        # 테스트 데이터 저장 (나중에 사용)
        self.test_df = test_df
        
        return True
    
    def _evaluate_models(self, val_df, models_trained):
        """검증 데이터로 모델 성능 평가"""
        results = {}
        
        # 정상/이상 라벨 생성 (에러율 기준)
        baseline_error_rate = self.detector.baseline_stats.get('error_rate_mean', 0)
        threshold = baseline_error_rate * 2  # 기준 에러율의 2배 이상이면 이상
        
        val_df = val_df.copy()
        val_df['true_label'] = (val_df['error_rate'] > threshold).astype(int)
        
        # 각 모델별 평가
        for model_type, is_trained in models_trained.items():
            if not is_trained or model_type not in self.detector.models:
                continue
            
            try:
                # 예측
                predictions = self.detector.predict_anomalies_ml(val_df, model_type=model_type)
                
                if predictions.empty:
                    continue
                
                # 예측 라벨 생성 (이상치로 탐지된 것)
                val_df['pred_label'] = 0
                val_df.loc[predictions.index, 'pred_label'] = 1
                
                # 성능 지표 계산
                true_labels = val_df['true_label'].values
                pred_labels = val_df['pred_label'].values
                
                # 라벨이 모두 같으면 평가 불가
                if len(set(true_labels)) == 1 and len(set(pred_labels)) == 1:
                    continue
                
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
                cm = confusion_matrix(true_labels, pred_labels)
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'true_anomalies': int(true_labels.sum()),
                    'predicted_anomalies': int(pred_labels.sum())
                }
            except Exception as e:
                print(f"   ⚠️ {model_type} 평가 실패: {e}")
                continue
        
        return results
    
    def _print_validation_results(self, validation_results):
        """검증 결과 출력"""
        if not validation_results:
            print("⚠️ 평가 가능한 모델이 없습니다.")
            return
        
        for model_type, metrics in validation_results.items():
            print(f"\n📊 {model_type.upper()} 모델 성능:")
            print(f"   정확도 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"   정밀도 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"   재현율 (Recall): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"   F1 점수: {metrics['f1_score']:.4f}")
            print(f"   실제 이상치: {metrics['true_anomalies']}개")
            print(f"   예측 이상치: {metrics['predicted_anomalies']}개")
            
            cm = metrics['confusion_matrix']
            print(f"   혼동 행렬:")
            print(f"      정상→정상: {cm[0][0]}, 정상→이상: {cm[0][1]}")
            print(f"      이상→정상: {cm[1][0]}, 이상→이상: {cm[1][1]}")
    
    def detect_all_anomalies(self, use_test_data=True):
        """
        모든 이상치 탐지
        
        Args:
            use_test_data: True면 테스트 데이터 사용, False면 전체 데이터 사용
        """
        print("\n" + "=" * 60)
        print("이상치 탐지 중...")
        print("=" * 60)
        
        # 테스트 데이터가 있으면 테스트 데이터만 사용
        if use_test_data and hasattr(self, 'test_df') and not self.test_df.empty:
            print("📝 테스트 데이터로 이상치 탐지 수행")
            test_features_df = self.test_df
        else:
            print("📝 전체 데이터로 이상치 탐지 수행")
            test_features_df = self.features_df
        
        results = {}
        
        # 1. 통계적 이상치
        print("\n1. 통계적 이상치 탐지...")
        stat_anomalies = self.detector.detect_statistical_anomalies(test_features_df)
        results['statistical'] = stat_anomalies
        print(f"   ✅ {len(stat_anomalies)}개 이상치 발견")
        
        # 2. 에러 급증
        print("\n2. 에러 급증 탐지...")
        error_spikes = self.detector.detect_error_spikes(test_features_df)
        results['error_spikes'] = error_spikes
        print(f"   ✅ {len(error_spikes)}개 에러 급증 발견")
        
        # 3. 비정상 패턴 (전체 로그 데이터 사용)
        print("\n3. 비정상 패턴 탐지...")
        unusual_patterns = self.detector.detect_unusual_patterns(self.logs_df)
        results['unusual_patterns'] = unusual_patterns
        print(f"   ✅ {len(unusual_patterns)}개 비정상 패턴 발견")
        
        # 4. ML 기반 이상치
        print("\n4. 머신러닝 기반 이상치 탐지...")
        ml_anomalies_if = self.detector.predict_anomalies_ml(
            test_features_df, model_type='isolation_forest'
        )
        results['ml_isolation_forest'] = ml_anomalies_if
        print(f"   ✅ Isolation Forest: {len(ml_anomalies_if)}개 이상치")
        
        # AutoEncoder는 학습이 성공한 경우에만 실행
        if 'autoencoder' in self.detector.models:
            ml_anomalies_ae = self.detector.predict_anomalies_ml(
                test_features_df, model_type='autoencoder'
            )
            results['ml_autoencoder'] = ml_anomalies_ae
            print(f"   ✅ AutoEncoder: {len(ml_anomalies_ae)}개 이상치")
        else:
            results['ml_autoencoder'] = pd.DataFrame()
            print(f"   ⚠️ AutoEncoder: 학습되지 않아 건너뜀")
        
        return results
    
    def save_model(self, model_path):
        """학습된 모델과 기준선 저장"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'baseline_stats': self.detector.baseline_stats,
            'scaler': self.detector.scaler,
            'models': self.detector.models
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 모델 저장 완료: {model_path}")
    
    def load_model(self, model_path):
        """저장된 모델과 기준선 로드"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.detector.baseline_stats = model_data['baseline_stats']
        self.detector.scaler = model_data['scaler']
        self.detector.models = model_data['models']
        
        print(f"✅ 모델 로드 완료: {model_path}")
    
    def detect_anomalies_on_new_data(self, new_log_directory, max_files=None, sample_lines=None):
        """
        새로운 로그 데이터에 대해 이상치 탐지
        
        Args:
            new_log_directory: 새로운 로그 파일이 있는 디렉토리 경로
            max_files: 최대 처리할 파일 수
            sample_lines: 파일당 최대 처리할 라인 수
        
        Returns:
            dict: 이상치 탐지 결과
        """
        print("=" * 60)
        print("새로운 로그 데이터 이상치 탐지")
        print("=" * 60)
        
        # 새로운 로그 파싱
        print("\n새로운 로그 파일 파싱 중...")
        new_logs_df = self.parser.parse_directory(
            new_log_directory,
            max_files=max_files,
            sample_lines=sample_lines
        )
        
        if new_logs_df.empty:
            print("⚠️ 파싱된 로그가 없습니다.")
            return {}
        
        print(f"✅ {len(new_logs_df)}개 로그 라인 파싱 완료")
        
        # 특징 추출
        print("\n특징 추출 중...")
        new_features_df = self.detector.extract_features(new_logs_df)
        
        if new_features_df.empty:
            print("⚠️ 추출된 특징이 없습니다.")
            return {}
        
        print(f"✅ {len(new_features_df)}개 시간 윈도우 특징 추출 완료")
        
        # 이상치 탐지
        print("\n이상치 탐지 중...")
        results = {}
        
        # 1. 통계적 이상치
        stat_anomalies = self.detector.detect_statistical_anomalies(new_features_df)
        results['statistical'] = stat_anomalies
        print(f"   ✅ 통계적 이상치: {len(stat_anomalies)}개")
        
        # 2. 에러 급증
        error_spikes = self.detector.detect_error_spikes(new_features_df)
        results['error_spikes'] = error_spikes
        print(f"   ✅ 에러 급증: {len(error_spikes)}개")
        
        # 3. 비정상 패턴
        unusual_patterns = self.detector.detect_unusual_patterns(new_logs_df)
        results['unusual_patterns'] = unusual_patterns
        print(f"   ✅ 비정상 패턴: {len(unusual_patterns)}개")
        
        # 4. ML 기반 이상치
        if 'isolation_forest' in self.detector.models:
            ml_anomalies_if = self.detector.predict_anomalies_ml(
                new_features_df, model_type='isolation_forest'
            )
            results['ml_isolation_forest'] = ml_anomalies_if
            print(f"   ✅ ML 이상치 (IF): {len(ml_anomalies_if)}개")
        else:
            results['ml_isolation_forest'] = pd.DataFrame()
            print(f"   ⚠️ ML 모델이 학습되지 않음")
        
        if 'autoencoder' in self.detector.models:
            ml_anomalies_ae = self.detector.predict_anomalies_ml(
                new_features_df, model_type='autoencoder'
            )
            results['ml_autoencoder'] = ml_anomalies_ae
            print(f"   ✅ ML 이상치 (AE): {len(ml_anomalies_ae)}개")
        else:
            results['ml_autoencoder'] = pd.DataFrame()
        
        return results
    
    def generate_report(self, results):
        """결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("이상치 탐지 결과 리포트")
        print("=" * 60)
        
        # 통계적 이상치
        if not results['statistical'].empty:
            print("\n📊 통계적 이상치:")
            for idx, row in results['statistical'].head(10).iterrows():
                print(f"   시간: {row['time_window']}")
                print(f"   이상 점수: {row['anomaly_score']:.2f}")
                print(f"   이유: {row['reasons']}")
                print()
        
        # 에러 급증
        if not results['error_spikes'].empty:
            print("\n🚨 에러 급증:")
            for idx, row in results['error_spikes'].head(10).iterrows():
                print(f"   시간: {row['time_window']}")
                print(f"   기준 에러율: {row['baseline_error_rate']:.2%}")
                print(f"   현재 에러율: {row['current_error_rate']:.2%}")
                print(f"   배수: {row['multiplier']:.1f}배")
                print(f"   에러 수: {row['error_count']}개 / 총 {row['total_logs']}개")
                print()
        
        # 비정상 패턴
        if not results['unusual_patterns'].empty:
            print("\n⚠️ 비정상 패턴:")
            for idx, row in results['unusual_patterns'].iterrows():
                if row['type'] == 'error_concentration':
                    print(f"   에러 집중: {row['class']}에서 {row['error_count']}개 ({row['percentage']:.1f}%)")
                elif row['type'] == 'frequency_anomaly':
                    print(f"   로그 빈도 이상: {row['time_window']} (Z-score: {row['z_score']:.2f})")
                elif row['type'] == 'new_exception_type':
                    print(f"   새로운 예외: {row['exception_type']} ({row['count']}회)")
                print()
        
        # ML 기반 이상치
        if not results['ml_isolation_forest'].empty:
            print("\n🤖 ML 기반 이상치 (Isolation Forest):")
            for idx, row in results['ml_isolation_forest'].head(10).iterrows():
                print(f"   시간: {row['time_window']}")
                print(f"   이상 점수: {row['anomaly_score']:.2f}")
                print(f"   에러 수: {row['error_count']}개")
                print()
        
        # 요약
        print("\n" + "=" * 60)
        print("요약")
        print("=" * 60)
        print(f"통계적 이상치: {len(results['statistical'])}개")
        print(f"에러 급증: {len(results['error_spikes'])}개")
        print(f"비정상 패턴: {len(results['unusual_patterns'])}개")
        print(f"ML 이상치 (IF): {len(results['ml_isolation_forest'])}개")
        print(f"ML 이상치 (AE): {len(results['ml_autoencoder'])}개")
        
        return results


def main():
    """메인 실행 함수"""
    log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    # 샘플링 옵션 (전체 분석을 원하면 None으로 설정)
    MAX_FILES = None  # None으로 설정하면 전체 파일 처리 (기존: 5개만 처리)
    SAMPLE_LINES = None  # None으로 설정하면 전체 라인 처리 (기존: 10000줄만 처리)
    
    # 시스템 초기화
    system = LogAnomalyDetectionSystem(
        log_directory,
        max_files=MAX_FILES,
        sample_lines=SAMPLE_LINES
    )
    
    # 로그 로드
    if not system.load_logs():
        return
    
    # 특징 추출
    if not system.extract_features():
        return
    
    # 기준선 학습
    if not system.train_baseline():
        return
    
    # 이상치 탐지 (테스트 데이터 사용)
    results = system.detect_all_anomalies(use_test_data=True)
    
    # 리포트 생성
    system.generate_report(results)
    
    # 결과 저장
    output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in results.items():
        if not df.empty:
            output_path = os.path.join(output_dir, f"anomalies_{name}.csv")
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 결과 저장: {output_path}")
    
    # 모델 저장
    model_path = os.path.join(output_dir, "trained_model.pkl")
    system.save_model(model_path)


def test_new_logs():
    """새로운 로그 파일에 대해 이상치 탐지 테스트"""
    # 학습된 모델 경로
    model_path = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results/trained_model.pkl"
    
    # 새로운 로그 디렉토리 (예시)
    new_log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    # 모델이 없으면 먼저 학습 필요
    if not os.path.exists(model_path):
        print("⚠️ 학습된 모델이 없습니다. 먼저 main()을 실행하여 모델을 학습하세요.")
        return
    
    # 시스템 초기화
    system = LogAnomalyDetectionSystem(new_log_directory)
    
    # 모델 로드
    system.load_model(model_path)
    
    # 새로운 로그 데이터로 이상치 탐지
    # 예: 최근 3개 파일만 테스트
    results = system.detect_anomalies_on_new_data(
        new_log_directory,
        max_files=3,  # 처음 3개 파일만 테스트
        sample_lines=5000  # 파일당 5000줄만 처리
    )
    
    # 리포트 생성
    system.generate_report(results)
    
    # 결과 저장
    output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, df in results.items():
        if not df.empty:
            output_path = os.path.join(output_dir, f"test_anomalies_{name}_{timestamp}.csv")
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 테스트 결과 저장: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 테스트 모드: 새로운 로그에 대해 이상치 탐지
        test_new_logs()
    else:
        # 학습 모드: 모델 학습 및 저장
        main()

