"""
로그 이상치 심각도 평가 시스템
로그 레벨과 예외 유형에 따른 위험 가중치를 기반으로 심각도를 계산합니다.
"""

import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class SeverityAssessment:
    """심각도 평가 클래스"""
    
    # 로그 레벨별 위험 가중치 (0~10)
    LEVEL_WEIGHTS = {
        'FATAL': 10,    # 최고 위험: 시스템이 사용 불가능한 상태
        'ERROR': 8,     # 고위험: 요청 실패, 예외 발생 등 기능 수행 불가
        'WARN': 5,      # 중위험: 잠재적 오류, 디스크 공간 부족, API 지연
        'INFO': 1,      # 저위험: 정상 작동 상태, 주요 이벤트 기록
        'DEBUG': 0,     # 디버깅 정보 (이상 탐지 시 주로 무시)
        'TRACE': 0,     # 상세 추적 정보
    }
    
    # 예외 유형별 위험 가중치
    EXCEPTION_WEIGHTS = {
        # JVM 치명적 오류 (Critical: 10)
        'OutOfMemoryError': 10,
        'StackOverflowError': 10,
        'VirtualMachineError': 10,
        'NoClassDefFoundError': 10,
        'ClassNotFoundException': 10,
        
        # 데이터베이스/연동 오류 (High: 8)
        'SQLException': 8,
        'ConnectException': 8,
        'SocketTimeoutException': 8,
        'ConnectionException': 8,
        'DatabaseException': 8,
        'JDBCException': 8,
        'SQLTimeoutException': 8,
        'SocketException': 8,
        'IOException': 8,
        'NetworkException': 8,
        
        # 런타임 코드 오류 (Medium: 6)
        'NullPointerException': 6,
        'IndexOutOfBoundsException': 6,
        'ClassCastException': 6,
        'ArrayIndexOutOfBoundsException': 6,
        'IllegalStateException': 6,
        'ConcurrentModificationException': 6,
        'UnsupportedOperationException': 6,
        'RuntimeException': 6,
        
        # 비즈니스 로직 오류 (Low: 4)
        'IllegalArgumentException': 4,
        'FileNotFoundException': 4,
        'ValidationException': 4,
        'BusinessException': 4,
        'InvalidParameterException': 4,
        'BadRequestException': 4,
    }
    
    # HTTP 상태 코드별 위험 가중치
    HTTP_STATUS_WEIGHTS = {
        # 5xx 서버 오류 (Critical/High: 9-10)
        '500': 10,  # Internal Server Error - 서버 내부 오류
        '503': 10,  # Service Unavailable - 서비스 불가
        '504': 9,   # Gateway Timeout - 게이트웨이 타임아웃
        '502': 9,   # Bad Gateway - 게이트웨이 오류
        '501': 8,   # Not Implemented - 미구현
        '505': 8,   # HTTP Version Not Supported
        
        # 4xx 클라이언트 오류 (Medium/Low: 4-7)
        '401': 7,   # Unauthorized - 인증 실패 (보안 이슈)
        '403': 7,   # Forbidden - 접근 금지 (보안 이슈)
        '429': 6,   # Too Many Requests - 과도한 요청
        '408': 6,   # Request Timeout - 요청 타임아웃
        '400': 5,   # Bad Request - 잘못된 요청
        '404': 4,   # Not Found - 리소스 없음 (일반적)
        '405': 4,   # Method Not Allowed - 메서드 불가
        '406': 4,   # Not Acceptable - 허용 불가
        '409': 5,   # Conflict - 충돌
        '410': 4,   # Gone - 영구적으로 없음
        '411': 4,   # Length Required
        '412': 5,   # Precondition Failed
        '413': 5,   # Payload Too Large
        '414': 4,   # URI Too Long
        '415': 4,   # Unsupported Media Type
        '416': 4,   # Range Not Satisfiable
        '417': 4,   # Expectation Failed
        '422': 5,   # Unprocessable Entity
        '423': 5,   # Locked
        '424': 5,   # Failed Dependency
        '426': 4,   # Upgrade Required
        '428': 5,   # Precondition Required
        '431': 5,   # Request Header Fields Too Large
        '451': 6,   # Unavailable For Legal Reasons
        
        # 3xx 리다이렉션 (Low: 1-2)
        '301': 1,   # Moved Permanently
        '302': 1,   # Found
        '303': 1,   # See Other
        '304': 0,   # Not Modified (정상)
        '307': 1,   # Temporary Redirect
        '308': 1,   # Permanent Redirect
        
        # 2xx 성공 (Info: 0)
        '200': 0,   # OK - 정상
        '201': 0,   # Created - 생성됨
        '202': 0,   # Accepted - 수락됨
        '204': 0,   # No Content - 내용 없음
        '206': 0,   # Partial Content - 부분 내용
    }
    
    # 예외 유형 패턴 (정규식)
    EXCEPTION_PATTERNS = {
        'jvm_critical': re.compile(
            r'(OutOfMemoryError|StackOverflowError|VirtualMachineError|'
            r'NoClassDefFoundError|ClassNotFoundException)',
            re.IGNORECASE
        ),
        'database_connection': re.compile(
            r'(SQLException|ConnectException|SocketTimeoutException|'
            r'ConnectionException|DatabaseException|JDBCException|'
            r'SQLTimeoutException|SocketException|IOException|NetworkException)',
            re.IGNORECASE
        ),
        'runtime_error': re.compile(
            r'(NullPointerException|IndexOutOfBoundsException|'
            r'ClassCastException|ArrayIndexOutOfBoundsException|'
            r'IllegalStateException|ConcurrentModificationException|'
            r'UnsupportedOperationException|RuntimeException)',
            re.IGNORECASE
        ),
        'business_logic': re.compile(
            r'(IllegalArgumentException|FileNotFoundException|'
            r'ValidationException|BusinessException|'
            r'InvalidParameterException|BadRequestException)',
            re.IGNORECASE
        ),
    }
    
    # HTTP 상태 코드 패턴 (정규식)
    HTTP_STATUS_PATTERN = re.compile(
        r'\b(?:HTTP/[\d.]+)?\s*(\d{3})\b|'  # HTTP/1.1 404 또는 단독 404
        r'status[:\s]+(\d{3})|'              # status: 404 또는 status 404
        r'code[:\s]+(\d{3})|'                # code: 404 또는 code 404
        r'response[:\s]+(\d{3})|'            # response: 404
        r'returned[:\s]+(\d{3})',            # returned: 404
        re.IGNORECASE
    )
    
    def __init__(self):
        """초기화"""
        pass
    
    def extract_http_status_code(self, message: str) -> Optional[Tuple[str, int]]:
        """
        메시지에서 HTTP 상태 코드 추출 및 가중치 반환
        
        Args:
            message: 로그 메시지
        
        Returns:
            (HTTP 상태 코드, 가중치) 튜플 또는 None
        """
        if not message:
            return None
        
        match = self.HTTP_STATUS_PATTERN.search(message)
        if match:
            # 여러 그룹 중 첫 번째 매칭된 상태 코드 사용
            status_code = None
            for group in match.groups():
                if group:
                    status_code = group
                    break
            
            if status_code:
                weight = self.HTTP_STATUS_WEIGHTS.get(status_code, 0)
                if weight > 0:
                    return (f'HTTP_{status_code}', weight)
        
        return None
    
    def extract_exception_type(self, message: str) -> Optional[Tuple[str, int]]:
        """
        메시지에서 예외 유형 추출 및 가중치 반환
        
        Args:
            message: 로그 메시지
        
        Returns:
            (예외 타입, 가중치) 튜플 또는 None
        """
        if not message:
            return None
        
        # 우선순위 순서대로 확인 (높은 위험부터)
        for category, pattern in self.EXCEPTION_PATTERNS.items():
            match = pattern.search(message)
            if match:
                exception_name = match.group(1)
                weight = self.EXCEPTION_WEIGHTS.get(exception_name, 0)
                if weight > 0:
                    return (exception_name, weight)
        
        # 패턴 매칭 실패 시 일반적인 Exception/Error 키워드 확인
        if 'Exception' in message or 'Error' in message:
            # 기본 가중치 (중간 수준)
            return ('UnknownException', 5)
        
        return None
    
    def calculate_log_severity(self, level: str, message: str = '') -> Dict:
        """
        단일 로그 라인의 심각도 계산
        
        Args:
            level: 로그 레벨 (FATAL, ERROR, WARN, INFO, DEBUG, TRACE)
            message: 로그 메시지
        
        Returns:
            심각도 정보 딕셔너리
        """
        # 레벨 가중치
        level_weight = self.LEVEL_WEIGHTS.get(level.upper(), 0)
        
        # HTTP 상태 코드 가중치 (우선순위 1)
        http_status_info = self.extract_http_status_code(message)
        if http_status_info:
            http_status_type, http_status_weight = http_status_info
        else:
            http_status_type = None
            http_status_weight = 0
        
        # 예외 유형 가중치 (우선순위 2)
        exception_info = self.extract_exception_type(message)
        if exception_info:
            exception_type, exception_weight = exception_info
        else:
            exception_type = None
            exception_weight = 0
        
        # 최종 심각도 점수 (레벨, HTTP 상태 코드, 예외 중 가장 높은 값 사용)
        severity_score = max(level_weight, http_status_weight, exception_weight)
        
        # 심각도 등급
        if severity_score >= 9:
            severity_level = 'CRITICAL'
            severity_description = '치명적: 즉각 조치 필요'
        elif severity_score >= 7:
            severity_level = 'HIGH'
            severity_description = '고위험: 빠른 조치 필요'
        elif severity_score >= 5:
            severity_level = 'MEDIUM'
            severity_description = '중위험: 모니터링 및 조치 검토'
        elif severity_score >= 1:
            severity_level = 'LOW'
            severity_description = '저위험: 정상 범위 내'
        else:
            severity_level = 'INFO'
            severity_description = '정보성: 이상 탐지 무시 가능'
        
        return {
            'severity_score': severity_score,
            'severity_level': severity_level,
            'severity_description': severity_description,
            'level_weight': level_weight,
            'http_status_code': http_status_type,
            'http_status_weight': http_status_weight,
            'exception_type': exception_type,
            'exception_weight': exception_weight,
        }
    
    def assess_anomaly_severity(self, anomaly_logs: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 탐지된 로그들의 심각도 평가
        
        Args:
            anomaly_logs: 이상치로 탐지된 로그 DataFrame
                          (level, message 컬럼 필요)
        
        Returns:
            심각도 정보가 추가된 DataFrame
        """
        if anomaly_logs.empty:
            return anomaly_logs
        
        result_df = anomaly_logs.copy()
        
        # 각 로그 라인에 대한 심각도 계산
        severity_info = []
        for idx, row in result_df.iterrows():
            level = row.get('level', 'INFO')
            message = str(row.get('message', ''))
            
            severity = self.calculate_log_severity(level, message)
            severity_info.append(severity)
        
        # 심각도 정보를 DataFrame에 추가
        severity_df = pd.DataFrame(severity_info)
        for col in severity_df.columns:
            result_df[col] = severity_df[col]
        
        return result_df
    
    def assess_time_window_severity(self, window_logs: pd.DataFrame) -> Dict:
        """
        시간 윈도우 내 로그들의 종합 심각도 평가
        
        Args:
            window_logs: 시간 윈도우 내의 로그 DataFrame
        
        Returns:
            종합 심각도 정보
        """
        if window_logs.empty:
            return {
                'max_severity_score': 0,
                'max_severity_level': 'INFO',
                'avg_severity_score': 0,
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
            }
        
        # 각 로그의 심각도 계산
        severity_scores = []
        severity_levels = []
        
        for idx, row in window_logs.iterrows():
            level = row.get('level', 'INFO')
            message = str(row.get('message', ''))
            severity = self.calculate_log_severity(level, message)
            severity_scores.append(severity['severity_score'])
            severity_levels.append(severity['severity_level'])
        
        # 통계 계산
        max_severity_score = max(severity_scores) if severity_scores else 0
        avg_severity_score = np.mean(severity_scores) if severity_scores else 0
        
        # 최고 심각도 등급
        if max_severity_score >= 9:
            max_severity_level = 'CRITICAL'
        elif max_severity_score >= 7:
            max_severity_level = 'HIGH'
        elif max_severity_score >= 5:
            max_severity_level = 'MEDIUM'
        elif max_severity_score >= 1:
            max_severity_level = 'LOW'
        else:
            max_severity_level = 'INFO'
        
        # 등급별 카운트
        level_counts = pd.Series(severity_levels).value_counts()
        
        return {
            'max_severity_score': max_severity_score,
            'max_severity_level': max_severity_level,
            'avg_severity_score': avg_severity_score,
            'critical_count': level_counts.get('CRITICAL', 0),
            'high_count': level_counts.get('HIGH', 0),
            'medium_count': level_counts.get('MEDIUM', 0),
            'low_count': level_counts.get('LOW', 0),
            'info_count': level_counts.get('INFO', 0),
        }
    
    def prioritize_anomalies(self, anomalies_df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치를 심각도에 따라 우선순위 정렬
        
        Args:
            anomalies_df: 심각도 정보가 포함된 이상치 DataFrame
        
        Returns:
            우선순위 정렬된 DataFrame
        """
        if anomalies_df.empty or 'severity_score' not in anomalies_df.columns:
            return anomalies_df
        
        # 심각도 점수 기준 내림차순 정렬
        sorted_df = anomalies_df.sort_values(
            'severity_score',
            ascending=False,
            na_position='last'
        )
        
        # 우선순위 추가
        sorted_df['priority'] = range(1, len(sorted_df) + 1)
        
        return sorted_df
    
    def generate_severity_summary(self, anomalies_df: pd.DataFrame) -> Dict:
        """
        심각도 요약 통계 생성
        
        Args:
            anomalies_df: 심각도 정보가 포함된 이상치 DataFrame
        
        Returns:
            요약 통계 딕셔너리
        """
        if anomalies_df.empty or 'severity_level' not in anomalies_df.columns:
            return {}
        
        summary = {
            'total_anomalies': len(anomalies_df),
            'by_severity': anomalies_df['severity_level'].value_counts().to_dict(),
            'avg_severity_score': anomalies_df['severity_score'].mean() if 'severity_score' in anomalies_df.columns else 0,
            'max_severity_score': anomalies_df['severity_score'].max() if 'severity_score' in anomalies_df.columns else 0,
        }
        
        # 예외 유형별 통계
        if 'exception_type' in anomalies_df.columns:
            exception_counts = anomalies_df['exception_type'].value_counts()
            summary['top_exceptions'] = exception_counts.head(10).to_dict()
        
        return summary


def add_severity_to_anomaly_results(
    anomaly_results: Dict,
    original_logs_df: pd.DataFrame
) -> Dict:
    """
    이상치 탐지 결과에 심각도 정보 추가
    
    Args:
        anomaly_results: 이상치 탐지 결과 딕셔너리
        original_logs_df: 원본 로그 DataFrame
    
    Returns:
        심각도 정보가 추가된 결과 딕셔너리
    """
    assessor = SeverityAssessment()
    
    enhanced_results = {}
    
    for key, value in anomaly_results.items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            # 시간 윈도우 기반 결과인 경우
            if 'time_window' in value.columns:
                # 해당 시간 윈도우의 로그들을 가져와서 심각도 평가
                enhanced_df = value.copy()
                
                severity_info_list = []
                for idx, row in enhanced_df.iterrows():
                    time_window = row['time_window']
                    window_logs = original_logs_df[
                        original_logs_df['timestamp'].dt.floor('10T') == time_window
                    ]
                    
                    severity_info = assessor.assess_time_window_severity(window_logs)
                    severity_info_list.append(severity_info)
                
                severity_df = pd.DataFrame(severity_info_list)
                for col in severity_df.columns:
                    enhanced_df[col] = severity_df[col]
                
                enhanced_results[key] = enhanced_df
            
            # 개별 로그 라인 기반 결과인 경우
            elif 'level' in value.columns and 'message' in value.columns:
                enhanced_results[key] = assessor.assess_anomaly_severity(value)
            
            else:
                enhanced_results[key] = value
        
        else:
            enhanced_results[key] = value
    
    return enhanced_results














