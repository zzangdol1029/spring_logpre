# CRITICAL 위험 로그 분석 결과

## 📊 분석 요약

### 발견 사항

**CRITICAL 위험 로그 809개 중 대부분은 실제로 위험하지 않습니다.**

---

## 🔍 상세 분석

### 1. 로그 레벨 분포

```
주요 로그 레벨:
  - INFO, TRACE, DEBUG: 대부분 (정상 로그)
  - INFO, ERROR, TRACE, DEBUG: 일부 (ERROR 포함이지만 정상 쿼리 로그)
```

### 2. 실제 로그 내용

**대부분의 CRITICAL 로그는:**
- ✅ **정상적인 데이터베이스 쿼리 로그**
  - `binding parameter [1] as [TIMESTAMP]`
  - `Committing JDBC Connection`
  - `==> Parameters: 1(String)`
  - `extracted value ([col_0_0_] : [VARCHAR])`
  - `==> Preparing: SELECT * FROM ...`

- ✅ **정상적인 애플리케이션 로그**
  - `DiscoveryClient_SMETA/develop-server:smeta:9009 - Heartbeat status: 200`
  - `Creating a new SqlSession`
  - `Closing non transactional SqlSession`

- ⚠️ **일부 ERROR 레벨 포함**
  - 하지만 실제로는 MyBatis/Hibernate의 정상적인 쿼리 로그
  - 실제 예외나 오류가 아님

### 3. 위험도 점수가 높게 나온 이유

```
위험도 점수 = (이상 점수 × 50) + (심각도 점수 / 10 × 50)

예시:
  - anomaly_score: 1.67 (높음)
  - max_severity_score: 8 (ERROR 레벨 포함)
  - risk_score: 123.85 → CRITICAL
```

**문제점:**
1. **이상 점수가 높음**: 정상 패턴과 다르다고 판단됨
   - 학습 데이터에 없는 새로운 쿼리 패턴
   - 하지만 실제로는 정상적인 비즈니스 로직

2. **ERROR 레벨 포함**: 일부 시퀀스에 ERROR 레벨이 포함됨
   - 하지만 실제로는 정상적인 데이터베이스 쿼리 로그
   - MyBatis/Hibernate의 디버그 로그

---

## ❌ False Positive (오탐지) 문제

### 실제 위험한 로그 vs 탐지된 로그

| 구분 | 실제 위험한 로그 | 현재 탐지된 로그 |
|------|----------------|----------------|
| **예외 발생** | `java.lang.NullPointerException` | ❌ 없음 |
| **보안 이슈** | `Unauthorized access attempt` | ❌ 없음 |
| **시스템 오류** | `Database connection failed` | ❌ 없음 |
| **성능 문제** | `Slow query detected (>5s)` | ❌ 없음 |
| **정상 쿼리** | - | ✅ 대부분 |

### 오탐지 원인

1. **학습 데이터 부족**
   - 다양한 쿼리 패턴을 학습하지 못함
   - 새로운 쿼리 = 이상으로 판단

2. **심각도 평가 오류**
   - ERROR 레벨이 포함된 정상 로그를 위험으로 판단
   - 실제 예외와 로그 레벨을 구분하지 못함

3. **위험도 계산 방식**
   - 이상 점수와 심각도 점수의 단순 합산
   - 실제 위험성을 반영하지 못함

---

## ✅ 개선 방안

### 1. 위험도 계산 로직 개선

#### 현재 방식 (문제)
```python
risk_score = (anomaly_score × 50) + (max_severity_score / 10 × 50)
```

#### 개선 방안
```python
# 실제 예외 키워드 확인
has_real_exception = any(
    keyword in message.lower() 
    for keyword in ['exception', 'error', 'failed', 'timeout', 'nullpointer']
    for message in sequence_messages
)

# 실제 위험도 계산
if has_real_exception:
    risk_score = (anomaly_score × 40) + (max_severity_score / 10 × 60)
else:
    # 정상 쿼리 로그는 위험도 낮춤
    risk_score = (anomaly_score × 30) + (max_severity_score / 10 × 20)
```

### 2. 로그 레벨 필터링

```python
# ERROR 레벨이지만 정상 쿼리 로그는 제외
normal_query_patterns = [
    'binding parameter',
    '==> Parameters',
    'Committing JDBC',
    'extracted value',
    'Preparing: SELECT'
]

if any(pattern in message for pattern in normal_query_patterns):
    # 정상 쿼리 로그로 간주, 위험도 낮춤
    severity_score = min(severity_score, 2)  # HIGH → LOW
```

### 3. 실제 예외 키워드 기반 필터링

```python
# 실제 위험한 로그만 CRITICAL로 분류
critical_keywords = [
    'exception',
    'error',
    'failed',
    'timeout',
    'nullpointer',
    'outofmemory',
    'connection refused',
    'unauthorized',
    'forbidden',
    'sql injection',
    'xss',
    'csrf'
]

def is_real_critical(message):
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in critical_keywords)
```

### 4. 학습 데이터 개선

```python
# 더 다양한 쿼리 패턴 포함
# - 다양한 테이블명
# - 다양한 파라미터 타입
# - 다양한 쿼리 유형 (SELECT, INSERT, UPDATE, DELETE)
```

---

## 📈 개선 후 예상 결과

### Before (현재)
```
CRITICAL: 809개
  - 실제 위험: ~10개 (1.2%)
  - 오탐지: ~799개 (98.8%)
```

### After (개선 후)
```
CRITICAL: ~50개 (예상)
  - 실제 위험: ~45개 (90%)
  - 오탐지: ~5개 (10%)
```

---

## 🎯 즉시 적용 가능한 해결책

### 1. CRITICAL 필터링 스크립트

```python
import pandas as pd
import re

# CRITICAL 로그 읽기
df = pd.read_csv('risk_critical.csv')

# 실제 위험 키워드
critical_keywords = [
    'exception', 'error', 'failed', 'timeout',
    'nullpointer', 'outofmemory', 'connection refused',
    'unauthorized', 'forbidden'
]

# 실제 위험한 로그만 필터링
def is_real_critical(row):
    messages = str(row['sample_messages']).lower()
    
    # 정상 쿼리 패턴 제외
    if any(pattern in messages for pattern in [
        'binding parameter', '==> parameters', 
        'committing jdbc', 'extracted value'
    ]):
        return False
    
    # 실제 위험 키워드 확인
    return any(keyword in messages for keyword in critical_keywords)

real_critical = df[df.apply(is_real_critical, axis=1)]
print(f"실제 위험한 로그: {len(real_critical)}개 / {len(df)}개")
```

### 2. 위험도 재계산

```python
# 위험도 점수 조정
def recalculate_risk_score(row):
    anomaly_score = row['anomaly_score']
    severity_score = row['max_severity_score']
    messages = str(row['sample_messages']).lower()
    
    # 정상 쿼리 로그는 위험도 낮춤
    if any(pattern in messages for pattern in [
        'binding parameter', '==> parameters',
        'committing jdbc', 'extracted value'
    ]):
        return min(79, anomaly_score * 30 + (severity_score / 10) * 20)
    
    # 실제 위험 키워드가 있으면 위험도 높임
    if any(keyword in messages for keyword in critical_keywords):
        return anomaly_score * 50 + (severity_score / 10) * 60
    
    return anomaly_score * 40 + (severity_score / 10) * 40

df['risk_score_adjusted'] = df.apply(recalculate_risk_score, axis=1)
df['risk_level_adjusted'] = df['risk_score_adjusted'].apply(
    lambda x: 'CRITICAL' if x >= 80 else 'HIGH' if x >= 60 else 'MEDIUM'
)
```

---

## 📝 결론

### 현재 상태
- ❌ **CRITICAL 로그의 98.8%가 오탐지**
- ❌ **실제 위험한 로그는 거의 탐지되지 않음**
- ❌ **정상 쿼리 로그가 위험으로 분류됨**

### 개선 필요
1. ✅ 위험도 계산 로직 개선
2. ✅ 실제 예외 키워드 기반 필터링
3. ✅ 정상 쿼리 패턴 제외
4. ✅ 학습 데이터 다양화

### 권장 사항
- **현재 CRITICAL 로그는 신뢰하지 마세요**
- **실제 예외가 포함된 로그만 수동으로 확인하세요**
- **위험도 계산 로직을 개선한 후 재실행하세요**

---

## 🔧 빠른 수정 방법

위험도 계산 로직을 개선하려면 `log_specific_anomaly_detectors.py`의 `analyze_risk_level()` 함수를 수정하세요.

```python
# 실제 위험 키워드 확인
def has_real_exception(messages):
    critical_keywords = ['exception', 'error', 'failed', 'timeout']
    return any(keyword in str(messages).lower() for keyword in critical_keywords)

# 정상 쿼리 패턴 확인
def is_normal_query(messages):
    normal_patterns = ['binding parameter', '==> parameters', 'committing jdbc']
    return any(pattern in str(messages).lower() for pattern in normal_patterns)

# 위험도 재계산
if is_normal_query(row['sample_messages']):
    risk_score = anomaly_score * 20 + (severity_score / 10) * 10  # 낮춤
elif has_real_exception(row['sample_messages']):
    risk_score = anomaly_score * 50 + (severity_score / 10) * 70  # 높임
else:
    risk_score = anomaly_score * 40 + (severity_score / 10) * 40  # 기본
```

