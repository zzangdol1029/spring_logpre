"""
로그 이상치 탐지 모델 성능 분석 시스템
- 정상 로그만 학습
- 에러 로그를 이상치로 탐지
- 여러 모델 성능 비교
"""

import re
import os
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from severity_assessment import SeverityAssessment, add_severity_to_anomaly_results


class LogFeatureExtractor:
    """로그 특징 추출 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, log_df):
        """
        로그 데이터에서 특징 추출
        
        Args:
            log_df: 파싱된 로그 DataFrame
        
        Returns:
            numpy array: 특징 벡터
        """
        features = []
        
        for idx, row in log_df.iterrows():
            feature = []
            
            # 1. 로그 레벨 원-핫 인코딩
            level_map = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'FATAL': 4}
            level = row.get('level', 'INFO')
            level_encoded = [0] * 5
            if level in level_map:
                level_encoded[level_map[level]] = 1
            feature.extend(level_encoded)
            
            # 2. 에러 여부 (이건 라벨이므로 제외할 수도 있음)
            # feature.append(1 if row.get('is_error', False) else 0)
            
            # 3. 예외 여부
            feature.append(1 if row.get('has_exception', False) else 0)
            
            # 4. 메시지 길이
            feature.append(row.get('message_length', 0))
            
            # 5. 프로세스 ID (정규화)
            pid = str(row.get('pid', '0'))
            try:
                feature.append(int(pid) % 1000 / 1000.0)
            except:
                feature.append(0.0)
            
            # 6. 클래스 경로 해시 (정규화)
            class_path = str(row.get('class_path', ''))
            feature.append((hash(class_path) % 1000) / 1000.0)
            
            # 7. 스레드명 해시 (정규화)
            thread = str(row.get('thread', ''))
            feature.append((hash(thread) % 1000) / 1000.0)
            
            # 8. 메시지에 특정 키워드 포함 여부
            message = str(row.get('message', '')).lower()
            keywords = ['exception', 'error', 'failed', 'timeout', 'connection', 
                       'null', 'stack', 'trace', 'warning', 'critical']
            for keyword in keywords:
                feature.append(1 if keyword in message else 0)
            
            # 9. 메시지 단어 수
            word_count = len(message.split())
            feature.append(word_count)
            
            # 10. 특수 문자 비율
            special_chars = sum(1 for c in message if not c.isalnum() and c != ' ')
            feature.append(special_chars / max(len(message), 1))
            
            features.append(feature)
        
        return np.array(features)
    
    def fit_transform(self, log_df):
        """특징 추출 및 정규화"""
        features = self.extract_features(log_df)
        features_scaled = self.scaler.fit_transform(features)
        self.is_fitted = True
        return features_scaled
    
    def transform(self, log_df):
        """새로운 데이터 특징 추출"""
        if not self.is_fitted:
            raise ValueError("Scaler가 학습되지 않았습니다. fit_transform()을 먼저 실행하세요.")
        features = self.extract_features(log_df)
        return self.scaler.transform(features)


class LogAnomalyModelComparator:
    """로그 이상치 탐지 모델 비교 클래스"""
    
    def __init__(self, models_config=None):
        """
        Args:
            models_config: 모델 설정 딕셔너리
                예: {'isolation_forest': IForest(...), 'autoencoder': AutoEncoder(...)}
        """
        if models_config is None:
            # 기본 모델 설정
            self.models_config = {
                'Isolation Forest': {
                    'model': IForest(contamination=0.1, random_state=42, n_estimators=100),
                    'description': 'Isolation Forest - 트리 기반 이상치 탐지'
                },
                'AutoEncoder': {
                    'model': None,  # 데이터 크기에 따라 동적 생성
                    'description': 'AutoEncoder - 신경망 기반 재구성 오차 탐지'
                },
                'LOF': {
                    'model': LOF(contamination=0.1, n_neighbors=20),
                    'description': 'Local Outlier Factor - 지역적 이상치 탐지'
                },
                'OCSVM': {
                    'model': OCSVM(contamination=0.1, kernel='rbf'),
                    'description': 'One-Class SVM - 서포트 벡터 기반 탐지'
                },
                'COPOD': {
                    'model': COPOD(contamination=0.1),
                    'description': 'COPOD - Copula 기반 이상치 탐지'
                }
            }
        else:
            self.models_config = models_config
        
        self.feature_extractor = LogFeatureExtractor()
        self.trained_models = {}
        self.results = {}
        
    def prepare_data(self, normal_logs_df, error_logs_df, train_ratio=0.8, valid_ratio=0.2):
        """
        데이터 준비 및 분할
        - 전체의 80% → train
        - train의 20% → valid (전체의 16%)
        - 나머지 20% → test (전체의 20%)
        
        Args:
            normal_logs_df: 정상 로그 DataFrame
            error_logs_df: 에러 로그 DataFrame
            train_ratio: 학습 데이터 비율 (기본 0.8 = 80%)
            valid_ratio: 검증 데이터 비율 (train의 비율, 기본 0.2 = 20%)
        
        Returns:
            dict: 학습/검증/테스트 데이터
        """
        print("=" * 60)
        print("데이터 준비 및 분할")
        print("=" * 60)
        
        # 시간 순서 정렬
        normal_logs_df = normal_logs_df.sort_values('timestamp').reset_index(drop=True)
        error_logs_df = error_logs_df.sort_values('timestamp').reset_index(drop=True)
        
        # 전체 로그 (정상 + 에러)를 시간 순서로 결합
        all_logs = pd.concat([normal_logs_df, error_logs_df], ignore_index=True)
        all_logs = all_logs.sort_values('timestamp').reset_index(drop=True)
        
        # 전체 데이터 분할: 80% train, 20% test
        total_split_idx = int(len(all_logs) * train_ratio)
        train_all = all_logs.iloc[:total_split_idx]
        test_all = all_logs.iloc[total_split_idx:]
        
        # Train 데이터에서 정상/에러 분리
        train_normal_df = train_all[train_all['is_error'] == False].copy()
        train_error_df = train_all[train_all['is_error'] == True].copy()
        
        # Train의 20%를 Valid로 분할 (정상 로그 기준)
        train_normal_split_idx = int(len(train_normal_df) * (1 - valid_ratio))
        train_normal_final = train_normal_df.iloc[:train_normal_split_idx]
        valid_normal_df = train_normal_df.iloc[train_normal_split_idx:]
        
        # 에러 로그도 train의 20%를 valid로
        if len(train_error_df) > 0:
            train_error_split_idx = int(len(train_error_df) * (1 - valid_ratio))
            train_error_final = train_error_df.iloc[:train_error_split_idx]
            valid_error_df = train_error_df.iloc[train_error_split_idx:]
        else:
            train_error_final = train_error_df
            valid_error_df = pd.DataFrame()
        
        # Test 데이터에서 정상/에러 분리
        test_normal_df = test_all[test_all['is_error'] == False].copy()
        test_error_df = test_all[test_all['is_error'] == True].copy()
        
        print(f"\n📊 데이터 분할 결과:")
        print(f"   Train (학습용):")
        print(f"      - 정상 로그: {len(train_normal_final)}개 ({len(train_normal_final)/len(all_logs)*100:.1f}%)")
        print(f"      - 에러 로그: {len(train_error_final)}개")
        print(f"      - 전체: {len(train_normal_final) + len(train_error_final)}개 ({len(train_all)/len(all_logs)*100:.1f}%)")
        print(f"\n   Valid (검증용, train의 {valid_ratio*100:.0f}%):")
        print(f"      - 정상 로그: {len(valid_normal_df)}개 ({len(valid_normal_df)/len(all_logs)*100:.1f}%)")
        print(f"      - 에러 로그: {len(valid_error_df)}개")
        print(f"      - 전체: {len(valid_normal_df) + len(valid_error_df)}개 ({(len(valid_normal_df) + len(valid_error_df))/len(all_logs)*100:.1f}%)")
        print(f"\n   Test (테스트용):")
        print(f"      - 정상 로그: {len(test_normal_df)}개 ({len(test_normal_df)/len(all_logs)*100:.1f}%)")
        print(f"      - 에러 로그: {len(test_error_df)}개 ({len(test_error_df)/len(all_logs)*100:.1f}%)")
        print(f"      - 전체: {len(test_normal_df) + len(test_error_df)}개 ({len(test_all)/len(all_logs)*100:.1f}%)")
        
        # 특징 추출
        print(f"\n특징 추출 중...")
        X_train = self.feature_extractor.fit_transform(train_normal_final)
        X_valid_normal = self.feature_extractor.transform(valid_normal_df) if not valid_normal_df.empty else np.array([]).reshape(0, X_train.shape[1])
        X_test_normal = self.feature_extractor.transform(test_normal_df)
        X_test_error = self.feature_extractor.transform(test_error_df)
        
        # Valid 데이터 결합
        if len(X_valid_normal) > 0:
            X_valid_error = self.feature_extractor.transform(valid_error_df) if not valid_error_df.empty else np.array([]).reshape(0, X_train.shape[1])
            if len(X_valid_error) > 0:
                X_valid = np.vstack([X_valid_normal, X_valid_error])
                y_valid = np.hstack([
                    np.zeros(len(X_valid_normal)),  # 정상 = 0
                    np.ones(len(X_valid_error))      # 이상 = 1
                ])
            else:
                X_valid = X_valid_normal
                y_valid = np.zeros(len(X_valid_normal))
        else:
            X_valid = np.array([]).reshape(0, X_train.shape[1])
            y_valid = np.array([])
        
        # Test 데이터 결합
        X_test = np.vstack([X_test_normal, X_test_error])
        y_test = np.hstack([
            np.zeros(len(X_test_normal)),  # 정상 = 0
            np.ones(len(X_test_error))      # 이상 = 1
        ])
        
        print(f"   - 학습 특징 차원: {X_train.shape}")
        if len(X_valid) > 0:
            print(f"   - 검증 특징 차원: {X_valid.shape}")
        print(f"   - 테스트 특징 차원: {X_test.shape}")
        if len(y_valid) > 0:
            print(f"   - 검증 라벨: 정상 {np.sum(y_valid==0)}개, 이상 {np.sum(y_valid==1)}개")
        print(f"   - 테스트 라벨: 정상 {np.sum(y_test==0)}개, 이상 {np.sum(y_test==1)}개")
        
        return {
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_valid': y_valid,
            'y_test': y_test,
            'train_normal_df': train_normal_final,
            'valid_normal_df': valid_normal_df,
            'valid_error_df': valid_error_df,
            'test_normal_df': test_normal_df,
            'test_error_df': test_error_df
        }
    
    def train_models(self, X_train, selected_models=None):
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터
            selected_models: 학습할 모델 리스트 (None이면 모든 모델)
        """
        print("\n" + "=" * 60)
        print("모델 학습")
        print("=" * 60)
        
        if selected_models is None:
            selected_models = list(self.models_config.keys())
        
        n_samples, n_features = X_train.shape
        
        for model_name in selected_models:
            if model_name not in self.models_config:
                print(f"⚠️ 알 수 없는 모델: {model_name}")
                continue
            
            print(f"\n[{model_name}] 학습 중...")
            print(f"   설명: {self.models_config[model_name]['description']}")
            
            try:
                # AutoEncoder는 동적 생성
                if model_name == 'AutoEncoder':
                    # 데이터 크기에 따라 파라미터 조정
                    if n_samples < 10:
                        print(f"   ⚠️ 데이터가 너무 적습니다 ({n_samples}개). 건너뜁니다.")
                        continue
                    
                    if n_features < 50:
                        hidden_neurons = [max(8, n_features//2), max(4, n_features//4), max(8, n_features//2)]
                    elif n_features < 200:
                        hidden_neurons = [64, 32, 16, 32, 64]
                    else:
                        hidden_neurons = [128, 64, 32, 64, 128]
                    
                    if n_samples < 50:
                        epochs = 20
                        batch_size = min(8, n_samples)
                    elif n_samples < 100:
                        epochs = 30
                        batch_size = 16
                    else:
                        epochs = 50
                        batch_size = 32
                    
                    try:
                        model = AutoEncoder(
                            contamination=0.1,
                            hidden_neurons=hidden_neurons,
                            epochs=epochs,
                            batch_size=batch_size,
                            dropout_rate=0.2,
                            verbose=0,
                            random_state=42
                        )
                    except TypeError:
                        model = AutoEncoder(
                            contamination=0.1,
                            hidden_neuron_list=hidden_neurons,
                            epoch_num=epochs,
                            batch_size=batch_size,
                            dropout_rate=0.2,
                            verbose=0,
                            random_state=42
                        )
                else:
                    model = self.models_config[model_name]['model']
                
                # 모델 학습
                model.fit(X_train)
                self.trained_models[model_name] = model
                print(f"   ✅ 학습 완료")
                
            except Exception as e:
                print(f"   ❌ 학습 실패: {e}")
                continue
    
    def evaluate_models(self, X_test, y_test, test_logs_df=None):
        """
        모델 성능 평가
        
        Args:
            X_test: 테스트 데이터
            y_test: 테스트 라벨 (0=정상, 1=이상)
            test_logs_df: 테스트 로그 DataFrame (심각도 평가용)
        """
        print("\n" + "=" * 60)
        print("모델 성능 평가")
        print("=" * 60)
        
        results = {}
        severity_assessor = SeverityAssessment()
        
        for model_name, model in self.trained_models.items():
            print(f"\n[{model_name}] 평가 중...")
            
            try:
                # 예측
                y_pred = model.predict(X_test)  # 1=이상, 0=정상
                y_scores = model.decision_function(X_test)  # 이상 점수 (낮을수록 이상)
                
                # 점수를 확률로 변환 (일부 모델은 음수 점수 사용)
                # 점수가 낮을수록 이상이므로, -score를 사용하거나 정규화
                if y_scores.min() < 0:
                    # 음수 점수를 양수로 변환 (낮은 점수 = 높은 이상 확률)
                    y_scores_normalized = -y_scores
                    y_scores_normalized = (y_scores_normalized - y_scores_normalized.min()) / (
                        y_scores_normalized.max() - y_scores_normalized.min() + 1e-8
                    )
                else:
                    y_scores_normalized = 1 - (y_scores - y_scores.min()) / (
                        y_scores.max() - y_scores.min() + 1e-8
                    )
                
                # 성능 지표 계산
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # ROC-AUC (일부 모델은 지원하지 않을 수 있음)
                try:
                    roc_auc = roc_auc_score(y_test, y_scores_normalized)
                except:
                    roc_auc = None
                
                # 혼동 행렬
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # 추가 지표
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                # 심각도 평가 (이상치로 탐지된 로그들)
                severity_info = None
                if test_logs_df is not None and len(test_logs_df) == len(y_pred):
                    # 이상치로 탐지된 로그들 (y_pred == 1)
                    detected_anomalies = test_logs_df[y_pred == 1].copy()
                    
                    if not detected_anomalies.empty:
                        # 심각도 평가
                        detected_anomalies = severity_assessor.assess_anomaly_severity(detected_anomalies)
                        
                        # 심각도 통계
                        severity_summary = severity_assessor.generate_severity_summary(detected_anomalies)
                        severity_info = {
                            'detected_anomalies': detected_anomalies,
                            'summary': severity_summary
                        }
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'specificity': specificity,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate,
                    'confusion_matrix': cm,
                    'y_pred': y_pred,
                    'y_scores': y_scores_normalized,
                    'tp': tp,
                    'tn': tn,
                    'fp': fp,
                    'fn': fn,
                    'severity_info': severity_info
                }
                
                print(f"   정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"   정밀도: {precision:.4f} ({precision*100:.2f}%)")
                print(f"   재현율: {recall:.4f} ({recall*100:.2f}%)")
                print(f"   F1 점수: {f1:.4f}")
                if roc_auc:
                    print(f"   ROC-AUC: {roc_auc:.4f}")
                print(f"   특이도: {specificity:.4f} ({specificity*100:.2f}%)")
                print(f"   혼동 행렬:")
                print(f"      [정상→정상: {tn:4d}  정상→이상: {fp:4d}]")
                print(f"      [이상→정상: {fn:4d}  이상→이상: {tp:4d}]")
                
                # 심각도 정보 출력
                if severity_info and severity_info['summary']:
                    summary = severity_info['summary']
                    print(f"\n   🔍 심각도 분석:")
                    print(f"      탐지된 이상치: {summary.get('total_anomalies', 0)}개")
                    if 'by_severity' in summary:
                        print(f"      심각도 분포:")
                        for level, count in summary['by_severity'].items():
                            print(f"        {level}: {count}개")
                    if 'avg_severity_score' in summary:
                        print(f"      평균 심각도 점수: {summary['avg_severity_score']:.2f}")
                    if 'max_severity_score' in summary:
                        print(f"      최고 심각도 점수: {summary['max_severity_score']:.2f}")
                
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
            return
        
        print("\n" + "=" * 60)
        print("모델 성능 비교 리포트")
        print("=" * 60)
        
        # 성능 지표 비교 테이블
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                '모델': model_name,
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
        
        print(f"   최고 F1 점수: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        print(f"   최고 정확도: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"   최고 재현율: {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        
        # 결과 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 비교 테이블 저장
            comparison_path = os.path.join(output_dir, "model_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 비교 결과 저장: {comparison_path}")
            
            # 상세 결과 저장
            for model_name, metrics in self.results.items():
                detail_path = os.path.join(output_dir, f"results_{model_name.replace(' ', '_')}.csv")
                detail_df = pd.DataFrame({
                    'y_true': [0] * len(metrics['y_pred']),  # 실제 라벨은 별도로 저장 필요
                    'y_pred': metrics['y_pred'],
                    'y_score': metrics['y_scores']
                })
                detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
                
                # 심각도 정보가 있으면 별도로 저장
                if metrics.get('severity_info') and metrics['severity_info'].get('detected_anomalies') is not None:
                    severity_path = os.path.join(output_dir, f"severity_{model_name.replace(' ', '_')}.csv")
                    severity_df = metrics['severity_info']['detected_anomalies']
                    # 우선순위 정렬
                    severity_df = SeverityAssessment().prioritize_anomalies(severity_df)
                    severity_df.to_csv(severity_path, index=False, encoding='utf-8-sig')
                    print(f"💾 심각도 분석 결과 저장: {severity_path}")
        
        return comparison_df
    
    def plot_roc_curves(self, y_test, output_path=None):
        """ROC 곡선 시각화"""
        if not self.results:
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if metrics['roc_auc'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test, metrics['y_scores'])
                    plt.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['roc_auc']:.3f})", linewidth=2)
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 ROC 곡선 저장: {output_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, output_path=None):
        """Precision-Recall 곡선 시각화"""
        if not self.results:
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            try:
                precision, recall, _ = precision_recall_curve(y_test, metrics['y_scores'])
                plt.plot(recall, precision, label=f"{model_name}", linewidth=2)
            except:
                continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Precision-Recall 곡선 저장: {output_path}")
        
        plt.show()


def main():
    """메인 실행 함수"""
    from log_anomaly_detector import SpringBootLogParser
    
    log_directory = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    print("=" * 60)
    print("로그 이상치 탐지 모델 성능 분석")
    print("=" * 60)
    
    # 1. 로그 파싱 (메모리 효율적)
    print("\n1단계: 로그 파일 파싱")
    parser = SpringBootLogParser()
    
    # 메모리 절약을 위한 파라미터
    # 필요에 따라 조정 가능:
    # - max_files: 처리할 최대 파일 수
    # - sample_lines: 파일당 최대 라인 수
    # - chunk_size: 청크 크기 (기본 10,000)
    # - max_total_lines: 전체 최대 라인 수
    logs_df = parser.parse_directory(
        log_directory,
        max_files=None,        # 전체 파일 처리 (메모리 부족 시 숫자로 제한)
        sample_lines=None,     # 전체 라인 처리 (메모리 부족 시 숫자로 제한)
        chunk_size=5000,       # 청크 크기 (메모리 절약, 기본값보다 작게)
        max_total_lines=None,  # 전체 최대 라인 수 (메모리 부족 시 설정)
        save_chunks_to_disk=True  # 파일로 저장하여 메모리 절약
    )
    
    if logs_df.empty:
        print("⚠️ 파싱된 로그가 없습니다.")
        return
    
    print(f"✅ 총 {len(logs_df)}개 로그 라인 파싱 완료")
    
    # 2. 정상/에러 로그 분리
    print("\n2단계: 정상/에러 로그 분리")
    # 정상 로그: INFO, DEBUG, TRACE만 포함 (WARN 제외)
    # 에러 로그: ERROR, FATAL, WARN 포함
    normal_logs_df = logs_df[
        (logs_df['is_error'] == False) & 
        (logs_df['level'].isin(['INFO', 'DEBUG', 'TRACE']))
    ].copy()
    error_logs_df = logs_df[
        (logs_df['is_error'] == True) | 
        (logs_df['level'].isin(['WARN', 'ERROR', 'FATAL']))
    ].copy()
    
    print(f"   - 정상 로그 (INFO/DEBUG/TRACE만): {len(normal_logs_df)}개")
    print(f"   - 에러 로그 (ERROR/FATAL/WARN 포함): {len(error_logs_df)}개")
    
    # 레벨별 통계
    level_counts = logs_df['level'].value_counts()
    print(f"\n   레벨별 분포:")
    for level, count in level_counts.items():
        print(f"      {level}: {count}개 ({count/len(logs_df)*100:.1f}%)")
    
    if len(normal_logs_df) == 0:
        print("⚠️ 정상 로그가 없습니다.")
        return
    
    if len(error_logs_df) == 0:
        print("⚠️ 에러 로그가 없습니다.")
        return
    
    # 3. 모델 비교 시스템 초기화
    print("\n3단계: 모델 비교 시스템 초기화")
    # 2개 모델 선택 (Isolation Forest, AutoEncoder)
    comparator = LogAnomalyModelComparator()
    selected_models = ['Isolation Forest', 'AutoEncoder']
    
    # 4. 데이터 준비
    print("\n4단계: 데이터 준비")
    data = comparator.prepare_data(
        normal_logs_df=normal_logs_df,
        error_logs_df=error_logs_df,
        train_ratio=0.8,    # 전체의 80% → train
        valid_ratio=0.2     # train의 20% → valid (전체의 16%)
    )
    
    # 5. 모델 학습
    print("\n5단계: 모델 학습")
    comparator.train_models(data['X_train'], selected_models=selected_models)
    
    if not comparator.trained_models:
        print("⚠️ 학습된 모델이 없습니다.")
        return
    
    # 6. 모델 평가
    print("\n6단계: 모델 평가")
    
    # 검증 데이터 평가 (선택적)
    if len(data['X_valid']) > 0 and len(data['y_valid']) > 0:
        print("\n6-1. 검증 데이터 평가")
        valid_logs_list = [data['valid_normal_df']]
        if 'valid_error_df' in data and not data['valid_error_df'].empty:
            valid_logs_list.append(data['valid_error_df'])
        valid_logs_df = pd.concat(valid_logs_list, ignore_index=True) if valid_logs_list else pd.DataFrame()
        
        if not valid_logs_df.empty:
            valid_results = comparator.evaluate_models(data['X_valid'], data['y_valid'], test_logs_df=valid_logs_df)
            print("   ✅ 검증 데이터 평가 완료")
    
    # 테스트 데이터 평가
    print("\n6-2. 테스트 데이터 평가")
    # 테스트 로그 DataFrame 준비 (심각도 평가용)
    test_logs_df = pd.concat([data['test_normal_df'], data['test_error_df']], ignore_index=True)
    results = comparator.evaluate_models(data['X_test'], data['y_test'], test_logs_df=test_logs_df)
    
    # 7. 비교 리포트 생성
    print("\n7단계: 비교 리포트 생성")
    output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results/performance"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df = comparator.generate_comparison_report(output_dir=output_dir)
    
    # 8. 시각화
    print("\n8단계: 성능 곡선 시각화")
    try:
        comparator.plot_roc_curves(
            data['y_test'],
            output_path=os.path.join(output_dir, "roc_curves.png")
        )
        comparator.plot_precision_recall_curves(
            data['y_test'],
            output_path=os.path.join(output_dir, "pr_curves.png")
        )
    except Exception as e:
        print(f"⚠️ 시각화 실패: {e}")
    
    # 9. 상세 리포트 저장
    print("\n9단계: 상세 리포트 저장")
    report_path = os.path.join(output_dir, "performance_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("로그 이상치 탐지 모델 성능 분석 리포트\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("데이터 정보:\n")
        f.write(f"  - 전체 로그: {len(logs_df)}개\n")
        f.write(f"  - 정상 로그: {len(normal_logs_df)}개\n")
        f.write(f"  - 에러 로그: {len(error_logs_df)}개\n")
        f.write(f"  - 학습 정상 로그: {len(data['train_normal_df'])}개 ({len(data['train_normal_df'])/len(logs_df)*100:.1f}%)\n")
        if 'valid_normal_df' in data:
            f.write(f"  - 검증 정상 로그: {len(data['valid_normal_df'])}개 ({len(data['valid_normal_df'])/len(logs_df)*100:.1f}%)\n")
        f.write(f"  - 테스트 정상 로그: {len(data['test_normal_df'])}개 ({len(data['test_normal_df'])/len(logs_df)*100:.1f}%)\n")
        f.write(f"  - 테스트 에러 로그: {len(data['test_error_df'])}개 ({len(data['test_error_df'])/len(logs_df)*100:.1f}%)\n\n")
        
        f.write("모델 성능 비교:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"[{model_name}] 상세 결과:\n")
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
            
            # 심각도 정보 추가
            if metrics.get('severity_info') and metrics['severity_info'].get('summary'):
                summary = metrics['severity_info']['summary']
                f.write(f"\n  심각도 분석:\n")
                f.write(f"    탐지된 이상치: {summary.get('total_anomalies', 0)}개\n")
                if 'by_severity' in summary:
                    f.write(f"    심각도 분포:\n")
                    for level, count in summary['by_severity'].items():
                        f.write(f"      {level}: {count}개\n")
                if 'avg_severity_score' in summary:
                    f.write(f"    평균 심각도 점수: {summary['avg_severity_score']:.2f}\n")
                if 'max_severity_score' in summary:
                    f.write(f"    최고 심각도 점수: {summary['max_severity_score']:.2f}\n")
                if 'top_exceptions' in summary:
                    f.write(f"    주요 예외 유형:\n")
                    for exc_type, count in list(summary['top_exceptions'].items())[:5]:
                        f.write(f"      {exc_type}: {count}회\n")
            f.write("\n")
    
    print(f"💾 상세 리포트 저장: {report_path}")
    
    print("\n" + "=" * 60)
    print("성능 분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

