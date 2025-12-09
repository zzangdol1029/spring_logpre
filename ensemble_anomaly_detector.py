"""
앙상블 이상치 탐지 클래스
여러 모델의 예측을 결합하여 더 정확한 이상치 탐지를 수행합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class EnsembleAnomalyDetector:
    """앙상블 이상치 탐지 클래스"""
    
    def __init__(self, models_dict: Dict, method: str = 'majority', weights: Optional[Dict] = None):
        """
        Args:
            models_dict: {모델명: 모델객체} 딕셔너리
            method: 앙상블 방법 ('majority', 'weighted', 'stacking', 'max', 'min')
            weights: 가중치 딕셔너리 (weighted 방법 사용 시)
        """
        self.models = models_dict
        self.method = method
        self.weights = weights or {}
        self.meta_model = None
        
        # 가중치가 없으면 균등 가중치
        if method == 'weighted' and not self.weights:
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models.keys()}
    
    def predict(self, X):
        """
        앙상블 예측
        
        Args:
            X: 입력 데이터
        
        Returns:
            예측 결과 (0=정상, 1=이상)
        """
        predictions = {}
        scores = {}
        
        # 각 모델의 예측 수집
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
                scores[name] = model.decision_function(X)
            except Exception as e:
                print(f"⚠️ 모델 {name} 예측 실패: {e}")
                continue
        
        if not predictions:
            raise ValueError("모든 모델의 예측이 실패했습니다.")
        
        # 앙상블 방법에 따라 결합
        if self.method == 'majority':
            return self._majority_vote(predictions)
        elif self.method == 'weighted':
            return self._weighted_vote(scores)
        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("스태킹을 사용하려면 먼저 fit_meta_model()을 호출하세요.")
            return self._stacking_predict(X, scores)
        elif self.method == 'max':
            return self._max_vote(predictions)
        elif self.method == 'min':
            return self._min_vote(predictions)
        else:
            raise ValueError(f"알 수 없는 앙상블 방법: {self.method}")
    
    def decision_function(self, X):
        """
        앙상블 이상 점수
        
        Args:
            X: 입력 데이터
        
        Returns:
            이상 점수 (높을수록 이상)
        """
        scores = {}
        
        # 각 모델의 점수 수집
        for name, model in self.models.items():
            try:
                score = model.decision_function(X)
                # 점수 정규화 (모델 간 스케일 통일)
                if score.min() < 0:
                    score = -score
                score = (score - score.min()) / (score.max() - score.min() + 1e-8)
                scores[name] = score
            except Exception as e:
                print(f"⚠️ 모델 {name} 점수 계산 실패: {e}")
                continue
        
        if not scores:
            raise ValueError("모든 모델의 점수 계산이 실패했습니다.")
        
        # 가중 평균
        if self.method == 'weighted':
            total_weight = sum(self.weights.get(name, 1.0) for name in scores.keys())
            ensemble_score = np.zeros(len(scores[list(scores.keys())[0]]))
            
            for name, score in scores.items():
                weight = self.weights.get(name, 1.0) / total_weight
                ensemble_score += weight * score
            
            return ensemble_score
        else:
            # 단순 평균
            return np.mean(list(scores.values()), axis=0)
    
    def _majority_vote(self, predictions: Dict) -> np.ndarray:
        """
        다수결 투표
        
        Args:
            predictions: {모델명: 예측결과} 딕셔너리
        
        Returns:
            앙상블 예측 결과
        """
        model_names = list(predictions.keys())
        n_samples = len(predictions[model_names[0]])
        
        ensemble_pred = []
        for i in range(n_samples):
            votes = sum(pred[i] for pred in predictions.values())
            # 과반수 이상이면 이상치
            threshold = len(model_names) / 2
            ensemble_pred.append(1 if votes >= threshold else 0)
        
        return np.array(ensemble_pred)
    
    def _weighted_vote(self, scores: Dict) -> np.ndarray:
        """
        가중 투표
        
        Args:
            scores: {모델명: 이상점수} 딕셔너리
        
        Returns:
            앙상블 예측 결과
        """
        # 점수 정규화
        normalized_scores = {}
        for name, score in scores.items():
            if score.min() < 0:
                score = -score
            normalized_scores[name] = (score - score.min()) / (score.max() - score.min() + 1e-8)
        
        # 가중 평균
        total_weight = sum(self.weights.get(name, 1.0) for name in scores.keys())
        ensemble_score = np.zeros(len(scores[list(scores.keys())[0]]))
        
        for name, score in normalized_scores.items():
            weight = self.weights.get(name, 1.0) / total_weight
            ensemble_score += weight * score
        
        # 임계값으로 이진 분류
        threshold = 0.5
        return (ensemble_score > threshold).astype(int)
    
    def _max_vote(self, predictions: Dict) -> np.ndarray:
        """
        최대 투표 (하나라도 이상이면 이상)
        
        Args:
            predictions: {모델명: 예측결과} 딕셔너리
        
        Returns:
            앙상블 예측 결과
        """
        model_names = list(predictions.keys())
        n_samples = len(predictions[model_names[0]])
        
        ensemble_pred = []
        for i in range(n_samples):
            # 하나라도 이상이면 이상
            max_vote = max(pred[i] for pred in predictions.values())
            ensemble_pred.append(max_vote)
        
        return np.array(ensemble_pred)
    
    def _min_vote(self, predictions: Dict) -> np.ndarray:
        """
        최소 투표 (모두 이상이어야 이상)
        
        Args:
            predictions: {모델명: 예측결과} 딕셔너리
        
        Returns:
            앙상블 예측 결과
        """
        model_names = list(predictions.keys())
        n_samples = len(predictions[model_names[0]])
        
        ensemble_pred = []
        for i in range(n_samples):
            # 모두 이상이어야 이상
            min_vote = min(pred[i] for pred in predictions.values())
            ensemble_pred.append(min_vote)
        
        return np.array(ensemble_pred)
    
    def _stacking_predict(self, X, scores: Dict) -> np.ndarray:
        """
        스태킹 예측
        
        Args:
            X: 입력 데이터
            scores: {모델명: 이상점수} 딕셔너리
        
        Returns:
            앙상블 예측 결과
        """
        # 점수를 특징으로 변환
        features = np.column_stack(list(scores.values()))
        return self.meta_model.predict(features)
    
    def fit_meta_model(self, X_train, y_train):
        """
        메타 모델 학습 (스태킹용)
        
        Args:
            X_train: 학습 데이터
            y_train: 학습 라벨
        """
        # 각 모델의 점수 수집
        scores_list = []
        for name, model in self.models.items():
            try:
                score = model.decision_function(X_train)
                # 점수 정규화
                if score.min() < 0:
                    score = -score
                score = (score - score.min()) / (score.max() - score.min() + 1e-8)
                scores_list.append(score)
            except Exception as e:
                print(f"⚠️ 모델 {name} 점수 계산 실패: {e}")
                continue
        
        if not scores_list:
            raise ValueError("모든 모델의 점수 계산이 실패했습니다.")
        
        # 특징 행렬 생성
        X_meta = np.column_stack(scores_list)
        
        # 메타 모델 학습
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(X_meta, y_train)
        
        print(f"✅ 메타 모델 학습 완료 (특징 수: {X_meta.shape[1]})")
    
    def evaluate(self, X_test, y_test):
        """
        앙상블 성능 평가
        
        Args:
            X_test: 테스트 데이터
            y_test: 테스트 라벨
        
        Returns:
            성능 지표 딕셔너리
        """
        y_pred = self.predict(X_test)
        y_scores = self.decision_function(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
        except:
            roc_auc = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_scores': y_scores
        }


def compare_models_and_ensemble(
    comparator,
    X_train, y_train, X_test, y_test,
    selected_models=None,
    ensemble_methods=['majority', 'weighted']
):
    """
    모델 비교 및 앙상블 성능 평가
    
    Args:
        comparator: LogAnomalyModelComparator 객체
        X_train, y_train: 학습 데이터
        X_test, y_test: 테스트 데이터
        selected_models: 선택할 모델 리스트
        ensemble_methods: 앙상블 방법 리스트
    
    Returns:
        비교 결과 딕셔너리
    """
    # 1. 개별 모델 학습 및 평가
    if selected_models:
        comparator.train_models(X_train, selected_models=selected_models)
    else:
        comparator.train_models(X_train)
    
    individual_results = comparator.evaluate_models(X_test, y_test)
    
    # 2. 앙상블 생성 및 평가
    ensemble_results = {}
    
    for method in ensemble_methods:
        print(f"\n{'='*60}")
        print(f"앙상블 방법: {method}")
        print(f"{'='*60}")
        
        try:
            if method == 'stacking':
                # 스태킹은 메타 모델 학습 필요
                ensemble = EnsembleAnomalyDetector(
                    models_dict=comparator.trained_models,
                    method=method
                )
                ensemble.fit_meta_model(X_train, y_train)
            elif method == 'weighted':
                # 가중치는 개별 모델의 F1 점수 기반
                weights = {}
                total_f1 = sum(metrics['f1_score'] for metrics in individual_results.values())
                for name, metrics in individual_results.items():
                    weights[name] = metrics['f1_score'] / total_f1 if total_f1 > 0 else 1.0 / len(individual_results)
                
                ensemble = EnsembleAnomalyDetector(
                    models_dict=comparator.trained_models,
                    method=method,
                    weights=weights
                )
                print(f"가중치: {weights}")
            else:
                ensemble = EnsembleAnomalyDetector(
                    models_dict=comparator.trained_models,
                    method=method
                )
            
            # 앙상블 평가
            ensemble_metrics = ensemble.evaluate(X_test, y_test)
            ensemble_results[method] = ensemble_metrics
            
            print(f"정확도: {ensemble_metrics['accuracy']:.4f}")
            print(f"정밀도: {ensemble_metrics['precision']:.4f}")
            print(f"재현율: {ensemble_metrics['recall']:.4f}")
            print(f"F1 점수: {ensemble_metrics['f1_score']:.4f}")
            if ensemble_metrics['roc_auc']:
                print(f"ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        except Exception as e:
            print(f"⚠️ 앙상블 {method} 실패: {e}")
            continue
    
    # 3. 종합 비교
    print(f"\n{'='*60}")
    print("종합 비교")
    print(f"{'='*60}")
    
    comparison_data = []
    
    # 개별 모델
    for name, metrics in individual_results.items():
        comparison_data.append({
            '모델': name,
            '방법': '개별',
            '정확도': f"{metrics['accuracy']:.4f}",
            'F1 점수': f"{metrics['f1_score']:.4f}",
            '재현율': f"{metrics['recall']:.4f}",
            '정밀도': f"{metrics['precision']:.4f}",
        })
    
    # 앙상블
    for method, metrics in ensemble_results.items():
        comparison_data.append({
            '모델': f'앙상블 ({method})',
            '방법': '앙상블',
            '정확도': f"{metrics['accuracy']:.4f}",
            'F1 점수': f"{metrics['f1_score']:.4f}",
            '재현율': f"{metrics['recall']:.4f}",
            '정밀도': f"{metrics['precision']:.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 성능 비교:")
    print(comparison_df.to_string(index=False))
    
    # 최고 성능
    all_results = {**individual_results, **{f'앙상블_{k}': v for k, v in ensemble_results.items()}}
    best_f1 = max(all_results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 최고 F1 점수: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    
    return {
        'individual': individual_results,
        'ensemble': ensemble_results,
        'comparison': comparison_df
    }














