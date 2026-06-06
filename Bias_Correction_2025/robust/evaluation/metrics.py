"""
Comprehensive Evaluation Metrics for Bias Correction
=====================================================

Complete suite of meteorological skill scores following
WMO and IMD verification standards.

Categories:
1. Continuous metrics (MAE, RMSE, MBE, etc.)
2. Skill scores (NSE, KGE, R)
3. Categorical metrics (POD, FAR, CSI, HSS, ETS)
4. Spatial metrics (FSS, SAL)
5. Distributional metrics (KL divergence, Wasserstein)

Author: IITM Pune - Government of India Project
Reference: WMO Guide on Verification of Forecasts (2008)
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter
from typing import Dict, List, Optional, Tuple, Union
import warnings


class BiasCorrectionMetrics:
    """
    Complete suite of meteorological verification metrics.
    
    Implements all standard metrics for precipitation forecast verification
    as recommended by WMO and IMD.
    """
    
    def __init__(
        self,
        thresholds: List[float] = [1.0, 5.0, 10.0, 20.0, 50.0],
        fss_scales: List[int] = [1, 3, 5, 9, 15],
        mask: Optional[np.ndarray] = None
    ):
        """
        Initialize metrics calculator.
        
        Args:
            thresholds: Precipitation thresholds for categorical metrics (mm)
            fss_scales: Spatial scales for FSS calculation
            mask: Optional mask for land/sea areas
        """
        self.thresholds = thresholds
        self.fss_scales = fss_scales
        self.mask = mask
        
    def compute_all(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            pred: Predictions [B, H, W] or [H, W]
            target: Targets [B, H, W] or [H, W]
            mask: Optional mask (True = valid)
            
        Returns:
            Dictionary of all metrics
        """
        if mask is None:
            mask = self.mask
            
        # Flatten if needed
        if pred.ndim > 2:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
        else:
            pred = pred.flatten()
            target = target.flatten()
            if mask is not None:
                mask = mask.flatten()
                
        # Apply mask
        if mask is not None:
            valid = mask.astype(bool)
            pred = pred[valid]
            target = target[valid]
            
        # Remove NaN values
        valid_idx = np.isfinite(pred) & np.isfinite(target)
        pred = pred[valid_idx]
        target = target[valid_idx]
        
        if len(pred) == 0:
            warnings.warn("No valid data points!")
            return {}
            
        results = {}
        
        # ========== CONTINUOUS METRICS ==========
        results.update(self._continuous_metrics(pred, target))
        
        # ========== SKILL SCORES ==========
        results.update(self._skill_scores(pred, target))
        
        # ========== CATEGORICAL METRICS ==========
        for threshold in self.thresholds:
            cat_metrics = self._categorical_metrics(pred, target, threshold)
            for key, value in cat_metrics.items():
                results[f"{key}_{threshold}mm"] = value
                
        return results
    
    def _continuous_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute continuous verification metrics."""
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['MAE'] = np.mean(np.abs(pred - target))
        
        # Root Mean Square Error
        metrics['RMSE'] = np.sqrt(np.mean((pred - target) ** 2))
        
        # Mean Bias Error (positive = over-prediction)
        metrics['MBE'] = np.mean(pred - target)
        
        # Percent Bias
        if np.sum(np.abs(target)) > 0:
            metrics['PBIAS'] = 100 * np.sum(pred - target) / np.sum(np.abs(target))
        else:
            metrics['PBIAS'] = 0.0
            
        # Mean Absolute Percent Error
        valid_target = target != 0
        if np.sum(valid_target) > 0:
            mape = np.mean(np.abs((pred[valid_target] - target[valid_target]) / target[valid_target])) * 100
            metrics['MAPE'] = mape
            
        # Pearson Correlation
        if np.std(pred) > 0 and np.std(target) > 0:
            metrics['Pearson_R'] = np.corrcoef(pred, target)[0, 1]
        else:
            metrics['Pearson_R'] = 0.0
            
        # Spearman Correlation
        try:
            metrics['Spearman_Rho'], _ = stats.spearmanr(pred, target)
        except:
            metrics['Spearman_Rho'] = 0.0
            
        # Coefficient of Determination (R)
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        if ss_tot > 0:
            metrics['R2'] = 1 - ss_res / ss_tot
        else:
            metrics['R2'] = 0.0
            
        # Standard deviation ratio
        if np.std(target) > 0:
            metrics['StdRatio'] = np.std(pred) / np.std(target)
        else:
            metrics['StdRatio'] = 1.0
            
        return metrics
    
    def _skill_scores(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute skill scores."""
        
        metrics = {}
        
        # Nash-Sutcliffe Efficiency
        # NSE = 1 means perfect model
        # NSE = 0 means model is as good as mean
        # NSE < 0 means model is worse than mean
        numerator = np.sum((target - pred) ** 2)
        denominator = np.sum((target - np.mean(target)) ** 2)
        if denominator > 0:
            metrics['NSE'] = 1 - numerator / denominator
        else:
            metrics['NSE'] = 0.0
            
        # Kling-Gupta Efficiency (KGE)
        # Combines correlation, bias, and variability
        r = metrics.get('Pearson_R', np.corrcoef(pred, target)[0, 1])
        
        mean_pred = np.mean(pred)
        mean_target = np.mean(target)
        std_pred = np.std(pred)
        std_target = np.std(target)
        
        if mean_target > 0:
            beta = mean_pred / mean_target  # Bias ratio
        else:
            beta = 1.0
            
        if std_target > 0:
            alpha = std_pred / std_target  # Variability ratio
        else:
            alpha = 1.0
            
        metrics['KGE'] = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        # Index of Agreement (d)
        numerator = np.sum((pred - target) ** 2)
        denominator = np.sum((np.abs(pred - mean_target) + np.abs(target - mean_target)) ** 2)
        if denominator > 0:
            metrics['IoA'] = 1 - numerator / denominator
        else:
            metrics['IoA'] = 0.0
            
        return metrics
    
    def _categorical_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Compute categorical metrics for a given threshold."""
        
        # Convert to binary
        pred_binary = (pred >= threshold).astype(int)
        target_binary = (target >= threshold).astype(int)
        
        # Contingency table
        TP = np.sum((pred_binary == 1) & (target_binary == 1))  # Hits
        TN = np.sum((pred_binary == 0) & (target_binary == 0))  # Correct negatives
        FP = np.sum((pred_binary == 1) & (target_binary == 0))  # False alarms
        FN = np.sum((pred_binary == 0) & (target_binary == 1))  # Misses
        
        N = TP + TN + FP + FN
        
        metrics = {}
        
        # Probability of Detection (POD) / Hit Rate
        if TP + FN > 0:
            metrics['POD'] = TP / (TP + FN)
        else:
            metrics['POD'] = 0.0
            
        # False Alarm Ratio (FAR)
        if TP + FP > 0:
            metrics['FAR'] = FP / (TP + FP)
        else:
            metrics['FAR'] = 0.0
            
        # Probability of False Detection (POFD)
        if FP + TN > 0:
            metrics['POFD'] = FP / (FP + TN)
        else:
            metrics['POFD'] = 0.0
            
        # Critical Success Index (CSI) / Threat Score
        if TP + FP + FN > 0:
            metrics['CSI'] = TP / (TP + FP + FN)
        else:
            metrics['CSI'] = 0.0
            
        # Bias Score
        if TP + FN > 0:
            metrics['BIAS'] = (TP + FP) / (TP + FN)
        else:
            metrics['BIAS'] = 1.0
            
        # Heidke Skill Score (HSS)
        expected_correct = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / N if N > 0 else 0
        if N - expected_correct > 0:
            metrics['HSS'] = (TP + TN - expected_correct) / (N - expected_correct)
        else:
            metrics['HSS'] = 0.0
            
        # Equitable Threat Score (ETS) / Gilbert Skill Score
        if N > 0:
            hits_random = (TP + FN) * (TP + FP) / N
        else:
            hits_random = 0
            
        if TP + FP + FN - hits_random > 0:
            metrics['ETS'] = (TP - hits_random) / (TP + FP + FN - hits_random)
        else:
            metrics['ETS'] = 0.0
            
        # Accuracy
        metrics['ACC'] = (TP + TN) / N if N > 0 else 0.0
        
        # F1 Score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = metrics['POD']
        if precision + recall > 0:
            metrics['F1'] = 2 * precision * recall / (precision + recall)
        else:
            metrics['F1'] = 0.0
            
        return metrics
    
    @staticmethod
    def compute_fss(
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float,
        scale: int
    ) -> float:
        """
        Compute Fractions Skill Score (FSS).
        
        FSS measures the skill of predicting the fraction of
        precipitation area within a neighborhood.
        
        Args:
            pred: Predictions [H, W]
            target: Targets [H, W]
            threshold: Precipitation threshold
            scale: Neighborhood size (in grid points)
            
        Returns:
            FSS value (0-1, higher is better)
        """
        # Convert to binary
        pred_binary = (pred >= threshold).astype(float)
        target_binary = (target >= threshold).astype(float)
        
        # Compute fractions using uniform filter
        pred_frac = uniform_filter(pred_binary, size=scale, mode='constant')
        target_frac = uniform_filter(target_binary, size=scale, mode='constant')
        
        # Fractions Skill Score
        mse = np.mean((pred_frac - target_frac) ** 2)
        mse_ref = np.mean(pred_frac ** 2) + np.mean(target_frac ** 2)
        
        if mse_ref > 0:
            fss = 1 - mse / mse_ref
        else:
            fss = 1.0 if mse == 0 else 0.0
            
        return fss
    
    @staticmethod
    def compute_sal(
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute SAL (Structure, Amplitude, Location) score.
        
        Object-based verification metric that separately evaluates:
        - S: Structure of precipitation objects
        - A: Amplitude (intensity)
        - L: Location of precipitation
        
        Args:
            pred: Predictions [H, W]
            target: Targets [H, W]
            threshold: Threshold for object identification
            
        Returns:
            Dictionary with S, A, L components
        """
        from scipy import ndimage
        
        # Amplitude component (-2 to 2, 0 is perfect)
        mean_pred = np.mean(pred)
        mean_target = np.mean(target)
        
        if mean_target + mean_pred > 0:
            A = (mean_pred - mean_target) / (0.5 * (mean_pred + mean_target))
        else:
            A = 0.0
            
        # Location component (0 to 2, 0 is perfect)
        # Using center of mass
        if np.sum(pred) > 0:
            pred_com = ndimage.center_of_mass(pred)
        else:
            pred_com = (pred.shape[0] / 2, pred.shape[1] / 2)
            
        if np.sum(target) > 0:
            target_com = ndimage.center_of_mass(target)
        else:
            target_com = (target.shape[0] / 2, target.shape[1] / 2)
            
        # Normalize by domain size
        domain_size = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
        com_distance = np.sqrt((pred_com[0] - target_com[0]) ** 2 + 
                               (pred_com[1] - target_com[1]) ** 2)
        L = (com_distance / domain_size) * 2
        
        # Structure component (-2 to 2, 0 is perfect)
        # Based on normalized volume of objects
        total_pred = np.sum(pred)
        total_target = np.sum(target)
        
        max_pred = np.max(pred) if np.max(pred) > 0 else 1
        max_target = np.max(target) if np.max(target) > 0 else 1
        
        V_pred = total_pred / max_pred if max_pred > 0 else 0
        V_target = total_target / max_target if max_target > 0 else 0
        
        if V_target + V_pred > 0:
            S = (V_pred - V_target) / (0.5 * (V_pred + V_target))
        else:
            S = 0.0
            
        return {'S': S, 'A': A, 'L': L, 'SAL': np.abs(S) + np.abs(A) + L}


def compute_improvement(
    original_metrics: Dict[str, float],
    corrected_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute improvement from original to bias-corrected predictions.
    
    Args:
        original_metrics: Metrics before bias correction
        corrected_metrics: Metrics after bias correction
        
    Returns:
        Dictionary of improvements (positive = better after correction)
    """
    improvements = {}
    
    # Metrics where lower is better
    lower_better = ['MAE', 'RMSE', 'MBE', 'PBIAS', 'MAPE', 'FAR', 'POFD']
    
    # Metrics where higher is better
    higher_better = ['Pearson_R', 'Spearman_Rho', 'R2', 'NSE', 'KGE', 'IoA',
                     'POD', 'CSI', 'HSS', 'ETS', 'ACC', 'F1']
    
    for key in original_metrics:
        if key in corrected_metrics:
            orig = original_metrics[key]
            corr = corrected_metrics[key]
            
            # Check if this metric exists in either category
            is_lower_better = any(lb in key for lb in lower_better)
            is_higher_better = any(hb in key for hb in higher_better)
            
            if is_lower_better:
                # Improvement = reduction in error
                improvements[f"{key}_improvement"] = orig - corr
                if orig != 0:
                    improvements[f"{key}_improvement_pct"] = 100 * (orig - corr) / abs(orig)
                    
            elif is_higher_better:
                # Improvement = increase in score
                improvements[f"{key}_improvement"] = corr - orig
                if orig != 0:
                    improvements[f"{key}_improvement_pct"] = 100 * (corr - orig) / abs(orig)
                    
    return improvements


def print_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics"
):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics
        title: Table title
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    # Group by category
    continuous = ['MAE', 'RMSE', 'MBE', 'PBIAS', 'Pearson_R', 'R2']
    skill = ['NSE', 'KGE', 'IoA']
    
    print("\n Continuous Metrics:")
    print("-" * 40)
    for key in continuous:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:12.4f}")
            
    print("\n Skill Scores:")
    print("-" * 40)
    for key in skill:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:12.4f}")
            
    print("\n Categorical Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if any(cat in key for cat in ['POD', 'FAR', 'CSI', 'HSS', 'ETS']):
            print(f"  {key:20s}: {value:12.4f}")
            
    print("=" * 60 + "\n")
