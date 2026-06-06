#!/usr/bin/env python
"""
Evaluation and Visualization Script
=====================================

Generates professional outputs for Government of India presentation:
1. Comprehensive metrics report
2. Before/after correction comparison maps
3. Skill score improvements
4. Spatial correlation maps
5. Time series analysis
6. Publication-quality figures

Author: IITM Pune - Government of India Project
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Try to import cartopy for map projections (optional)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print(" Cartopy not installed. Using basic plots.")

from robust.models.msua_net import MSUANet, create_model
from robust.data.datasets import RobustNormalizer
from robust.evaluation.metrics import BiasCorrectionMetrics, print_metrics_table, compute_improvement


class BiasCorrectionEvaluator:
    """
    Complete evaluation pipeline for bias correction model.
    
    Generates all outputs needed for Government presentation.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        output_dir: str = "evaluation_results",
        device: str = 'cuda'
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = self._load_model()
        
        # Load normalizer
        normalizer_path = self.checkpoint_path.parent / 'normalizer.json'
        if normalizer_path.exists():
            self.normalizer = RobustNormalizer()
            self.normalizer.load(str(normalizer_path))
        else:
            self.normalizer = None
            
        # Metrics calculator
        self.metrics = BiasCorrectionMetrics()
        
        # Results storage
        self.results = {}
        
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        print(f" Loading model from: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Create model with same config
        model_config = config.get('model', {
            'in_channels': 10,
            'out_channels': 1,
            'base_filters': 64,
            'num_levels': 4,
            'uncertainty': True
        })
        
        model = create_model(model_config, self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f" Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return model
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        nwp_data: np.ndarray,
        obs_data: np.ndarray,
        batch_size: int = 32
    ) -> dict:
        """
        Evaluate model on full dataset.
        
        Args:
            nwp_data: NWP model data [T, H, W] or [T, C, H, W]
            obs_data: Observation data [T, H, W]
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n Evaluating on dataset...")
        
        # Normalize
        if self.normalizer:
            nwp_norm = self.normalizer.transform(nwp_data)
        else:
            nwp_norm = nwp_data
            
        # Prepare input shape
        if nwp_norm.ndim == 3:
            # [T, H, W] -> [T, 1, H, W]
            nwp_norm = nwp_norm[:, np.newaxis, :, :]
            
        # Run inference in batches
        all_predictions = []
        all_variances = []
        
        for i in range(0, len(nwp_norm), batch_size):
            batch = torch.from_numpy(nwp_norm[i:i+batch_size]).float().to(self.device)
            mean, var = self.model(batch)
            all_predictions.append(mean.cpu().numpy())
            if var is not None:
                all_variances.append(var.cpu().numpy())
                
        predictions = np.concatenate(all_predictions, axis=0)
        
        if all_variances:
            variances = np.concatenate(all_variances, axis=0)
        else:
            variances = None
            
        # Denormalize
        if self.normalizer:
            predictions = self.normalizer.inverse_transform(predictions)
            
        # Squeeze to [T, H, W]
        predictions = predictions.squeeze(1)
        
        # Compute metrics
        print(" Computing metrics...")
        
        # Original NWP vs Observations
        original_metrics = self.metrics.compute_all(nwp_data, obs_data)
        
        # Bias-corrected vs Observations
        corrected_metrics = self.metrics.compute_all(predictions, obs_data)
        
        # Improvements
        improvements = compute_improvement(original_metrics, corrected_metrics)
        
        results = {
            'predictions': predictions,
            'variances': variances,
            'original_metrics': original_metrics,
            'corrected_metrics': corrected_metrics,
            'improvements': improvements
        }
        
        self.results = results
        return results
    
    def generate_metrics_report(self) -> str:
        """Generate comprehensive metrics report."""
        
        report_path = self.output_dir / 'metrics_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  IITM PUNE BIAS CORRECTION - EVALUATION REPORT\n")
            f.write("  Government of India Project\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Original metrics
            f.write("-" * 70 + "\n")
            f.write("  ORIGINAL NWP MODEL PERFORMANCE (Before Correction)\n")
            f.write("-" * 70 + "\n")
            for key, value in self.results['original_metrics'].items():
                if isinstance(value, float):
                    f.write(f"    {key:30s}: {value:12.4f}\n")
                    
            # Corrected metrics
            f.write("\n" + "-" * 70 + "\n")
            f.write("  BIAS-CORRECTED MODEL PERFORMANCE (After Correction)\n")
            f.write("-" * 70 + "\n")
            for key, value in self.results['corrected_metrics'].items():
                if isinstance(value, float):
                    f.write(f"    {key:30s}: {value:12.4f}\n")
                    
            # Improvements
            f.write("\n" + "-" * 70 + "\n")
            f.write("  IMPROVEMENTS\n")
            f.write("-" * 70 + "\n")
            for key, value in self.results['improvements'].items():
                if isinstance(value, float):
                    if 'pct' in key:
                        f.write(f"    {key:30s}: {value:+12.2f}%\n")
                    else:
                        f.write(f"    {key:30s}: {value:+12.4f}\n")
                        
            f.write("\n" + "=" * 70 + "\n")
            
        print(f" Report saved: {report_path}")
        return str(report_path)
    
    def generate_comparison_figure(
        self,
        nwp_data: np.ndarray,
        obs_data: np.ndarray,
        time_idx: int = 0,
        save_name: str = "comparison_map.png"
    ):
        """
        Generate side-by-side comparison figure.
        
        Shows: Original NWP | Bias-Corrected | Observation | Difference
        """
        pred = self.results['predictions'][time_idx]
        nwp = nwp_data[time_idx]
        obs = obs_data[time_idx]
        
        fig = plt.figure(figsize=(20, 5))
        
        # Color settings for rainfall
        cmap = plt.cm.Blues
        vmax = max(nwp.max(), obs.max(), pred.max())
        
        # Common plot settings
        titles = [
            'Original NWP (GFS)',
            'Bias-Corrected (MSUA-Net)',
            'Observation (IMDAA)',
            'Correction Applied (BC - NWP)'
        ]
        
        data = [nwp, pred, obs, pred - nwp]
        cmaps = [cmap, cmap, cmap, plt.cm.RdBu_r]
        vmaxs = [vmax, vmax, vmax, None]
        
        for i, (d, title, cm, vm) in enumerate(zip(data, titles, cmaps, vmaxs)):
            ax = fig.add_subplot(1, 4, i + 1)
            
            if i == 3:  # Difference plot
                limit = np.abs(d).max()
                im = ax.imshow(d, cmap=cm, origin='lower', vmin=-limit, vmax=limit)
            else:
                im = ax.imshow(d, cmap=cm, origin='lower', vmin=0, vmax=vm)
                
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Bias Correction Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Figure saved: {save_path}")
        
    def generate_skill_scores_figure(self, save_name: str = "skill_scores.png"):
        """Generate bar chart comparing skill scores before/after correction."""
        
        # Key metrics to show
        metrics_to_show = ['Pearson_R', 'NSE', 'KGE', 'R2']
        
        original = [self.results['original_metrics'].get(m, 0) for m in metrics_to_show]
        corrected = [self.results['corrected_metrics'].get(m, 0) for m in metrics_to_show]
        
        x = np.arange(len(metrics_to_show))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, original, width, label='Original NWP', color='#ff6b6b', alpha=0.8)
        bars2 = ax.bar(x + width/2, corrected, width, label='Bias-Corrected', color='#4ecdc4', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Skill Score Improvement\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_show, fontsize=11)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.2, 1.1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Figure saved: {save_path}")
        
    def generate_error_metrics_figure(self, save_name: str = "error_metrics.png"):
        """Generate bar chart comparing error metrics before/after correction."""
        
        metrics_to_show = ['MAE', 'RMSE', 'MBE']
        
        original = [self.results['original_metrics'].get(m, 0) for m in metrics_to_show]
        corrected = [self.results['corrected_metrics'].get(m, 0) for m in metrics_to_show]
        
        x = np.arange(len(metrics_to_show))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars1 = ax.bar(x - width/2, original, width, label='Original NWP', color='#ff6b6b', alpha=0.8)
        bars2 = ax.bar(x + width/2, corrected, width, label='Bias-Corrected', color='#4ecdc4', alpha=0.8)
        
        ax.set_ylabel('Error (mm)', fontsize=12)
        ax.set_title('Error Reduction\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_show, fontsize=11)
        ax.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Figure saved: {save_path}")
        
    def generate_categorical_metrics_figure(self, save_name: str = "categorical_metrics.png"):
        """Generate bar chart for categorical metrics at different thresholds."""
        
        thresholds = ['1.0mm', '5.0mm', '10.0mm', '20.0mm']
        metrics = ['POD', 'CSI', 'ETS']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            original = []
            corrected = []
            
            for thresh in thresholds:
                key = f"{metric}_{thresh}"
                original.append(self.results['original_metrics'].get(key, 0))
                corrected.append(self.results['corrected_metrics'].get(key, 0))
                
            x = np.arange(len(thresholds))
            width = 0.35
            
            ax.bar(x - width/2, original, width, label='Original', color='#ff6b6b', alpha=0.8)
            ax.bar(x + width/2, corrected, width, label='Corrected', color='#4ecdc4', alpha=0.8)
            
            ax.set_ylabel('Score')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(thresholds, fontsize=9)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)
            
        plt.suptitle('Categorical Metrics by Precipitation Threshold', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Figure saved: {save_path}")
        
    def generate_correlation_map(
        self,
        nwp_data: np.ndarray,
        obs_data: np.ndarray,
        save_name: str = "correlation_improvement.png"
    ):
        """Generate spatial correlation improvement map."""
        
        pred = self.results['predictions']
        
        # Compute pixel-wise correlation
        H, W = obs_data.shape[1], obs_data.shape[2]
        
        orig_corr = np.zeros((H, W))
        corr_corr = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):
                # Original correlation
                if np.std(nwp_data[:, i, j]) > 0 and np.std(obs_data[:, i, j]) > 0:
                    orig_corr[i, j] = np.corrcoef(nwp_data[:, i, j], obs_data[:, i, j])[0, 1]
                    corr_corr[i, j] = np.corrcoef(pred[:, i, j], obs_data[:, i, j])[0, 1]
                    
        improvement = corr_corr - orig_corr
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original correlation
        im1 = axes[0].imshow(orig_corr, cmap='RdYlGn', vmin=-1, vmax=1, origin='lower')
        axes[0].set_title('Original NWP Correlation', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Corrected correlation
        im2 = axes[1].imshow(corr_corr, cmap='RdYlGn', vmin=-1, vmax=1, origin='lower')
        axes[1].set_title('Bias-Corrected Correlation', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Improvement
        im3 = axes[2].imshow(improvement, cmap='RdBu', vmin=-0.5, vmax=0.5, origin='lower')
        axes[2].set_title('Correlation Improvement', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        for ax in axes:
            ax.axis('off')
            
        plt.suptitle('Spatial Correlation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Figure saved: {save_path}")
        
    def generate_summary_figure(self, save_name: str = "summary_figure.png"):
        """Generate a single summary figure with key results."""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('IITM Pune Bias Correction - Results Summary\nGovernment of India Project', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Skill scores comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['R', 'NSE', 'KGE']
        original = [self.results['original_metrics'].get('Pearson_R', 0),
                   self.results['original_metrics'].get('NSE', 0),
                   self.results['original_metrics'].get('KGE', 0)]
        corrected = [self.results['corrected_metrics'].get('Pearson_R', 0),
                    self.results['corrected_metrics'].get('NSE', 0),
                    self.results['corrected_metrics'].get('KGE', 0)]
        
        x = np.arange(len(metrics))
        ax1.bar(x - 0.2, original, 0.4, label='Original', color='#ff6b6b')
        ax1.bar(x + 0.2, corrected, 0.4, label='Corrected', color='#4ecdc4')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_title('Skill Scores', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.set_ylim(-0.2, 1)
        
        # 2. Error reduction
        ax2 = fig.add_subplot(gs[0, 1])
        errors = ['MAE', 'RMSE']
        orig_err = [self.results['original_metrics'].get(e, 0) for e in errors]
        corr_err = [self.results['corrected_metrics'].get(e, 0) for e in errors]
        
        x = np.arange(len(errors))
        ax2.bar(x - 0.2, orig_err, 0.4, label='Original', color='#ff6b6b')
        ax2.bar(x + 0.2, corr_err, 0.4, label='Corrected', color='#4ecdc4')
        ax2.set_xticks(x)
        ax2.set_xticklabels(errors)
        ax2.set_title('Error Metrics (mm)', fontweight='bold')
        ax2.legend(fontsize=8)
        
        # 3. Key improvements text box
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        mae_imp = self.results['improvements'].get('MAE_improvement_pct', 0)
        rmse_imp = self.results['improvements'].get('RMSE_improvement_pct', 0)
        r_imp = self.results['improvements'].get('Pearson_R_improvement_pct', 0)
        
        text = f"""KEY IMPROVEMENTS
        
MAE Reduction: {mae_imp:+.1f}%
RMSE Reduction: {rmse_imp:+.1f}%
Correlation Increase: {r_imp:+.1f}%

Model: MSUA-Net
Parameters: ~15M
Training: Mixed Precision"""
        
        ax3.text(0.1, 0.5, text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Summary saved: {save_path}")
        
    def generate_all_outputs(
        self,
        nwp_data: np.ndarray,
        obs_data: np.ndarray
    ):
        """Generate all evaluation outputs."""
        
        print("\n" + "=" * 60)
        print("  Generating Government of India Report Outputs")
        print("=" * 60)
        
        # Run evaluation
        self.evaluate_dataset(nwp_data, obs_data)
        
        # Generate all outputs
        self.generate_metrics_report()
        self.generate_comparison_figure(nwp_data, obs_data, time_idx=0)
        self.generate_comparison_figure(nwp_data, obs_data, time_idx=len(nwp_data)//2, 
                                        save_name="comparison_map_mid.png")
        self.generate_skill_scores_figure()
        self.generate_error_metrics_figure()
        self.generate_categorical_metrics_figure()
        self.generate_correlation_map(nwp_data, obs_data)
        self.generate_summary_figure()
        
        # Save results as JSON
        results_json = {
            'original_metrics': {k: float(v) for k, v in self.results['original_metrics'].items() if isinstance(v, (int, float))},
            'corrected_metrics': {k: float(v) for k, v in self.results['corrected_metrics'].items() if isinstance(v, (int, float))},
            'improvements': {k: float(v) for k, v in self.results['improvements'].items() if isinstance(v, (int, float))}
        }
        
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f" Results JSON saved: {json_path}")
        
        print("\n" + "=" * 60)
        print("   All outputs generated!")
        print(f"   Output directory: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bias Correction Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--nwp_data', type=str, required=True, help='Path to NWP data NetCDF')
    parser.add_argument('--obs_data', type=str, required=True, help='Path to observation NetCDF')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    import xarray as xr
    
    # Load data
    print(" Loading data...")
    with xr.open_dataset(args.nwp_data) as ds:
        nwp_data = ds.rf.values
    with xr.open_dataset(args.obs_data) as ds:
        obs_data = ds.rf.values
        
    # Create evaluator
    evaluator = BiasCorrectionEvaluator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Generate all outputs
    evaluator.generate_all_outputs(nwp_data, obs_data)


if __name__ == '__main__':
    main()
