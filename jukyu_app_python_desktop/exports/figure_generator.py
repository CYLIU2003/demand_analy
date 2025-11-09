"""
学術論文用の高品質な図表生成モジュール
Publication-ready figures for academic papers
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


class FigureGenerator:
    """学術論文用の図表生成器"""
    
    def __init__(self, output_dir: str = "figures", style: str = 'seaborn-v0_8-paper'):
        """
        Args:
            output_dir: 出力ディレクトリ
            style: matplotlibのスタイル
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 論文用の設定
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 論文品質の設定
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 13
    
    def plot_time_series(self, data: pd.DataFrame, time_column: str,
                        value_columns: List[str], title: str = "",
                        xlabel: str = "時刻", ylabel: str = "電力 (MW)",
                        figsize: Tuple[float, float] = (10, 6),
                        filename: Optional[str] = None,
                        show_grid: bool = True,
                        legend_loc: str = 'best') -> str:
        """
        時系列データのプロット（論文品質）
        
        Args:
            data: データフレーム
            time_column: 時刻カラム
            value_columns: プロットする値のカラムリスト
            title: タイトル
            xlabel: X軸ラベル
            ylabel: Y軸ラベル
            figsize: 図のサイズ
            filename: ファイル名（Noneの場合は自動生成）
            show_grid: グリッドを表示するか
            legend_loc: 凡例の位置
        
        Returns:
            保存されたファイルのパス
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # カラーマップ
        colors = plt.cm.tab10(np.linspace(0, 1, len(value_columns)))
        
        for idx, col in enumerate(value_columns):
            if col in data.columns:
                ax.plot(data[time_column], data[col], label=col,
                       color=colors[idx], linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if title:
            ax.set_title(title, fontweight='bold', pad=15)
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax.legend(loc=legend_loc, framealpha=0.9)
        
        # 日付フォーマット
        if pd.api.types.is_datetime64_any_dtype(data[time_column]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            fig.autofmt_xdate(rotation=45)
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timeseries_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # PDFも保存（ベクター形式）
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        print(f"PDF version saved to: {pdf_path}")
        
        return str(filepath)
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame,
                               title: str = "相関行列",
                               figsize: Tuple[float, float] = (10, 8),
                               filename: Optional[str] = None,
                               cmap: str = 'coolwarm',
                               vmin: float = -1.0, vmax: float = 1.0) -> str:
        """
        相関行列のヒートマップ
        
        Args:
            correlation_matrix: 相関行列
            title: タイトル
            figsize: 図のサイズ
            filename: ファイル名
            cmap: カラーマップ
            vmin: 最小値
            vmax: 最大値
        
        Returns:
            保存されたファイルのパス
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(correlation_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect='auto', interpolation='nearest')
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相関係数', rotation=270, labelpad=20)
        
        # 軸ラベル
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.index)
        
        # 値を表示
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title, fontweight='bold', pad=15)
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlation_matrix_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        return str(filepath)
    
    def plot_decomposition(self, decomposition_result: Dict[str, pd.Series],
                          title: str = "時系列分解",
                          figsize: Tuple[float, float] = (12, 10),
                          filename: Optional[str] = None) -> str:
        """
        時系列分解結果のプロット
        
        Args:
            decomposition_result: 分解結果（observed, trend, seasonal, residual）
            title: タイトル
            figsize: 図のサイズ
            filename: ファイル名
        
        Returns:
            保存されたファイルのパス
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        labels = ['観測値', 'トレンド', '季節性', '残差']
        
        for ax, comp, label in zip(axes, components, labels):
            if comp in decomposition_result:
                data = decomposition_result[comp]
                ax.plot(data.index, data.values, linewidth=1.2)
                ax.set_ylabel(label, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        axes[0].set_title(title, fontweight='bold', pad=15)
        axes[-1].set_xlabel('時刻', fontweight='bold')
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"decomposition_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        return str(filepath)
    
    def plot_acf_pacf(self, acf_values: np.ndarray, pacf_values: np.ndarray,
                     title: str = "自己相関と偏自己相関",
                     figsize: Tuple[float, float] = (12, 6),
                     filename: Optional[str] = None,
                     confidence_level: float = 0.95) -> str:
        """
        ACFとPACFのプロット
        
        Args:
            acf_values: 自己相関係数
            pacf_values: 偏自己相関係数
            title: タイトル
            figsize: 図のサイズ
            filename: ファイル名
            confidence_level: 信頼区間
        
        Returns:
            保存されたファイルのパス
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 信頼区間
        confidence_interval = 1.96 / np.sqrt(len(acf_values))
        
        # ACF
        lags = np.arange(len(acf_values))
        ax1.stem(lags, acf_values, basefmt=' ', linefmt='C0-', markerfmt='C0o')
        ax1.axhline(y=0, color='black', linewidth=0.8)
        ax1.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('ラグ', fontweight='bold')
        ax1.set_ylabel('ACF', fontweight='bold')
        ax1.set_title('自己相関関数', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # PACF
        lags = np.arange(len(pacf_values))
        ax2.stem(lags, pacf_values, basefmt=' ', linefmt='C1-', markerfmt='C1o')
        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('ラグ', fontweight='bold')
        ax2.set_ylabel('PACF', fontweight='bold')
        ax2.set_title('偏自己相関関数', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontweight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"acf_pacf_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        return str(filepath)
    
    def plot_prediction_comparison(self, time_index: pd.Index,
                                   actual: np.ndarray,
                                   predicted: np.ndarray,
                                   title: str = "予測結果の比較",
                                   xlabel: str = "時刻",
                                   ylabel: str = "電力 (MW)",
                                   figsize: Tuple[float, float] = (12, 6),
                                   filename: Optional[str] = None) -> str:
        """
        実測値と予測値の比較プロット
        
        Args:
            time_index: 時刻インデックス
            actual: 実測値
            predicted: 予測値
            title: タイトル
            xlabel: X軸ラベル
            ylabel: Y軸ラベル
            figsize: 図のサイズ
            filename: ファイル名
        
        Returns:
            保存されたファイルのパス
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 上段: 実測値 vs 予測値
        ax1.plot(time_index, actual, label='実測値', color='C0',
                linewidth=1.5, alpha=0.8)
        ax1.plot(time_index, predicted, label='予測値', color='C1',
                linewidth=1.5, alpha=0.8, linestyle='--')
        ax1.set_ylabel(ylabel, fontweight='bold')
        ax1.set_title(title, fontweight='bold', pad=15)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 下段: 誤差
        error = actual - predicted
        ax2.plot(time_index, error, label='誤差', color='C2', linewidth=1.2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel(xlabel, fontweight='bold')
        ax2.set_ylabel('誤差', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 日付フォーマット
        if isinstance(time_index, pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            fig.autofmt_xdate(rotation=45)
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_comparison_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        return str(filepath)
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "訓練履歴",
                             figsize: Tuple[float, float] = (10, 6),
                             filename: Optional[str] = None) -> str:
        """
        モデルの訓練履歴をプロット
        
        Args:
            history: 訓練履歴（train_loss, val_lossなど）
            title: タイトル
            figsize: 図のサイズ
            filename: ファイル名
        
        Returns:
            保存されたファイルのパス
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], label='訓練損失',
               color='C0', linewidth=1.5, marker='o', markersize=3)
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], label='検証損失',
                   color='C1', linewidth=1.5, marker='s', markersize=3)
        
        ax.set_xlabel('エポック', fontweight='bold')
        ax.set_ylabel('損失', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
        print(f"Figure saved to: {filepath}")
        return str(filepath)
