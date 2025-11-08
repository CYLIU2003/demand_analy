"""
学術論文用のエクスポート機能
データ、統計レポート、グラフを論文品質でエクスポート
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json


class AcademicExporter:
    """学術研究用データエクスポーター"""
    
    def __init__(self, output_dir: str = "research_outputs"):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataset(self, data: pd.DataFrame, filename: str,
                      format: str = 'csv', include_metadata: bool = True) -> str:
        """
        データセットをエクスポート
        
        Args:
            data: エクスポートするデータフレーム
            filename: ファイル名（拡張子なし）
            format: 'csv', 'excel', 'parquet'
            include_metadata: メタデータを含めるか
        
        Returns:
            保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filepath = self.output_dir / f"{filename}_{timestamp}.csv"
            data.to_csv(filepath, index=True, encoding='utf-8-sig')
        
        elif format == 'excel':
            filepath = self.output_dir / f"{filename}_{timestamp}.xlsx"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=True)
                
                if include_metadata:
                    metadata = self._generate_metadata(data)
                    metadata_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata')
        
        elif format == 'parquet':
            filepath = self.output_dir / f"{filename}_{timestamp}.parquet"
            data.to_parquet(filepath, index=True)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Dataset exported to: {filepath}")
        return str(filepath)
    
    def export_statistics_report(self, statistics: Dict[str, Any],
                                 filename: str = "statistics_report",
                                 format: str = 'txt') -> str:
        """
        統計レポートをエクスポート
        
        Args:
            statistics: 統計情報の辞書
            filename: ファイル名
            format: 'txt', 'json', 'latex'
        
        Returns:
            保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'txt':
            filepath = self.output_dir / f"{filename}_{timestamp}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("電力需給データ統計分析レポート\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                self._write_statistics_recursive(f, statistics)
        
        elif format == 'json':
            filepath = self.output_dir / f"{filename}_{timestamp}.json"
            
            # DataFrameとSeriesをJSON互換形式に変換
            json_data = self._convert_to_json_serializable(statistics)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        elif format == 'latex':
            filepath = self.output_dir / f"{filename}_{timestamp}.tex"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\\documentclass{article}\n")
                f.write("\\usepackage[utf8]{inputenc}\n")
                f.write("\\usepackage{booktabs}\n")
                f.write("\\usepackage{array}\n")
                f.write("\\begin{document}\n\n")
                f.write("\\section{電力需給データ統計分析レポート}\n\n")
                
                self._write_statistics_latex(f, statistics)
                
                f.write("\\end{document}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Statistics report exported to: {filepath}")
        return str(filepath)
    
    def export_model_results(self, model_name: str,
                            training_history: Dict[str, List[float]],
                            evaluation_metrics: Dict[str, float],
                            config: Dict[str, Any],
                            filename: Optional[str] = None) -> str:
        """
        モデルの訓練結果をエクスポート
        
        Args:
            model_name: モデル名
            training_history: 訓練履歴
            evaluation_metrics: 評価指標
            config: モデル設定
            filename: ファイル名（オプション）
        
        Returns:
            保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{model_name}_results_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.json"
        
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config': config,
            'training_history': training_history,
            'evaluation_metrics': evaluation_metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Model results exported to: {filepath}")
        return str(filepath)
    
    def export_comparison_table(self, results: List[Dict[str, Any]],
                               filename: str = "model_comparison",
                               format: str = 'csv') -> str:
        """
        複数モデルの比較表をエクスポート
        
        Args:
            results: モデル結果のリスト
            filename: ファイル名
            format: 'csv', 'excel', 'latex'
        
        Returns:
            保存されたファイルのパス
        """
        # 比較表を作成
        comparison_data = []
        for result in results:
            row = {
                'モデル名': result.get('model_name', 'Unknown'),
                **result.get('evaluation_metrics', {})
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filepath = self.output_dir / f"{filename}_{timestamp}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        elif format == 'excel':
            filepath = self.output_dir / f"{filename}_{timestamp}.xlsx"
            df.to_excel(filepath, index=False)
        
        elif format == 'latex':
            filepath = self.output_dir / f"{filename}_{timestamp}.tex"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(df.to_latex(index=False, float_format="%.4f"))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Comparison table exported to: {filepath}")
        return str(filepath)
    
    def _generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """データフレームのメタデータを生成"""
        metadata = {
            'レコード数': len(data),
            'カラム数': len(data.columns),
            'カラム名': ', '.join(data.columns.tolist()),
            'データ型': ', '.join([str(dtype) for dtype in data.dtypes]),
            'メモリ使用量（MB）': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}",
            '欠損値の総数': data.isnull().sum().sum(),
            '生成日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 時刻カラムがあれば期間を追加
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            first_col = datetime_cols[0]
            metadata['開始日時'] = str(data[first_col].min())
            metadata['終了日時'] = str(data[first_col].max())
        
        return metadata
    
    def _write_statistics_recursive(self, f, data: Any, indent: int = 0):
        """統計情報を再帰的にテキストファイルに書き込み"""
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, pd.DataFrame, pd.Series)):
                    f.write(f"{prefix}【{key}】\n")
                    self._write_statistics_recursive(f, value, indent + 1)
                else:
                    f.write(f"{prefix}{key}: {value}\n")
            f.write("\n")
        
        elif isinstance(data, pd.DataFrame):
            f.write(prefix + data.to_string() + "\n\n")
        
        elif isinstance(data, pd.Series):
            f.write(prefix + data.to_string() + "\n\n")
        
        elif isinstance(data, (list, np.ndarray)):
            f.write(f"{prefix}{data}\n\n")
        
        else:
            f.write(f"{prefix}{data}\n\n")
    
    def _write_statistics_latex(self, f, data: Any):
        """統計情報をLaTeX形式で書き込み"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    f.write(f"\\subsection{{{key}}}\n")
                    f.write(value.to_latex(float_format="%.4f"))
                    f.write("\n")
                elif isinstance(value, dict):
                    f.write(f"\\subsection{{{key}}}\n")
                    self._write_statistics_latex(f, value)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """オブジェクトをJSON互換形式に変換"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
