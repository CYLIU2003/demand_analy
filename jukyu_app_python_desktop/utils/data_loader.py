"""
データ読み込みユーティリティ
複数ファイルの一括読み込み、マージ機能を提供
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import re


class DataLoader:
    """電力需給データの読み込みを行うクラス"""
    
    FNAME_PATTERN = re.compile(r"^eria_jukyu_(\d{6})_(\d{2})\.csv$")
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
    
    def read_csv(self, filepath: Path, 
                 encodings: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        CSVファイルを読み込み
        
        Args:
            filepath: ファイルパス
            encodings: 試行するエンコーディングのリスト
        
        Returns:
            (DataFrame, 時刻カラム名)のタプル
        """
        if encodings is None:
            encodings = ["shift_jis", "cp932", "utf-8", "utf-8-sig"]
        
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, engine="python", skiprows=0)
                # 単位行をスキップ
                if "単位" in str(df.columns[0]) or "MW" in str(df.columns[0]):
                    df = pd.read_csv(filepath, encoding=encoding, engine="python", skiprows=1)
                break
            except (UnicodeDecodeError, Exception):
                continue
        
        if df is None:
            raise ValueError(f"ファイルを読み込めませんでした: {filepath}")
        
        # 時刻カラムを検出
        time_col = self._detect_time_column(df)
        
        # 数値カラムを変換
        for column in df.columns:
            if column != time_col:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except Exception:
                    continue
        
        return df, time_col
    
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """時刻カラムを検出"""
        date_col = None
        time_col = None
        
        for column in df.columns:
            column_upper = str(column).upper()
            if "DATE" in column_upper or "日付" in str(column):
                date_col = column
            if "TIME" in column_upper or "時刻" in str(column) or "時間" in str(column):
                time_col = column
        
        # 日付と時刻を結合
        if date_col and time_col:
            try:
                df["datetime"] = pd.to_datetime(
                    df[date_col].astype(str) + " " + df[time_col].astype(str),
                    errors="coerce",
                )
                if df["datetime"].notna().sum() > 0:
                    return "datetime"
            except Exception:
                pass
        
        # その他の時刻カラムを探す
        for column in df.columns:
            if any(keyword in str(column).lower() for keyword in ["datetime", "date", "time", "日時"]):
                try:
                    parsed = pd.to_datetime(df[column], errors="coerce")
                    if parsed.notna().sum() > 0:
                        df[column] = parsed
                        return column
                except Exception:
                    continue
        
        # 最初のカラムを試す
        try:
            first_column = df.columns[0]
            parsed = pd.to_datetime(df[first_column], errors="coerce")
            if parsed.notna().sum() > 0:
                df[first_column] = parsed
                return first_column
        except Exception:
            pass
        
        return None
    
    def load_multiple_files(self, area_code: str, 
                           year_months: Optional[List[str]] = None) -> pd.DataFrame:
        """
        複数ファイルを読み込んで結合
        
        Args:
            area_code: エリアコード（01-10）
            year_months: 年月のリスト（YYYYMM形式）。Noneの場合は全て
        
        Returns:
            結合されたDataFrame
        """
        if not self.data_dir.exists():
            raise ValueError(f"データディレクトリが見つかりません: {self.data_dir}")
        
        dfs = []
        
        for filepath in sorted(self.data_dir.iterdir()):
            match = self.FNAME_PATTERN.match(filepath.name)
            if not match:
                continue
            
            file_ym, file_area = match.group(1), match.group(2)
            
            if file_area != area_code:
                continue
            
            if year_months is not None and file_ym not in year_months:
                continue
            
            try:
                df, time_col = self.read_csv(filepath)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"該当するファイルが見つかりませんでした（エリア: {area_code}）")
        
        # 結合
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 時刻でソート
        time_col = self._detect_time_column(combined_df)
        if time_col and time_col in combined_df.columns:
            combined_df = combined_df.sort_values(time_col).reset_index(drop=True)
        
        return combined_df
    
    def get_available_files(self) -> List[Tuple[str, str, Path]]:
        """
        利用可能なファイルのリストを取得
        
        Returns:
            (年月, エリアコード, パス)のタプルのリスト
        """
        files = []
        
        if not self.data_dir.exists():
            return files
        
        for filepath in sorted(self.data_dir.iterdir()):
            match = self.FNAME_PATTERN.match(filepath.name)
            if match:
                year_month, area_code = match.group(1), match.group(2)
                files.append((year_month, area_code, filepath))
        
        return files
