# ============================================================================
# ANALYZER.PY - PHÂN TÍCH KHÁC BIỆT THU NHẬP THEO GIỚI TÍNH
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class IncomeAnalyzer:
    """
    Phân tích và kiểm định sự khác biệt thu nhập theo giới tính
    Phương pháp: Two-Proportion Z-Test
    """
    
    def __init__(self):
        self.df = None
        self.df_clean = None
        self.results = {}
        self.income_col = 'income'
    
    # ========================================================================
    # 1. LOAD VÀ LÀM SẠCH DỮ LIỆU
    # ========================================================================
    def load_data(self, file_path: str) -> bool:
        """Load dataset từ file CSV"""
        try:
            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, na_values=['?', ' ?', ''], encoding=encoding)
                    return True
                except UnicodeDecodeError:
                    continue
            return False
        except Exception as e:
            print(f"Error loading  {str(e)}")
            return False
    
    def load_from_dataframe(self, df: pd.DataFrame) -> bool:
        """Load dataset từ DataFrame"""
        try:
            self.df = df.copy()
            return True
        except Exception as e:
            print(f"Error loading dataframe: {str(e)}")
            return False
    
    def clean_data(self) -> bool:
        """Làm sạch dữ liệu"""
        if self.df is None:
            return False
        
        self.df_clean = self.df.copy()
        
        # Xử lý missing values
        categorical_cols = ['workclass', 'occupation', 'native-country']
        for col in categorical_cols:
            if col in self.df_clean.columns:
                mode_value = self.df_clean[col].mode()[0] if len(self.df_clean[col].mode()) > 0 else 'Unknown'
                self.df_clean[col].fillna(mode_value, inplace=True)
        
        self.df_clean = self.df_clean.dropna().reset_index(drop=True)
        
        # Xác định cột thu nhập
        self.income_col = 'income' if 'income' in self.df_clean.columns else 'salary'
        
        # Chuẩn hóa thu nhập
        self.df_clean[self.income_col] = self.df_clean[self.income_col].astype(str).str.strip()
        self.df_clean['income_binary'] = self.df_clean[self.income_col].map({
            '<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1
        })
        
        # Chuẩn hóa giới tính
        if 'sex' in self.df_clean.columns:
            self.df_clean['sex'] = self.df_clean['sex'].astype(str).str.strip()
            self.df_clean = self.df_clean[self.df_clean['sex'].isin(['Male', 'Female'])]
        
        self.df_clean = self.df_clean.dropna(subset=['income_binary', 'sex'])
        self.df_clean = self.df_clean.reset_index(drop=True)
        
        return len(self.df_clean) > 0
    
    # ========================================================================
    # 2. THỐNG KÊ MÔ TẢ
    # ========================================================================
    def get_descriptive_stats(self) -> Dict:
        """Thống kê mô tả thu nhập theo giới tính"""
        if self.df_clean is None or 'sex' not in self.df_clean.columns:
            return {'n_male': 0, 'n_female': 0, 'p_male': 0.0, 'p_female': 0.0, 
                    'x_male': 0, 'x_female': 0, 'total': 0}
        
        male_sample = self.df_clean[self.df_clean['sex'] == 'Male']['income_binary']
        female_sample = self.df_clean[self.df_clean['sex'] == 'Female']['income_binary']
        
        return {
            'n_male': int(len(male_sample)),
            'n_female': int(len(female_sample)),
            'p_male': float(male_sample.mean()) if len(male_sample) > 0 else 0.0,
            'p_female': float(female_sample.mean()) if len(female_sample) > 0 else 0.0,
            'x_male': int(male_sample.sum()) if len(male_sample) > 0 else 0,
            'x_female': int(female_sample.sum()) if len(female_sample) > 0 else 0,
            'total': int(len(self.df_clean))
        }
    
    # ========================================================================
    # 3. KIỂM ĐỊNH THỐNG KÊ (ĐÃ LOẠI BỎ ODDS RATIO & COHEN'S H)
    # ========================================================================
    def run_z_test(self) -> Dict:
        """
        Two-Proportion Z-Test
        Áp dụng: Chương 3.5, 4.6
        """
        stats_desc = self.get_descriptive_stats()
        
        if stats_desc['n_male'] == 0 or stats_desc['n_female'] == 0:
            return {'error': 'Không đủ dữ liệu cho cả 2 giới tính'}
        
        n_male = stats_desc['n_male']
        n_female = stats_desc['n_female']
        x_male = stats_desc['x_male']
        x_female = stats_desc['x_female']
        p_male = stats_desc['p_male']
        p_female = stats_desc['p_female']
        
        # Pooled proportion
        p_pooled = (x_male + x_female) / (n_male + n_female)
        
        # Standard Error
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_male + 1/n_female))
        
        # Z-statistic
        z_stat = (p_male - p_female) / se_pooled if se_pooled > 0 else 0.0
        
        # P-value (one-tailed: H₁: pₘ > pբ)
        if z_stat > 37:
            p_value_one_tail = 1e-300
        else:
            p_value_one_tail = 1 - stats.norm.cdf(z_stat)
        
        # P-value (two-tailed)
        if abs(z_stat) > 37:
            p_value_two_tail = 2e-300
        else:
            p_value_two_tail = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Khoảng tin cậy 95% (Chương 4.6.3)
        se_diff = np.sqrt(p_male*(1-p_male)/n_male + p_female*(1-p_female)/n_female)
        ci_lower = (p_male - p_female) - 1.96 * se_diff
        ci_upper = (p_male - p_female) + 1.96 * se_diff
        
        # Kết luận
        alpha = 0.05
        reject_h0 = p_value_one_tail < alpha
        
        self.results = {
            'z_statistic': float(z_stat),
            'p_value_one_tail': float(p_value_one_tail),
            'p_value_two_tail': float(p_value_two_tail),
            'ci_lower_95': float(ci_lower),
            'ci_upper_95': float(ci_upper),
            'reject_h0': bool(reject_h0),
            'alpha': float(alpha),
            'se_pooled': float(se_pooled),
            'se_diff': float(se_diff),
            'p_pooled': float(p_pooled)
        }
        
        return self.results
    
    # ========================================================================
    # 4. KIỂM TRA ĐIỀU KIỆN (Chương 3.1.2, 3.2.5)
    # ========================================================================
    def check_assumptions(self) -> Dict:
        """Kiểm tra điều kiện áp dụng Z-test"""
        stats_desc = self.get_descriptive_stats()
        
        conditions = {
            'nam_income_gt_50k': stats_desc['n_male'] * stats_desc['p_male'],
            'nam_income_le_50k': stats_desc['n_male'] * (1 - stats_desc['p_male']),
            'nu_income_gt_50k': stats_desc['n_female'] * stats_desc['p_female'],
            'nu_income_le_50k': stats_desc['n_female'] * (1 - stats_desc['p_female'])
        }
        
        z_test_valid = all(v >= 5 for v in conditions.values())
        large_sample = stats_desc['n_male'] > 30 and stats_desc['n_female'] > 30
        
        return {
            'conditions': conditions,
            'z_test_valid': bool(z_test_valid),
            'large_sample': bool(large_sample),
            'all_valid': bool(z_test_valid and large_sample)
        }
    
    # ========================================================================
    # 5. POWER ANALYSIS (Chương 3.2.4, 3.5.2, 4.6.2)
    # ========================================================================
    def calculate_power(self, d_min: float = 0.03) -> Dict:
        """Tính Power và Beta error"""
        if not self.results:
            self.run_z_test()
        
        stats_desc = self.get_descriptive_stats()
        diff = stats_desc['p_male'] - stats_desc['p_female']
        se_pooled = self.results.get('se_pooled', 0.01)
        
        if se_pooled <= 0:
            return {'d_min': d_min, 'beta': 0.0, 'power': 1.0, 'adequate': True}
        
        z_alpha = stats.norm.ppf(1 - 0.05)
        z_beta = (d_min - diff) / se_pooled - z_alpha
        z_beta = np.clip(z_beta, -37, 37)
        beta = stats.norm.cdf(z_beta)
        power = 1 - beta
        
        return {
            'd_min': d_min,
            'beta': float(beta),
            'power': float(power),
            'adequate': bool(power > 0.80)
        }
    
    # ========================================================================
    # 6. EXPORT KẾT QUẢ (ĐÃ LOẠI BỎ ODDS RATIO & COHEN'S H)
    # ========================================================================
    def get_results_table(self) -> pd.DataFrame:
        """Xuất bảng kết quả tổng hợp"""
        stats_desc = self.get_descriptive_stats()
        
        if stats_desc['n_male'] == 0 and stats_desc['n_female'] == 0:
            return pd.DataFrame({
                'Chỉ số': ['Không có dữ liệu'],
                'Giá trị': ['Vui lòng tải dataset trước']
            })
        
        if not self.results:
            self.run_z_test()
        
        p_value_formatted = self.format_p_value(self.results.get('p_value_one_tail', 0))
        p_value_scientific = self.format_scientific(self.results.get('p_value_one_tail', 0))
        
        return pd.DataFrame({
            'Chỉ số': [
                'Kích thước mẫu (Nam)',
                'Kích thước mẫu (Nữ)',
                'Tỷ lệ >50K (Nam)',
                'Tỷ lệ >50K (Nữ)',
                'Chênh lệch',
                'Z-statistic',
                'P-value',
                'Khoảng tin cậy 95%',
                'Kết luận'
            ],
            'Giá trị': [
                f"{stats_desc['n_male']:,}",
                f"{stats_desc['n_female']:,}",
                f"{stats_desc['p_male']*100:.2f}%",
                f"{stats_desc['p_female']*100:.2f}%",
                f"{(stats_desc['p_male']-stats_desc['p_female'])*100:+.2f} điểm %",
                f"{self.results.get('z_statistic', 0):.4f}",
                p_value_formatted,
                f"[{self.results.get('ci_lower_95', 0)*100:.2f}%, {self.results.get('ci_upper_95', 0)*100:.2f}%]",
                "Bác bỏ H₀" if self.results.get('reject_h0', False) else "Không bác bỏ H₀"
            ]
        })
    
    def save_results(self, file_path: str) -> bool:
        """Lưu kết quả vào file CSV"""
        try:
            table = self.get_results_table()
            table.to_csv(file_path, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    @staticmethod
    def format_p_value(p_value: float) -> str:
        """Format p-value để hiển thị"""
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            return "N/A"
        if p_value < 0.0001:
            return "< 0.0001"
        elif p_value < 0.001:
            return f"{p_value:.4f}"
        elif p_value < 0.01:
            return f"{p_value:.3f}"
        else:
            return f"{p_value:.2f}"
    
    @staticmethod
    def format_scientific(p_value: float, decimals: int = 2) -> str:
        """Format p-value dạng khoa học"""
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            return "N/A"
        return f"{p_value:.{decimals}e}"
    
    def get_clean_data(self) -> Optional[pd.DataFrame]:
        """Trả về dữ liệu đã làm sạch"""
        return self.df_clean


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("PHÂN TÍCH KHÁC BIỆT THU NHẬP THEO GIỚI TÍNH")
    print("="*80)
    
    analyzer = IncomeAnalyzer()
    
    if analyzer.load_data('adult.csv'):
        print("✓ Loaded data")
        
        if analyzer.clean_data():
            print("✓ Cleaned data")
            
            stats_desc = analyzer.get_descriptive_stats()
            print(f"\n=== THỐNG KÊ MÔ TẢ ===")
            print(f"Năm: {stats_desc['n_male']:,} ({stats_desc['p_male']*100:.2f}%)")
            print(f"Nữ: {stats_desc['n_female']:,} ({stats_desc['p_female']*100:.2f}%)")
            
            results = analyzer.run_z_test()
            print(f"\n=== KIỂM ĐỊNH Z-TEST ===")
            print(f"Z-statistic: {results['z_statistic']:.4f}")
            print(f"P-value: {analyzer.format_p_value(results['p_value_one_tail'])}")
            print(f"Kết luận: {'Bác bỏ H₀' if results['reject_h0'] else 'Không bác bỏ H₀'}")
            
            assumptions = analyzer.check_assumptions()
            print(f"\n=== ĐIỀU KIỆN ===")
            print(f"Z-test valid: {assumptions['z_test_valid']}")
            
            analyzer.save_results('ket_qua_phan_tich.csv')
            print("\n✓ Đã lưu kết quả!")
        else:
            print("✗ Failed to clean data")
    else:
        print("✗ Failed to load data")
    
    print("="*80)
