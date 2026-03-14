# ============================================================================
# ANALYZER.PY - XỬ LÝ LOGIC THỐNG KÊ
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IncomeAnalyzer:
    """Class xử lý phân tích thu nhập theo giới tính"""
    
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
            self.df = pd.read_csv(file_path, na_values=['?', ' ?', ''])
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
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
        
        # Drop rows không có income_binary
        self.df_clean = self.df_clean.dropna(subset=['income_binary'])
        
        return len(self.df_clean) > 0
    
    # ========================================================================
    # 2. THỐNG KÊ MÔ TẢ
    # ========================================================================
    def get_descriptive_stats(self) -> Dict:
        """Trả về thống kê mô tả"""
        if self.df_clean is None or 'sex' not in self.df_clean.columns:
            return {}
        
        male_sample = self.df_clean[self.df_clean['sex'] == 'Male']['income_binary']
        female_sample = self.df_clean[self.df_clean['sex'] == 'Female']['income_binary']
        
        return {
            'n_male': len(male_sample),
            'n_female': len(female_sample),
            'p_male': male_sample.mean() if len(male_sample) > 0 else 0,
            'p_female': female_sample.mean() if len(female_sample) > 0 else 0,
            'x_male': male_sample.sum() if len(male_sample) > 0 else 0,
            'x_female': female_sample.sum() if len(female_sample) > 0 else 0,
            'total': len(self.df_clean)
        }
    
    # ========================================================================
    # 3. KIỂM ĐỊNH THỐNG KÊ
    # ========================================================================
    def run_z_test(self) -> Dict:
        """Two-proportion Z-test"""
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
        
        # Standard error
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_male + 1/n_female))
        
        # Z-statistic
        z_stat = (p_male - p_female) / se_pooled if se_pooled > 0 else 0
        
        # P-value (one-tailed)
        p_value_one_tail = 1 - stats.norm.cdf(z_stat)
        
        # P-value (two-tailed)
        p_value_two_tail = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Khoảng tin cậy 95%
        se_diff = np.sqrt(p_male*(1-p_male)/n_male + p_female*(1-p_female)/n_female)
        ci_lower = (p_male - p_female) - 1.96 * se_diff
        ci_upper = (p_male - p_female) + 1.96 * se_diff
        
        # Effect size (Cohen's h)
        cohens_h = 2 * (np.arcsin(np.sqrt(p_male)) - np.arcsin(np.sqrt(p_female)))
        
        # Odds Ratio
        odds_male = p_male / (1 - p_male) if p_male < 1 else np.inf
        odds_female = p_female / (1 - p_female) if p_female < 1 else np.inf
        odds_ratio = odds_male / odds_female if odds_female > 0 else np.inf
        
        # Kết luận
        alpha = 0.05
        reject_h0 = p_value_one_tail < alpha
        
        self.results = {
            'z_statistic': z_stat,
            'p_value_one_tail': p_value_one_tail,
            'p_value_two_tail': p_value_two_tail,
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
            'cohens_h': cohens_h,
            'odds_ratio': odds_ratio,
            'reject_h0': reject_h0,
            'alpha': alpha,
            'se_pooled': se_pooled,
            'se_diff': se_diff,
            'p_pooled': p_pooled
        }
        
        return self.results
    
    def run_t_test(self) -> Dict:
        """Independent T-test (Welch's)"""
        if self.df_clean is None:
            return {'error': 'Chưa có dữ liệu'}
        
        male_sample = self.df_clean[self.df_clean['sex'] == 'Male']['income_binary']
        female_sample = self.df_clean[self.df_clean['sex'] == 'Female']['income_binary']
        
        if len(male_sample) == 0 or len(female_sample) == 0:
            return {'error': 'Không đủ dữ liệu'}
        
        t_stat, p_value_two = stats.ttest_ind(male_sample, female_sample, equal_var=False)
        p_value_one = p_value_two / 2
        
        # Degrees of freedom (Welch-Satterthwaite)
        var_male = male_sample.var()
        var_female = female_sample.var()
        n_male, n_female = len(male_sample), len(female_sample)
        
        df_welch = ((var_male/n_male + var_female/n_female)**2 / 
                   ((var_male/n_male)**2/(n_male-1) + (var_female/n_female)**2/(n_female-1)))
        
        return {
            't_statistic': t_stat,
            'p_value_one_tail': p_value_one,
            'p_value_two_tail': p_value_two,
            'df_welch': df_welch
        }
    
    def check_assumptions(self) -> Dict:
        """Kiểm tra điều kiện áp dụng kiểm định"""
        stats_desc = self.get_descriptive_stats()
        
        conditions = {
            'nam_income_gt_50k': stats_desc['n_male'] * stats_desc['p_male'],
            'nam_income_le_50k': stats_desc['n_male'] * (1 - stats_desc['p_male']),
            'nu_income_gt_50k': stats_desc['n_female'] * stats_desc['p_female'],
            'nu_income_le_50k': stats_desc['n_female'] * (1 - stats_desc['p_female']),
            'sample_size_male': stats_desc['n_male'],
            'sample_size_female': stats_desc['n_female']
        }
        
        # Kiểm tra np >= 5
        z_test_valid = all(v >= 5 for k, v in conditions.items() if 'income' in k)
        
        # Kiểm tra mẫu lớn
        large_sample = stats_desc['n_male'] > 30 and stats_desc['n_female'] > 30
        
        return {
            'conditions': conditions,
            'z_test_valid': z_test_valid,
            'large_sample': large_sample,
            'all_valid': z_test_valid and large_sample
        }
    
    def calculate_power(self, d_min: float = 0.03) -> Dict:
        """Tính Power và Beta error"""
        if not self.results:
            self.run_z_test()
        
        stats_desc = self.get_descriptive_stats()
        diff = stats_desc['p_male'] - stats_desc['p_female']
        se_pooled = self.results.get('se_pooled', 0.01)
        
        z_alpha = stats.norm.ppf(1 - 0.05)  # one-tailed
        z_beta = (d_min - diff) / se_pooled - z_alpha if se_pooled > 0 else 0
        beta = stats.norm.cdf(z_beta)
        power = 1 - beta
        
        return {
            'd_min': d_min,
            'beta': beta,
            'power': power,
            'adequate': power > 0.80
        }
    
    # ========================================================================
    # 4. EXPORT KẾT QUẢ
    # ========================================================================
    def get_results_table(self) -> pd.DataFrame:
        """Trả về bảng kết quả dưới dạng DataFrame"""
        stats_desc = self.get_descriptive_stats()
        
        if not self.results:
            self.run_z_test()
        
        table = pd.DataFrame({
            'Chỉ số': [
                'Kích thước mẫu (Nam)',
                'Kích thước mẫu (Nữ)',
                'Tỷ lệ >50K (Nam)',
                'Tỷ lệ >50K (Nữ)',
                'Chênh lệch',
                'Z-statistic',
                'P-value (one-tail)',
                'Khoảng tin cậy 95%',
                'Cohen\'s h',
                'Odds Ratio',
                'Kết luận'
            ],
            'Giá trị': [
                f"{stats_desc['n_male']:,}",
                f"{stats_desc['n_female']:,}",
                f"{stats_desc['p_male']*100:.2f}%",
                f"{stats_desc['p_female']*100:.2f}%",
                f"{(stats_desc['p_male']-stats_desc['p_female'])*100:+.2f} điểm %",
                f"{self.results.get('z_statistic', 0):.4f}",
                f"{self.results.get('p_value_one_tail', 0):.8f}",
                f"[{self.results.get('ci_lower_95', 0)*100:.2f}%, {self.results.get('ci_upper_95', 0)*100:.2f}%]",
                f"{self.results.get('cohens_h', 0):.4f}",
                f"{self.results.get('odds_ratio', 0):.4f}",
                "Bác bỏ H₀" if self.results.get('reject_h0', False) else "Không bác bỏ H₀"
            ]
        })
        
        return table
    
    def save_results(self, file_path: str) -> bool:
        """Lưu kết quả vào file CSV"""
        try:
            table = self.get_results_table()
            table.to_csv(file_path, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    def get_clean_data(self) -> Optional[pd.DataFrame]:
        """Trả về dữ liệu đã làm sạch"""
        return self.df_clean


# ============================================================================
# HÀM TIỆN ÍCH ĐỂ TEST NHANH
# ============================================================================
if __name__ == "__main__":
    # Test nhanh analyzer
    analyzer = IncomeAnalyzer()
    
    # Test với file
    if analyzer.load_data('adult.csv'):
        print("✓ Loaded data")
        
        if analyzer.clean_data():
            print("✓ Cleaned data")
            
            stats_desc = analyzer.get_descriptive_stats()
            print(f"\n=== THỐNG KÊ MÔ TẢ ===")
            print(f"Năm: {stats_desc['n_male']:,} ({stats_desc['p_male']*100:.2f}%)")
            print(f"Nữ: {stats_desc['n_female']:,} ({stats_desc['p_female']*100:.2f}%)")
            
            results = analyzer.run_z_test()
            print(f"\n=== Z-TEST ===")
            print(f"Z-statistic: {results['z_statistic']:.4f}")
            print(f"P-value: {results['p_value_one_tail']:.8f}")
            print(f"Kết luận: {'Bác bỏ H₀' if results['reject_h0'] else 'Không bác bỏ H₀'}")
            
            # Lưu kết quả
            analyzer.save_results('ket_qua_phan_tich.csv')
            print("\n✓ Đã lưu kết quả!")
