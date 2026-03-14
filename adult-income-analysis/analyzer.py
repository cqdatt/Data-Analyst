# ============================================================================
# ANALYZER.PY - XỬ LÝ LOGIC THỐNG KÊ (FINAL - CLEAN)
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
    # UTILITY: FORMAT P-VALUE
    # ========================================================================
    @staticmethod
    def format_p_value(p_value: float) -> str:
        """Format p-value để hiển thị chuyên nghiệp"""
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
        if p_value < 0.001 or p_value > 0.999:
            return f"{p_value:.{decimals}e}"
        return f"{p_value:.{decimals}f}"
    
    # ========================================================================
    # 1. LOAD VÀ LÀM SẠCH DỮ LIỆU
    # ========================================================================
    def load_data(self, file_path: str) -> bool:
        """Load dataset từ file CSV"""
        try:
            encodings = ['utf-8', 'latin1', 'cp1252', 'utf-8-sig']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, na_values=['?', ' ?', ''], encoding=encoding)
                    return True
                except UnicodeDecodeError:
                    continue
            return False
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
            '<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1,
            '<=50K ': 0, '>50K ': 1
        })
        
        # Chuẩn hóa giới tính
        if 'sex' in self.df_clean.columns:
            self.df_clean['sex'] = self.df_clean['sex'].astype(str).str.strip()
        
        # Lọc chỉ giữ Male và Female
        if 'sex' in self.df_clean.columns:
            self.df_clean = self.df_clean[self.df_clean['sex'].isin(['Male', 'Female'])]
        
        # Drop rows không có income_binary
        self.df_clean = self.df_clean.dropna(subset=['income_binary', 'sex'])
        self.df_clean = self.df_clean.reset_index(drop=True)
        
        return len(self.df_clean) > 0
    
    # ========================================================================
    # 2. THỐNG KÊ MÔ TẢ
    # ========================================================================
    def get_descriptive_stats(self) -> Dict:
        """Trả về thống kê mô tả"""
        if self.df_clean is None or 'sex' not in self.df_clean.columns:
            return {
                'n_male': 0,
                'n_female': 0,
                'p_male': 0.0,
                'p_female': 0.0,
                'x_male': 0,
                'x_female': 0,
                'total': 0
            }
        
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
    # 3. KIỂM ĐỊNH THỐNG KÊ (Z-TEST)
    # ========================================================================
    def run_z_test(self) -> Dict:
        """Two-proportion Z-test (one-tailed cho H1: p_male > p_female)"""
        stats_desc = self.get_descriptive_stats()
        
        if stats_desc['n_male'] == 0 or stats_desc['n_female'] == 0:
            return {
                'error': 'Không đủ dữ liệu cho cả 2 giới tính',
                'z_statistic': 0,
                'p_value_one_tail': 1.0,
                'p_value_two_tail': 1.0,
                'ci_lower_95': 0,
                'ci_upper_95': 0,
                'cohens_h': 0,
                'odds_ratio': 0,
                'reject_h0': False,
                'alpha': 0.05,
                'se_pooled': 0,
                'se_diff': 0,
                'p_pooled': 0
            }
        
        n_male = stats_desc['n_male']
        n_female = stats_desc['n_female']
        x_male = stats_desc['x_male']
        x_female = stats_desc['x_female']
        p_male = stats_desc['p_male']
        p_female = stats_desc['p_female']
        
        # Pooled proportion (dùng cho standard error của kiểm định)
        p_pooled = (x_male + x_female) / (n_male + n_female)
        
        # Standard Error (pooled) cho z-test
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_male + 1/n_female))
        
        # Z-statistic
        if se_pooled > 0:
            z_stat = (p_male - p_female) / se_pooled
        else:
            z_stat = 0.0
        
        # P-value (one-tailed) cho H1: p_male > p_female
        if z_stat > 37:
            p_value_one_tail = 1e-300
        else:
            p_value_one_tail = 1 - stats.norm.cdf(z_stat)
        
        # P-value (two-tailed) cho tham khảo
        if abs(z_stat) > 37:
            p_value_two_tail = 2e-300
        else:
            p_value_two_tail = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Khoảng tin cậy 95% cho chênh lệch (không dùng pooled)
        se_diff = np.sqrt(p_male*(1-p_male)/n_male + p_female*(1-p_female)/n_female)
        ci_lower = (p_male - p_female) - 1.96 * se_diff
        ci_upper = (p_male - p_female) + 1.96 * se_diff
        
        # Effect size (Cohen's h)
        # arcsin trả về giá trị trong [0, pi/2], an toàn với p=0 hoặc p=1
        cohens_h = 2 * (np.arcsin(np.sqrt(p_male)) - np.arcsin(np.sqrt(p_female)))
        
        # Odds Ratio (cẩn thận với trường hợp mẫu số bằng 0)
        if p_female == 1:
            odds_ratio = np.inf if p_male > p_female else 0
        elif p_female == 0:
            odds_ratio = np.inf if p_male > 0 else 1.0
        else:
            odds_male = p_male / (1 - p_male) if p_male < 1 else np.inf
            odds_female = p_female / (1 - p_female)
            odds_ratio = odds_male / odds_female if odds_female > 0 else np.inf
        
        # Kết luận (mức ý nghĩa α = 0.05, one-tailed)
        alpha = 0.05
        reject_h0 = p_value_one_tail < alpha
        
        self.results = {
            'z_statistic': float(z_stat),
            'p_value_one_tail': float(p_value_one_tail),
            'p_value_two_tail': float(p_value_two_tail),
            'ci_lower_95': float(ci_lower),
            'ci_upper_95': float(ci_upper),
            'cohens_h': float(cohens_h),
            'odds_ratio': float(odds_ratio) if odds_ratio != np.inf else np.inf,
            'reject_h0': bool(reject_h0),
            'alpha': float(alpha),
            'se_pooled': float(se_pooled),
            'se_diff': float(se_diff),
            'p_pooled': float(p_pooled)
        }
        
        return self.results
    
    def check_assumptions(self) -> Dict:
        """Kiểm tra điều kiện áp dụng z-test"""
        stats_desc = self.get_descriptive_stats()
        
        conditions = {
            'nam_income_gt_50k': stats_desc['n_male'] * stats_desc['p_male'],
            'nam_income_le_50k': stats_desc['n_male'] * (1 - stats_desc['p_male']),
            'nu_income_gt_50k': stats_desc['n_female'] * stats_desc['p_female'],
            'nu_income_le_50k': stats_desc['n_female'] * (1 - stats_desc['p_female']),
            'sample_size_male': stats_desc['n_male'],
            'sample_size_female': stats_desc['n_female']
        }
        
        # Điều kiện: expected counts >= 5
        expected_ok = all(v >= 5 for k, v in conditions.items() if 'income' in k)
        # Điều kiện: mẫu lớn (thường dùng cho z-test)
        large_sample = stats_desc['n_male'] > 30 and stats_desc['n_female'] > 30
        
        return {
            'conditions': conditions,
            'z_test_valid': bool(expected_ok),
            'large_sample': bool(large_sample),
            'all_valid': bool(expected_ok and large_sample)
        }
    
    def calculate_power(self, d_min: float = 0.03) -> Dict:
        """Tính post-hoc power cho kiểm định một phía (dựa trên chênh lệch quan sát)"""
        if not self.results:
            self.run_z_test()
        
        stats_desc = self.get_descriptive_stats()
        diff = stats_desc['p_male'] - stats_desc['p_female']
        se_pooled = self.results.get('se_pooled', 0.01)
        
        if se_pooled <= 0:
            return {'d_min': d_min, 'beta': 0.0, 'power': 1.0, 'adequate': True}
        
        # z_alpha cho one-tailed test với alpha = 0.05
        z_alpha = stats.norm.ppf(1 - 0.05)
        
        # Post-hoc power: xác suất bác bỏ H0 khi H1 đúng với effect size quan sát diff
        # Công thức: power = 1 - norm.cdf(z_alpha - diff / se_pooled)
        # (diff dương vì H1 là p_male > p_female)
        z_beta = z_alpha - diff / se_pooled
        z_beta = np.clip(z_beta, -37, 37)
        power = 1 - stats.norm.cdf(z_beta)
        beta = 1 - power
        
        return {
            'd_min': d_min,                # effect size tối thiểu mong muốn (tham khảo)
            'beta': float(beta),
            'power': float(power),
            'adequate': bool(power > 0.80)
        }
    
    # ========================================================================
    # 4. EXPORT KẾT QUẢ
    # ========================================================================
    def get_results_table(self) -> pd.DataFrame:
        """Trả về bảng kết quả dưới dạng DataFrame"""
        stats_desc = self.get_descriptive_stats()
        
        # Kiểm tra nếu không có dữ liệu
        if stats_desc['n_male'] == 0 and stats_desc['n_female'] == 0:
            return pd.DataFrame({
                'Chỉ số': ['Không có dữ liệu'],
                'Giá trị': ['Vui lòng tải dataset trước']
            })
        
        if not self.results:
            self.run_z_test()
        
        p_value_formatted = self.format_p_value(self.results.get('p_value_one_tail', 0))
        p_value_scientific = self.format_scientific(self.results.get('p_value_one_tail', 0))
        
        or_val = self.results.get('odds_ratio', 0)
        or_formatted = f"{or_val:.4f}" if or_val != np.inf else "> 999"
        
        table = pd.DataFrame({
            'Chỉ số': [
                'Kích thước mẫu (Nam)',
                'Kích thước mẫu (Nữ)',
                'Tỷ lệ >50K (Nam)',
                'Tỷ lệ >50K (Nữ)',
                'Chênh lệch (Nam - Nữ)',
                'Z-statistic',
                'P-value (one-tailed) - formatted',
                'P-value (one-tailed) - scientific',
                'Khoảng tin cậy 95% cho chênh lệch',
                "Cohen's h",
                'Odds Ratio',
                'Kết luận (α = 0.05, one-tailed)'
            ],
            'Giá trị': [
                f"{stats_desc['n_male']:,}",
                f"{stats_desc['n_female']:,}",
                f"{stats_desc['p_male']*100:.2f}%",
                f"{stats_desc['p_female']*100:.2f}%",
                f"{(stats_desc['p_male']-stats_desc['p_female'])*100:+.2f} điểm %",
                f"{self.results.get('z_statistic', 0):.4f}",
                p_value_formatted,
                p_value_scientific,
                f"[{self.results.get('ci_lower_95', 0)*100:.2f}%, {self.results.get('ci_upper_95', 0)*100:.2f}%]",
                f"{self.results.get('cohens_h', 0):.4f}",
                or_formatted,
                "Bác bỏ H₀ (nam > nữ)" if self.results.get('reject_h0', False) else "Không đủ cơ sở bác bỏ H₀"
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
# TEST
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("TEST ANALYZER.PY (FINAL CLEAN VERSION)")
    print("="*80)
    
    analyzer = IncomeAnalyzer()
    
    if analyzer.load_data('adult.csv'):
        print("✓ Loaded data")
        
        if analyzer.clean_data():
            print("✓ Cleaned data")
            
            stats_desc = analyzer.get_descriptive_stats()
            print(f"\n=== THỐNG KÊ MÔ TẢ ===")
            print(f"Nam: {stats_desc['n_male']:,} ({stats_desc['p_male']*100:.2f}%)")
            print(f"Nữ: {stats_desc['n_female']:,} ({stats_desc['p_female']*100:.2f}%)")
            
            results = analyzer.run_z_test()
            print(f"\n=== Z-TEST (ONE-TAILED) ===")
            print(f"Z-statistic: {results['z_statistic']:.4f}")
            print(f"P-value (formatted): {analyzer.format_p_value(results['p_value_one_tail'])}")
            print(f"Kết luận: {'Bác bỏ H₀ (nam > nữ)' if results['reject_h0'] else 'Không bác bỏ H₀'}")
            
            # Kiểm tra điều kiện
            assumptions = analyzer.check_assumptions()
            print(f"\n=== KIỂM TRA ĐIỀU KIỆN ===")
            print(f"Z-test valid: {'✓' if assumptions['z_test_valid'] else '✗'} (expected counts >=5)")
            print(f"Large sample: {'✓' if assumptions['large_sample'] else '✗'} (n > 30)")
            
            # Tính power
            power_info = analyzer.calculate_power(d_min=0.03)
            print(f"\n=== POST-HOC POWER ===")
            print(f"Power: {power_info['power']:.2%}")
            print(f"{'Đủ mạnh (≥80%)' if power_info['adequate'] else 'Chưa đủ mạnh'}")
            
            analyzer.save_results('ket_qua_phan_tich.csv')
            print("\n✓ Đã lưu kết quả!")
        else:
            print("✗ Failed to clean data")
    else:
        print("✗ Failed to load data")
    
    print("="*80)
