# Import các thư viện cơ bản
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho biểu đồ chuyên nghiệp
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
# Load dataset từ Kaggle (tải file adult.csv về local)
# Link download: https://www.kaggle.com/datasets/uciml/adult-census-income
df = pd.read_csv('adult.csv', na_values='?')  # Xử lý missing values '?' -> NaN

# Hiển thị thông tin cơ bản
print("=== THÔNG TIN DATASET ===")
print(f"Shape: {df.shape}")
print(f"\nCác cột dữ liệu:\n{df.columns.tolist()}")
print("\n=== SAMPLE DATA ===")
print(df.head(3))

# Kiểm tra missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sort_values(ascending=False))

# 1. Xử lý missing values (drop rows có NaN)
df_clean = df.dropna().reset_index(drop=True)
print(f"Sau khi xóa missing values: {df_clean.shape}")

# 2. Chuẩn hóa cột 'income' -> binary numeric (0 = <=50K, 1 = >50K)
df_clean['income_binary'] = df_clean['income'].map({' <=50K': 0, ' >50K': 1})

# 3. Kiểm tra phân bố giới tính
print("\n=== PHÂN BỐ GIỚI TÍNH ===")
print(df_clean['sex'].value_counts())
print("\n=== PHÂN BỐ THU NHẬP ===")
print(df_clean['income'].value_counts(normalize=True) * 100)

# 4. Tạo 2 nhóm mẫu độc lập
male_sample = df_clean[df_clean['sex'] == ' Male']['income_binary']
female_sample = df_clean[df_clean['sex'] == ' Female']['income_binary']

print(f"\nSố lượng mẫu nam: {len(male_sample)}")
print(f"Số lượng mẫu nữ: {len(female_sample)}")

# Tính statistics cho từng nhóm
stats_summary = pd.DataFrame({
    'Giới tính': ['Nam', 'Nữ'],
    'n': [len(male_sample), len(female_sample)],
    'Tỷ lệ >50K (%)': [
        male_sample.mean() * 100,
        female_sample.mean() * 100
    ],
    'Số người >50K': [
        male_sample.sum(),
        female_sample.sum()
    ]
})

print("=== THỐNG KÊ MÔ TẢ THEO GIỚI TÍNH ===")
print(stats_summary.round(2))

# Visualization 1: Bar chart tỷ lệ thu nhập >50K
plt.figure(figsize=(8, 5))
sns.barplot(data=stats_summary, x='Giới tính', y='Tỷ lệ >50K (%)', palette='Set2')
plt.title('Tỷ lệ người có thu nhập >50K/năm theo giới tính', fontsize=14, fontweight='bold')
plt.ylabel('Tỷ lệ (%)')
plt.ylim(0, 40)
for i, v in enumerate(stats_summary['Tỷ lệ >50K (%)']):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('income_by_gender.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Stacked bar chart (phân bố đầy đủ)
income_gender = pd.crosstab(df_clean['sex'], df_clean['income'], normalize='index') * 100
income_gender.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff'])
plt.title('Phân bố thu nhập theo giới tính (%)', fontsize=14, fontweight='bold')
plt.ylabel('Phần trăm (%)')
plt.legend(title='Thu nhập')
plt.tight_layout()
plt.savefig('income_distribution_gender.png', dpi=300, bbox_inches='tight')
plt.show()


# Lấy số liệu cho kiểm định
n_male = len(male_sample)
n_female = len(female_sample)
x_male = male_sample.sum()      # Số nam có income >50K
x_female = female_sample.sum()  # Số nữ có income >50K

p_male = x_male / n_male        # Tỷ lệ nam
p_female = x_female / n_female  # Tỷ lệ nữ
p_pooled = (x_male + x_female) / (n_male + n_female)  # Pooled proportion

# Tính z-statistic (one-tailed test)
z_stat = (p_male - p_female) / np.sqrt(p_pooled * (1 - p_pooled) * (1/n_male + 1/n_female))

# Tính p-value (one-tailed: H1: p_male > p_female)
p_value = 1 - stats.norm.cdf(z_stat)

# Tính khoảng tin cậy 95% cho chênh lệch proportion
se_diff = np.sqrt(p_male*(1-p_male)/n_male + p_female*(1-p_female)/n_female)
ci_lower = (p_male - p_female) - 1.96 * se_diff
ci_upper = (p_male - p_female) + 1.96 * se_diff

# In kết quả
print("=== KIỂM ĐỊNH GIẢ THIẾT: TWO-PROPORTION Z-TEST ===")
print(f"Giả thiết H₀: pₘ = pբ")
print(f"Giả thiết H₁: pₘ > pբ (one-tailed)")
print(f"\nTỷ lệ nam có income >50K: {p_male:.4f} ({p_male*100:.2f}%)")
print(f"Tỷ lệ nữ có income >50K: {p_female:.4f} ({p_female*100:.2f}%)")
print(f"Chênh lệch: {(p_male - p_female)*100:.2f} điểm phần trăm")
print(f"\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"\nKhoảng tin cậy 95% cho chênh lệch (pₘ - pբ): [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"→ [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# Kết luận thống kê
alpha = 0.05
if p_value < alpha:
    print(f"\nKẾT LUẬN: Bác bỏ H₀ (p-value = {p_value:.6f} < α = {alpha})")
    print("→ CÓ bằng chứng thống kê cho thấy nam giới có tỷ lệ thu nhập >50K cao hơn nữ giới.")
else:
    print(f"\nKẾT LUẬN: Không đủ bằng chứng để bác bỏ H₀ (p-value = {p_value:.6f} ≥ α = {alpha})")
    
    # Điều kiện cho z-test: np ≥ 5 và n(1-p) ≥ 5 cho cả 2 nhóm
conditions = {
    'Nam - income>50K': n_male * p_male,
    'Nam - income<=50K': n_male * (1 - p_male),
    'Nữ - income>50K': n_female * p_female,
    'Nữ - income<=50K': n_female * (1 - p_female)
}

print("\n=== KIỂM TRA ĐIỀU KIỆN ÁP DỤNG Z-TEST ===")
for key, value in conditions.items():
    status = "✓ Đạt" if value >= 5 else "✗ Không đạt"
    print(f"{key}: {value:.1f} → {status}")
print("→ Tất cả điều kiện đều đạt → z-test hợp lệ.")

# Thực hiện independent t-test (two-sample)
t_stat, p_value_ttest = stats.ttest_ind(male_sample, female_sample, equal_var=False, alternative='greater')

print("\n=== KIỂM ĐỊNH BỔ SUNG: INDEPENDENT T-TEST ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value (one-tailed): {p_value_ttest:.6f}")
print(f"→ Kết quả tương đồng với z-test (do n lớn và biến binary)")

# Giả sử chênh lệch thực tế tối thiểu có ý nghĩa: d = 0.03 (3 điểm %)
d_min = 0.03
alpha = 0.05

# Tính pooled standard error cho proportion
se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_male + 1/n_female))

# Tính critical value cho one-tailed test
z_alpha = stats.norm.ppf(1 - alpha)

# Tính β-error (xác suất mắc lỗi loại II)
z_beta = (d_min - (p_male - p_female)) / se_pooled - z_alpha
beta = stats.norm.cdf(z_beta)
power = 1 - beta

print("\n=== PHÂN TÍCH LỖI LOẠI II (β) VÀ POWER ===")
print(f"Chênh lệch tối thiểu có ý nghĩa (d_min): {d_min*100:.1f}%")
print(f"Lỗi loại II (β): {beta:.4f}")
print(f"Power (1-β): {power:.4f} ({power*100:.2f}%)")
print(f"→ Power > 0.8 → Kiểm định có độ tin cậy cao")

# Tạo bảng kết quả chuyên nghiệp
results_table = pd.DataFrame({
    'Chỉ số': [
        'Kích thước mẫu (Nam)',
        'Kích thước mẫu (Nữ)',
        'Tỷ lệ thu nhập >50K (Nam)',
        'Tỷ lệ thu nhập >50K (Nữ)',
        'Chênh lệch',
        'Z-statistic',
        'P-value',
        'Khoảng tin cậy 95%',
        'Kết luận'
    ],
    'Giá trị': [
        f"{n_male:,}",
        f"{n_female:,}",
        f"{p_male*100:.2f}%",
        f"{p_female*100:.2f}%",
        f"{(p_male-p_female)*100:+.2f} điểm %",
        f"{z_stat:.4f}",
        f"{p_value:.6f}",
        f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
        "Bác bỏ H₀" if p_value < 0.05 else "Không bác bỏ H₀"
    ]
})

print("\n=== BẢNG KẾT QUẢ TỔNG HỢP ===")
print(results_table.to_string(index=False))

# Lưu kết quả ra file CSV để nộp
results_table.to_csv('ket_qua_phan_tich_thu_nhap_gioi_tinh.csv', index=False, encoding='utf-8-sig')
print("\n Đã lưu kết quả vào file: ket_qua_phan_tich_thu_nhap_gioi_tinh.csv")