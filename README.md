# Data-Analyst
## Mục tiêu:
## Kiểm định giả thuyết thu nhập nam và nữ

| Giả thuyết        | Ký hiệu   | Phát biểu                                      | Ý nghĩa               |
|-------------------|-----------|------------------------------------------------|-----------------------|
| **H₀ (Null)**     | pₘ = pₚ   | Tỷ lệ thu nhập >50K của nam **bằng** nữ        | Không có sự khác biệt |
| **H₁ (Alternative)** | pₘ > pₚ | Tỷ lệ thu nhập >50K của nam **cao hơn** nữ     | Có sự khác biệt       |
## Quick Start
1. Vào thư mục
cd adult-income-analysis

2. Kích hoạt venv
.\venv\Scripts\Activate.ps1

3. Chạy Streamlit
streamlit run app.py

4. Truy cập
http://localhost:8501

# Sơ đồ quy trình phân tích
1. **Thu thập dữ liệu**
2. **Làm sạch & chuẩn hóa**
3. **Thống kê mô tả**
4. **Kiểm tra điều kiện kiểm định**
5. **Tính toán kiểm định** (Z-test, T-test)
6. **Tính khoảng tin cậy & Effect Size**
7. **Ra quyết định** (bác bỏ/không bác bỏ H₀)
8. **Diễn giải kết quả**
