# ============================================================================
# APP.PY - GIAO DIỆN STREAMLIT (PROFESSIONAL DESIGN)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analyzer import IncomeAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="Phân Tích Thu Nhập Theo Giới Tính",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - TỐI ƯU GIAO DIỆN
# ============================================================================
st.markdown("""
<style>
    /* Giảm padding giữa các section */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style cho metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Style cho headers */
    h1, h2, h3 {
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Style cho dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    
    /* Style cho expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
    }
    
    /* Style cho success/warning boxes */
    .stSuccess, .stWarning, .stInfo, .stError {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# KHỞI TẠO ANALYZER
# ============================================================================
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = IncomeAnalyzer()
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

analyzer = st.session_state.analyzer

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### Điều Hướng")
    st.markdown("---")
    
    option = st.radio(
        "Chọn chức năng",
        ["Tải Dataset", "Nhập Dữ Liệu", "Phân Tích", "Kết Quả"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Thông Tin Project")
    st.info(
        "Project: Adult Income Analysis\n\n"
        "Chương: 3 & 4 - Thống kê\n\n"
        "Phương pháp: Two-Proportion Z-Test"
    )

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.5rem;">Phân Tích Thu Nhập Theo Giới Tính</h1>
    <p style="color: #666; margin-top: 0;">Nghiên cứu sự khác biệt về thu nhập giữa nam và nữ</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TAB 1: TẢI DATASET
# ============================================================================
if option == "Tải Dataset":
    col_main, col_info = st.columns([3, 1])
    
    with col_main:
        st.markdown("### Tải File Dữ Liệu")
        
        uploaded_file = st.file_uploader(
            "Chọn file CSV",
            type=['csv'],
            help="Tải dataset từ: https://www.kaggle.com/datasets/uciml/adult-census-income"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, na_values=['?', ' ?', ''])
                analyzer.load_from_dataframe(df)
                analyzer.clean_data()
                
                st.success(f"Đã tải thành công {len(df):,} dòng dữ liệu")
                
                # Preview data
                st.markdown("#### Xem Trước Dữ Liệu")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data info
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Số hàng", f"{df.shape[0]:,}")
                with col_b:
                    st.metric("Số cột", df.shape[1])
                with col_c:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    
    with col_info:
        st.markdown("### Hoặc Dùng Dữ Liệu Mẫu")
        st.markdown("Dữ liệu mẫu được tạo ngẫu nhiên để kiểm tra nhanh")
        
        if st.button("Tải Dữ Liệu Mẫu", use_container_width=True):
            np.random.seed(42)
            n = 1000
            
            data = {
                'age': np.random.randint(18, 80, n),
                'sex': np.random.choice(['Male', 'Female'], n, p=[0.67, 0.33]),
                'income': np.random.choice(['<=50K', '>50K'], n, p=[0.75, 0.25])
            }
            
            df = pd.DataFrame(data)
            mask = df['sex'] == 'Male'
            df.loc[mask, 'income'] = np.random.choice(['<=50K', '>50K'], mask.sum(), p=[0.67, 0.33])
            mask = df['sex'] == 'Female'
            df.loc[mask, 'income'] = np.random.choice(['<=50K', '>50K'], mask.sum(), p=[0.88, 0.12])
            
            analyzer.load_from_dataframe(df)
            analyzer.clean_data()
            st.success("Đã tải dữ liệu mẫu")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Hướng Dẫn")
        st.markdown("""
        1. Tải file adult.csv từ Kaggle
        2. Hoặc dùng dữ liệu mẫu để test
        3. Chuyển sang tab Phân Tích để tiếp tục
        """)

# ============================================================================
# TAB 2: NHẬP DỮ LIỆU
# ============================================================================
elif option == "Nhập Dữ Liệu":
    st.markdown("### Nhập Dữ Liệu Thủ Công")
    st.markdown("Thêm từng quan sát vào dataset")
    
    # Input form
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=35)
        
        with col2:
            sex = st.selectbox("Giới tính", ["Male", "Female"])
        
        with col3:
            income = st.selectbox("Thu nhập", ["<=50K", ">50K"])
        
        col_btn, col_space = st.columns([1, 3])
        with col_btn:
            if st.button("Thêm Dòng", use_container_width=True):
                new_row = pd.DataFrame({'age': [age], 'sex': [sex], 'income': [income]})
                
                if analyzer.df is None:
                    analyzer.df = new_row
                else:
                    analyzer.df = pd.concat([analyzer.df, new_row], ignore_index=True)
                
                analyzer.clean_data()
                st.success(f"Đã thêm! Tổng: {len(analyzer.df)} dòng")
                st.rerun()
    
    st.markdown("---")
    
    # Data preview
    if analyzer.df is not None:
        st.markdown("#### Dữ Liệu Đã Nhập")
        st.dataframe(analyzer.df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xóa Tất Cả", use_container_width=True):
                analyzer.df = None
                analyzer.df_clean = None
                st.rerun()
        with col2:
            if st.button("Chuyển Sang Phân Tích", use_container_width=True):
                st.session_state.analysis_done = True
                st.rerun()
    else:
        st.info("Chưa có dữ liệu. Vui lòng thêm dòng hoặc tải dataset.")

# ============================================================================
# TAB 3: PHÂN TÍCH
# ============================================================================
elif option == "Phân Tích":
    if analyzer.df_clean is None:
        st.warning("Vui lòng tải dữ liệu ở tab Tải Dataset trước")
        st.stop()
    
    # Chạy phân tích
    stats_desc = analyzer.get_descriptive_stats()
    results = analyzer.run_z_test()
    t_results = analyzer.run_t_test()
    assumptions = analyzer.check_assumptions()
    power = analyzer.calculate_power()
    
    st.session_state.analysis_done = True
    
    # === SECTION 1: TỔNG QUAN ===
    st.markdown("### Tổng Quan Mẫu")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mẫu Nam", f"{stats_desc['n_male']:,}")
    with col2:
        st.metric("Mẫu Nữ", f"{stats_desc['n_female']:,}")
    with col3:
        st.metric("Tỷ lệ Nam >50K", f"{stats_desc['p_male']*100:.1f}%")
    with col4:
        st.metric("Tỷ lệ Nữ >50K", f"{stats_desc['p_female']*100:.1f}%")
    
    st.markdown("---")
    
    # === SECTION 2: BIỂU ĐỒ ===
    st.markdown("### Trực Quan Hóa")
    
    tab1, tab2, tab3 = st.tabs(["Biểu Đồ Cột", "Biểu Đồ Tròn", "Heatmap"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Nam', 'Nữ'],
            y=[stats_desc['p_male']*100, stats_desc['p_female']*100],
            marker_color=['#2563eb', '#dc2626'],
            text=[f'{stats_desc["p_male"]*100:.1f}%', f'{stats_desc["p_female"]*100:.1f}%'],
            textposition='outside'
        ))
        fig.update_layout(
            title='Tỷ lệ thu nhập trên 50K theo giới tính',
            xaxis_title='Giới tính',
            yaxis_title='Tỷ lệ (%)',
            yaxis_range=[0, 40],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=[stats_desc['x_male'], stats_desc['n_male']-stats_desc['x_male']],
                names=['>50K', '<=50K'],
                title=f'Nam (n={stats_desc["n_male"]:,})',
                color_discrete_sequence=['#16a34a', '#9ca3af'],
                hole=0.3
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(
                values=[stats_desc['x_female'], stats_desc['n_female']-stats_desc['x_female']],
                names=['>50K', '<=50K'],
                title=f'Nữ (n={stats_desc["n_female"]:,})',
                color_discrete_sequence=['#16a34a', '#9ca3af'],
                hole=0.3
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'age' in analyzer.df_clean.columns:
            df = analyzer.df_clean.copy()
            df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100], labels=['<30','30-45','45-60','60+'])
            pivot = pd.crosstab([df['age_group'], df['sex']], df['income_binary'], normalize='index') * 100
            
            if 1 in pivot.columns:
                fig = px.imshow(
                    pivot[1].unstack(),
                    text_auto='.1f',
                    color_continuous_scale='YlOrRd',
                    labels={'color': 'Tỷ lệ (%)'}
                )
                fig.update_layout(title='Tỷ lệ thu nhập >50K theo tuổi và giới tính', height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dữ liệu không có cột tuổi")
    
    st.markdown("---")
    
    # === SECTION 3: KIỂM ĐỊNH THỐNG KÊ ===
    st.markdown("### Kết Quả Kiểm Định")
    
    # Metrics hàng 1
    col1, col2, col3, col4 = st.columns(4)
    
    p_value_display = analyzer.format_p_value(results['p_value_one_tail'])
    p_scientific = analyzer.format_scientific(results['p_value_one_tail'])
    
    with col1:
        st.metric("Z-statistic", f"{results['z_statistic']:.4f}")
    with col2:
        st.metric("P-value", p_value_display, help=f"Dạng khoa học: {p_scientific}")
    with col3:
        st.metric("Power (1-β)", f"{power['power']*100:.1f}%")
    with col4:
        st.metric("Effect Size (h)", f"{results.get('cohens_h', 0):.4f}")
    
    # Kết luận
    st.markdown("#### Kết Luận")
    
    if results['reject_h0']:
        diff = (stats_desc['p_male']-stats_desc['p_female'])*100
        ci_lower = results['ci_lower_95']*100
        ci_upper = results['ci_upper_95']*100
        or_val = results.get('odds_ratio', 0)
        
        st.success(
            f"**Bác bỏ giả thuyết H₀** (p-value {p_value_display})\n\n"
            f"Có bằng chứng thống kê cho thấy nam giới có tỷ lệ thu nhập trên 50K cao hơn nữ giới.\n\n"
            f"- Chênh lệch: {diff:+.1f} điểm phần trăm\n"
            f"- Khoảng tin cậy 95%: [{ci_lower:.1f}%, {ci_upper:.1f}%]\n"
            f"- Odds Ratio: {or_val:.2f} lần"
        )
    else:
        st.warning(
            f"**Không bác bỏ giả thuyết H₀** (p-value {p_value_display})\n\n"
            f"Không đủ bằng chứng thống kê để kết luận có sự khác biệt về thu nhập giữa nam và nữ."
        )
    
    # Điều kiện kiểm định
    st.markdown("#### Điều Kiện Kiểm Định")
    
    with st.container():
        conditions = assumptions['conditions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Nam**")
            st.write(f"- Thu nhập >50K: {conditions['nam_income_gt_50k']:.1f}")
            st.write(f"- Thu nhập ≤50K: {conditions['nam_income_le_50k']:.1f}")
        
        with col2:
            st.markdown("**Nữ**")
            st.write(f"- Thu nhập >50K: {conditions['nu_income_gt_50k']:.1f}")
            st.write(f"- Thu nhập ≤50K: {conditions['nu_income_le_50k']:.1f}")
        
        col1, col2 = st.columns(2)
        with col1:
            valid = "Đạt" if assumptions['z_test_valid'] else "Không đạt"
            st.write(f"**Điều kiện np ≥ 5:** {valid}")
        with col2:
            large = "Đạt" if assumptions['large_sample'] else "Không đạt"
            st.write(f"**Mẫu lớn (n>30):** {large}")
    
    # Giải thích p-value
    with st.expander("Giải thích về P-value"):
        st.markdown("""
        **P-value hiển thị bằng 0 có nghĩa là gì?**
        
        P-value thực tế rất nhỏ (nhỏ hơn 0.0001), do:
        - Kích thước mẫu rất lớn
        - Chênh lệch giữa 2 nhóm rõ rệt
        - Z-statistic rất cao
        
        Python hiển thị làm tròn thành 0, nhưng ý nghĩa thống kê là rất mạnh.
        
        **Cách diễn giải:**
        - p < 0.05: Có ý nghĩa thống kê
        - p < 0.01: Rất có ý nghĩa
        - p < 0.001: Cực kỳ có ý nghĩa
        """)

# ============================================================================
# TAB 4: KẾT QUẢ
# ============================================================================
elif option == "Kết Quả":
    if not st.session_state.analysis_done:
        st.warning("Vui lòng thực hiện phân tích ở tab Phân Tích trước")
        st.stop()
    
    st.markdown("### Bảng Kết Quả Tổng Hợp")
    
    table = analyzer.get_results_table()
    st.dataframe(table, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Download buttons
    st.markdown("### Xuất Dữ Liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Tải Bảng Kết Quả (CSV)",
            data=csv,
            file_name='ket_qua_phan_tich.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        if analyzer.df_clean is not None:
            csv_full = analyzer.df_clean.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Tải Dữ Liệu Đã Làm Sạch (CSV)",
                data=csv_full,
                file_name='data_cleaned.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Phương pháp
    st.markdown("### Phương Pháp Phân Tích")
    
    with st.container():
        st.markdown("""
        **Kiểm định sử dụng:** Two-Proportion Z-Test
        
        **Giả thuyết:**
        - H₀: Tỷ lệ thu nhập >50K của nam bằng nữ
        - H₁: Tỷ lệ thu nhập >50K của nam cao hơn nữ
        
        **Công thức:**
        ```
        z = (p₁ - p₂) / √[p̂(1-p̂)(1/n₁ + 1/n₂)]
        
        Trong đó:
        - p̂ = (x₁ + x₂) / (n₁ + n₂)  (pooled proportion)
        - n₁, n₂: Kích thước mẫu
        - x₁, x₂: Số quan sát thành công
        ```
        
        **Khoảng tin cậy 95%:**
        ```
        CI = (p₁ - p₂) ± 1.96 × √[p₁(1-p₁)/n₁ + p₂(1-p₂)/n₂]
        ```
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;'>"
    "Adult Income Analysis - Chương 3 & 4 Thống kê<br>"
    "Phương pháp: Two-Proportion Z-Test, Welch's T-Test, Power Analysis"
    "</div>",
    unsafe_allow_html=True
)
