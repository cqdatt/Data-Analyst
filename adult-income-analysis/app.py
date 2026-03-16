# ============================================================================
# APP.PY - GIAO DIỆN STREAMLIT (3 MỤC: TẢI DATA - PHƯƠNG PHÁP - KẾT QUẢ)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
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
# CUSTOM CSS - DARK MODE OPTIMIZED
# ============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9ca3af;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #f3f4f6;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 8px;
    }
    .streamlit-expanderHeader {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 6px;
    }
    .stAlert {
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
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

analyzer = st.session_state.analyzer

# ============================================================================
# HELPER 1: BIỂU ĐỒ CỘT SO SÁNH TỶ LỆ (BAR CHART)
# ============================================================================
def plot_income_bar_chart(n_male, n_female, p_male, p_female) -> go.Figure:
    """
    Biểu đồ cột so sánh tỷ lệ thu nhập >50K giữa nam và nữ
    """
    fig = go.Figure()
    
    # Cột cho Nam
    fig.add_trace(go.Bar(
        x=['Nam'],
        y=[p_male * 100],
        name='>50K',
        marker_color='#3b82f6',
        text=[f'{p_male * 100:.1f}%'],
        textposition='outside',
        textfont_color='#f3f4f6',
        textfont_size=14,
        textfont_weight='bold'
    ))
    
    # Cột cho Nữ
    fig.add_trace(go.Bar(
        x=['Nữ'],
        y=[p_female * 100],
        name='>50K',
        marker_color='#ec4899',
        text=[f'{p_female * 100:.1f}%'],
        textposition='outside',
        textfont_color='#f3f4f6',
        textfont_size=14,
        textfont_weight='bold'
    ))
    
    # Chênh lệch
    diff = (p_male - p_female) * 100
    
    # Layout
    fig.update_layout(
        title='Tỷ Lệ Thu Nhập >50K Theo Giới Tính',
        title_font_color='#f3f4f6',
        title_font_size=16,
        xaxis_title='Giới tính',
        yaxis_title='Tỷ lệ (%)',
        height=450,
        showlegend=False,
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#f3f4f6',
        xaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280'),
        yaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280', range=[0, 40])
    )
    
    # Annotation chênh lệch
    fig.add_annotation(
        x=0.5, y=35,
        text=f'Chênh lệch: {diff:+.1f} điểm %',
        showarrow=False,
        bgcolor='#1f2937',
        bordercolor='#6b7280',
        borderwidth=1,
        font_color='#f3f4f6',
        font_size=12
    )
    
    return fig


# ============================================================================
# HELPER 2: BIỂU ĐỒ PHÂN PHỐI CHUẨN VỚI MIỀN BÁC BỎ (COMBINED)
# ============================================================================
def plot_normal_with_rejection(z_stat: float, p_value: float, alpha: float = 0.05) -> go.Figure:
    """
    Biểu đồ phân phối chuẩn với miền bác bỏ
    """
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig = go.Figure()
    
    # Đường cong phân phối chuẩn
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Phân phối chuẩn N(0,1)',
        line=dict(color='#9ca3af', width=2),
        showlegend=True
    ))
    
    # Critical value
    z_critical = stats.norm.ppf(1 - alpha)
    
    # Vùng bác bỏ
    x_reject = np.linspace(z_critical, 4, 100)
    y_reject = stats.norm.pdf(x_reject, 0, 1)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_reject, x_reject[::-1]]),
        y=np.concatenate([y_reject, np.zeros_like(y_reject)]),
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.5)',
        line=dict(color='rgba(239, 68, 68, 0)'),
        name=f'Vùng bác bỏ H₀ (α = {alpha})',
        showlegend=True
    ))
    
    # Vùng không bác bỏ
    x_accept = np.linspace(-4, z_critical, 100)
    y_accept = stats.norm.pdf(x_accept, 0, 1)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_accept, x_accept[::-1]]),
        y=np.concatenate([y_accept, np.zeros_like(y_accept)]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(59, 130, 246, 0)'),
        name='Vùng không bác bỏ',
        showlegend=True
    ))
    
    # Đường critical value
    fig.add_vline(x=z_critical, line_dash='dash', line_color='#ef4444', line_width=2,
                 annotation_text=f'Critical Z = {z_critical:.2f}', annotation_position='top',
                 annotation_font_color='#f3f4f6')
    
    # Z-statistic
    marker_color = '#ef4444' if z_stat > z_critical else '#10b981'
    marker_symbol = 'x' if z_stat > z_critical else 'circle'
    
    fig.add_trace(go.Scatter(
        x=[z_stat],
        y=[stats.norm.pdf(z_stat, 0, 1)],
        mode='markers+text',
        marker=dict(color=marker_color, size=14, symbol=marker_symbol, line=dict(width=2, color='white')),
        name=f'Z-observed = {z_stat:.2f}',
        text=[f'{z_stat:.2f}'],
        textposition='top center',
        textfont_color='#f3f4f6',
        showlegend=True
    ))
    
    # Kết luận
    conclusion = "Bác bỏ H₀" if z_stat > z_critical else "Không bác bỏ H₀"
    conclusion_color = '#ef4444' if z_stat > z_critical else '#10b981'
    
    fig.add_annotation(
        x=0, y=0.3,
        text=f"Kết luận: {conclusion}",
        font=dict(color=conclusion_color, size=14, weight='bold'),
        bgcolor='#1f2937',
        bordercolor=conclusion_color,
        borderwidth=2,
        showarrow=False
    )
    
    fig.update_layout(
        title='Biểu Đồ Phân Phối Chuẩn Với Miền Bác Bỏ',
        title_font_color='#f3f4f6',
        title_font_size=16,
        xaxis_title='Giá trị Z',
        yaxis_title='Mật độ xác suất',
        height=550,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#f3f4f6',
        xaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280', range=[-4, 4]),
        yaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280')
    )
    
    fig.add_annotation(
        x=-3.5, y=0.05,
        text="🔴 Vùng bác bỏ (α)<br>🔵 Vùng không bác bỏ (1-α)<br>✕ Z-observed",
        showarrow=False,
        bgcolor='#1f2937',
        bordercolor='#6b7280',
        borderwidth=1,
        align='left',
        font_color='#f3f4f6',
        font_size=10
    )
    
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### Điều Hướng")
    st.markdown("---")
    
    option = st.radio(
        "Chọn chức năng",
        ["Tải Dữ Liệu", "Phương Pháp", "Kết Quả"],
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
    <h1 style="margin-bottom: 0.5rem; color: #f3f4f6;">Phân Tích Thu Nhập Theo Giới Tính</h1>
    <p style="color: #9ca3af; margin-top: 0;">Nghiên cứu sự khác biệt về thu nhập giữa nam và nữ</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TAB 1: TẢI DỮ LIỆU
# ============================================================================
if option == "Tải Dữ Liệu":
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
                
                if analyzer.clean_data():
                    st.session_state.data_loaded = True
                    st.success(f"Đã tải và xử lý thành công {len(df):,} dòng dữ liệu")
                    
                    st.markdown("#### Xem Trước Dữ Liệu")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Số hàng", f"{df.shape[0]:,}")
                    with col_b:
                        st.metric("Số cột", df.shape[1])
                    with col_c:
                        st.metric("Missing Values", df.isnull().sum().sum())
                else:
                    st.error("Không thể làm sạch dữ liệu. Kiểm tra file CSV.")
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    
    with col_info:
        st.markdown("### Hoặc Dùng Dữ Liệu Mẫu")
        
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
            st.session_state.data_loaded = True
            st.success("Đã tải dữ liệu mẫu")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Hướng Dẫn")
        st.markdown("""
        **Các bước thực hiện:**
        
        1. Tải file adult.csv từ Kaggle
        2. Hoặc dùng dữ liệu mẫu để test
        3. Chuyển sang tab Phương Pháp để xem lý thuyết
        4. Chuyển sang tab Kết Quả để xem phân tích
        """)

# ============================================================================
# TAB 2: PHƯƠNG PHÁP
# ============================================================================
elif option == "Phương Pháp":
    st.markdown("### Phương Pháp Phân Tích")
    st.markdown("Two-Proportion Z-Test - So sánh tỷ lệ thu nhập cao giữa nam và nữ")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Giả thuyết kiểm định:**")
        st.markdown("""
        - **H₀:** pₘ = p (Tỷ lệ thu nhập >50K của nam bằng nữ)
        - **H₁:** pₘ > p (Tỷ lệ thu nhập >50K của nam cao hơn nữ)
        - **Mức ý nghĩa:** α = 0.05
        """)
        
        st.markdown("**Loại kiểm định:**")
        st.info("Kiểm định một phía (one-tailed test)")
    
    with col2:
        st.markdown("**Công thức Z-test:**")
        st.latex(r"""
        z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}}
        """)
        st.markdown("Trong đó: $\\hat{p} = (x_1 + x_2) / (n_1 + n_2)$")
    
    st.markdown("---")
    
    st.markdown("### Các Công Thức Liên Quan")
    
    tab1, tab2 = st.tabs(["Khoảng Tin Cậy (CI)", "Power Analysis"])
    
    with tab1:
        st.markdown("**Khoảng tin cậy 95%:**")
        st.latex(r"""
        CI = (\hat{p}_1 - \hat{p}_2) \pm 1.96 \times \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
        """)
    
    with tab2:
        st.markdown("**Power (1-β):**")
        st.latex(r"""
        \text{Power} = 1 - \beta = 1 - \Phi(z_\beta)
        """)
        st.markdown("Power > 0.80: Kiểm định có độ tin cậy cao")
    
    st.markdown("---")
    
    st.markdown("### Điều Kiện Áp Dụng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Điều kiện cho Z-test:**")
        st.markdown("""
        - n₁p̂₁ ≥ 5 và n₁(1-p̂₁) ≥ 5
        - n₂p̂₂ ≥ 5 và n₂(1-p̂₂) ≥ 5
        - n₁ > 30 và n₂ > 30
        """)
    
    with col2:
        st.markdown("**Chương áp dụng:**")
        st.markdown("""
        - Chương 2: Phân bố chuẩn
        - Chương 3.1, 3.5: Giả thuyết thống kê
        - Chương 4.1, 4.6: Mẫu kép
        """)
    
    st.markdown("---")
    
    st.markdown("### Quy Trình Kiểm Định")
    
    st.markdown("""
    1. **Phát biểu giả thuyết** H₀ và H₁
    2. **Chọn mức ý nghĩa** α = 0.05
    3. **Tính Z-statistic** từ dữ liệu mẫu
    4. **Tính P-value** từ phân phối chuẩn
    5. **So sánh P-value với α:**
       - P-value < α → Bác bỏ H₀
       - P-value ≥ α → Không bác bỏ H₀
    6. **Kết luận** dựa trên quyết định
    """)

# ============================================================================
# TAB 3: KẾT QUẢ
# ============================================================================
elif option == "Kết Quả":
    if not st.session_state.get('data_loaded', False) or analyzer.df_clean is None or len(analyzer.df_clean) == 0:
        st.warning("Vui lòng tải dữ liệu ở tab Tải Dữ Liệu trước")
        st.stop()
    
    stats_desc = analyzer.get_descriptive_stats()
    
    if stats_desc['n_male'] == 0 or stats_desc['n_female'] == 0:
        st.error("Dữ liệu không đủ cả 2 giới tính để phân tích")
        st.stop()
    
    results = analyzer.run_z_test()
    assumptions = analyzer.check_assumptions()
    power = analyzer.calculate_power()
    
    st.session_state.analysis_done = True
    
    # === KẾT QUẢ CUỐI CÙNG ===
    st.markdown("### Kết Quả Cuối Cùng")
    
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
    
    # === BIỂU ĐỒ ===
    st.markdown("### Trực Quan Hóa")
    
    tab1, tab2 = st.tabs(["Biểu Đồ Cột", "Phân Phối Chuẩn"])
    
    # TAB 1: BAR CHART
    with tab1:
        st.markdown("#### Biểu Đồ Cột So Sánh Tỷ Lệ")
        
        fig_bar = plot_income_bar_chart(
            stats_desc['n_male'],
            stats_desc['n_female'],
            stats_desc['p_male'],
            stats_desc['p_female']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        with st.expander("Giải thích biểu đồ"):
            st.markdown("""
            **Cách đọc:**
            
            - Mỗi cột thể hiện tỷ lệ người có thu nhập >50K trong mỗi giới
            - Chiều cao cột = tỷ lệ phần trăm
            - Chênh lệch giữa 2 cột cho thấy mức độ khác biệt
            
            **Màu sắc:**
            
            - Xanh: Nam
            - Hồng: Nữ
            """)
    
    # TAB 2: NORMAL DISTRIBUTION
    with tab2:
        st.markdown("#### Biểu Đồ Phân Phối Chuẩn Với Miền Bác Bỏ")
        
        z_stat = results.get('z_statistic', 0)
        p_value = results.get('p_value_one_tail', 1)
        alpha = 0.05
        
        fig_combined = plot_normal_with_rejection(z_stat, p_value, alpha)
        st.plotly_chart(fig_combined, use_container_width=True)
        
        with st.expander("Giải thích biểu đồ"):
            st.markdown("""
            **Các thành phần:**
            
            - **Đường cong xám:** Phân phối chuẩn N(0,1)
            - **Vùng đỏ:** Vùng bác bỏ H₀ (α = 0.05)
            - **Vùng xanh nhạt:** Vùng không bác bỏ H₀ (1-α)
            - **Đường nét đứt đỏ:** Critical value (Z = 1.645)
            - **Marker X:** Z-statistic quan sát được
            
            **Cách đọc:**
            
            - Nếu Z-observed > Critical Z → Bác bỏ H₀
            - Nếu Z-observed ≤ Critical Z → Không bác bỏ H₀
            """)
    
    st.markdown("---")
    
    # === KẾT QUẢ KIỂM ĐỊNH ===
    st.markdown("### Kết Quả Kiểm Định")
    
    col1, col2, col3 = st.columns(3)
    
    p_value_display = analyzer.format_p_value(results['p_value_one_tail'])
    p_scientific = analyzer.format_scientific(results['p_value_one_tail'])
    
    with col1:
        st.metric("Z-statistic", f"{results['z_statistic']:.4f}")
    with col2:
        st.metric("P-value", p_value_display, help=f"Dạng khoa học: {p_scientific}")
    with col3:
        st.metric("Power (1-β)", f"{power['power']*100:.1f}%")
    
    st.markdown("#### Kết Luận")
    
    if results['reject_h0']:
        diff = (stats_desc['p_male']-stats_desc['p_female'])*100
        ci_lower = results['ci_lower_95']*100
        ci_upper = results['ci_upper_95']*100
        
        st.success(
            f"**Bác bỏ giả thuyết H₀** (p-value {p_value_display})\n\n"
            f"Có bằng chứng thống kê cho thấy nam giới có tỷ lệ thu nhập trên 50K cao hơn nữ giới.\n\n"
            f"- Chênh lệch: {diff:+.1f} điểm phần trăm\n"
            f"- Khoảng tin cậy 95%: [{ci_lower:.1f}%, {ci_upper:.1f}%]"
        )
    else:
        st.warning(
            f"**Không bác bỏ giả thuyết H₀** (p-value {p_value_display})\n\n"
            f"Không đủ bằng chứng thống kê để kết luận có sự khác biệt về thu nhập giữa nam và nữ."
        )
    
    st.markdown("---")
    
    # === ĐIỀU KIỆN KIỂM ĐỊNH ===
    st.markdown("### Điều Kiện Kiểm Định")
    
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
    
    st.markdown("---")
    
    # === BẢNG KẾT QUẢ + EXPORT ===
    st.markdown("### Bảng Kết Quả Tổng Hợp")
    
    table = analyzer.get_results_table()
    st.dataframe(table, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
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
        if analyzer.df_clean is not None and len(analyzer.df_clean) > 0:
            csv_full = analyzer.df_clean.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Tải Dữ Liệu Đã Làm Sạch (CSV)",
                data=csv_full,
                file_name='data_cleaned.csv',
                mime='text/csv',
                use_container_width=True
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem;'>"
    "Adult Income Analysis - Chương 3 & 4 Thống kê<br>"
    "Phương pháp: Two-Proportion Z-Test, Power Analysis"
    "</div>",
    unsafe_allow_html=True
)
