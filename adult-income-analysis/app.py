# ============================================================================
# APP.PY - GIAO DIỆN STREAMLIT (DARK MODE - FIXED)
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
# HELPER: VẼ BIỂU ĐỒ PHÂN PHỐI CHUẨN (DARK MODE)
# ============================================================================
def plot_normal_distribution(z_stat: float, p_value: float, test_type: str = 'one-tailed') -> go.Figure:
    """Vẽ biểu đồ phân phối chuẩn - Dark mode optimized"""
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig = go.Figure()
    
    # Đường cong phân phối chuẩn - màu sáng cho dark mode
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Phân phối chuẩn N(0,1)',
        line=dict(color='#9ca3af', width=2)
    ))
    
    if test_type == 'one-tailed':
        x_fill = np.linspace(z_stat, 4, 100)
        y_fill = stats.norm.pdf(x_fill, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.4)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            name=f'P-value',
            showlegend=True
        ))
        
        # Z-statistic line - màu sáng
        fig.add_vline(x=z_stat, line_dash='dash', line_color='#ef4444', line_width=2,
                     annotation_text=f'Z = {z_stat:.2f}', annotation_position='top',
                     annotation_font_color='#f3f4f6')
        
        z_critical = stats.norm.ppf(0.95)
        fig.add_vline(x=z_critical, line_dash='dot', line_color='#3b82f6', line_width=2,
                     annotation_text=f'Critical Z = {z_critical:.2f}', annotation_position='top',
                     annotation_font_color='#f3f4f6')
    else:
        x_fill_right = np.linspace(abs(z_stat), 4, 100)
        y_fill_right = stats.norm.pdf(x_fill_right, 0, 1)
        x_fill_left = np.linspace(-4, -abs(z_stat), 100)
        y_fill_left = stats.norm.pdf(x_fill_left, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill_right, x_fill_right[::-1]]),
            y=np.concatenate([y_fill_right, np.zeros_like(y_fill_right)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.4)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            name=f'P-value',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill_left, x_fill_left[::-1]]),
            y=np.concatenate([y_fill_left, np.zeros_like(y_fill_left)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.4)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            showlegend=False
        ))
    
    fig.update_layout(
        title='Phân Phối Chuẩn và Vị Trí Z-Statistic',
        title_font_color='#f3f4f6',
        xaxis_title='Giá trị Z',
        yaxis_title='Mật độ xác suất',
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#f3f4f6',
        xaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280'),
        yaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280')
    )
    
    fig.add_annotation(
        x=0, y=0.35,
        text=f"Z = {z_stat:.2f}<br>P = {analyzer.format_p_value(p_value)}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#ef4444',
        bgcolor='#1f2937',
        bordercolor='#ef4444',
        borderwidth=1,
        font_color='#f3f4f6'
    )
    
    return fig


# ============================================================================
# HELPER: VẼ BIỂU ĐỒ VÙNG BÁC BỎ (DARK MODE)
# ============================================================================
def plot_critical_region(z_stat: float, alpha: float = 0.05, test_type: str = 'one-tailed') -> go.Figure:
    """Vẽ biểu đồ vùng bác bỏ - Dark mode optimized"""
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Phân phối H₀',
        line=dict(color='#9ca3af', width=2),
        showlegend=True
    ))
    
    if test_type == 'one-tailed':
        z_critical = stats.norm.ppf(1 - alpha)
        
        # Vùng bác bỏ - màu đỏ sáng
        x_reject = np.linspace(z_critical, 4, 100)
        y_reject = stats.norm.pdf(x_reject, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_reject, x_reject[::-1]]),
            y=np.concatenate([y_reject, np.zeros_like(y_reject)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.5)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            name='Vùng bác bỏ H₀',
            showlegend=True
        ))
        
        # Vùng không bác bỏ - màu xanh sáng
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
        
        fig.add_vline(x=z_critical, line_dash='dash', line_color='#ef4444', line_width=2,
                     annotation_text=f'Critical Z = {z_critical:.2f}', annotation_position='top',
                     annotation_font_color='#f3f4f6')
        
        marker_color = '#ef4444' if z_stat > z_critical else '#10b981'
        marker_symbol = 'x' if z_stat > z_critical else 'circle'
        
        fig.add_trace(go.Scatter(
            x=[z_stat],
            y=[stats.norm.pdf(z_stat, 0, 1)],
            mode='markers+text',
            marker=dict(color=marker_color, size=12, symbol=marker_symbol, line=dict(width=2, color='white')),
            name=f'Z = {z_stat:.2f}',
            text=[f'{z_stat:.2f}'],
            textposition='top center',
            textfont_color='#f3f4f6',
            showlegend=True
        ))
        
        conclusion = "Bác bỏ H₀" if z_stat > z_critical else "Không bác bỏ H₀"
        conclusion_color = '#ef4444' if z_stat > z_critical else '#10b981'
        
        fig.add_annotation(
            x=0, y=0.3,
            text=f"Kết luận: {conclusion}",
            font=dict(color=conclusion_color, size=12, weight='bold'),
            bgcolor='#1f2937',
            bordercolor=conclusion_color,
            borderwidth=2,
            showarrow=False
        )
    else:
        z_critical_lower = stats.norm.ppf(alpha / 2)
        z_critical_upper = stats.norm.ppf(1 - alpha / 2)
        
        x_reject_r = np.linspace(z_critical_upper, 4, 100)
        y_reject_r = stats.norm.pdf(x_reject_r, 0, 1)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_reject_r, x_reject_r[::-1]]),
            y=np.concatenate([y_reject_r, np.zeros_like(y_reject_r)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.5)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            name='Vùng bác bỏ H₀',
            showlegend=True
        ))
        
        x_reject_l = np.linspace(-4, z_critical_lower, 100)
        y_reject_l = stats.norm.pdf(x_reject_l, 0, 1)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_reject_l, x_reject_l[::-1]]),
            y=np.concatenate([y_reject_l, np.zeros_like(y_reject_l)]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.5)',
            line=dict(color='rgba(239, 68, 68, 0)'),
            showlegend=False
        ))
        
        fig.add_vline(x=z_critical_upper, line_dash='dash', line_color='#ef4444',
                     annotation_text=f'+{z_critical_upper:.2f}', annotation_position='top',
                     annotation_font_color='#f3f4f6')
        fig.add_vline(x=z_critical_lower, line_dash='dash', line_color='#ef4444',
                     annotation_text=f'{z_critical_lower:.2f}', annotation_position='top',
                     annotation_font_color='#f3f4f6')
        
        fig.add_trace(go.Scatter(
            x=[z_stat],
            y=[stats.norm.pdf(z_stat, 0, 1)],
            mode='markers+text',
            marker=dict(color='#ef4444' if abs(z_stat) > z_critical_upper else '#10b981', 
                       size=12, symbol='x' if abs(z_stat) > z_critical_upper else 'circle'),
            name=f'Z = {z_stat:.2f}',
            text=[f'{z_stat:.2f}'],
            textposition='top center',
            textfont_color='#f3f4f6'
        ))
    
    fig.update_layout(
        title=f'Biểu Đồ Vùng Bác Bỏ (α = {alpha})',
        title_font_color='#f3f4f6',
        xaxis_title='Giá trị Z',
        yaxis_title='Mật độ xác suất',
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#f3f4f6',
        xaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280'),
        yaxis=dict(gridcolor='#374151', zerolinecolor='#6b7280')
    )
    
    fig.add_annotation(
        x=-3.5, y=0.05,
        text="🔴 Vùng bác bỏ (α)<br>🔵 Vùng không bác bỏ (1-α)",
        showarrow=False,
        bgcolor='#1f2937',
        bordercolor='#6b7280',
        borderwidth=1,
        align='left',
        font_color='#f3f4f6'
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
        ["Tải Dataset", "Nhập Dữ Liệu", "Phương Pháp & Kết Quả", "Kết Quả"],
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
                
                # Clean data ngay sau khi load
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
            st.session_state.data_loaded = True
            st.success("Đã tải dữ liệu mẫu")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Hướng Dẫn")
        st.markdown("""
        1. Tải file adult.csv từ Kaggle
        2. Hoặc dùng dữ liệu mẫu để test
        3. Chuyển sang tab Phương Pháp & Kết Quả để tiếp tục
        """)

# ============================================================================
# TAB 2: NHẬP DỮ LIỆU
# ============================================================================
elif option == "Nhập Dữ Liệu":
    st.markdown("### Nhập Dữ Liệu Thủ Công")
    st.markdown("Thêm từng quan sát vào dataset")
    
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
                st.session_state.data_loaded = len(analyzer.df) > 0
                st.success(f"Đã thêm! Tổng: {len(analyzer.df)} dòng")
                st.rerun()
    
    st.markdown("---")
    
    if analyzer.df is not None and len(analyzer.df) > 0:
        st.markdown("#### Dữ Liệu Đã Nhập")
        st.dataframe(analyzer.df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xóa Tất Cả", use_container_width=True):
                analyzer.df = None
                analyzer.df_clean = None
                st.session_state.data_loaded = False
                st.session_state.analysis_done = False
                st.rerun()
        with col2:
            if st.button("Lưu", use_container_width=True):
                if analyzer.clean_data():
                    st.session_state.data_loaded = True
                    st.session_state.analysis_done = True
                    st.success("Đã lưu dữ liệu thành công")
                    st.rerun()
                else:
                    st.error("Không thể lưu. Dữ liệu không hợp lệ.")
    else:
        st.info("Chưa có dữ liệu. Vui lòng thêm dòng hoặc tải dataset.")

# ============================================================================
# TAB 3: PHƯƠNG PHÁP & KẾT QUẢ
# ============================================================================
elif option == "Phương Pháp & Kết Quả":
    # Kiểm tra dữ liệu đã được load chưa
    if not st.session_state.get('data_loaded', False) or analyzer.df_clean is None or len(analyzer.df_clean) == 0:
        st.warning("Vui lòng tải dữ liệu ở tab Tải Dataset trước")
        st.stop()
    
    # Chạy phân tích
    stats_desc = analyzer.get_descriptive_stats()
    
    # Kiểm tra nếu không có dữ liệu hợp lệ
    if stats_desc['n_male'] == 0 or stats_desc['n_female'] == 0:
        st.error("Dữ liệu không đủ cả 2 giới tính để phân tích")
        st.stop()
    
    results = analyzer.run_z_test()
    assumptions = analyzer.check_assumptions()
    power = analyzer.calculate_power()
    
    st.session_state.analysis_done = True
    
    # === SECTION 0: TÓM TẮT PHƯƠNG PHÁP ===
    st.markdown("### Phương Pháp Phân Tích")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giả thuyết kiểm định:**")
            st.markdown("""
            - **H₀:** pₘ = pբ (Tỷ lệ thu nhập >50K của nam bằng nữ)
            - **H₁:** pₘ > pբ (Tỷ lệ thu nhập >50K của nam cao hơn nữ)
            - **Mức ý nghĩa:** α = 0.05
            """)
        
        with col2:
            st.markdown("**Công thức Z-test:**")
            st.latex(r"""
            z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}}
            """)
            st.markdown("Trong đó: p̂ = (x₁ + x₂) / (n₁ + n₂)")
    
    st.markdown("---")
    
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
    
    # === SECTION 2: BIỂU ĐỒ THỐNG KÊ ===
    st.markdown("### Trực Quan Hóa Kiểm Định")
    
    tab1, tab2 = st.tabs(["Phân Phối Chuẩn", "Vùng Bác Bỏ"])
    
    with tab1:
        st.markdown("#### Biểu Đồ Phân Phối Chuẩn")
        st.markdown("Hiển thị vị trí Z-statistic trên đường cong phân phối chuẩn N(0,1)")
        
        z_stat = results.get('z_statistic', 0)
        p_value = results.get('p_value_one_tail', 1)
        
        fig_normal = plot_normal_distribution(z_stat, p_value, test_type='one-tailed')
        st.plotly_chart(fig_normal, use_container_width=True)
        
        with st.expander("Giải thích biểu đồ"):
            st.markdown("""
            **Đường cong xám:** Phân phối chuẩn chuẩn hóa N(0,1)
            
            **Vùng đỏ:** Diện tích tương ứng với p-value
            
            **Đường nét đứt đỏ:** Vị trí Z-statistic quan sát được
            
            **Đường chấm xanh:** Critical value tại α = 0.05
            """)
    
    with tab2:
        st.markdown("#### Biểu Đồ Vùng Bác Bỏ")
        st.markdown("Hiển thị vùng quyết định bác bỏ hoặc không bác bỏ giả thuyết H₀")
        
        alpha = 0.05
        fig_critical = plot_critical_region(z_stat, alpha=alpha, test_type='one-tailed')
        st.plotly_chart(fig_critical, use_container_width=True)
        
        with st.expander("Giải thích biểu đồ"):
            st.markdown(f"""
            **Mức ý nghĩa α:** {alpha} ({alpha*100}%)
            
            **Vùng đỏ:** Vùng bác bỏ H₀
            
            **Vùng xanh:** Vùng không bác bỏ H₀
            
            **Critical Z:** {stats.norm.ppf(1-alpha):.2f}
            
            **Z-observed:** {z_stat:.2f}
            """)
    
    st.markdown("---")
    
    # === SECTION 3: KẾT QUẢ KIỂM ĐỊNH ===
    st.markdown("### Kết Quả Kiểm Định")
    
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
    
    with st.expander("Giải thích về P-value"):
        st.markdown("""
        **P-value hiển thị bằng 0 có nghĩa là gì?**
        
        P-value thực tế rất nhỏ (nhỏ hơn 0.0001), do:
        - Kích thước mẫu rất lớn
        - Chênh lệch giữa 2 nhóm rõ rệt
        - Z-statistic rất cao
        
        **Cách diễn giải:**
        - p < 0.05: Có ý nghĩa thống kê
        - p < 0.01: Rất có ý nghĩa
        - p < 0.001: Cực kỳ có ý nghĩa
        """)
    
    st.markdown("---")
    
    # === SECTION 4: CÁCH RA KẾT QUẢ ===
    st.markdown("### Cách Ra Kết Quả")
    
    with st.container():
        st.markdown("""
        **Quy trình kiểm định:**
        
        1. **Tính Z-statistic** từ dữ liệu mẫu
        2. **Tính P-value** = P(Z > Z-observed) từ phân phối chuẩn
        3. **So sánh P-value với α:**
           - Nếu P-value < α (0.05) → Bác bỏ H₀
           - Nếu P-value ≥ α (0.05) → Không bác bỏ H₀
        4. **Kết luận:** Dựa trên quyết định bác bỏ/không bác bỏ H₀
        
        **Hoặc so sánh Z-statistic với Critical Value:**
        
        - Critical Z (α = 0.05, one-tailed) = 1.645
        - Nếu Z-observed > 1.645 → Bác bỏ H₀
        - Nếu Z-observed ≤ 1.645 → Không bác bỏ H₀
        """)
        
        z_stat = results.get('z_statistic', 0)
        z_critical = 1.645
        reject = z_stat > z_critical
        
        st.info(f"""
        **Áp dụng vào dữ liệu hiện tại:**
        
        - Z-observed = {z_stat:.4f}
        - Critical Z = {z_critical:.2f}
        - P-value = {analyzer.format_p_value(results.get('p_value_one_tail', 0))}
        
        → **Kết luận:** {'Bác bỏ H₀' if reject else 'Không bác bỏ H₀'}
        """)

# ============================================================================
# TAB 4: KẾT QUẢ
# ============================================================================
elif option == "Kết Quả":
    if not st.session_state.get('analysis_done', False):
        st.warning("Vui lòng thực hiện phân tích ở tab Phương Pháp & Kết Quả trước")
        st.stop()
    
    # Kiểm tra dữ liệu
    stats_desc = analyzer.get_descriptive_stats()
    if stats_desc['n_male'] == 0 and stats_desc['n_female'] == 0:
        st.warning("Không có dữ liệu để hiển thị kết quả")
        st.stop()
    
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
    
    st.markdown("---")
    
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
    "<div style='text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem;'>"
    "Adult Income Analysis - Chương 3 & 4 Thống kê<br>"
    "Phương pháp: Two-Proportion Z-Test, Welch's T-Test, Power Analysis"
    "</div>",
    unsafe_allow_html=True
)
