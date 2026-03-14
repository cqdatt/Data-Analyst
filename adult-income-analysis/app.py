# ============================================================================
# APP.PY - GIAO DIỆN STREAMLIT
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import IncomeAnalyzer  # Import logic từ analyzer.py
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Phân Tích Thu Nhập Theo Giới Tính",
    page_icon="📊",
    layout="wide"
)

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
st.sidebar.title("🎯 MENU")
st.sidebar.markdown("---")

option = st.sidebar.radio(
    "Chọn chức năng:",
    ["📁 Tải Dataset", "✍️ Nhập Dữ Liệu", "📊 Phân Tích", "📄 Kết Quả"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** Adult Income Analysis  
**Chương:** 3 & 4 - Thống kê  
**Method:** Z-test, T-test  
""")

# ============================================================================
# TAB 1: TẢI DATASET
# ============================================================================
if option == "📁 Tải Dataset":
    st.title("📁 TẢI DATASET")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tải file CSV")
        uploaded_file = st.file_uploader(
            "Chọn file adult.csv",
            type=['csv'],
            help="https://www.kaggle.com/datasets/uciml/adult-census-income"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, na_values=['?', ' ?', ''])
                analyzer.load_from_dataframe(df)
                analyzer.clean_data()
                
                st.success(f"✓ Đã tải {len(df):,} dòng!")
                
                st.subheader("📋 Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Số hàng", f"{df.shape[0]:,}")
                col_b.metric("Số cột", df.shape[1])
                col_c.metric("Missing", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    with col2:
        st.subheader("Dữ liệu mẫu")
        if st.button("📊 Load Data Mẫu", use_container_width=True):
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
            st.success("✓ Đã load!")
            st.rerun()

# ============================================================================
# TAB 2: NHẬP DỮ LIỆU
# ============================================================================
elif option == "✍️ Nhập Dữ Liệu":
    st.title("✍️ NHẬP DỮ LIỆU THỦ CÔNG")
    st.markdown("---")
    
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Tuổi", 18, 100, 35)
        with col2:
            sex = st.selectbox("Giới tính", ["Male", "Female"])
        with col3:
            income = st.selectbox("Thu nhập", ["<=50K", ">50K"])
        
        submitted = st.form_submit_button("➕ Thêm")
        
        if submitted:
            new_row = pd.DataFrame({'age': [age], 'sex': [sex], 'income': [income]})
            
            if analyzer.df is None:
                analyzer.df = new_row
            else:
                analyzer.df = pd.concat([analyzer.df, new_row], ignore_index=True)
            
            analyzer.clean_data()
            st.success(f"✓ Đã thêm! Tổng: {len(analyzer.df)} dòng")
    
    if analyzer.df is not None:
        st.subheader("📋 Dữ liệu đã nhập")
        st.dataframe(analyzer.df, use_container_width=True)
        
        if st.button("🗑️ Xóa tất cả"):
            analyzer.df = None
            analyzer.df_clean = None
            st.rerun()

# ============================================================================
# TAB 3: PHÂN TÍCH
# ============================================================================
elif option == "📊 Phân Tích":
    st.title("📊 PHÂN TÍCH & BIỂU ĐỒ")
    st.markdown("---")
    
    if analyzer.df_clean is None:
        st.warning("⚠ Vui lòng tải dữ liệu trước!")
        st.stop()
    
    # Chạy phân tích
    stats_desc = analyzer.get_descriptive_stats()
    results = analyzer.run_z_test()
    t_results = analyzer.run_t_test()
    assumptions = analyzer.check_assumptions()
    power = analyzer.calculate_power()
    
    st.session_state.analysis_done = True
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mẫu Nam", f"{stats_desc['n_male']:,}")
    col2.metric("Mẫu Nữ", f"{stats_desc['n_female']:,}")
    col3.metric("Tỷ lệ Nam", f"{stats_desc['p_male']*100:.1f}%")
    col4.metric("Tỷ lệ Nữ", f"{stats_desc['p_female']*100:.1f}%")
    
    # Tabs cho biểu đồ
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📈 Heatmap"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Nam', 'Nữ'],
            y=[stats_desc['p_male']*100, stats_desc['p_female']*100],
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{stats_desc["p_male"]*100:.1f}%', f'{stats_desc["p_female"]*100:.1f}%'],
            textposition='outside'
        ))
        fig.update_layout(
            title='Tỷ lệ thu nhập >50K theo giới tính',
            yaxis_title='Tỷ lệ (%)',
            yaxis_range=[0, 40],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=[stats_desc['x_male'], stats_desc['n_male']-stats_desc['x_male']],
                names=['>50K', '<=50K'],
                title=f'Nam (n={stats_desc["n_male"]:,})'
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(
                values=[stats_desc['x_female'], stats_desc['n_female']-stats_desc['x_female']],
                names=['>50K', '<=50K'],
                title=f'Nữ (n={stats_desc["n_female"]:,})'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'age' in analyzer.df_clean.columns:
            df = analyzer.df_clean.copy()
            df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100], labels=['<30','30-45','45-60','60+'])
            pivot = pd.crosstab([df['age_group'], df['sex']], df['income_binary'], normalize='index') * 100
            
            fig = px.imshow(
                pivot[1].unstack(),
                text_auto='.1f',
                color_continuous_scale='YlOrRd',
                labels={'color': 'Tỷ lệ >50K (%)'}
            )
            fig.update_layout(title='Heatmap theo tuổi và giới tính', height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Kiểm định thống kê
    st.markdown("---")
    st.subheader("🧮 KIỂM ĐỊNH THỐNG KÊ")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Z-statistic", f"{results['z_statistic']:.4f}")
    col2.metric("P-value", f"{results['p_value_one_tail']:.6f}")
    col3.metric("Power", f"{power['power']*100:.1f}%")
    
    if results['reject_h0']:
        st.success("✅ **Bác bỏ H₀** - Nam có thu nhập cao hơn nữ (p < 0.05)")
    else:
        st.warning("⚠ **Không bác bỏ H₀** - Không đủ bằng chứng (p ≥ 0.05)")
    
    # Điều kiện kiểm định
    with st.expander("📋 Kiểm tra điều kiện kiểm định"):
        st.write("### Điều kiện Z-test (np ≥ 5)")
        for k, v in assumptions['conditions'].items():
            status = "✓" if v >= 5 else "✗"
            st.write(f"{status} {k}: {v:.1f}")
        
        st.write(f"\n**Z-test hợp lệ:** {'✓' if assumptions['z_test_valid'] else '✗'}")
        st.write(f"**Mẫu lớn (n>30):** {'✓' if assumptions['large_sample'] else '✗'}")

# ============================================================================
# TAB 4: KẾT QUẢ
# ============================================================================
elif option == "📄 Kết Quả":
    st.title("📄 KẾT QUẢ PHÂN TÍCH")
    st.markdown("---")
    
    if not st.session_state.analysis_done:
        st.warning("⚠ Vui lòng phân tích trước!")
        st.stop()
    
    table = analyzer.get_results_table()
    st.dataframe(table, use_container_width=True, hide_index=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        csv = table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Tải kết quả (CSV)",
            data=csv,
            file_name='ket_qua_phan_tich.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        if analyzer.df_clean is not None:
            csv_full = analyzer.df_clean.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 Tải dữ liệu sạch (CSV)",
                data=csv_full,
                file_name='data_cleaned.csv',
                mime='text/csv',
                use_container_width=True
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    📚 Adult Income Analysis | Chương 3 & 4 - Thống kê
</div>
""", unsafe_allow_html=True)