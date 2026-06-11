import streamlit as pd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# 1) SET PAGE CONFIG LÀ LỆNH STREAMLIT ĐẦU TIÊN
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Dự Báo Rủi Ro Gian Lận Giao Dịch",
    page_icon="🛡️"
)

# 2) IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ file bytes để đảm bảo tính hashable cho cache_data.
    Hỗ trợ cả định dạng CSV và Excel.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo đặc tả mô hình trong notebook
FEATURES = [f'X_{i}' for i in range(1, 15)]
TARGET = 'default'

# 3) SIDEBAR (TP1) — VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu mẫu (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn tệp chứa các cột đặc trưng từ X_1 đến X_14 và cột nhãn 'default'."
    )
    
    st.divider()
    
    # Lựa chọn mô hình AI thuật toán
    model_choice = st.selectbox(
        "Lựa chọn Thuật toán Mô hình",
        options=["Logistic Regression", "Decision Tree", "Random Forest"],
        help="Chọn thuật toán phân loại muốn sử dụng để huấn luyện dữ liệu."
    )
    
    st.subheader("Tham số mô hình AI")
    
    # Hiển thị tham số động theo mô hình lựa chọn, cấu hình giá trị mặc định từ notebook
    model_params = {}
    if model_choice == "Logistic Regression":
        model_params['C'] = st.slider("C (Inverse of regularization strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, help="Tham số hiệu chỉnh độ chính xác phạt lỗi.")
        model_params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=2000, value=1000, step=100, help="Số lượt chạy tối ưu hóa cực đại.")
        model_params['random_state'] = st.number_input("Random State", value=32, step=1, help="Gạt seed cố định tính ngẫu nhiên.")
        
    elif model_choice == "Decision Tree":
        model_params['criterion'] = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách.")
        model_params['max_depth'] = st.slider("Chiều sâu tối đa (max_depth)", min_value=1, max_value=50, value=10, help="Độ sâu tối đa của cây quyết định.")
        model_params['random_state'] = st.number_input("Random State", value=32, step=1, help="Seed cố định của mô hình.")
        
    elif model_choice == "Random Forest":
        model_params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=500, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        model_params['max_depth'] = st.slider("Chiều sâu tối đa (max_depth)", min_value=1, max_value=50, value=15, help="Độ sâu giới hạn tối đa của mỗi cây phụ.")
        model_params['random_state'] = st.number_input("Random State", value=32, step=1, help="Seed ngẫu nhiên cố định từ thiết lập notebook.")

    st.divider()
    
    # Nút hành động duy nhất kích hoạt huấn luyện mô hình ở dưới cùng sidebar
    trigger_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# 4) HEADER + KIỂM TRA DỮ LIỆU (TP2) — VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Hệ Thống Phát Hiện & Dự Báo Rủi Ro Gian Lận Giao Dịch")
st.caption("Ứng dụng hỗ trợ phân tích thông tin giao dịch tài chính, quản lý rủi ro tín dụng và chấm điểm gian lận tự động dựa trên các mô hình Machine Learning.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu huấn luyện (.csv hoặc .xlsx) tại Sidebar bên trái để kích hoạt hệ thống.")
    st.stop()

# Đã có dữ liệu đầu vào -> Đọc dữ liệu
df_data = load_data(uploaded_file, uploaded_file.name)

if df_data is None:
    st.stop()

# Kiểm tra tính toàn vẹn của schema cấu trúc cột
missing_cols = [col for col in FEATURES + [TARGET] if col not in df_data.columns]
if missing_cols:
    st.error(f"⚠️ Tệp dữ liệu thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp dữ liệu: **{uploaded_file.name}**")
st.divider()

# 5) KHỐI TRAIN (KHI NGƯỜI DÙNG BẤM NÚT SẼ LƯU VÀO SESSION STATE)
if trigger_train:
    with st.spinner("Đang tiến hành phân tách và huấn luyện mô hình..."):
        X = df_data[FEATURES]
        y = df_data[TARGET]
        
        # Chia dữ liệu theo tỷ lệ huấn luyện/kiểm định chuẩn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32, stratify=y)
        
        # Khởi tạo thuật toán theo tham số chọn động trên UI
        if model_choice == "Logistic Regression":
            model = LogisticRegression(C=model_params['C'], max_iter=model_params['max_iter'], random_state=model_params['random_state'])
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(criterion=model_params['criterion'], max_depth=model_params['max_depth'], random_state=model_params['random_state'])
        else:
            model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=model_params['random_state'])
            
        # Fit mô hình
        model.fit(X_train, y_train)
        
        # Dự báo đánh giá trên tập test
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu kết quả vào session_state để tái sử dụng qua các tab khác nhau không bị rerun train
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_choice
        st.session_state['features_list'] = FEATURES
        st.session_state['evaluation_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_probs': y_probs.tolist() if y_probs is not None else None
        }
        st.success(f"🎉 Đã huấn luyện thành công mô hình **{model_choice}**!")

# 6) KHỐI TABS HIỂN THỊ NỘI DUNG CHÍNH (TP3 -> TP6)
tabs = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Kết quả kiểm định", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tabs[0]:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    # 1. Kích thước dữ liệu thông qua metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng dòng giao dịch", f"{df_data.shape[0]:,}")
    col2.metric("Số lượng cột đặc trưng", f"{df_data.shape[1]}")
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    # 2. Xem dữ liệu thô dạng bảng cuộn gọn
    st.markdown("##### Danh sách bản ghi dữ liệu mẫu đầu tiên (Head)")
    st.dataframe(df_data.head(10), use_container_width=True, height=250)
    
    # 3. Thống kê mô tả chỉ cho các biến đưa vào mô hình X và y
    st.markdown("##### Thống kê mô tả các biến đặc trưng đưa vào mô hình AI")
    st.dataframe(df_data[FEATURES + [TARGET]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tabs[1]:
    st.subheader("Trực quan hóa đồ thị các biến đặc trưng")
    
    # Lựa chọn hiển thị linh hoạt nếu số lượng biến quá nhiều hoặc phân bổ lưới
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng muốn hiển thị biểu đồ phân phối (Mặc định chọn sẵn 3 biến đầu):",
        options=FEATURES,
        default=FEATURES[:3]
    )
    
    # Bố trí biểu đồ phân phối biến mục tiêu 'default' lên hàng đầu trước tiên
    st.markdown("##### 1. Phân phối của biến mục tiêu (Nhãn phân loại rủi ro gian lận)")
    class_counts = df_data[TARGET].value_counts().reset_index()
    class_counts.columns = ['Trạng thái Rủi ro', 'Số lượng']
    class_counts['Trạng thái Rủi ro'] = class_counts['Trạng thái Rủi ro'].map({0: 'Hợp lệ (0)', 1: 'Gian lận (1)'})
    
    fig_target = px.bar(class_counts, x='Trạng thái Rủi ro', y='Số lượng', color='Trạng thái Rủi ro',
                        color_discrete_map={'Hợp lệ (0)': '#2ecc71', 'Gian lận (1)': '#e74c3c'},
                        title="Tỷ lệ phân bổ Nhãn Giao dịch Gian lận vs Hợp lệ", height=300)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("##### 2. Phân phối động của các biến đầu vào liên tục")
    if selected_features:
        # Tổ chức hiển thị dạng lưới 2 cột
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(selected_features):
                    feat = selected_features[i + j]
                    with cols[j]:
                        fig_feat = px.histogram(df_data, x=feat, color=TARGET, barmode='overlay',
                                                title=f"Phân phối tần suất của biến {feat} theo nhãn mục tiêu",
                                                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                                                height=300)
                        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến đặc trưng để vẽ đồ thị hiển thị phân phối.")

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tabs[2]:
    st.subheader("Chỉ số đo lường hiệu năng kiểm định thuật toán")
    
    # Kiểm tra điều phối trạng thái, nếu chưa bấm huấn luyện -> Dừng tab hướng dẫn người dùng
    if 'evaluation_metrics' not in st.session_state:
        st.info("ℹ️ Hệ thống chưa ghi nhận mô hình được huấn luyện. Vui lòng thiết lập cấu hình tham số và bấm nút **🚀 Huấn luyện mô hình** tại sidebar.")
    else:
        metrics = st.session_state['evaluation_metrics']
        model_name_trained = st.session_state['model_name']
        
        st.markdown(f"Mô hình hiện tại đang đánh giá: **{model_name_trained}**")
        
        # Trình bày chỉ tiêu phân loại vô hướng qua st.metric
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        m_col2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4f}")
        m_col3.metric("Độ nhạy (Recall)", f"{metrics['recall']:.4f}")
        m_col4.metric("Điểm F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("##### Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
            fig_cm = px.imshow(
                cm, text_auto=True, 
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                x=['Hợp lệ (0)', 'Gian lận (1)'],
                y=['Hợp lệ (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_right:
            st.markdown("##### Đường cong ROC & Chỉ số AUC")
            if metrics['y_probs'] is not None:
                fpr, tpr, thresholds = roc_curve(metrics['y_test'], metrics['y_probs'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.4f})', line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ngẫu nhiên (Baseline)', line=dict(dash='dash', color='navy')))
                fig_roc.update_layout(xaxis_title='Tỷ lệ báo động giả (FPR)', yaxis_title='Tỷ lệ nhận diện đúng (TPR)', margin=dict(l=20, r=20, t=30, b=20), height=350)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Mô hình được chọn không hỗ trợ xuất xác suất phân lớp để tính toán trực quan hóa đồ thị ROC-AUC.")

# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tabs[3]:
    st.subheader("Chẩn đoán dự báo giao dịch rủi ro trực tuyến")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Vui lòng chạy huấn luyện mô hình ở sidebar để kích hoạt tính năng dự báo rủi ro.")
    else:
        model = st.session_state['trained_model']
        
        # Lựa chọn chế độ nhập trực tiếp hay xử lý file hàng loạt
        predict_mode = st.radio(
            "Phương thức nạp dữ liệu cần dự báo đoán:",
            options=["Nhập thủ công thông số trực tiếp", "Tải tệp danh sách cần chấm điểm hàng loạt (X_test)"],
            horizontal=True
        )
        
        if predict_mode == "Nhập thủ công thông số trực tiếp":
            st.markdown("##### Vui lòng điều chỉnh thông số cho các biến đặc trưng đầu vào:")
            
            # Sử dụng st.form để bao đóng tránh việc tương tác gạt slider gây rerun liên tục app
            with st.form("manual_prediction_form"):
                input_data = {}
                
                # Chia giao diện làm 3 cột nhập dữ liệu cho gọn gàng
                f_cols = st.columns(3)
                for index, feat in enumerate(FEATURES):
                    # Tính toán giá trị mặc định dựa trên median thực tế của tập dữ liệu mẫu để tối ưu trải nghiệm người dùng
                    default_val = float(df_data[feat].median())
                    min_val = float(df_data[feat].min())
                    max_val = float(df_data[feat].max())
                    
                    col_idx = index % 3
                    with f_cols[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Giá trị đặc trưng {feat}",
                            min_value=min_val * 2.0 if min_val < 0 else 0.0,
                            max_value=max_value * 2.0,
                            value=default_val,
                            format="%.6f"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành dự báo rủi ro")
                
            if submit_predict:
                # Chuyển đổi dữ liệu input sang cấu trúc DataFrame khớp thứ tự cột
                input_df = pd.DataFrame([input_data])[FEATURES]
                
                # Thực hiện dự báo nhãn
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
                
                st.markdown("---")
                st.markdown("#### Kết quả phân tích chẩn đoán hệ thống:")
                if prediction == 1:
                    st.error("🚨 **CẢNH BÁO: Giao dịch có dấu hiệu nguy cơ rủi ro GIAN LẬN CAO!**")
                else:
                    st.success("✅ **AN TOÀN: Giao dịch được thẩm định ở mức Rủi ro thấp (Hợp lệ).**")
                    
                if probabilities is not None:
                    p_cols = st.columns(2)
                    p_cols[0].metric("Xác suất Hợp lệ (0)", f"{probabilities[0]*100:.2f}%")
                    p_cols[1].metric("Xác suất Gian lận (1)", f"{probabilities[1]*100:.2f}%")
                    st.progress(float(probabilities[1]))
                    
        elif predict_mode == "Tải tệp danh sách cần chấm điểm hàng loạt (X_test)":
            st.markdown("##### Nạp tệp dữ liệu kiểm định tổng hợp (Thiết kế cấu trúc chứa đủ từ cột X_1 đến X_14):")
            bulk_file = st.file_uploader("Tải lên file danh sách mới cần chấm điểm rủi ro hàng loạt", type=["csv", "xlsx"])
            
            if bulk_file is not None:
                df_bulk = load_data(bulk_file, bulk_file.name)
                if df_bulk is not None:
                    # Kiểm tra tính đồng bộ cấu trúc cột biến đầu vào
                    missing_bulk_cols = [c for c in FEATURES if c not in df_bulk.columns]
                    if missing_bulk_cols:
                        st.error(f"⚠️ Bản mẫu file tải lên không hợp lệ, thiếu các cột biến đặc trưng: {', '.join(missing_bulk_cols)}")
                    else:
                        X_bulk = df_bulk[FEATURES]
                        
                        # Thực hiện dự báo hàng loạt mà không cần huấn luyện lại
                        bulk_preds = model.predict(X_bulk)
                        
                        # Đóng gói tạo bảng kết quả đầu ra tổng hợp
                        df_result = df_bulk.copy()
                        df_result['Predicted_Default'] = bulk_preds
                        
                        if hasattr(model, "predict_proba"):
                            bulk_probs = model.predict_proba(X_bulk)[:, 1]
                            df_result['Fraud_Probability'] = bulk_probs
                            
                        st.success(f"🔥 Đã xử lý phân tích và chấm điểm thành công cho {df_result.shape[0]} dòng giao dịch tài chính.")
                        
                        # Hiển thị kết quả trong container cuộn gọn gàng gọn nhẹ
                        st.dataframe(df_result, use_container_width=True, height=300)
                        
                        # Cho phép xuất dữ liệu kết quả chấm điểm rủi ro về máy định dạng CSV mã hóa utf-8-sig
                        csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo rủi ro hàng loạt (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv"
                        )
