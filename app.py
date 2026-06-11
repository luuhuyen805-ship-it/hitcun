import streamlit as st  # <-- ĐÃ SỬA: Đổi từ 'as pd' thành 'as st'
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
st.set_page_config(  # <-- ĐÃ SỬA: pd. -> st.
    layout="wide",
    page_title="Hệ Thống Dự Báo Rủi Ro Gian Lận Giao Dịch",
    page_icon="🛡️"
)

# 2) IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@st.cache_data  # <-- ĐÃ SỬA: pd. -> st.
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng
FEATURES = [f'X_{i}' for i in range(1, 15)]
TARGET = 'default'

# 3) SIDEBAR (TP1) — VÙNG CẤU HÌNH
with st.sidebar:  # <-- ĐÃ SỬA: with pd.sidebar: -> with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu mẫu (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn tệp chứa các cột đặc trưng từ X_1 đến X_14 và cột nhãn 'default'."
    )
    
    st.divider()
    
    model_choice = st.selectbox(
        "Lựa chọn Thuật toán Mô hình",
        options=["Logistic Regression", "Decision Tree", "Random Forest"],
        help="Chọn thuật toán phân loại muốn sử dụng để huấn luyện dữ liệu."
    )
    
    st.subheader("Tham số mô hình AI")
