# 🛡️ Ứng Dụng Dự Báo Rủi Ro & Phát Hiện Gian Lận Giao Dịch

Ứng dụng web được xây dựng hoàn toàn bằng thư viện **Streamlit** giúp trực quan hóa dữ liệu khám phá (EDA), huấn luyện động các mô hình học máy phân loại kiểm định cấu trúc tài chính, đồng thời hỗ trợ chấm điểm rủi ro trực tuyến (hoặc hàng loạt) dựa theo luồng xử lý từ mã nguồn Jupyter Notebook.

## ✨ Tính năng chính của ứng dụng
1. **⚙️ Cấu hình Linh hoạt & Tải dữ liệu**: Hỗ trợ nạp tệp định dạng Excel hoặc CSV. Tùy biến lựa chọn 3 thuật toán học máy từ notebook: *Logistic Regression, Decision Tree,* và *Random Forest* kèm việc can thiệp sâu các siêu tham số mô hình ngay trên giao diện Sidebar.
2. **📊 Phân tích Tổng quan Dữ liệu**: Thống kê nhanh phân bổ kích thước, xem trước cấu trúc bảng ghi và thống kê thuộc tính tập trung chỉ định trực diện cho các biến mô hình ($X_1 \rightarrow X_{14}$ và `default`).
3. **📈 Trực quan hóa Biến**: Sử dụng các biểu đồ động trực quan hóa Plotly tương tác, phân tích phân phối mật độ nhãn rủi ro gian lận kết hợp phân bổ tần suất giao dịch một cách trực quan.
4. **🎯 Đo lường Kiểm định Mô hình**: Tái hiện toàn vẹn chỉ số đánh giá của mô hình phân loại nhị phân bao gồm các điểm đo chính xác như *Accuracy, Precision, Recall, F1-Score* kèm Ma trận nhầm lẫn (Confusion Matrix) và Đồ thị đường cong ROC-AUC.
5. **🔮 Ứng dụng Dự báo thực tế**: Hỗ trợ 2 chế độ: Nhập liệu thông số thủ công trực tiếp thông qua form tiện ích (có gợi ý giá trị trung vị mặc định) hoặc tải lên file dữ liệu kiểm định kiểm thử lớn (`X_test`) nhằm chấm điểm hàng loạt rồi xuất file kết quả.

## 🚀 Hướng dẫn cài đặt và khởi chạy ứng dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã cài đặt sẵn Python phiên bản ổn định (Khuyến nghị sử dụng Python từ `3.9` đến `3.12`).

### Bước 2: Cài đặt các thư viện phụ thuộc cần thiết
Mở Terminal / Command Prompt tại thư mục chứa mã nguồn ứng dụng này và chạy lệnh cài đặt sau:
```bash
pip install -r requirements.txt
