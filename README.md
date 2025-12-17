## 📂 Cấu trúc dự án (Project Structure)

Dưới đây là sơ đồ tổ chức thư mục và giải thích chi tiết chức năng của từng thành phần trong dự án:

```text
├── LSTNet/                     # Thư mục mã nguồn chính (Source Code)
│   ├── data/                   # Chứa 4 bộ dữ liệu đầu vào
│   └── save/                   # Lưu trữ kết quả huấn luyện (Checkpoints & Logs)
│       └── [Model_Variants]    # (Chi tiết bên dưới)
├── Plots/                      # Chứa các biểu đồ trực quan hóa kết quả (Images)
└── reconstructed_logs/         # Notebooks tái hiện quá trình huấn luyện
```

## 📂 Chi tiết cấu trúc thư mục

Dưới đây là mô tả chi tiết về chức năng và nội dung của từng thư mục trong dự án:

### 1. `LSTNet/`
Thư mục chứa mã nguồn chính (Source Code) để triển khai mô hình.

* **`data/`**:
    * Chứa **04 bộ dữ liệu** chuỗi thời gian được sử dụng cho các thực nghiệm trong dự án.
* **`save/`**:
    * Nơi lưu trữ kết quả huấn luyện (checkpoints) của tổng cộng **64 mô hình LSTNet**.
    * Các mô hình này được chia thành **4 nhóm biến thể** kiến trúc để thực hiện *Ablation Study* (nghiên cứu lược bỏ):
        1.  `Full`: Mô hình LSTNet đầy đủ các thành phần.
        2.  `no-ar`: Mô hình lược bỏ thành phần Auto-regressive (AR).
        3.  `no-skip`: Mô hình lược bỏ thành phần Skip-RNN.
        4.  `no-cnn`: Mô hình lược bỏ thành phần Convolutional Layer.
    * 📄 **Các File `history.csv`**: Trong mỗi thư mục con sẽ có các file này, dùng để lưu lại log quá trình huấn luyện và sự thay đổi của các chỉ số (metrics/loss) qua từng epoch. Tất cả 64 mô hình đều có riêng 1 file history.

### 2. `Plots/`
* Thư mục chứa các tệp hình ảnh (.png/.jpg) biểu diễn các biểu đồ trực quan hóa kết quả (Visualization), giúp so sánh hiệu suất giữa các mô hình.

### 3. `reconstructed_logs/`
* **Mục đích:** Do quá trình huấn luyện ban đầu được nhóm thực hiện trực tiếp trên Terminal, thư mục này chứa các file **Jupyter Notebook (.ipynb)** nhằm tái hiện lại các log kết quả đó từ history.csv để thuận tiện cho việc báo cáo.
* **Cấu trúc:** Tương tự như thư mục `save`, các notebook này cũng được chia thành **4 file** tương ứng với 4 biến thể mô hình (`Full`, `no-ar`, `no-skip`, `no-cnn`).

## 🔗 Acknowledgements
This project was conducted as part of the coursework for **IS403** at **[University of Information Technology - Vietnam National University]**.

The primary objective of this project is to reproduce and evaluate the performance of the LSTNet model based on the original paper.

### 1. Original Paper
This project is based on the method proposed in the following paper:
> **Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks**
> *Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu.*
> SIGIR 2018.
> [Link to arXiv](https://arxiv.org/abs/1703.07015)

### 2. Acknowledgements
We utilized the original source code and datasets provided by the authors to reproduce the results. The core model implementation is taken from the following:
* **Source Code:** [https://github.com/fbadine/LSTNet](https://github.com/fbadine/LSTNet)
* **Datasets:** [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)
