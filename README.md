# ViCare – Trợ lý Hỏi đáp Sức khỏe Tiếng Việt (RAG)

Hệ thống hỏi đáp thông tin y tế tiếng Việt xây dựng trên bộ dữ liệu **ViHealthQA**, nâng cấp từ kiến trúc tìm kiếm vector cơ bản lên **Hybrid Retrieval + Cross-Encoder Reranking** nhằm tăng độ chính xác truy xuất, trích dẫn nguồn minh bạch và từ chối an toàn các câu hỏi ngoài phạm vi.

## Kiến trúc & Cải tiến chính
- **Hybrid Retrieval (Dense + Sparse):** Kết hợp tìm kiếm ngữ nghĩa từ **ChromaDB** (`intfloat/multilingual-e5-small`) và tìm kiếm từ khóa **BM25Okapi** thông qua thuật toán dung hợp thứ hạng **Reciprocal Rank Fusion (RRF)**.
- **Cross-Encoder Reranking:** Tái xếp hạng top-20 ứng viên sau RRF bằng mô hình đa ngôn ngữ **`BAAI/bge-reranker-v2-m3`**.
- **Context Pruning & Safety Guardrails:** Hiệu chỉnh ngưỡng xác suất `MIN_SCORE = 0.08`, lọc tối đa 3 đoạn đạt ngưỡng trước khi đưa vào **`Qwen2.5-3B`** (Ollama), tích hợp từ chối ngoài phạm vi và cảnh báo khẩn cấp 115.

## Kết quả thực nghiệm (100 câu `dev` & 10 câu `LLM-as-a-Judge`)

| Chỉ số | Baseline (Trước) | Cải tiến (Sau) | Thay đổi |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 36.0% | **47.0%** | **+11.0 đ.%** |
| **Recall@3** | 46.0% | **53.0%** | **+7.0 đ.%** |
| **Recall@5** | 50.0% | **54.0%** | **+4.0 đ.%** |
| **Recall@10** | 54.0% | **57.0%** | **+3.0 đ.%** |
| **Coverage** | 100.0% | 100.0% | +0.0 đ.% |
| **Judge@10** | 70.0% | 70.0% | +0.0 đ.% |
| **Refusal Rate** | 30.0% | 30.0% | +0.0 đ.% |

## Hướng dẫn chạy
Mở trực tiếp `ViCare_Bootcamp24h.ipynb` trên Google Colab (chọn runtime **T4 GPU**), chạy tuần tự các cell từ Mục 0 đến Mục 8 để đánh giá chỉ số, chạy Mục 9 (`ENABLE_GRADIO = True`) để mở giao diện chat demo và Mục 11 để xuất bảng so sánh.
