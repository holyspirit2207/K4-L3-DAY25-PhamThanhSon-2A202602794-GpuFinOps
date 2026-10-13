# Báo Cáo Kỹ Thuật FinOps — Tối Ưu Hóa Chi Phí GPU (NimbusAI)

> **Khóa học:** AI Cloud & Big Data (AICB) — Phase 2 · Track 2 · Day 25  
> **Dự án:** Lab 25 — GPU FinOps Optimization Workshop  
> **Tác giả:** Phạm Thanh Sơn (2A202602794) — NimbusAI FinOps Lead  

---

## 1. Tổng quan Dự án & Kết quả Đạt được

Startup LLM **NimbusAI** đang gặp tình trạng chi phí GPU tăng vọt trong khi chưa kiểm soát được hiệu suất thực tế. Áp dụng khung quản trị **FinOps**, mục tiêu của dự án là tối ưu hóa toàn diện từ kiến trúc mô hình, hạ tầng GPU, chiến lược mua bán đến phân bổ chi phí theo đội ngũ.

### Bảng Chỉ số Chính (Before vs After)

| Chỉ số FinOps | Trước Tối ưu (Baseline) | Sau Tối ưu (Optimized) | Mức Cải thiện |
|---|---|---|---|
| **Tổng chi phí hàng tháng** | **$27,133 / tháng** | **$14,626 / tháng** | **Tiết kiệm $12,507 (46.1%)** |
| **Đơn vị kinh tế Inference (`$/1M-token`)** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **Giảm 82.6%** |
| **Chi phí Inference hàng ngày** | $48.87 / ngày | $8.48 / ngày | Tiết kiệm $40.39 / ngày |
| **Chi phí Workload GPU (M3)** | $25,667 / tháng | $15,627 / tháng | Tiết kiệm $10,040 (39.1%) |
| **Lãng phí GPU rảnh rỗi (Idle Waste)** | $600 / tháng | $0 / tháng | Thu hồi 100% |
| **Tag Coverage & Sẵn sàng Chargeback** | 92.0% | 92.0% | Vượt ngưỡng cổng 80% (`True`) |

> [!NOTE]
> Kết quả kiểm thử tự động đạt **11/11 checks passed** trên `verify.py` và **19/19 tests passed** trên `pytest`.

---

## 2. Phân tích Chi tiết Từng Đòn bẩy Tiết kiệm (Savings Breakdown)

Dự án triển khai đồng bộ 4 đòn bẩy FinOps chính, mang lại tổng mức tiết kiệm **$12,507/tháng**:

| Đòn bẩy FinOps | Mức Tiết kiệm (USD/tháng) | Tỷ trọng trên Tổng Savings | Cơ chế & Tác động |
|---|---|---|---|
| **1. Purchasing Optimization (Spot & Reserved)** | **$10,040** | **80.3%** | Chuyển 5 job batch/train sang Spot kèm checkpointing; Cam kết 3yr Reserved cho 3 cụm inference 24/7. |
| **2. Inference Architecture (Cascade + Cache + Batch)** | **$1,212** | **9.7%** | Định tuyến Small Model (-93%), Prompt Caching (-90% input), Batch API (-50% batch jobs). |
| **3. Right-sizing 'Util-Lies'** | **$655** | **5.2%** | Hạ cấp `gpu-h100-4` xuống `A100` và `gpu-a10g-1` xuống `L4` theo đúng yêu cầu VRAM & Bandwidth. |
| **4. Kill Idle GPUs** | **$600** | **4.8%** | Tự động tắt instance `gpu-h100-5` khi GPU không chạy tác vụ (>1 giờ idle). |

### Phân tích Đòn bẩy Đóng góp Lớn nhất
- **Purchasing Strategy đóng góp lớn nhất ($10,040/tháng, chiếm 80.3%)**: Trong cụm tính toán AI, các workload huấn luyện mô hình lớn (`job-train-llm`, 8x H100) tiêu tốn ngân sách khổng lồ khi chạy on-demand ($12,000/tháng). Bằng cách khai thác bản chất **interruptible** và thiết lập cơ chế **automated checkpointing**, chuyển sang Spot instance giúp cắt giảm ngay 36.7% chi phí trên job này mà không làm mất trạng thái huấn luyện. Đồng thời, các cụm inference chạy liên tục (duty cycle 100%) đạt điểm hòa vốn cam kết (`break-even = 55%`), giúp hưởng trọn chiết khấu 45% của Reserved 3-Year.

---

## 3. Phân tích Bản chất: Cơ chế "GPU-Util Lie" & Tác động Tài chính

### 1. Hiện tượng phát hiện trong Mission 1
Trong tổng số 11 GPU giám sát, có 2 GPU bị gắn cờ "GPU-Util Lie" (`util >= 90%` nhưng `MFU < 0.30`):
- **`gpu-h100-4`**: `gpu_util_pct = 98.2%`, nhưng **`MFU = 0.194` (19.4%)** và **`MBU = 0.207` (20.7%)**.
- **`gpu-a10g-1`**: `gpu_util_pct = 96.9%`, nhưng **`MFU = 0.268` (26.8%)** và **`MBU = 0.302` (30.2%)**.

### 2. Bản chất kỹ thuật tại sao GPU-Util 98% nhưng MFU chỉ 19.4%?
- Chỉ số `gpu_util_pct` trong công cụ `nvidia-smi` đo thời gian hoạt động của GPU clock: tỷ lệ thời gian có ít nhất 1 kernel đang cư trú trên SM. **Nó không đo lượng phép tính thực hiện**.
- Khi chạy suy luận sinh từ (LLM decode phase) với batch size nhỏ (BS=1), workload mang tính chất **memory-bound** (arithmetic intensity chỉ 1–2 FLOP/byte, thấp hơn rất nhiều so với điểm ridge ~295 FLOP/byte của H100).
- Các Tensor Core trên SM hầu như phải **chờ (stall)** nạp trọng số mô hình từ bộ nhớ HBM. Dù GPU clock bận 98.2% thời gian, GPU chỉ đạt chưa tới 20% công suất tính toán lý thuyết.

### 3. Tác động tài chính
Doanh nghiệp đang trả đủ **100% chi phí on-demand ($2.50/giờ = $1,800/tháng)** cho một GPU H100 nhưng chỉ nhận được chưa đầy 1/5 hiệu năng thực tế. Hơn 80% số tiền bỏ ra bị lãng phí do chờ nạp dữ liệu.

---

## 4. Trả lời Câu hỏi Phân tích Chuyên sâu (Guide.md)

### Mục 4.3 — Mission 1 (Hiệu quả GPU)
1. **GPU nào có `GPU-Util` cao nhất? MFU là bao nhiêu?**
   - `gpu-h100-4` có `gpu_util_pct = 98.2%` (cao nhất hệ thống), nhưng MFU chỉ đạt **0.194 (19.4%)**.
2. **Tại sao `GPU-Util 98%` lại đi kèm `MFU 20%`?**
   - Do GPU bị nghẽn băng thông bộ nhớ (memory-bound stall), kernel launch overhead và I/O bottleneck. SM bận chờ dữ liệu từ HBM đưa vào SRAM thay vì liên tục nhân ma trận trên Tensor Cores.
3. **Lãng phí idle tính ra bao nhiêu $/tháng? Chiếm bao nhiêu % chi phí?**
   - `gpu-h100-5` bị bỏ không 8 giờ/ngày khi workload kết thúc, gây lãng phí **$20.00/ngày = $600/tháng**. Chiếm 2.2% tổng ngân sách ban đầu.

### Mục 5.4 — Mission 2 (Inference Levers)
1. **Trong 3 đòn bẩy (cascade, cache, batch), đòn bẩy nào đóng góp lớn nhất?**
   - **Model Cascade** đóng góp lớn nhất: Chuyển phần lớn truy vấn đơn giản từ Large Model ($3.00/1M in, $15.00/1M out) sang Small Model ($0.20/1M in, $0.40/1M out) giúp giảm giá vốn tới **15–37 lần**.
2. **Tại sao `discount_stack(batch=True, cache_hit_frac=1.0) = 0.05`?**
   - Vì các mức chiết khấu áp dụng theo cấp số nhân: Prompt caching chiết khấu 90% (chỉ trả 10% = 0.10); Batch API chiết khấu 50% (trả 0.50). Kết hợp cả hai: `0.10 * 0.50 = 0.05` (tương đương giảm **95%** chi phí!).
3. **Khi nào KHÔNG NÊN dùng Batch API?**
   - Khi dịch vụ yêu cầu phản hồi thời gian thực (real-time chat, autocomplete, voice agent) với ràng buộc độ trễ SLA khắt khe (< 1–2 giây), vì Batch API gom lệnh xử lý bất đồng bộ theo cửa sổ thời gian (thường từ vài phút đến 24 giờ).

### Mục 6.4 — Mission 3 (Purchasing Strategy)
1. **Job nào được đề xuất dùng Spot? Tại sao?**
   - Các job huấn luyện và đánh giá (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval`) vì đều có cờ `interruptible = 1` và chạy theo đợt (<24h/ngày). Khi bị nhà cung cấp thu hồi, hệ thống khôi phục từ checkpoint mà không hỏng cả tiến trình.
2. **Với Spot, tại sao "effective hours" cao hơn "job hours" thực tế?**
   - Vì cần cộng thêm: (1) Chi phí ghi checkpoint định kỳ (`ckpt_overhead_frac = 3%`), và (2) Thời gian chạy lại phần việc bị mất giữa 2 lần checkpoint khi bị thu hồi (`rework_hours`).
3. **Job nào đề xuất Reserved? Tại sao?**
   - `job-infer-chat` (A10G, 24h), `job-infer-rag` (A100, 24h), và `job-infer-search` (L4, 18h). Tỷ lệ sử dụng (`duty cycle`) đạt từ 75% đến 100%, vượt xa ngưỡng hòa vốn 55% (`break-even`) của mức giảm giá 45%.

### Mục 7.4 — Mission 4 (Cost Allocation & FOCUS)
1. **Team nào tốn nhiều nhất? Tỷ lệ bao nhiêu?**
   - Đội **`assistant`** tốn nhiều nhất với **$2.59/ngày** (chiếm **30.6%** tổng chi phí inference), tiếp theo là `search` ($2.49/ngày, 29.4%), `eval` ($1.79/ngày, 21.1%) và `rag` ($1.60/ngày, 18.9%).
2. **Tag coverage là bao nhiêu %? Có đủ để chargeback không?**
   - Tag coverage đạt **92.0%** (vượt ngưỡng điều kiện tiên quyết **80%**), do đó hệ thống **đã sẵn sàng kích hoạt cơ chế Chargeback** thực tế.
3. **Tại sao chuẩn FOCUS lại quan trọng trong môi trường Multi-Cloud?**
   - FOCUS (FinOps Open Cost & Usage Specification) chuẩn hóa tên trường dữ liệu chi phí (như `BilledCost`, `ServiceCategory`, `ResourceId`, `ChargePeriodStart`) thành một schema thống nhất, loại bỏ sự khác biệt về thuật ngữ giữa AWS, GCP, Azure và Neocloud, cho phép lập báo cáo tài chính hợp nhất không phụ thuộc vào nền tảng.

---

## 5. Kết quả Thực hiện Toàn bộ 5 Phần Mở Rộng ("Your Turn" Extensions)

### Extension 1: Cải thiện `recommend_tier()` — Matrix Policy & Interruption Risk
- **Cải tiến Code:** Bổ sung bảng tỷ lệ gián đoạn `GPU_SPOT_INTERRUPT_RATES` (H100: 8%, H200: 12%, A100: 5%, A10G: 3%, L4: 2%) và đánh giá thời hạn cam kết (1-Year vs 3-Year Commitment).
- **Kết quả đo lường:**
  - Nếu một GPU có tỷ lệ thu hồi vượt ngưỡng an toàn (>20%), thuật toán sẽ từ chối Spot để tránh rủi ro vỡ tiến độ.
  - Phân loại rõ ràng giữa workload dự án ngắn hạn (<=30 ngày) và dịch vụ dài hạn (>=365 ngày), ngăn ngừa việc mua cam kết dài hạn cho dự án thử nghiệm ngắn hạn.

### Extension 2: Right-sizing theo MBU & Băng thông Bộ nhớ
- **Cải tiến Code:** Thêm module tính toán `$/GB-VRAM` và so sánh nhu cầu băng thông đạt được (`achieved_bw_tbs`) với các GPU có sẵn trong `data/price_catalog.csv`.
- **Kết quả đo lường:**
  - `gpu-h100-4` (80GB VRAM, peak BW 3.35 TB/s, giá $2.50/h, achieved BW = 0.69 TB/s): Đề xuất chuyển sang **`A100-80GB`** (giá $1.79/h, peak BW 2.0 TB/s). Vẫn dư thừa băng thông (MBU kỳ vọng 34.7%), giúp **tiết kiệm $511.20/tháng**.
  - `gpu-a10g-1` (24GB VRAM, peak BW 0.6 TB/s, giá $1.00/h, achieved BW = 0.18 TB/s): Đề xuất chuyển sang **`L4`** (giá $0.80/h, peak BW 0.3 TB/s). MBU tăng lên 60.4% tối ưu, giúp **tiết kiệm $144.00/tháng**.
  - **Tổng tiết kiệm từ Right-sizing MBU:** **$655.20 / tháng**.

### Extension 3: `cache_is_worth_it()` — Phân tích Điểm Hòa vốn Ghi Cache
- **Cải tiến Code:** Xây dựng hàm `cache_breakeven_reads()` và `cache_is_worth_it()`.
- **Kết quả đo lường:**
  - Với tỷ lệ chiết khấu đọc 90% (tiết kiệm $2.70/1M token trên Large Model), và chi phí ghi cache $3.75/1M token: **Ngưỡng hòa vốn tối thiểu là 1.39 lần đọc lại**.
  - Nếu một tiền tố chỉ được đọc 1 lần (`reads = 1`), ghi cache gây **lỗ** $1.05/1M token.
  - Nếu tiền tố được đọc từ 2 lần trở lên (`reads >= 2`), ghi cache mang lại lợi nhuận ròng.
  - Trong tập dữ liệu của NimbusAI, 100% request có tiền tố cache được tái sử dụng qua nhiều phiên, khẳng định quyết định bật Prompt Caching là hoàn toàn chính xác.

### Extension 4: Ngân sách Reasoning (USD & Wh)
- **Cải tiến Code:** Tách nhóm truy vấn suy luận sâu `is_reasoning = 1` và áp dụng hệ số năng lượng `REASONING_ENERGY_MULTIPLIER = 80.0`.
- **Kết quả đo lường định lượng:**
  - Số lượng truy vấn reasoning: 201 / 2,400 (**8.4% tổng lưu lượng**).
  - Số token reasoning: 1,241,156 tokens (**16.5% tổng token**).
  - Chi phí reasoning: $1.40 / ngày (**16.5% hóa đơn inference**).
  - Năng lượng tiêu thụ reasoning: **29,787.7 Wh / ngày (chiếm tới 94.0% TỔNG NĂNG LƯỢNG INFERENCE!)**.
  - Trong khi đó, 91.6% lưu lượng truy vấn thông thường chỉ tiêu tốn 1,887.6 Wh (6.0%).
- **Đề xuất chính sách:** Thiết lập bộ phân loại intent để chỉ kích hoạt chế độ reasoning khi điểm độ phức tạp bài toán (code generation, multi-step math) vượt ngưỡng, tránh lãng phí năng lượng cho các câu hỏi hội thoại thông thường.

### Extension 5: Carbon-Aware Scheduling (Lập lịch Huấn luyện Theo Vùng)
- **Cải tiến Code:** Tích hợp lưới phát thải carbon (`REGION_CARBON`) và đơn giá điện (`REGION_PRICE_KWH`) trên 5 vùng điện toán đám mây cho toàn bộ các tác vụ interruptible.
- **Kết quả đo lường định lượng (cho 1,789 kWh tác vụ gián đoạn):**
  - Vùng mặc định `us-east-1` (Virginia): Phát thải **679.8 kg CO2e**, tiền điện $214.68.
  - Vùng sạch nhất **`europe-north1`** (Na Uy - Thủy điện, 30 g/kWh): Phát thải **53.7 kg CO2e** -> **Giảm 626.1 kg CO2e (giảm 92.1% lượng phát thải carbon!)**.
  - Vùng rẻ nhất **`us-east-wa`** (Washington - 90 g/kWh, $0.055/kWh): Tiền điện chỉ **$98.39** -> **Tiết kiệm 54.2% chi phí tiền điện** và vẫn giảm 76.3% carbon.

---

## 6. Khuyến nghị Chiến lược cho Ban Lãnh đạo NimbusAI

Nếu đảm nhận vai trò FinOps Lead tại NimbusAI, 3 hành động tiên quyết trong 30 ngày đầu tiên bao gồm:

1. **Hành động 1 — Kích hoạt Chính sách Mua tập trung (Centralized Purchasing Policy):**
   - Ngay lập tức di chuyển các job training định kỳ sang Spot instances với checkpointing tự động lưu vào S3/GCS.
   - Ký kết cam kết 3-Year Compute Savings Plan / Reserved Instances cho toàn bộ dung lượng inference chạy ổn định, khóa mức tiết kiệm $10,040/tháng ngay trong tháng đầu tiên.

2. **Hành động 2 — Chuẩn hóa Kiến trúc Tách biệt Prefill / Decode & Cascade Router:**
   - Triển khai Intelligent Gateway: 100% prompt ngắn chuyển sang mô hình nhỏ, kích hoạt prompt caching tự động cho system prompt.
   - Với các cụm máy chủ tự host, tách riêng GPU phục vụ Prefill (compute-bound trên H100) và Decode (memory-bound trên A100/L4) để triệt tiêu hiện tượng "GPU-Util Lie".

3. **Hành động 3 — Thiết lập Cổng Kiểm soát Chargeback & Thẻ Định danh (Tagging Enforcement):**
   - Vì tag coverage đã đạt 92%, chuyển đổi ngay từ chế độ Showback sang **Chargeback thực tế** trong chu kỳ tài chính tới.
   - Áp dụng chính sách tự động hủy (auto-terminate) các instance GPU có mức sử dụng < 10% kéo dài quá 60 phút nhằm xóa bỏ hoàn toàn lãng phí rảnh rỗi.
