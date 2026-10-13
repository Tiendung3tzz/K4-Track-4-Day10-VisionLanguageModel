Họ tên: Tống Trần Tiến Dũng
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 3,7785 | 10,3079 | 0,1005 | [0,0738; 0,1292] | 0,9807 | 0,1310 | 0,0271 | 0,2311 |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Tôi đếm trên cây Qwen3-VL-4B-Instruct thật, dùng model trên thiết bị meta để không nhầm số phần tử storage NF4 với số tham số kiến trúc. Tổng model gốc là 4.437.815.808 tham số; LLM cùng embedding có 4.022.468.096. Khối visual tổng cộng 415.347.712, gồm backbone ViT 306.242.560 và projector 109.105.152 (merger 27.271.680; ba deepstack merger cộng 81.833.472).

| Setup | Phần cập nhật | Module LoRA | Tham số huấn luyện đo trên T4 |
|---|---|---:|---:|
| E1 | LoRA attention/MLP của LLM; embedding, lm_head, ViT và projector đóng băng | 252 | 33.030.144 |
| E2 | E1 và toàn bộ merger/deepstack merger | 252 | 142.135.296 |
| E3 | E2 và LoRA attention/MLP của ViT | 348 | 148.426.752 |

E2 − E1 = 109.105.152, đúng bằng số tham số projector. E3 thêm 96 module LoRA vision và 6.291.456 tham số; đây là LoRA trên ViT, không mở toàn bộ trọng số ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có parse_valid_rate = 300/311 = 96,4630%; đủ 311 dự đoán, không thiếu ảnh. F1 = 0,1005, precision = 0,1335, recall = 0,0806; có 47 TP, 305 FP và 536 FN. Hai lỗi nổi bật là bỏ sót hư hại và định vị box quá rộng/lệch vị trí: 11 ảnh trả `[]` dù có GT, còn crack chỉ có 7 TP nhưng 210 FP và 292 FN. Ví dụ output thô của `vlcsnap-00203` là `[{"label":"crack","bbox_2d":[0,457,1000,999]}]`; GT crack là khoảng [321,875; 630,556; 759,375; 775,000], nên box gần phủ cả nửa dưới ảnh không đạt IoU 0,5. Với `vlcsnap-00200`, output `[{"label":"manhole","bbox_2d":[759,634,845,687]}]` vừa bỏ sót crack vừa lệch khỏi manhole thật. Năm output đầu không bọc markdown và không có chữ thừa; 11 output không parse được trên toàn test cho thấy lỗi định dạng vẫn có.

E0 dùng Tesla T4: VRAM đỉnh 3,7785 GiB, thời gian suy luận 10,3079 phút, đo bằng torch.cuda.max_memory_allocated và perf_counter trong cell E0. Cột Phút của E0 là suy luận; các thí nghiệm huấn luyện sẽ ghi thời gian train.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Cùng 311 output E0, chấm đúng `coord="norm1000"` cho F1 = 0,1005; gọi evaluate của scripts.score với `coord="pixel"` cho F1 = 0,0000. Tôi giữ nguyên ảnh, nhãn và output, chỉ đổi cách diễn giải toạ độ để kiểm chứng. Ví dụ x = 1000 theo norm1000 là mép phải ảnh 640 pixel; nếu coi là 1000 pixel rồi đổi lại về norm1000, nó thành 1562,5. Với trục y, 1000 pixel thành 2777,8 trên ảnh cao 360 pixel. Box bị phóng to/dịch vị trí nên không đạt IoU 0,5. Thang 0–1000 mô tả vị trí tương đối độc lập độ phân giải; bước quy ngược toàn bộ nhãn đã đo sai số làm tròn tối đa 0,32 pixel và vẽ hai bộ box chồng nhau.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Tôi kiểm tra trên Colab T4: ảnh 640×360 có đúng 220 token ảnh và toàn bộ labels tại token ảnh đều bằng -100. Giải mã riêng các token được tính loss cho JSON GT của ví dụ đầu tiên (9 box) rồi kết thúc bằng `<|im_end|>`; tôi cũng assert chuỗi prompt là tiền tố chính xác của hội thoại đầy đủ. Batch hai ảnh có input_ids/labels/attention_mask/mm_token_type_ids cùng shape (2,585), pixel_values (1760,1536) và image_grid_thw (2,3); padding của labels là -100.

Mục tiêu là học P(JSON đáp án | ảnh, prompt), nên loss chỉ tính trên đáp án và token kết thúc. Prompt và token ảnh là điều kiện đã cung cấp, không phải nội dung cần dự đoán. Nếu tính loss trên cả prompt/220 token ảnh, phần cố định có thể chiếm nhiều vị trí, làm loss giảm nhờ học lặp lại mẫu hội thoại và placeholder ảnh thay vì học box/lớp. Điều đó làm loss tổng khó diễn giải và phân tán cập nhật khỏi nhiệm vụ grounding; tôi không chạy đối chứng unmasked nên không gán một mức giảm F1 cụ thể cho lỗi này.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trước khi gắn LoRA, tôi assert không có Linear4bit nào trong visual hoặc lm_head. Runtime Transformers 5.19 ban đầu vẫn lượng tử hoá visual khi chỉ truyền tên ngắn `visual`; tôi sửa danh sách bỏ qua thành ["visual","model.visual","lm_head"] theo tên đầy đủ của model. ViT/projector nền giữ fp16, còn 109.105.152 tham số projector của E2/E3 được chuyển float32 và mở requires_grad; kiểm tra dtype/đóng băng đã qua trên cả ba setup.

Trọng số 4-bit được lưu dạng mã lượng tử cùng hệ số scale, không phải từng trọng số float độc lập để AdamW cập nhật đầy đủ. Gắn LoRA vẫn có thể học các ma trận bổ sung, nhưng không thay thế yêu cầu học toàn bộ projector của E2/E3. Projector float32 giữ gradient và cập nhật nhỏ đủ chính xác; GradScaler của vòng lặp này cũng không unscale gradient fp16. Tôi giữ các tham số LoRA học được ở float32 và dùng autocast fp16 cho phép tính, còn LLM nền NF4 đóng băng. Khi lưu E2/E3, notebook lưu riêng merger/deepstack merger vào extra_trainable.pt vì adapter PEFT không chứa các trọng số projector được mở trực tiếp. E3 còn cần gỡ torchao 0.10 có sẵn (không dùng trong lab) để PEFT 0.21.2 gắn LoRA vào các lớp vision thông thường.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Dữ liệu gồm các frame cắt từ video, nên các frame gần nhau thường cùng mặt đường, ánh sáng và vị trí hư hại. Chia ngẫu nhiên có thể đưa các ảnh gần trùng vào cả train và test; mô hình nhận lại cảnh đã học khiến F1 cao giả và không phản ánh khả năng tổng quát sang phiên quay khác. Tôi dùng split cố định do prepare_data.py sinh và kiểm tra giao tập session của train/val/test đều rỗng.

| Split | Ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|
| train | 1422 | 1074 | 1750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Các số được đo lại trên dataset_version d10-21e4070301. Tập val chỉ có 66 box pothole, so với 470 crack và 103 manhole; vì vậy cần đọc F1 từng lớp thay vì chỉ nhìn F1 micro. Có một box crack trong train với chiều cao bằng 0; notebook thông báo và bỏ box suy biến khỏi đáp án SFT, còn dữ liệu/split và GT của trọng tài giữ nguyên.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
