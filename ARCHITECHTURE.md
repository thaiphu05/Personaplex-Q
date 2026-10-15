# Kiến trúc PersonaPlex-Q: Helium gốc → Qwen (trước sửa) → Qwen (sau sửa) → Qwen + audio heads

Tài liệu này mô tả 4 phiên bản kiến trúc và chỉ ra chỗ thay đổi giữa chúng. Lý do và kế hoạch chi tiết nằm ở [PLAN.md](PLAN.md).

Code liên quan:
- [qwen_lm.py](src/personaplex_finetuning/qwen_lm.py): wrapper `QwenMoshiLM`
- [runtime.py](src/personaplex_finetuning/runtime.py): ghép Qwen với các phần lấy từ PersonaPlex
- [peft_adapter.py](src/personaplex_finetuning/peft_adapter.py): LoRA, tham số train theo stage, checkpoint
- [train.py](src/personaplex_finetuning/train.py): loss, optimizer, schedule
- [objective.py](src/personaplex_finetuning/objective.py): trọng số từng stream

---

## 1. Phần chung cho cả 4 phiên bản

### 1.1 Mimi codec (frozen, lấy từ PersonaPlex)
- Audio 24 kHz được nén thành 12.5 frame/giây (80 ms/frame).
- Mỗi frame có 8 codebook, mỗi codebook có 2048 giá trị (`card = 2048`).
  - CB0 là **semantic**: nội dung, nói hay im lặng.
  - CB1–7 là **acoustic**: âm sắc, chi tiết giọng.

### 1.2 Mười bảy stream mỗi frame
```
stream 0      : text agent      (token chữ, hoặc PAD / EPAD)
stream 1..8   : audio agent     CB0..CB7   ← model phải sinh
stream 9..16  : audio user      CB0..CB7   ← input (nghe); chỉ tính loss khi user_loss=true
```

### 1.3 Delay
`delays = [0, 0, 1,1,1,1,1,1,1, 0, 1,1,1,1,1,1,1]`

Text và CB0 không trễ. CB1–7 trễ 1 frame. Model quyết định "nói gì" trước, rồi mới điền chi tiết âm thanh.

### 1.4 Chuỗi train (PersonaPlex hybrid prompt)
```
| voice prompt | pause | text system prompt | pause |      hội thoại       |
  agent audio = giọng mẫu + im lặng                    agent audio thật
  user audio  = sine                                   user audio thật
  text        = PAD ... prompt ... PAD                 từ của agent, căn theo thời gian
  loss mask   = False ──────────────────────────────── True
```

Trong vùng prompt, "sine" (audio user) và "im lặng" (audio agent sau giọng mẫu) là **bộ 8 mã cố định** `SINE_TOKENS` / `SILENCE_TOKENS` ([src/moshi/models/lm.py](src/moshi/models/lm.py)), lặp lại mỗi frame. Text ở các frame voice prompt / pause là PAD của tokenizer. Đây đúng là những gì LMGen đưa vào lúc inference và cách reference PersonaPlex finetune làm. Mọi khác biệt ở vùng này giữa train và inference đều làm model trượt khi sinh tự do.

---

## 2. Phiên bản A: PersonaPlex gốc (backbone Helium)

```
                 frame t-1: 17 token
                        │
        ┌───────────────▼────────────────┐
        │ Σ 16 emb_audio  +  text_emb     │  tất cả học cùng Helium
        └───────────────┬────────────────┘
                        │ x_t (4096)
        ┌───────────────▼────────────────┐
        │ Helium 7B, 32 layer, dim 4096   │  pretrain ~7M giờ audio
        │ (temporal, nhìn toàn lịch sử)   │
        └───────────────┬────────────────┘
                        │
                   out_norm (RMSNorm)
                        │ z_t (4096)
          ┌─────────────┼───────────────────────► text_linear → text_logits (32k vocab)
          │
          ▼ depformer_in[k] : 4096 → 1024
        ┌────────────────────────────────┐
        │ Depformer, 6 layer, dim 1024    │  bước 0: + depformer_text_emb(text) → CB0
        │ sinh 16 CB trong 1 frame        │  bước k: + depformer_emb(CB k-1)    → CB k
        └───────────────┬────────────────┘
                        ▼ linears[k] : 1024 → 2048
              CB0..CB7 agent, CB0..CB7 user
```

**Điểm mấu chốt:** mọi khối được train cùng nhau. Bảng embedding, `out_norm`, `depformer_in` và depformer đều khớp nhau.

---

## 3. Phiên bản B: Qwen, trước khi sửa

```
                 frame t-1: 17 token
                        │
        ┌───────────────▼────────────────┐
   ✗    │ Σ 16 emb_audio MỚI, N(0,1)      │  std ≈ 4 sau khi cộng → át text ~100x
        │  + qwen.embed_tokens(text)      │  std ~0.01–0.05
        └───────────────┬────────────────┘
                        │ x_t (hidden Qwen: 1024 / 2048 / 4096)
        ┌───────────────▼────────────────┐
        │ Qwen (frozen + LoRA q,v)        │  chưa từng thấy audio
        └───────────────┬────────────────┘
                        │ h_t
          ┌─────────────┼───────────────────────► qwen.lm_head → text_logits (~151k vocab)
          │
   ✗      ▼ depformer_in MỚI random : hidden → 1024
          │   (8B: dùng lại depformer_in của Helium, sai không gian)
        ┌────────────────────────────────┐
        │ Depformer Helium (frozen+LoRA)  │  bước 0: + text_depth_adapter MỚI random(qwen_emb(text))
        └───────────────┬────────────────┘
                        ▼
                    16 codebook

Loss = 1.0·text (PAD 0.3) + audio (CB0 1.0, CB1-7 0.02)
Train: tất cả cùng lúc, cosine schedule
```

**Vấn đề:**
1. Audio át text, nên Qwen frozen nhận input lạ và gần như không thấy text.
2. Depformer chỉ nhận ngữ cảnh qua một layer random, nên chỉ dựa vào frame trước, rồi collapse về im lặng.
3. Qwen không có đường gradient trực tiếp nào từ audio.
4. Loss text bị PAD chi phối.

---

## 4. Phiên bản C: Qwen, sau khi sửa (`audio_decoder: depformer`, mặc định)

```
                 frame t-1: 17 token
                        │
        ┌───────────────▼─────────────────────────────────────────────┐
   ①    │ x_t = g_text · qwen_embed(text)                              │  A1: 3 gain học được,
        │     + g_agent · RMS(Σ emb[0..7]  agent)                      │      audio chuẩn hóa từng nhóm
        │     + g_user  · RMS(Σ emb[8..15] user)                       │  A2: emb khởi tạo = PCA bảng
        └───────────────┬─────────────────────────────────────────────┘      emb của Helium
                        │ x_t (hidden Qwen)
        ┌───────────────▼────────────────┐
        │ Qwen decoder (frozen + LoRA q,v)│  HF KV cache khi inference
        └───────────────┬────────────────┘
                        │ hidden_states[0..L] (embedding + từng layer)
          ┌─────────────┴──────────────┐
          │ last_hidden_state           │ B2: layer_mix = Σ softmax(w)_l · final_norm(state_l)
          ▼                             ▼
   qwen.lm_head → text_logits         h_t (feature audio)
                                        │
          ┌─────────────────────────────┤
          │                             │
   ②      ▼ cb0_head(cb0_mlp(h_t + cb0_text_adapter(qwen_embed(text_t))))   B1: CB0 nhìn text của frame
          CB0 agent theo `cb0_source` (mục 4.4): qwen = nhánh này | depformer = linears[0](bước 0) | merge = cả hai
                                        │
   ③                                    ▼ backbone_proj : hidden → 4096 (MỚI, init orthogonal)
                                        ▼ out_norm (LẤY TỪ HELIUM)
                                        │ z_t (4096, cùng dạng với Helium)
                                        ▼ depformer_in[k] : 4096 → 1024 (LẤY TỪ HELIUM, train full)
        ┌────────────────────────────────┐
        │ Depformer Helium                │  `depformer_mode`: lora (frozen + LoRA) | full (dense FP32)
        │                                 │  bước 0: + text_depth_adapter(qwen_embed(text))
        │                                 │  bước k: + depformer_emb(CB k-1)
        └───────────────┬────────────────┘
                        ▼ linears[k], k ≥ 1 (train full); linears[0] chỉ dùng khi cb0_source = depformer | merge
                    CB1..CB7 agent, CB0..CB7 user
```

### 4.1 Các thay đổi (đánh số theo sơ đồ)

| # | Thay đổi | Code | Vì sao |
|---|---|---|---|
| ① A1 | `StreamGains`: chuẩn hóa RMS tổng audio của agent và của user, nhân với 3 gain học được (text, agent, user). Gain audio khởi tạo = RMS của embedding text | `StreamGains`, `embed_codes` | Tỉ lệ audio/text giữ ổn định suốt quá trình train, không phụ thuộc độ lớn bảng `emb`. Model tự cân mức nghe user so với mức nghe chính mình |
| ① A2 | `emb[k]` khởi tạo từ bảng `emb[k]` của Helium, chiếu PCA xuống hidden của Qwen, chỉnh RMS = 1 | `_pca_init`, `load_qwen_runtime` | Giữ cấu trúc "mã nào giống mã nào" mà Helium đã học qua ~7M giờ audio |
| B2 | `HiddenLayerMix`: trộn có trọng số mọi hidden state (các layer giữa đi qua norm cuối của Qwen trước khi trộn), khởi tạo trung bình đều. Text vẫn lấy từ layer cuối | `HiddenLayerMix`, `_backbone` | Layer cuối của LM chuyên đoán text; các layer giữa giữ nhiều chi tiết âm thanh và vị trí hơn |
| ② | `cb0_head` là nơi duy nhất đoán CB0 | `cb0_logits`, `forward_depformer_training`, `forward_depformer` | Qwen nhận gradient CB0 trực tiếp và tự quyết định "nói gì" (giống Cohere). Depformer chỉ lo âm sắc |
| ② B1 | `cb0_text_adapter` (khởi tạo bằng 0): CB0 nhìn text token của cùng frame | `cb0_logits` | Text và CB0 không còn được lấy mẫu độc lập. Khôi phục điều kiện "CB0 phụ thuộc text" của thiết kế gốc |
| ② B3 | `cb0_mlp`: `ResidualMLP` = `x + Linear(SiLU(Linear(RMSNorm(x))))` trước `cb0_head`, Linear cuối khởi tạo bằng 0 | `ResidualMLP`, `cb0_logits` | Khi chỉ có một Linear, CB0 chững ở khoảng 50% accuracy và loss ngừng giảm sớm. Lúc mới tạo, khối này là identity, nên checkpoint cũ (không có `cb0_mlp`) vẫn nạp được và giữ nguyên hàm; khi resume, optimizer khởi tạo lại |
| ③ | `backbone_proj` + `out_norm` của Helium + `depformer_in` của Helium | `depth_context()`, `load_qwen_runtime` | Depformer nhận lại context đúng dạng nó đã được train. Chỉ projection phải học |

Ghi chú:
- `_backbone` gọi thẳng decoder của Qwen (`get_decoder()`). Text lấy qua `lm_head(last_hidden_state)`, vì `last_hidden_state` luôn là output đã qua norm cuối, bất kể phiên bản transformers đặt gì ở cuối `hidden_states`.
- `load_qwen_runtime` chạy thử decoder 1 lần để đếm số hidden state thật, thay vì tin config.
- Bộ nhớ: B2 giữ các state đã norm (float32) cho backward, khoảng 170MB với 0.6B và 850MB với 8B (chuỗi ~1400 frame).

### 4.2 Luồng khi train (`forward_train`)
```
codes [B,17,T]
  → _delay_sequence + chèn initial token ở đầu
  → embed_codes (A1) → decoder Qwen → text_logits (layer cuối), h = layer_mix(states) (B2)
  → cb0_logits(h, text_t) (B1)  +  depth_context(h) → forward_depformer_training → logits [B,16,T,2048]
       (logits[:,0] = cb0 logits; logits[:,1:] = linears[k](depformer))
  → _undelay_sequence cho logits, text_logits (kèm mask)
```

### 4.3 Luồng khi inference (LMGen)
```
mỗi frame: forward_codes → forward_embeddings
             = (depth_context(h), text_logits)   ← LMGen coi đây là "transformer_out"
           bước 0: depformer chạy (cập nhật state) nhưng trả logits cb0_logits(h_t, text vừa lấy mẫu) → LMGen lấy mẫu CB0
           bước 1..15: depformer sinh CB1.. dựa trên CB0 đó (CB của user lấy từ input thật)
```

### 4.4 Nguồn CB0 (`cb0_source`) và chế độ depformer (`lora.depformer_mode`)

Hai tùy chọn độc lập trong config, mặc định giữ hành vi trước đó (`qwen` / `lora`). Chúng được thêm sau khi `tools.audio_probe` cho thấy CB0 là nút thắt audio lớn nhất (`force_text_cb0` rõ hơn hẳn `force_text`), và CB1–7 vẫn méo kể cả khi teacher-forced.

| `cb0_source` | CB0 logits | Train | Freeze |
|---|---|---|---|
| `qwen` | `cb0_head(cb0_mlp(h_t + cb0_text_adapter(text_t)))` | `cb0_head`, `cb0_text_adapter`, `cb0_mlp` | `linears[0]`, `cb0_depth_gate` |
| `depformer` | `linears[0](depformer bước 0)`, như PersonaPlex gốc | `linears[0]` | `cb0_head`, `cb0_text_adapter`, gate |
| `merge` | `cb0_head(...) + g · linears[0](...)`, `g` khởi tạo 0 | cả ba | — |

- Một hàm duy nhất `_agent_cb0_logits` quyết định CB0 cho cả đường train và đường chạy từng bước (bước 0 của `forward_depformer`).
- `cb0_depth_gate` luôn có trong model và checkpoint. Checkpoint cũ thiếu key này vẫn load được, với `g = 0`, đúng hành vi cũ.
- Mode `merge` log thêm `diag/cb0_head_only_acc`, `diag/cb0_depth_only_acc`, `diag/cb0_depth_gate`.

`depformer_mode`:
- `lora`: lõi depformer frozen + LoRA.
- `full`: lõi depformer dense FP32 (autocast bf16), train ở stage `joint` với `depformer_learning_rate`, frozen ở stage `interface`. Checkpoint chứa toàn bộ `depformer.*`.

**Resume:**
- Một checkpoint `lora` resume vào run `full` được **gộp LoRA vào dense** (`W + alpha/rank · B@A`), nên tiếp tục từ đúng hàm cũ.
- Đổi `cb0_source` / `depformer_mode` / stage: chỉ nạp weight, optimizer mới, step đếm lại từ 0.
- Inference / evaluate kiểm tra `adapter.json`, và báo lỗi nếu config khác mode mà checkpoint đã train.

Config mẫu: `configs/qwen3-0.6b-audio.yaml` (`cb0_source: merge`, `depformer_mode: full`).

---

## 5. Phiên bản D: Qwen + 8 audio heads, không Depformer (`audio_decoder: heads`)

```
                 frame t-1: 17 token
                        │
        ┌───────────────▼─────────────────────────────┐
        │ embed_codes (A1 StreamGains + A2 PCA emb)    │  giống C
        └───────────────┬─────────────────────────────┘
        ┌───────────────▼────────────────┐
        │ Qwen decoder (frozen + LoRA)    │  giống C
        └───────────────┬────────────────┘
                        │ hidden_states[0..L]
          ┌─────────────┴──────────────┐
          ▼ last_hidden_state           ▼ layer_mix (B2) → h_t
   qwen.lm_head → text_logits           │
                ┌───────────────────────┼──────────────────────────┐
                ▼                       ▼                          ▼
   cb0_head(cb0_mlp(h_t + cb0_text_adapter(text_t)))   acoustic_heads[1]  …  acoustic_heads[7]
                │                  (MLP residual cho từng CB, chạy song song)
                ▼                       ▼                          ▼
          CB0 agent (frame t)     CB1 … CB7 agent (frame t-1, delay 1)
```

Mỗi `AudioHead` là `Linear(dim, 2048)` đặt sau một `ResidualMLP`, cùng kiểu khối với `cb0_mlp`: `x + Linear(SiLU(Linear(RMSNorm(x))))`, Linear cuối của MLP khởi tạo bằng 0. Lúc mới tạo, head chỉ là một linear đọc từ `h_t`. CB0 và CB1–7 vì vậy có cùng dạng head.

### 5.1 Thay đổi so với C

| Thay đổi | Code | Vì sao |
|---|---|---|
| Bỏ `backbone_proj`, `out_norm`, `depformer_in`, `depformer`, `depformer_emb`, `linears`, `text_depth_adapter`, `cb0_depth_gate` | `QwenMoshiLM.__init__`, `load_qwen_runtime` | Bỏ lớp nối từ Qwen sang không gian Helium, nơi phân bố bị lệch. Không graft khối nào của Helium ngoài bảng `emb` dùng cho PCA init |
| Thêm `acoustic_heads` (7 × `AudioHead`) cho CB1–7 | `AudioHead`, `audio_head_logits` | Mọi codebook audio nhận gradient trực tiếp từ hidden của Qwen |
| CB0 giữ nguyên `cb0_head` + `cb0_text_adapter`, `cb0_source` bắt buộc `qwen` | `cb0_logits` | Checkpoint C dùng lại được cho CB0 |
| `dep_q` 16 → 8 | `load_qwen_runtime` | Chỉ sinh audio agent. Audio user luôn lấy từ input thật |
| Inference: một bước, sample 8 token cùng lúc | `forward_audio_heads`, `LMGen.depformer_step` (nhánh heads) | Thay 16 bước depth tuần tự |

**Đánh đổi:** CB1–7 của cùng một frame được đoán độc lập với nhau. Chúng chỉ thấy CB0 của frame đó qua input (delay 1), không thấy nhau. Codebook RVQ phụ thuộc nhau mạnh, nên chất âm có thể kém C. Phải A/B nghe thử với C trước khi chọn.

Cohere TinyAya cũng khai báo 8 `audio_heads` dạng `Linear(2048, 2048)` trên layer cuối, nhưng chỉ dùng `audio_heads[0]` cho CB0. `audio_heads[1:8]` không bao giờ vào loss, còn CB1–7 vẫn do depth decoder sinh. Vì vậy chưa có bằng chứng nào cho kiểu "chỉ dùng head" này.

### 5.2 Luồng khi train và inference
```
train:     codes → delay → embed_codes → Qwen → h = layer_mix(states)
           → audio_head_logits(h, text_t) → logits [B, 8, T, 2048] → undelay
inference: forward_embeddings trả (h_t, text_logits)   ← LMGen coi h_t là "transformer_out"
           depformer_step (nhánh heads): forward_audio_heads(text vừa lấy mẫu, h_t) → sample 8 token
```
Ở bước p, model trả CB0 của frame p và CB1–7 của frame p-1, khớp với delay `[0, 0, 1×7]`.

### 5.3 Checkpoint và warm-start
- `adapter.json` có thêm `audio_decoder`. Checkpoint cũ không có key này được coi là `depformer`.
- Resume checkpoint C vào run D (`--resume-from`): bỏ các key chỉ thuộc depformer, nạp các key còn lại (LoRA Qwen, `emb`, `stream_gains`, `layer_mix`, `cb0_head`, `cb0_text_adapter`) và vẫn kiểm tra strict. `acoustic_heads` khởi tạo mới. Optimizer, scheduler và step đếm lại từ 0.
- Resume D sang C báo lỗi, vì D không có depformer để khôi phục.
- Inference/evaluate báo lỗi nếu `audio_decoder` trong config khác với checkpoint.
- `lora.depformer_*` bị bỏ qua ở D.

Config mẫu: `configs/qwen3-0.6b-heads.yaml` (loss weights và LR giống `qwen3-0.6b.yaml` để so sánh công bằng).

---

## 6. Loss

```
total = text_loss_weight · CE_text
      + (Σ CE_audio có trọng số) / (Σ trọng số audio)     ← agent + user dùng chung mẫu số
                                                           (CB0 agent: logits từ cb0_head)
```

| Khóa config | Helium (`config.yaml`) | Qwen trước sửa | Qwen sau sửa (`qwen*.yaml`) |
|---|---|---|---|
| `text_padding_weight` | 0.3 | 0.3 | **0.01** |
| `text_loss_weight` | 1.0 | (không có, = 1.0) | **0.2** |
| `first_codebook_weight_multiplier` | 1.0 | 1.0 | 1.0 |
| `nonsemantic_audio_weight` | 0.02 | 0.02 (cố định trong code) | 0.02 (giờ chỉnh được trong config) |
| `epad_as_padding` | false | false | false |
| `user_loss` | false | false | false |

Log:
- `loss/text`, `loss/audio_semantic`, `loss/audio_nonsemantic`.
- `diag/*` và `val/diag/*`: CE và accuracy không trọng số (text tách từ thật / PAD, từng codebook của agent).
- `audio_semantic` là khoảng 0.88 × CE thật của CB0, do chia chung mẫu số với acoustic.
- Phiên bản D: logits chỉ có 8 hàng agent, không có nhóm user, nên `user_loss` phải là `false`. Khi `user_loss=false`, kết quả loss giống hệt C, vì nhóm user ở C vốn có trọng số 0.

---

## 7. Phần nào được train

| Nhóm tham số | Helium gốc | Qwen trước sửa | Qwen sau (C), stage `interface` | Qwen sau (C), stage `joint` | Qwen + heads (D), `interface` / `joint` |
|---|---|---|---|---|---|
| Backbone (Helium / Qwen) | LoRA | LoRA q,v | frozen | LoRA q,v | frozen / LoRA q,v |
| `emb` audio | frozen | nếu `ft_embed` | **train** | nếu `ft_embed` | **train** / nếu `ft_embed` |
| `stream_gains`, `layer_mix`, `cb0_head`, `cb0_text_adapter`, `cb0_mlp` | n/a | n/a | train | train | train |
| `backbone_proj`, `out_norm` | n/a | n/a | train | train | — |
| `depformer_in`, `depformer_emb`, `linears[1:]`, `text_depth_adapter` | frozen | train | train | train | — |
| `linears[0]`, `cb0_head` / `cb0_text_adapter` / `cb0_mlp`, `cb0_depth_gate` | — | — | theo `cb0_source` (mục 4.4) | theo `cb0_source` (mục 4.4) | chỉ `cb0_head` / `cb0_text_adapter` / `cb0_mlp` |
| `acoustic_heads` | — | — | — | — | train |
| Depformer core | LoRA (tùy stage) | LoRA | frozen | LoRA (`depformer_mode: lora`) hoặc dense FP32 (`full`) | — |

Learning rate (Qwen):
- `qwen.*` dùng `learning_rate`.
- `emb.*` dùng `audio_embed_lr`.
- `depformer.*` (LoRA hoặc dense khi `full`) dùng `depformer_learning_rate`, nếu không có thì dùng `interface_lr`.
- Còn lại (kể cả `acoustic_heads` ở D) dùng `interface_lr`.

Schedule: `train.lr_schedule` là `cosine` (mặc định) hoặc `wsd`.

**Checkpoint Qwen** (`lora.safetensors`):
- Luôn chứa toàn bộ LoRA, các lớp nối và `emb`, kể cả phần đang freeze. Nhờ vậy checkpoint stage `interface` resume được sang `joint`.
- Khi resume mà stage thay đổi: chỉ nạp weight, optimizer mới, step đếm lại từ 0.
- Checkpoint Qwen cũ (phiên bản B) **không tương thích**.

---

## 8. So sánh với Cohere (TinyAya + Moshi)

| | Cohere | Qwen sau sửa (C) | Qwen + heads (D) |
|---|---|---|---|
| Audio embedding | Chung bảng với backbone | Bảng riêng, scale theo text | Như C |
| Ai đoán CB0 | Backbone (`audio_heads[0]` = `Linear(2048, 2048)` trên layer cuối), cả khi inference | Backbone (`cb0_head` trên `layer_mix`, nhìn text của frame), cả khi train và inference | Như C |
| Nối sang depth decoder | Projection 2048→4096, I/O train full | Projection →4096 + `out_norm` Helium, I/O train full | Không có |
| Depth decoder | Frozen, chỉ sinh CB1–7 | Frozen + LoRA, sinh 16 CB | Không có; 7 head MLP song song cho CB1–7 |
| Delay | CB_k trễ k frame | `[0, 0, 1×7]` | `[0, 0, 1×7]` |
| LoRA backbone | q, v, embed, r=64 | q, v, r=64 | q, v, r=64 |
| Text / padding weight | 0.1–0.2 / 0.01 | 0.2 / 0.01 | 0.2 / 0.01 |

---

## 9. Chưa làm / rủi ro

- **PAD/EPAD trùng nhau** ở Qwen2.5-3B base (`<|endoftext|>`). Qwen3 dùng `<|im_end|>` làm EPAD. Phần này thuộc phase 1, chưa sửa.
- **Phase 0:** script baseline (unigram/bigram CB0, ablation thay output Qwen bằng 0) chưa làm.
- **Chưa chạy trên GPU** với checkpoint thật. Nên chạy `--smoke` trước.
- Qwen không có pretrain audio. Kỳ vọng thực tế: phía text tốt, âm thanh kém tự nhiên hơn PersonaPlex gốc.
- **Phiên bản D chưa chạy trên GPU.** Chưa biết các head song song cho CB1–7 có thắng depformer hay không. `nonsemantic_audio_weight=0.02` có thể quá thấp cho head mới học từ đầu: theo dõi `diag/agent_cb1..7` trước khi tăng.
