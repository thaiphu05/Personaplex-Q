# Kiến trúc PersonaPlex-Q: Helium gốc → Qwen (trước sửa) → Qwen (sau sửa)

Tài liệu này mô tả 3 phiên bản kiến trúc và chỉ ra chỗ thay đổi giữa chúng. Lý do và kế hoạch chi tiết nằm ở [PLAN.md](PLAN.md).

Code liên quan:
- [qwen_lm.py](src/personaplex_finetuning/qwen_lm.py): wrapper `QwenMoshiLM`
- [runtime.py](src/personaplex_finetuning/runtime.py): ghép Qwen với các phần lấy từ PersonaPlex
- [peft_adapter.py](src/personaplex_finetuning/peft_adapter.py): LoRA, tham số train theo stage, checkpoint
- [train.py](src/personaplex_finetuning/train.py): loss, optimizer, schedule
- [objective.py](src/personaplex_finetuning/objective.py): trọng số từng stream

---

## 1. Phần chung cho cả 3 phiên bản

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

## 4. Phiên bản C: Qwen, sau khi sửa (hiện tại)

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
   ②      ▼ cb0_head(h_t + cb0_text_adapter(qwen_embed(text_t)))     B1: CB0 nhìn text của frame
          CB0 agent (nơi DUY NHẤT đoán CB0, train + infer)
                                        │
   ③                                    ▼ backbone_proj : hidden → 4096 (MỚI, init orthogonal)
                                        ▼ out_norm (LẤY TỪ HELIUM)
                                        │ z_t (4096, cùng dạng với Helium)
                                        ▼ depformer_in[k] : 4096 → 1024 (LẤY TỪ HELIUM, train full)
        ┌────────────────────────────────┐
        │ Depformer Helium (frozen+LoRA)  │  bước 0: + text_depth_adapter(qwen_embed(text))  (chỉ làm ngữ cảnh)
        │                                 │  bước k: + depformer_emb(CB k-1)
        └───────────────┬────────────────┘
                        ▼ linears[k], k ≥ 1 (train full; linears[0] không dùng, freeze)
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

---

## 5. Loss

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

---

## 6. Phần nào được train

| Nhóm tham số | Helium gốc | Qwen trước sửa | Qwen sau, stage `interface` | Qwen sau, stage `joint` |
|---|---|---|---|---|
| Backbone (Helium / Qwen) | LoRA | LoRA q,v | frozen | LoRA q,v |
| `emb` audio | frozen | nếu `ft_embed` | **train** | nếu `ft_embed` |
| `stream_gains`, `layer_mix`, `backbone_proj`, `out_norm`, `cb0_head`, `cb0_text_adapter` | n/a | n/a | train | train |
| `depformer_in`, `depformer_emb`, `linears[1:]`, `text_depth_adapter` | frozen | train | train | train |
| `linears[0]` (CB0 của depformer) | frozen | train | **freeze, không dùng** | **freeze, không dùng** |
| Depformer core | LoRA (tùy stage) | LoRA | frozen | LoRA |

Learning rate (Qwen):
- `qwen.*` dùng `learning_rate`.
- `emb.*` dùng `audio_embed_lr`.
- `depformer.*` (LoRA) dùng `depformer_learning_rate`, nếu không có thì dùng `interface_lr`.
- Còn lại dùng `interface_lr`.

Schedule: `train.lr_schedule` là `cosine` (mặc định) hoặc `wsd`.

**Checkpoint Qwen** (`lora.safetensors`):
- Luôn chứa toàn bộ LoRA, các lớp nối và `emb`, kể cả phần đang freeze. Nhờ vậy checkpoint stage `interface` resume được sang `joint`.
- Khi resume mà stage thay đổi: chỉ nạp weight, optimizer mới, step đếm lại từ 0.
- Checkpoint Qwen cũ (phiên bản B) **không tương thích**.

---

## 7. So sánh với Cohere (TinyAya + Moshi)

| | Cohere | Qwen sau sửa (C) |
|---|---|---|
| Audio embedding | Chung bảng với backbone | Bảng riêng, scale theo text |
| Ai đoán CB0 | Backbone (audio head), cả khi inference | Backbone (`cb0_head`), cả khi train và inference |
| Nối sang depth decoder | Projection 2048→4096, I/O train full | Projection →4096 + `out_norm` Helium, I/O train full |
| Depth decoder | Frozen, chỉ sinh CB1–7 | Frozen + LoRA, sinh 16 CB |
| LoRA backbone | q, v, embed, r=64 | q, v, r=64 |
| Text / padding weight | 0.1–0.2 / 0.01 | 0.2 / 0.01 |

---

## 8. Chưa làm / rủi ro

- **PAD/EPAD trùng nhau** ở Qwen2.5-3B base (`<|endoftext|>`). Qwen3 dùng `<|im_end|>` làm EPAD. Phần này thuộc phase 1, chưa sửa.
- **Phase 0:** script baseline (unigram/bigram CB0, ablation thay output Qwen bằng 0) chưa làm.
- **Chưa chạy trên GPU** với checkpoint thật. Nên chạy `--smoke` trước.
- Qwen không có pretrain audio. Kỳ vọng thực tế: phía text tốt, âm thanh kém tự nhiên hơn PersonaPlex gốc.
