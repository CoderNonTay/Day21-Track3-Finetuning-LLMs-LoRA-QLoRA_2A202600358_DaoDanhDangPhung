# Lab 21 - Evaluation Report

**Hoc vien**: Dao Danh Dang Phung - 2A202600358  
**Ngay nop**: 2026-05-07  
**Submission option**: A - Lightweight ZIP  

## 1. Setup

- **Base model**: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit + LoRA adapters with Unsloth and TRL `SFTTrainer`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples
- **Split**: 170 train / 30 eval, seed = 42
- **Format**: Alpaca style with `instruction`, optional `input`, and `output`
- **Token length analysis**: p50 = 227, p95 = 562, p99 = 704
- **max_seq_length**: 1024, rounded up from p95 and capped for T4 profile
- **GPU**: Tesla T4, approximately 14.6 GB available VRAM
- **Training config**: 3 epochs, learning rate = 2e-4, cosine schedule, warmup ratio = 0.10, train batch size = 1, gradient accumulation = 8, effective batch size = 8
- **LoRA config**: `target_modules=["q_proj", "v_proj"]`, `lora_dropout=0`, `bias="none"`, gradient checkpointing enabled
- **Estimated training cost**: about $0.08, computed from 14.04 training minutes at $0.35/hour
- **HF Hub adapter**: https://huggingface.co/codernontay/lab21-qwen3-4b-lora-r16

Token length figure: `token_length_distribution.png`

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| Base | - | - | - | - | 1.8970 | 6.6661 |
| 8 | 16 | 2,949,120 | 4.48 min | 13.99 GB | 1.4358 | 4.2031 |
| 16 | 32 | 5,898,240 | 4.77 min | 10.32 GB | 1.3982 | 4.0479 |
| 64 | 128 | 23,592,960 | 4.79 min | 15.03 GB | 1.3275 | 3.7714 |

The experiment successfully produced adapters for all three required ranks: `r8`, `r16`, and `r64`, and the base model was also evaluated on the same eval split. All LoRA adapters improved perplexity over the base model. Increasing rank increased the number of trainable parameters linearly. Rank 64 had 8 times more trainable parameters than rank 8 and 4 times more than rank 16. In this run, rank 64 achieved the lowest eval loss and perplexity, but it also used the most VRAM.

## 3. Loss Curve Analysis

The training loss generally decreased across all three ranks. The final logged training losses were:

| Rank | First Logged Loss | Final Logged Loss |
|---:|---:|---:|
| 8 | 1.8983 | 1.3521 |
| 16 | 1.8876 | 1.3056 |
| 64 | 1.8373 | 1.1616 |

The rank 64 adapter learned fastest and ended with the lowest training loss. Rank 16 was consistently better than rank 8 after the first few logging steps. The loss curves show normal convergence under the cosine learning rate schedule: loss decreases quickly in the first half, then improves more slowly near the end as the learning rate becomes small.

There is no strong evidence of overfitting from the exported logs, but the conclusion is limited because evaluation was disabled during training on T4 to avoid OOM. The safer interpretation is: training loss decreased and final eval loss also improved as rank increased, so the adapters were learning useful behavior. A more rigorous overfitting check would require logging eval loss after each epoch.

Loss curve figure: `loss_curve.png`

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Gave a formal definition of machine learning and described learning from data.  
**Fine-tuned r16**: Answered in a more tutorial-like tone, starting with a beginner-friendly explanation and emphasizing pattern recognition from data.  
**Nhận xét**: Improved readability for beginners.

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Produced correct code, but also continued into an unwanted explanation prompt.  
**Fine-tuned r16**: Returned a compact Python function only.  
**Nhận xét**: Fine-tuned output followed the requested format better.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Listed principles but the response was long and dense.  
**Fine-tuned r16**: Produced a more direct numbered list covering usability, consistency, ease of use, and related UX principles.  
**Nhận xét**: Slightly better structure, though the answer could still be cleaner.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Explained LoRA and QLoRA generally, but the answer was verbose.  
**Fine-tuned r16**: Focused more directly on parameter-efficient adaptation and reduced training cost.  
**Nhận xét**: Improved relevance, but the explanation should explicitly mention 4-bit quantization for QLoRA.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Explained the three methods but became long and less organized.  
**Fine-tuned r16**: Started with clearer definitions and separated the concepts more directly.  
**Nhận xét**: Improved clarity, but a table format would be even better.

Overall, the r16 fine-tuned model often produced shorter and more instruction-following responses. The biggest improvement was in format control, especially for the Fibonacci coding prompt. However, some outputs were still incomplete or not perfectly structured, so the fine-tuned model is not uniformly better on all prompts.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, cả ba adapter đều cải thiện rõ rệt so với base model: perplexity giảm từ 6.6661 xuống 4.2031 với rank 8, 4.0479 với rank 16, và 3.7714 với rank 64. Rank 64 cho kết quả định lượng tốt nhất: eval loss thấp nhất và perplexity thấp nhất. Điều này hợp lý vì rank cao hơn cho adapter nhiều năng lực biểu diễn hơn, giúp mô hình học được nhiều thay đổi hơn từ dataset. Tuy nhiên, rank 64 cũng có chi phí cao nhất: 23.6M tham số trainable và peak VRAM khoảng 15.03 GB, gần giới hạn của Tesla T4. Rank 8 là lựa chọn nhẹ nhất nhưng perplexity cao nhất trong ba adapter, cho thấy capacity hơi thấp cho dataset này. Rank 16 là điểm cân bằng tốt: perplexity tốt hơn rank 8, số tham số chỉ bằng một phần tư rank 64, và VRAM thấp nhất trong log đã export. Nếu mục tiêu là nộp lab và chạy ổn định trên Colab Free T4, tôi sẽ chọn rank 16 làm adapter chính. Nếu mục tiêu là tối ưu chất lượng offline và GPU đủ ổn định, rank 64 đáng cân nhắc, nhưng lợi ích perplexity cần được so với rủi ro OOM và thời gian train.

## 6. What I Learned

- LoRA rank ảnh hưởng trực tiếp đến số tham số trainable: rank càng cao thì adapter càng có nhiều capacity, nhưng VRAM cũng tăng.
- QLoRA giúp fine-tune model nhiều tỷ tham số trên T4 bằng cách giữ base model ở 4-bit và chỉ train LoRA adapter.
- Perplexity là metric hữu ích để so sánh định lượng, nhưng qualitative prompts vẫn cần thiết vì mô hình có thể có perplexity tốt hơn nhưng format câu trả lời chưa chắc tốt hơn.

## Submission Readiness Check

| Requirement | Status |
|---|---|
| Train adapter r=8 | Done |
| Train adapter r=16 | Done |
| Train adapter r=64 | Done |
| Save adapter weights | Done |
| Rank comparison CSV | Done |
| Qualitative comparison CSV | Done |
| Token length distribution image | Done |
| Loss curve image | Done |
| Base model perplexity | Done |
| At least 5 qualitative examples | Done |
| Report with 6 required sections | Done |

Before final submission, strip notebook outputs if the upload size is too large, and package only the required files for Option A.
