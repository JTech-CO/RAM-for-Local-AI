# 로컬 AI 파인튜닝 메모리 가이드
> Full fine-tuning·LoRA·QLoRA·DoRA·SFT·DPO·GRPO·FSDP2·ZeRO를 RAM·VRAM·Apple 통합 메모리 기준으로 설계·측정·운영하는 가이드

[← 메인 README](../../README.md) · [생산성·문서·RAG](../domains/productivity-rag.md) · [데이터 분석](../domains/data-analysis.md) · [비전·OCR](../modalities/vision-ocr.md) · [이미지 생성](../modalities/image-generation.md) · [오디오·음성](../modalities/audio-speech.md)

> **최종 검증일:** 2026-07-21 (KST)
> **주요 도구:** Transformers·PEFT·TRL, bitsandbytes, torchtune, Accelerate·FSDP2, DeepSpeed ZeRO, TorchTitan, Axolotl, LLaMA-Factory, Unsloth, MLX-LM, Diffusers
> **범위:** 언어·코드·수학 모델, VLM·OCR, 이미지 생성, 음성·오디오, 임베딩·reranker의 full fine-tuning·PEFT·선호학습·분산학습 메모리 계산과 검증
> **관련 문서:** [양자화](./quantization.md) · [서빙·동시성](./serving-concurrency.md) (예정) · [런타임·하드웨어](./runtime-hardware.md) (예정)

이 문서는 “모델 파일이 VRAM에 들어가면 학습할 수 있는가?”, “7B Q4가 4GB이므로 8GB GPU에서 QLoRA가 항상 가능한가?”, “GPU 두 장이면 메모리가 정확히 두 배가 되는가?” 같은 오해를 피하면서, 보유한 **GPU VRAM**, **시스템 RAM**, **Apple Silicon 통합 메모리**에 맞는 파인튜닝 방식을 선택하기 위한 실전 가이드다.

추론은 대체로 가중치와 KV 캐시를 중심으로 계산하지만, 학습에는 역전파를 위한 activation, gradient, optimizer state, 통신 buffer와 checkpoint가 추가된다. 따라서 같은 모델이라도 추론과 학습의 peak memory는 크게 다르다.

```text
학습 peak accelerator memory
≈ resident model weights
+ trainable parameter copies
+ gradients
+ optimizer states
+ saved activations
+ logits·loss·temporary workspace
+ quantization·dequantization buffer
+ distributed communication buffer
+ allocator fragmentation
+ modality-specific encoder·decoder state
```

시스템 전체로는 다음도 포함한다.

```text
전체 학습 자원
≈ accelerator memory
+ CPU-side model·optimizer offload
+ dataloader·tokenization·augmentation buffer
+ pinned memory
+ dataset cache
+ checkpoint staging
+ 운영체제·IDE·브라우저·디스플레이 여유
```

> **핵심 원칙:** 파라미터 수만으로 구매 결정을 하지 않는다. 먼저 실제 dataset의 길이 분포와 목표 `micro_batch_size`, `max_length`, modality 입력 크기를 정하고, 가장 작은 대표 batch로 한 step을 실행해 peak memory를 측정한다. 가중치가 들어간다는 사실은 학습 가능성을 보장하지 않는다.

모델 구조, attention kernel, framework, optimizer, precision과 allocator 동작은 빠르게 변한다. 이 문서의 수치표는 **용량 계획용 근사치**이며, 실제 도입 전에는 대상 모델·runtime·hardware에서 dry run과 profiler로 검증해야 한다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [먼저 파인튜닝이 필요한지 판단하기](#2-먼저-파인튜닝이-필요한지-판단하기)
3. [학습 방식 선택](#3-학습-방식-선택)
4. [학습 메모리 해부](#4-학습-메모리-해부)
5. [파라미터당 바이트 계산](#5-파라미터당-바이트-계산)
6. [Activation·context·batch 메모리](#6-activationcontextbatch-메모리)
7. [Full fine-tuning](#7-full-fine-tuning)
8. [LoRA](#8-lora)
9. [QLoRA](#9-qlora)
10. [DoRA·rsLoRA·초기화 기법](#10-dorarslora초기화-기법)
11. [Optimizer·gradient·precision](#11-optimizergradientprecision)
12. [메모리 최적화 기법](#12-메모리-최적화-기법)
13. [Packing·padding-free·loss 최적화](#13-packingpadding-freeloss-최적화)
14. [RAM·VRAM별 30초 모델 선택](#14-ramvram별-30초-모델-선택)
15. [파라미터 규모별 모델 상태 용량표](#15-파라미터-규모별-모델-상태-용량표)
16. [단일 GPU 권장 구성](#16-단일-gpu-권장-구성)
17. [Apple Silicon과 MLX-LM](#17-apple-silicon과-mlx-lm)
18. [CPU RAM·NVMe offload](#18-cpu-ramnvme-offload)
19. [다중 GPU: DDP·FSDP2·ZeRO](#19-다중-gpu-ddpfsdp2zero)
20. [TP·PP·CP·SP·EP](#20-tpppcpspep)
21. [Dense와 MoE 모델](#21-dense와-moe-모델)
22. [SFT·continued pretraining](#22-sftcontinued-pretraining)
23. [DPO·KTO·ORPO·GRPO·PPO](#23-dpoktoorpogrpoppo)
24. [VLM·OCR 파인튜닝](#24-vlmocr-파인튜닝)
25. [이미지 생성 파인튜닝](#25-이미지-생성-파인튜닝)
26. [오디오·음성 파인튜닝](#26-오디오음성-파인튜닝)
27. [임베딩·reranker·분류 모델](#27-임베딩reranker분류-모델)
28. [프레임워크 선택과 실행 예제](#28-프레임워크-선택과-실행-예제)
29. [측정·벤치마크·OOM 문제 해결](#29-측정벤치마크oom-문제-해결)
30. [데이터·보안·라이선스·재현성](#30-데이터보안라이선스재현성)
31. [주요 출처와 저장소](#31-주요-출처와-저장소)
32. [최종 권장안과 갱신 주의](#32-최종-권장안과-갱신-주의)

---

## 1. 30초 선택표

### 1.1 학습 방식

| 상황 | 우선 시작점 | 이유 | 다음 단계 |
| --- | --- | --- | --- |
| 말투·출력 형식·업무 절차 적응 | LoRA | base 품질을 유지하면서 gradient·optimizer 메모리를 크게 줄임 | rank·target module A/B 평가 |
| VRAM이 매우 제한됨 | QLoRA NF4 | frozen base를 4-bit로 저장 | context·activation이 남는지 확인 |
| 작은 모델을 크게 바꾸거나 새 domain을 깊게 학습 | Full fine-tuning | 모든 weight를 업데이트 | FSDP2·ZeRO·8-bit optimizer 검토 |
| 새로운 지식·언어·코드 분포를 대량 corpus로 주입 | Continued pretraining | next-token 학습으로 분포 자체를 적응 | data mixture·forgetting 평가 |
| instruction-following 개선 | SFT + LoRA/QLoRA | 가장 단순하고 검증하기 쉬움 | preference 학습은 SFT 기준선 후 |
| 선호 응답 쌍이 있음 | DPO·KTO·ORPO 계열 | reward model 없이 선호 최적화 가능 | policy/reference 메모리 확인 |
| 검증 가능한 reward로 reasoning·agent 행동 최적화 | GRPO·RLOO 등 | critic 없는 구성 가능 | rollout·KV·generation engine 예산 |
| 이미지 스타일·캐릭터·제품 적응 | Diffusion LoRA | UNet·DiT 일부만 학습 | text encoder·resolution별 peak 측정 |
| VLM 문서·OCR 적응 | projector/LLM LoRA부터 | vision encoder full FT를 피함 | visual token 수와 image size 제한 |
| ASR·TTS 화자·domain 적응 | adapter·LoRA·부분 동결 | 긴 audio activation과 codec 비용 억제 | duration bucket·streaming 평가 |

### 1.2 장비만 알고 있을 때

아래는 **단일 accelerator에서 일반적인 text SFT·QLoRA를 시작하기 위한 보수적 범위**다. 모델 architecture, vocabulary, context와 kernel에 따라 크게 달라진다. 시스템 RAM만 많고 VRAM이 작으면 offload는 가능하지만 학습 속도는 크게 낮아질 수 있다.

| 실사용 가능 accelerator memory | 보수적인 시작점 | 공격적인 실험 | 기본 설정 |
| ---: | --- | --- | --- |
| 6–8 GB | 0.5–1.5B LoRA·QLoRA | 3B QLoRA, 512–1K tokens | micro-batch 1, checkpointing |
| 10–12 GB | 1–3B LoRA, 3B QLoRA | 7–8B QLoRA, 짧은 context | NF4, batch 1, 1–2K tokens |
| 16 GB | 3B LoRA, 7–8B QLoRA | 14B QLoRA, 512–1K tokens | gradient checkpointing, packing |
| 20–24 GB | 7–8B LoRA 또는 QLoRA | 14B QLoRA, 27B 극저 context | BF16 compute, batch 1–2 |
| 32 GB | 14B LoRA·QLoRA | 27–32B QLoRA | 2–4K tokens부터 측정 |
| 40–48 GB | 27–32B QLoRA | 65–70B QLoRA의 엄격한 설정 | CPU RAM 128GB 이상 권장 |
| 64 GB | 32B LoRA·QLoRA | 70B QLoRA | 긴 context보다 안정성 우선 |
| 80 GB | 70B QLoRA | 32B BF16 LoRA·소형 full FT | distributed rollout과 분리 |
| 96–128 GB | 70B QLoRA 여유, 120B QLoRA 실험 | 7–8B full FT, 구현별 차이 큼 | model state + activation 실측 |
| 192 GB 이상 | 235B QLoRA 연구, 14B급 full FT 후보 | 대형 preference/RL | sharding·TP·checkpoint 설계 필수 |

> 위 표의 “가능”은 throughput이나 수렴 품질을 보장하지 않는다. 7B QLoRA가 10GB 미만에 들어가는 공식 예제가 있어도, 8K context·큰 vocabulary·VLM 입력·다중 adapter를 추가하면 같은 장비에서 OOM이 발생할 수 있다.

### 1.3 즉시 판단 규칙

```text
1. 같은 dataset에서 prompt/RAG로 목표 달성 가능?
   → 가능하면 파인튜닝하지 않음

2. base BF16 weights + 20% 여유가 VRAM에 들어감?
   → LoRA부터 시도

3. BF16 base가 들어가지 않음?
   → QLoRA NF4 + BF16 compute

4. QLoRA base는 들어가지만 첫 backward에서 OOM?
   → max_length ↓ → micro-batch 1 → checkpointing → packing
   → logits/loss 최적화 → activation offload

5. model state 자체가 들어가지 않음?
   → FSDP2/ZeRO-3 또는 CPU/NVMe offload

6. 성능이 부족하다고 즉시 rank·epoch를 늘리지 않음
   → data 품질·loss masking·chat template·evaluation부터 점검
```

---

## 2. 먼저 파인튜닝이 필요한지 판단하기

파인튜닝은 사실 검색, 도구 호출, prompt engineering 또는 규칙 기반 검증으로 해결할 문제를 가중치에 영구적으로 넣는 수단이 아니다. 다음 순서로 판단한다.

### 2.1 파인튜닝보다 먼저 검토할 것

| 목표 | 먼저 시도할 방법 | 파인튜닝이 필요한 신호 |
| --- | --- | --- |
| 최신 사내 지식 답변 | RAG·검색·citation | 검색은 정확하지만 답변 형식·절차가 반복적으로 실패 |
| JSON·SQL·함수 호출 | schema constrained decoding·validator | 충분한 few-shot 후에도 특정 domain 구조가 지속 실패 |
| 브랜드 말투 | system prompt·style examples | context 비용이 너무 크거나 장문 전체에서 일관성이 깨짐 |
| 도구 사용 순서 | workflow engine·state machine | 다양한 입력에 대한 정책 일반화가 필요 |
| 보안 정책 준수 | 외부 policy engine·allowlist | 분류·거절 스타일 보조가 필요하되 최종 통제는 외부 유지 |
| 새 언어·도메인 | retrieval·terminology glossary | tokenizer·표현 분포 자체가 부족하고 대량 corpus가 있음 |
| 계산·코드 정확도 | 실행기·테스트·formal checker | 오류 패턴이 반복되고 고품질 정답 데이터가 충분함 |

### 2.2 파인튜닝이 잘하는 것

- 출력 형식과 답변 스타일을 일관되게 만들기
- domain별 문장·용어·코드 관습 적응
- 장황함, 거절 방식, 도구 호출 패턴 조정
- 특정 입력에서 필요한 추론 절차를 습관화
- 작은 분류·추출·reranking task 최적화
- 다른 base 모델보다 작은 모델이 특정 task를 잘 수행하도록 압축

### 2.3 파인튜닝이 보장하지 않는 것

- 최신 사실성
- 검색되지 않은 내부 문서의 정확한 기억
- 수학·코드의 형식적 정확성
- prompt injection 방어
- 개인정보 삭제 보장
- license 문제 해결
- 모델 전체의 안전성
- 충분한 validation 없이 production 성능 개선

### 2.4 비용 판단

```text
총 비용
= 데이터 수집·정제·검수
+ 학습 compute
+ 실험 실패와 hyperparameter 탐색
+ 평가·red-team
+ adapter·checkpoint 저장
+ serving 호환성
+ 주기적 재학습·rollback
```

작은 LoRA 파일은 저장 비용을 줄이지만, 데이터와 평가 비용까지 작게 만들지는 않는다.

---

## 3. 학습 방식 선택

### 3.1 방식 비교

| 방식 | 업데이트 대상 | base 저장 정밀도 | 메모리 특징 | 적합한 경우 |
| --- | --- | --- | --- | --- |
| Full fine-tuning | 모든 파라미터 | BF16·FP16·FP32 | gradient·optimizer state가 전체 모델에 필요 | 작은 모델, 깊은 domain shift, 충분한 cluster |
| Partial fine-tuning | 일부 layer·head·norm | 보통 BF16 | 선택 layer만 train state 보유 | encoder 상단·projector·head 적응 |
| LoRA | 저랭크 adapter | BF16·FP16 | base는 frozen, adapter만 gradient·optimizer | 범용 SFT·domain adaptation |
| QLoRA | 저랭크 adapter | frozen base NF4 등 4-bit | base resident memory 최소화 | 단일 GPU 대형 모델 적응 |
| DoRA | direction LoRA + magnitude | BF16 또는 4-bit base | LoRA보다 trainable state·연산 증가 | LoRA 품질이 부족하고 여유가 있을 때 |
| Prompt/prefix tuning | 학습 가능한 virtual token·prefix | frozen | trainable state는 매우 작음 | task가 단순하고 base가 강함 |
| Adapter layer | bottleneck module | frozen | layer마다 작은 module 추가 | 여러 task adapter 관리 |
| Full/partial multimodal FT | encoder·projector·LLM 일부/전체 | 혼합 | visual/audio activation이 큼 | 충분한 paired data와 GPU cluster |
| Distillation | student 전체 또는 일부 | teacher는 inference | teacher logits·generation 비용 추가 | 작은 production model 구축 |
| QAT | full/partial weight + fake quant state | 고정밀 train copy | PTQ보다 복잡, 추가 observer·scale | 최종 low-bit serving 품질 회복 |

### 3.2 SFT와 continued pretraining

| 항목 | SFT | Continued pretraining |
| --- | --- | --- |
| 데이터 | instruction-response, chat, task examples | raw text·code·domain corpus |
| loss | 응답 token 중심 next-token loss | 전체 token next-token loss |
| 목적 | 행동·형식·task 수행 | 분포·용어·지식 표현 적응 |
| 데이터 품질 | 적은 양도 가능하지만 정답성이 중요 | 대량·중복 제거·mixture 설계 중요 |
| forgetting 위험 | 비교적 낮음 | 높을 수 있음 |
| 메모리 | 같은 token 수라면 유사 | 긴 sequence·대량 throughput 요구가 흔함 |

### 3.3 PEFT를 기본으로 삼을 조건

다음 중 하나라도 해당하면 LoRA 또는 QLoRA를 먼저 사용한다.

- base 모델이 3B 이상
- 여러 task별 adapter가 필요
- 원본 weights를 보존해야 함
- 단일 GPU·단일 Mac에서 반복 실험
- checkpoint 저장과 배포 비용이 중요
- full FT 기준선의 model state가 accelerator memory를 초과

### 3.4 Full fine-tuning을 검토할 조건

- 모델이 작고 전체 state가 충분히 들어감
- base distribution과 target distribution 차이가 큼
- adapter 표현력 부족이 반복 평가로 확인됨
- 최종 모델을 단일 self-contained artifact로 운영해야 함
- tokenizer·embedding·output head를 크게 바꿔야 함
- 연구 목적상 full-parameter update가 필요

---

## 4. 학습 메모리 해부

### 4.1 구성요소

| 구성요소 | 생존 기간 | 증가 요인 | 주요 절감 방법 |
| --- | --- | --- | --- |
| base weights | 전체 run | 파라미터 수·dtype | QLoRA·sharding·offload |
| trainable weights | 전체 run | full/partial/adapter 범위 | LoRA·target 축소 |
| FP32 master copy | 전체 run 또는 optimizer step | mixed precision 구현 | native BF16·sharding·optimizer 선택 |
| gradients | backward 이후 step까지 | trainable parameter 수·dtype | PEFT·sharding·optimizer-in-backward |
| Adam first/second moments | 전체 run | trainable parameter 수 | 8-bit optimizer·Adafactor·sharding |
| saved activations | forward부터 backward까지 | batch·sequence·layer·hidden·modality | checkpointing·shorter sequence·offload |
| attention workspace | 각 op peak | attention backend·sequence | FlashAttention·SDPA·sequence parallel |
| logits | loss 계산까지 | batch × sequence × vocabulary | chunked CE·selective logit |
| quantization metadata | 전체 run | group size·format | scheme 선택, 일반적으로 작음 |
| dequant workspace | 각 linear op | QLoRA kernel·dtype | 지원 kernel·batch 조정 |
| communication buffer | collective 동안 | FSDP·TP·DDP bucket | bucket·prefetch·wrap tuning |
| dataloader buffer | CPU | workers·prefetch·sample size | worker·prefetch_factor 축소 |
| checkpoint staging | save 시점 | optimizer·shard·serialization | sharded checkpoint·CPU/disk 여유 |

### 4.2 peak는 합계가 아니라 시간축 문제다

모든 tensor가 동시에 존재하지는 않지만, 특정 op에서 temporary tensor와 통신 all-gather가 겹치면 OOM이 발생한다. 평균 사용량보다 **최대 allocated·reserved memory**를 기록해야 한다.

```text
steady resident memory
  < forward peak
  < backward peak
  < optimizer-step peak
  < checkpoint save·model gather peak가 될 수 있음
```

분산학습에서는 checkpoint를 단일 full state dict로 모으는 순간, 평소 학습보다 큰 CPU RAM·VRAM이 필요할 수 있다. 가능한 경우 sharded state dict를 유지하고 별도 merge job을 사용한다.

### 4.3 RAM·VRAM·통합 메모리 해석

#### 전용 GPU

- VRAM은 학습 tensor와 CUDA context가 사용한다.
- 시스템 RAM은 dataset, dataloader, offload, checkpoint staging에 사용한다.
- CPU offload는 VRAM을 줄이지만 PCIe 전송과 CPU optimizer 비용을 늘린다.
- display GPU는 desktop·browser·IDE가 VRAM을 점유한다.

#### Apple Silicon

- CPU와 GPU가 통합 메모리를 공유한다.
- “64GB Mac”에서 64GB 전체를 model에 배정할 수 없다.
- macOS, display, filesystem cache와 Python process가 같은 pool을 사용한다.
- swap 발생은 동작 가능성과 실용 속도를 구분하는 신호다.

#### 다중 GPU

- DDP는 각 GPU에 전체 모델 state를 복제한다.
- FSDP2·ZeRO-3는 state를 shard하지만 compute 시 layer를 all-gather한다.
- activation과 일부 buffer는 rank마다 복제된다.
- `N × VRAM`이 단일 연속 메모리처럼 동작하지 않는다.

### 4.4 최소 여유

| 환경 | 권장 여유 | 이유 |
| --- | ---: | --- |
| 단일 NVIDIA GPU | peak 측정값 대비 10–15% | allocator·kernel·batch 길이 변동 |
| display 겸용 GPU | 15–25% | desktop·browser·video memory |
| Apple 통합 메모리 | 20–30% | OS와 CPU 작업이 같은 pool 사용 |
| FSDP·ZeRO | 10–20% per rank | all-gather·reduce-scatter·prefetch peak |
| CPU/NVMe offload | RAM 20% + disk 2× checkpoint | pinned buffer·serialization·resume |
| online RL | 20–35% | generation KV cache·rollout 길이 변동 |

---

## 5. 파라미터당 바이트 계산

### 5.1 기본식

파라미터 수를 `P`, 구성요소별 bytes per parameter를 `b_i`라 하면 activation을 제외한 model-state 용량은 다음과 같다.

```text
model_state_GiB
= P × Σ(b_i) / 1024³
```

`P`가 billion 단위라면:

```text
model_state_GiB
≈ P_billion × bytes_per_parameter × 0.9313
```

예를 들어 7B 모델의 BF16 weight만 적재하면 약 `7 × 2 × 0.9313 = 13.0 GiB`다.

### 5.2 Full fine-tuning의 대표 범위

| 구성 | weights | gradients | optimizer | master copy | 합계·activation 제외 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 AdamW | 4 B/P | 4 B/P | 8 B/P | 없음 | 약 16 B/P |
| 고전적 FP16 mixed precision AdamW | FP16 2 B/P | FP32 4 B/P | 8 B/P | FP32 4 B/P | 약 18 B/P |
| native BF16, BF16 grad, FP32 Adam moments | 2 B/P | 2 B/P | 8 B/P | 구현에 따라 없음 | 약 12 B/P |
| native BF16 + 8-bit Adam | 2 B/P | 2 B/P | 약 2 B/P + metadata | 구현에 따라 없음 | 약 6 B/P 이상 |
| BF16 + SGD momentum | 2 B/P | 2 B/P | 4 B/P | 구현별 | 약 8 B/P 이상 |

Hugging Face의 현재 model memory anatomy는 고전적 mixed-precision AdamW를 **weights 6 + optimizer 8 + gradients 4 = 18 B/P**로 설명한다. 그러나 최신 native BF16·fused optimizer·optimizer-in-backward 구현은 master copy와 gradient 수명을 다르게 관리할 수 있으므로, 12 B/P는 “항상 보장되는 값”이 아니라 실측 전의 낮은 계획치다.

### 5.3 LoRA model state

Base 파라미터를 `P_base`, trainable adapter 파라미터를 `P_adapter`라 하면:

```text
LoRA model state
≈ P_base × base_weight_bytes
+ P_adapter × train_state_bytes
```

대표적인 BF16 LoRA는:

```text
≈ P_base × 2 B
+ P_adapter × 12~18 B
+ activation
```

base에는 gradient와 optimizer state가 없지만, forward/backward에 필요한 BF16 weights는 그대로 resident한다. 따라서 **LoRA는 BF16 base 자체가 들어갈 수 있을 때 가장 단순하고 빠르다.**

### 5.4 QLoRA model state

QLoRA는 frozen base를 4-bit NF4 등으로 저장하고 compute 시 BF16·FP16으로 dequantize한다.

```text
QLoRA model state
≈ P_base × 0.55~0.70 B
+ P_adapter × 12~18 B
+ quant metadata·dequant workspace
+ activation
```

`0.55–0.70 B/P`는 nominal 4-bit `0.5 B/P`에 quantization constants, block metadata, 일부 비양자화 tensor와 구현 차이를 더한 계획 범위다. 실제 checkpoint·runtime footprint는 architecture와 group size에 따라 달라진다.

### 5.5 LoRA 파라미터 수

선형층 `W ∈ R^(d_out × d_in)`에 rank `r` LoRA를 붙이면 trainable parameter는 대략 다음과 같다.

```text
P_lora = r × (d_in + d_out)
```

정사각형 `d × d` projection 하나의 원본 weight 대비 비율은:

```text
ratio = 2r / d
```

| hidden dimension `d` | r=8 | r=16 | r=32 | r=64 |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.78% | 1.56% | 3.12% | 6.25% |
| 4,096 | 0.39% | 0.78% | 1.56% | 3.12% |
| 5,120 | 0.31% | 0.62% | 1.25% | 2.50% |
| 8,192 | 0.20% | 0.39% | 0.78% | 1.56% |

이 비율은 projection 하나의 계산이다. 전체 모델 비율은 어떤 module에 adapter를 붙이는지에 따라 달라진다. `q_proj`·`v_proj`만 대상으로 할 때와 `all-linear`로 attention·MLP 전체를 대상으로 할 때는 trainable parameter와 optimizer memory가 크게 다르다.

### 5.6 adapter checkpoint 용량

adapter 저장 크기는 대략 다음과 같다.

```text
adapter_size
≈ trainable_parameters × storage_bytes
+ config·metadata
```

예를 들어 100M trainable parameters를 BF16으로 저장하면 raw weight는 약 191 MiB다. optimizer checkpoint까지 저장하면 수 배로 늘 수 있다.

### 5.7 계산기용 의사코드

```python
from dataclasses import dataclass

GIB = 1024 ** 3

@dataclass(frozen=True)
class Plan:
    base_params: int
    trainable_params: int
    base_bytes: float
    trainable_state_bytes: float
    activation_gib: float
    overhead_fraction: float = 0.15


def estimate_peak_gib(plan: Plan) -> float:
    state_bytes = (
        plan.base_params * plan.base_bytes
        + plan.trainable_params * plan.trainable_state_bytes
    )
    subtotal = state_bytes / GIB + plan.activation_gib
    return subtotal * (1.0 + plan.overhead_fraction)

# 예: 7B QLoRA, 80M adapter, activation 6 GiB
plan = Plan(
    base_params=7_000_000_000,
    trainable_params=80_000_000,
    base_bytes=0.62,
    trainable_state_bytes=14.0,
    activation_gib=6.0,
)
print(f"estimated peak: {estimate_peak_gib(plan):.1f} GiB")
```

이 계산은 시작점일 뿐이며, 실제 activation·temporary peak는 한 step을 실행해야 알 수 있다.

---

## 6. Activation·context·batch 메모리

### 6.1 activation이 결정하는 변수

메모리 효율적인 attention backend를 사용할 때 transformer activation은 대략 다음 변수에 비례한다.

```text
activation memory
∝ micro_batch
× sequence_length
× num_layers
× hidden_size
× saved_tensor_factor
```

naive attention이 전체 score matrix를 materialize하면 다음 항이 추가될 수 있다.

```text
attention score memory
∝ micro_batch × num_heads × sequence_length²
```

따라서 4K에서 8K로 context를 두 배 늘리면 activation이 단순히 두 배만 늘지 않을 수 있다. FlashAttention·SDPA 같은 backend가 중요한 이유다.

### 6.2 global batch와 micro-batch

```text
global_batch_examples
= micro_batch_per_device
× data_parallel_world_size
× gradient_accumulation_steps
```

packed training에서는 example 수보다 token 수가 더 정확하다.

```text
global_tokens_per_optimizer_step
= tokens_per_micro_batch
× data_parallel_world_size
× gradient_accumulation_steps
```

> **중요:** gradient accumulation은 한 micro-batch의 peak activation을 줄이지 않는다. `micro_batch=1`의 메모리가 이미 넘치면 accumulation 값을 늘려도 OOM은 해결되지 않는다.

### 6.3 길이 분포가 peak를 만든다

동적 padding에서 batch 하나의 길이는 가장 긴 sample에 맞춰진다. 평균 길이가 600 token이어도 일부 8K sample이 섞이면 해당 batch의 모든 sample이 8K에 가까운 메모리를 사용할 수 있다.

권장 절차:

```text
1. tokenizer로 실제 길이 분포 계산
2. p50·p90·p95·p99 기록
3. 과도하게 긴 sample 분리·chunking
4. length bucket sampler 적용
5. max_length를 p95 근처에서 시작
6. 긴 문서 task만 별도 run
```

### 6.4 logits와 vocabulary

언어 모델의 logits는 대략 `[batch, sequence, vocabulary]` 형태다.

```text
logits_bytes
≈ B × S × V × dtype_bytes
```

예를 들어 큰 vocabulary 모델에서 long context를 사용하면 LM head 출력이 주요 activation이 될 수 있다. chunked cross-entropy나 label이 있는 위치만 계산하는 loss가 peak를 크게 낮출 수 있다.

### 6.5 sequence length 변경 시 재측정

| 변경 | model state | activation | step time | 품질·데이터 영향 |
| --- | --- | --- | --- | --- |
| batch 2 → 1 | 동일 | 대체로 감소 | sample당 overhead 증가 | global batch는 accumulation으로 보완 |
| 4K → 2K | 동일 | 크게 감소 | 빨라짐 | 긴 dependency 손실 가능 |
| rank 16 → 8 | base 동일 | 거의 동일 | 약간 감소 | adapter 표현력 감소 가능 |
| QLoRA → LoRA | base 크게 증가 | 비슷하거나 일부 감소 | 대체로 빨라질 수 있음 | quantization 영향 제거 |
| checkpointing 켜기 | 동일 | 감소 | recompute로 느려짐 | 수학적으로 같은 objective |
| gradient accumulation 증가 | 동일 | micro-batch peak 동일 | optimizer step당 느려짐 | global batch 증가 |
| packing 켜기 | 동일 | 같은 token budget에서 효율 향상 | 보통 throughput 향상 | sample boundary 처리 확인 |

### 6.6 멀티모달 activation

```text
VLM activation
≈ text token activation
+ visual token activation
+ vision encoder intermediate
+ projector output

Audio activation
≈ text decoder activation
+ acoustic frame activation
+ feature extractor·codec state
```

이미지 해상도, crop 수, video frame 수, audio duration이 token 수와 동일한 역할을 한다. 텍스트 model의 `max_length`만 줄여서는 멀티모달 OOM이 해결되지 않을 수 있다.

### 6.7 첫 dry run

첫 실행은 전체 epoch가 아니라 다음처럼 짧게 한다.

```text
representative longest batch 1개
+ forward
+ backward
+ optimizer step
+ zero_grad
+ checkpoint save 1회
```

모든 단계를 통과해야 실제 학습 가능하다고 판단한다. forward만 성공한 상태는 충분하지 않다.


---

## 7. Full fine-tuning

Full fine-tuning은 모든 trainable weight를 업데이트한다. 가장 직접적인 방식이지만, model state가 전체 파라미터에 대해 생성되므로 메모리 요구량이 가장 크다.

### 7.1 언제 유리한가

- 0.1–3B급 작은 모델을 특정 task에 강하게 적응
- 새로운 tokenizer token과 embedding을 대량 추가
- domain shift가 크고 PEFT 한계가 평가로 확인됨
- representation 전체를 바꾸는 encoder·scientific model 연구
- distillation student를 처음부터 강하게 적응
- 최종 artifact에서 adapter 의존성을 제거해야 함

### 7.2 언제 피할 것인가

- 7B 이상 모델을 단일 24GB GPU에서 학습
- 데이터가 적거나 label 품질이 불확실
- 최신 지식 주입이 주목적
- 여러 task마다 별도 full checkpoint가 필요
- model license가 파생 weights 재배포를 제한
- evaluation 없이 “더 많이 업데이트하면 더 좋다”는 가정만 있음

### 7.3 메모리 예산

```text
보수적 classic mixed-precision AdamW
≈ 18 B/P + activation + temporary

native BF16 lower planning bound
≈ 12 B/P + activation + temporary

native BF16 + 8-bit optimizer lower planning bound
≈ 6 B/P 이상 + activation + temporary
```

7B 모델의 경우 activation을 제외해도 대략 다음과 같다.

| 구성 | model state 근사치 |
| --- | ---: |
| native BF16 lower bound, 12 B/P | 약 78.2 GiB |
| classic mixed precision, 18 B/P | 약 117.3 GiB |
| native BF16 + 8-bit optimizer, 6 B/P | 약 39.1 GiB 이상 |

8-bit optimizer를 사용해도 activation, temporary workspace, 일부 FP32 state와 framework overhead가 남는다. “7B full FT가 40GB에 들어간다”는 결론을 표만 보고 내리면 안 된다.

### 7.4 작은 모델 single-GPU full FT 절차

```text
1. BF16 지원 확인
2. micro-batch 1
3. max_length 512~2048에서 시작
4. memory-efficient attention
5. gradient checkpointing
6. 8-bit optimizer 비교
7. representative longest batch로 peak 측정
8. batch·length를 한 번에 하나씩 증가
```

### 7.5 부분 동결

전체 모델 대신 다음과 같이 일부만 훈련할 수 있다.

- 마지막 `N` transformer blocks
- LM head·classification head
- embedding·새 token rows
- normalization layer
- VLM projector
- speech adapter·speaker embedding
- diffusion text encoder 또는 DiT 일부 block

부분 동결은 LoRA와 다르다. 선택한 원본 weight 자체에 gradient와 optimizer state가 생성된다. 선택 layer가 크면 메모리 절감이 제한적이다.

### 7.6 catastrophic forgetting 완화

- base domain replay data 혼합
- 낮은 learning rate
- 짧은 run과 frequent evaluation
- domain별 held-out benchmark
- general capability regression test
- KL regularization 또는 reference model 비교
- tokenizer·embedding 변경을 최소화

### 7.7 checkpoint 크기

Full FT resume checkpoint는 보통 다음을 포함한다.

```text
model weights
+ optimizer states
+ scheduler state
+ gradient scaler
+ RNG states
+ data sampler position
+ distributed metadata
```

BF16 최종 weights가 14GB여도 Adam optimizer를 포함한 resume checkpoint는 훨씬 클 수 있다. 저장 중 임시 파일과 shard 병합까지 고려해 **최종 모델 파일의 2–4배 이상 disk 여유**를 잡는다.

---

## 8. LoRA

LoRA는 pretrained weight를 동결하고 linear projection에 저랭크 update를 추가한다. PEFT의 기본 선택이며, base weight가 BF16·FP16으로 accelerator에 들어갈 때 QLoRA보다 단순하고 빠른 경우가 많다.

### 8.1 기본 구조

```text
W' = W + ΔW
ΔW = B × A
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
```

원본 `W`는 frozen이고 `A`, `B`만 업데이트한다.

### 8.2 주요 설정

| 설정 | 역할 | 메모리 영향 | 시작점 |
| --- | --- | --- | --- |
| `r` | adapter rank | 선형 증가 | 8·16·32 비교 |
| `lora_alpha` | update scaling | 거의 없음 | `r` 또는 `2r`를 기준선으로 두고 평가 |
| `lora_dropout` | adapter regularization | 작음 | 0–0.05 |
| `target_modules` | adapter를 붙일 layer | 가장 큰 영향 | attention부터, 필요 시 `all-linear` |
| `bias` | bias 학습 여부 | 작음 | 기본 `none` |
| `modules_to_save` | adapter 외 full module 저장 | 해당 module state 증가 | LM head·classifier가 필요할 때만 |
| `use_rslora` | rank-stabilized scaling | 메모리는 유사 | 높은 rank에서 검토 |
| `use_dora` | magnitude 추가 학습 | 증가 | 품질 개선이 검증될 때 |

### 8.3 target module 선택

| 범위 | 대표 module | 장점 | 단점 |
| --- | --- | --- | --- |
| 최소 | `q_proj`, `v_proj` | adapter 작음 | task 표현력 제한 가능 |
| attention 전체 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 일반적인 균형 | 최소 구성보다 메모리 증가 |
| attention + MLP | 위 + gate/up/down projection | 강한 적응 | trainable params·optimizer 증가 |
| `all-linear` | 모든 linear layer | QLoRA식 넓은 coverage | architecture별 제외 module 확인 |
| head·embedding 포함 | LM head, embedding | 새 token·domain vocab 적응 | 큰 vocabulary에서 급격히 증가 |

architecture마다 module 이름이 다르다. 문자열을 추정하지 말고 다음을 확인한다.

```python
for name, module in model.named_modules():
    if module.__class__.__name__ == "Linear":
        print(name)
```

### 8.4 rank를 선택하는 방법

무조건 높은 rank를 쓰지 않는다.

```text
r=8:
  출력 스타일·간단한 task·작은 데이터

r=16~32:
  일반 domain SFT·코드·도구 사용

r=64 이상:
  강한 domain shift·복잡한 multimodal 적응
  → data 양과 overfitting·memory를 함께 검증
```

rank보다 target module coverage가 더 큰 영향을 주는 경우가 많다. `r=64`로 attention 일부만 학습하는 것과 `r=16`으로 all-linear를 학습하는 것을 직접 비교한다.

### 8.5 adapter 메모리 검사

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,}")
print(f"ratio: {100 * trainable / total:.4f}%")
```

실행 전 다음을 확인한다.

- frozen base에 `requires_grad=True`가 남지 않았는가
- `modules_to_save`가 큰 embedding·LM head 전체를 포함하는가
- tied weights가 중복 저장되는가
- adapter dtype이 FP32로 남아 있는가
- optimizer가 frozen parameter를 포함하는가

### 8.6 embedding과 새 token

새 token을 추가하면 embedding과 LM head row를 학습해야 할 수 있다. vocabulary 전체 matrix를 trainable로 만들면 LoRA보다 큰 optimizer state가 생길 수 있다.

가능하면:

- 선택 row만 학습하는 기능 사용
- tokenizer 변경 없이 기존 subword 조합 평가
- `modules_to_save`가 전체 head를 저장하는지 확인
- tied embedding 여부 확인

### 8.7 여러 adapter

여러 task adapter를 한 base에 보관할 수 있지만, training 중 동시에 활성화한 adapter마다 state가 증가한다.

```text
resident base
+ active trainable adapter
+ loaded evaluation adapters
+ optimizer for active adapter
```

한 번에 하나만 학습한다면 불필요한 adapter를 GPU에서 내리거나 process를 분리한다.

### 8.8 merge 주의

LoRA merge는 inference artifact를 단순화하지만:

- merged weight를 다시 저장할 disk·RAM이 필요
- quantized base에 바로 merge할 때 형식별 제약이 있음
- merge 후 unmerge가 항상 lossless하지 않음
- base revision과 adapter revision을 함께 기록해야 함
- 여러 adapter 합성은 순서와 scale에 민감

최종 배포 전에는 **unmerged base+adapter**와 **merged model**을 같은 평가셋에서 비교한다.

---

## 9. QLoRA

QLoRA는 frozen base model을 4-bit로 저장하고, 역전파는 base를 통과해 LoRA adapter에만 적용한다. 원 논문은 NF4, double quantization과 paged optimizer를 조합해 65B 모델을 단일 48GB GPU에서 fine-tuning한 결과를 제시했다. 이는 특정 model·sequence·implementation의 연구 결과이지 모든 65–70B 모델에 대한 보장 수치가 아니다.

### 9.1 핵심 구성

| 요소 | 역할 | 메모리 효과 |
| --- | --- | --- |
| NF4 | 정규분포 weight에 맞춘 4-bit storage | base weight 감소 |
| double quantization | quantization constant를 다시 양자화 | metadata 감소 |
| BF16/FP16 compute | matrix 연산 정밀도 | storage와 compute dtype 분리 |
| LoRA adapters | trainable update | gradient·optimizer를 소수 parameter로 제한 |
| paged optimizer | memory spike 완화 | host unified memory를 활용할 수 있음 |

### 9.2 기본 Transformers·PEFT 예제

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "OWNER/MODEL"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

실제 model에서 `all-linear` 지원과 output head 제외 여부를 확인한다. 일부 architecture는 remote code 또는 전용 PEFT mapping이 필요하다.

### 9.3 저장 dtype과 compute dtype

```text
base storage: 4-bit NF4
adapter storage: 보통 BF16·FP16·FP32
forward compute: BF16 또는 FP16
optimizer state: adapter에 대해서만 8-bit·FP32 등
```

QLoRA는 “모든 계산을 4-bit로 수행”하는 방식이 아니다. dequantization과 activation은 고정밀이므로, context가 길면 base weight 절감보다 activation이 더 커질 수 있다.

### 9.4 QLoRA가 LoRA보다 느릴 수 있는 이유

- 매 linear 연산에서 dequantization 필요
- kernel·architecture 지원 차이
- small model에서는 weight 절감보다 overhead가 큼
- CPU paging이 발생하면 host-device transfer 증가
- compile·fused kernel 호환성 제한

BF16 base가 충분히 들어간다면 LoRA와 QLoRA를 모두 측정한다.

### 9.5 QLoRA의 흔한 오류

| 오류 | 결과 | 수정 |
| --- | --- | --- |
| `bnb_4bit_compute_dtype=float32` | 메모리·속도 악화 | BF16 지원 시 BF16 비교 |
| NF4 대신 임의 INT4 | 학습 품질 차이 | 공식 지원 quant type 확인 |
| double quant를 무조건 끔 | base metadata 증가 | A/B 측정 후 결정 |
| `use_cache=True` | checkpointing 충돌·메모리 증가 | training에서 `False` |
| 모든 base parameter가 trainable | 즉시 OOM | trainable parameter 검사 |
| GGUF Q4를 QLoRA input으로 사용 | runtime 불일치 | Transformers·framework 지원 4-bit format 사용 |
| 너무 긴 `max_length` | 첫 backward OOM | 길이 분포 기반 축소 |
| adapter를 Q4로 저장한다고 가정 | 품질·호환 문제 | adapter는 일반적으로 고정밀 유지 |

### 9.6 QLoRA + FSDP

단일 GPU를 넘는 모델은 QLoRA base와 adapter를 FSDP2로 shard할 수 있다. Hugging Face PEFT 가이드는 70B QLoRA를 2×24GB 구성에서 다루는 예제를 제공하지만, 다음 조건이 중요하다.

- quantized parameter가 shard 가능한 dtype·wrapper로 표현되어야 함
- FSDP2·bitsandbytes·PEFT 버전 호환
- loading 시 각 rank가 전체 checkpoint를 중복 로드하지 않도록 설정
- `use_orig_params` 또는 FSDP2의 partial freezing 지원 확인
- all-gather peak와 activation은 별도로 남음
- CPU RAM이 checkpoint보다 충분해야 함

### 9.7 QLoRA 품질 검증

```text
BF16 LoRA baseline
vs NF4 QLoRA
vs 작은 모델 BF16 LoRA
```

최소한 다음을 비교한다.

- validation loss
- task accuracy·pass rate
- JSON·tool-call schema validity
- long-context retention
- code compile·test pass
- math verifier pass
- latency와 peak memory

---

## 10. DoRA·rsLoRA·초기화 기법

LoRA 생태계에는 rank scaling, magnitude decomposition과 data-aware initialization 기법이 추가되어 있다. 최신 PEFT에서 옵션이 존재하더라도 모든 model·task에서 기본값으로 사용할 이유는 없다.

### 10.1 주요 기법

| 기법 | 개념 | 메모리·속도 | 권장 사용 시점 |
| --- | --- | --- | --- |
| rsLoRA | 높은 rank에서 scaling을 안정화 | LoRA와 유사 | rank 32–128 비교 시 |
| DoRA | weight magnitude와 direction을 분리 | magnitude parameter·연산 증가 | LoRA 품질 한계가 명확할 때 |
| LoftQ | quantized base와 LoRA 초기화를 함께 최적화 | 초기화 비용·호환성 확인 | QLoRA quantization error 완화 실험 |
| PiSSA | SVD 계열 initialization | 초기 분석·메모리 필요 | 빠른 수렴·품질 연구 |
| EVA | activation 기반 rank allocation·initialization | calibration data·forward 필요 | layer별 rank를 데이터 기반으로 조절 |
| OLoRA | orthogonal initialization 계열 | 구현별 차이 | 연구·A/B 목적 |
| CorDA | context-oriented decomposition | preprocessing 필요 | 지식 보존·domain adaptation 연구 |
| AdaLoRA | layer별 rank를 동적으로 할당 | scheduler·추가 상태 | fixed rank가 비효율적일 때 |

### 10.2 DoRA 메모리

DoRA는 LoRA adapter 외에 magnitude component를 학습한다. 따라서:

```text
DoRA state
= LoRA A/B state
+ magnitude parameter
+ 추가 temporary computation
```

PEFT는 inference를 위해 merge할 수 있지만 training overhead는 LoRA보다 크다. GPU가 간신히 맞는 상황에서 DoRA를 먼저 선택하지 않는다.

### 10.3 높은 rank의 함정

- adapter optimizer state 선형 증가
- checkpoint 크기 증가
- 작은 dataset에서 overfitting
- all-linear + high rank 조합은 PEFT 장점을 줄임
- 더 긴 sequence를 사용할 메모리를 빼앗음

`rank ↑`보다 다음 순서를 권장한다.

```text
data error 수정
→ target module coverage 조정
→ learning rate·scheduler 조정
→ rank 8/16/32 비교
→ 필요할 때만 DoRA·data-aware init
```

### 10.4 초기화 비교 실험

```yaml
experiment:
  base_revision: <sha>
  dataset_revision: <sha>
  common:
    target_modules: all-linear
    rank: 16
    alpha: 32
    seed: 42
  variants:
    - init: default
    - init: pissa
    - init: eva
    - init: loftq
  metrics:
    - validation_loss
    - task_success
    - peak_vram_gib
    - tokens_per_second
    - adapter_size_mib
```

초기화 방식마다 preprocessing과 supported layer가 다르므로, framework documentation과 정확한 release를 고정한다.

---

## 11. Optimizer·gradient·precision

### 11.1 optimizer 선택

| optimizer | state 근사 | 장점 | 주의점 |
| --- | ---: | --- | --- |
| AdamW FP32 moments | 8 B/trainable P | 강한 기본선 | 가장 큰 state |
| 8-bit AdamW | 약 2 B/P + metadata | 큰 trainable model에서 절감 | 작은 tensor는 FP32 유지 가능, kernel 확인 |
| Adafactor | matrix factorization으로 감소 | encoder-decoder·큰 matrix에 유리 | tuning과 convergence 특성 다름 |
| SGD | 0 B/P | state 최소 | LLM 적응에서 수렴이 어려울 수 있음 |
| SGD + momentum | 4 B/P | 단순 | AdamW보다 tuning 민감 |
| optimizer-in-backward | gradient 수명 단축 | full FT peak 감소 | gradient accumulation과 충돌 가능 |

optimizer state는 **trainable parameter**에 대해서만 생성되어야 한다. PEFT에서 frozen parameter가 optimizer param group에 들어가면 메모리와 step overhead가 불필요하게 증가한다.

### 11.2 `zero_grad(set_to_none=True)`

```python
optimizer.zero_grad(set_to_none=True)
```

gradient buffer를 zero tensor로 유지하는 대신 `None`으로 해 다음 iteration에서 필요한 시점에 할당할 수 있다. 구현과 graph capture에 따라 효과가 다르지만 일반적인 안전한 시작점이다.

### 11.3 BF16·FP16·FP32

| dtype | weight bytes | 수치 특성 | 일반적 용도 |
| --- | ---: | --- | --- |
| FP32 | 4 | 넓은 mantissa·range | optimizer state·검증·CPU |
| FP16 | 2 | exponent range가 좁음 | 구형 GPU mixed precision, loss scaling 필요 가능 |
| BF16 | 2 | FP32와 유사한 exponent range | 최신 accelerator 학습 기본선 |
| TF32 | 저장 dtype 아님 | FP32 matmul 경로의 연산 모드 | NVIDIA matmul 가속 |
| FP8 | 1 nominal | scale 관리 필요 | 지원 GPU·kernel의 대규모 학습 |

BF16 지원 장비라면 일반적으로 FP16보다 loss scaling 부담이 적다. 단, 특정 layer·optimizer·normalization은 FP32 accumulation을 유지할 수 있다.

### 11.4 FP8 학습

FP8은 가중치 저장만 8-bit로 만드는 단순 PTQ가 아니다. training framework가 activation·gradient scaling, delayed/dynamic scale과 supported operator를 관리해야 한다.

검토 항목:

- GPU 세대와 FP8 tensor core 지원
- Transformer Engine·MS-AMP·torchao 등 backend
- FSDP2·TP와의 조합
- master weights·optimizer dtype
- unsupported op fallback
- convergence와 gradient overflow
- checkpoint portability

FP8을 켰다는 이유만으로 model state가 정확히 절반이 되지 않는다. optimizer와 master weight가 BF16·FP32로 남을 수 있다.

### 11.5 gradient clipping

```text
NaN·overflow 완화에는 도움
메모리 절감 기법은 아님
```

FSDP·TP 환경에서는 local shard가 아니라 global norm을 올바르게 계산하는 framework API를 사용한다.

### 11.6 loss scaling

FP16 training은 작은 gradient underflow를 막기 위해 dynamic loss scaling을 사용할 수 있다. BF16은 보통 필요성이 낮지만, framework가 자동으로 관리하는 설정을 임의로 중복 적용하지 않는다.

### 11.7 optimizer checkpoint

resume가 필요하지 않은 final adapter만 배포할 경우 optimizer state는 배포 artifact에 포함하지 않는다. 그러나 학습 중에는 다음을 분리해 보존한다.

```text
adapter-only inference checkpoint
training resume checkpoint
merged deployment checkpoint
```

---

## 12. 메모리 최적화 기법

### 12.1 적용 우선순위

OOM이 발생하면 다음 순서를 권장한다.

```text
1. micro-batch 1 확인
2. max_length·image size·audio duration 축소
3. 불필요한 trainable module 제거
4. gradient checkpointing
5. memory-efficient attention
6. padding/packing 최적화
7. QLoRA 또는 8-bit optimizer
8. activation offload
9. FSDP2·ZeRO-3
10. CPU/NVMe offload·TP·PP
```

한 번에 여러 옵션을 바꾸면 어떤 변경이 품질·속도·메모리에 영향을 줬는지 알기 어렵다.

### 12.2 gradient checkpointing

forward activation 일부만 저장하고 backward 때 다시 계산한다.

| 효과 | 결과 |
| --- | --- |
| VRAM | 감소 |
| compute | 증가 |
| step time | 느려짐 |
| model state | 변화 없음 |
| objective | 원칙적으로 동일 |

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
```

일부 architecture는 non-reentrant checkpointing, input gradient 또는 frozen layer 조합에 추가 설정이 필요하다.

### 12.3 selective activation checkpointing

모든 layer를 같은 방식으로 checkpoint하지 않고 큰 MLP·attention block만 선택한다. 최신 PyTorch 계열은 selective checkpoint 정책을 제공하므로, recompute 비용과 memory를 절충할 수 있다.

### 12.4 activation offloading

saved activation을 CPU RAM으로 옮겼다가 backward 전에 다시 가져온다.

```text
장점: GPU VRAM 감소
비용: PCIe·interconnect transfer, pinned RAM, latency
```

TRL의 현재 memory guide는 SFT 구성에서 `activation_offloading=True` 경로를 제공한다. CPU RAM과 bus bandwidth가 충분한지 확인한다.

### 12.5 optimizer·parameter offload

| 방식 | GPU에서 내리는 것 | 적합한 경우 | 병목 |
| --- | --- | --- | --- |
| ZeRO-Offload | optimizer state·compute | full FT state가 큰 경우 | CPU 연산·PCIe |
| FSDP CPU offload | parameter·gradient | VRAM이 절대 부족 | 매 layer transfer |
| ZeRO-Infinity | parameter·optimizer를 CPU/NVMe | 매우 큰 model | NVMe IOPS·latency |
| QLoRA paged optimizer | spike 시 unified memory | 단일 GPU adapter 학습 | paging 발생 시 속도 급락 |

### 12.6 optimizer step in backward

gradient가 생성되는 즉시 optimizer update를 수행해 전체 gradient를 동시에 유지하지 않는 방식이다. PyTorch 공식 tutorial은 gradient memory 절감 경로를 제공하지만, 일반적인 gradient accumulation과 양립하지 않을 수 있다.

사용 전 확인:

- accumulation 필요 여부
- distributed optimizer 지원
- checkpoint state 저장
- multiple parameter group
- gradient clipping 시점

### 12.7 memory-efficient attention

- PyTorch SDPA
- FlashAttention 2·3
- vendor fused attention
- sequence/context parallel attention

attention backend는 model architecture, mask, head dimension, dropout과 dtype을 지원해야 한다. fallback이 발생하면 메모리가 갑자기 증가할 수 있다.

### 12.8 fused·custom kernel

Liger, fused RMSNorm, fused MLP, fused optimizer 같은 kernel은 temporary tensor와 launch overhead를 줄일 수 있다. 그러나:

- 특정 GPU·dtype·architecture만 지원
- compile 시간이 늘어남
- custom extension 공급망 위험
- 수치 결과가 미세하게 달라질 수 있음
- PEFT·VLM·chunked loss와 조합 제한

공식 benchmark 수치를 장비 계획에 그대로 사용하지 말고 target setup에서 측정한다.

### 12.9 `torch.compile`

compile은 throughput을 높일 수 있지만 첫 iteration compile memory와 graph cache가 추가된다. dynamic sequence·custom autograd·quantized module에서 graph break가 많으면 이득이 줄어든다.

```text
eager peak 측정
→ compile 적용
→ warmup 이후 peak·throughput 재측정
```

### 12.10 dataloader 메모리

```python
DataLoader(
    dataset,
    batch_size=...,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=False,
)
```

큰 이미지·오디오·Parquet row에서 worker와 prefetch를 늘리면 시스템 RAM이 급증한다. OOM이 GPU가 아닌 host memory인지 구분한다.

### 12.11 fragmentation

reserved memory가 allocated보다 크게 남아 OOM이 발생할 수 있다.

- batch shape를 안정화
- 지나친 dynamic shape 축소
- 긴 sample을 별도 bucket
- 불필요한 tensor reference 제거
- allocator snapshot 확인
- process 재시작으로 실험 간 잔여 cache 제거

환경 변수는 framework·PyTorch 버전에 따라 달라지므로 최신 공식 CUDA semantics 문서를 확인한다.

---

## 13. Packing·padding-free·loss 최적화

### 13.1 truncation

`max_length`가 너무 작으면 중요한 token을 버리고, 너무 크면 padding과 activation이 늘어난다. 데이터 길이 분포를 보지 않고 model maximum context를 그대로 사용하지 않는다.

```python
lengths = [len(tokenizer(x)["input_ids"]) for x in texts]
# p50, p90, p95, p99를 계산해 max_length 후보를 결정
```

### 13.2 packing

여러 짧은 sample을 하나의 fixed-length row에 채워 padding을 줄인다.

| 전략 | 동작 | 주의점 |
| --- | --- | --- |
| BFD | 길이 기반으로 빈 공간을 효율적으로 채움 | overlength sample 처리 확인 |
| BFD split | 긴 sample을 max length 이하로 분할 후 packing | continuity 손실 가능 |
| wrapped/concat-split | 모든 token을 stream으로 이어 자름 | unrelated sample boundary 혼합 |

TRL의 현재 SFT 문서는 BFD 기반 packing을 제공한다.

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="out",
    max_length=2048,
    packing=True,
    packing_strategy="bfd",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

### 13.3 sequence boundary와 attention mask

packing은 단순 concatenate가 아니다. sample 간 attention 차단 또는 position id 처리가 objective에 맞아야 한다. framework가 지원하는 packing path를 사용하고, custom collator는 다음을 검증한다.

- sample A가 sample B token을 보지 않는가
- EOS가 정확히 들어가는가
- labels의 ignore index가 맞는가
- position id가 model 기대와 일치하는가
- chat template가 중복 적용되지 않는가

### 13.4 padding-free

padding-free batching은 batch의 실제 token만 flatten해 계산한다. 현재 TRL은 FlashAttention 2·3 계열과 함께 사용하도록 권장한다. 지원되지 않는 attention에서 batch contamination이 발생하지 않는지 확인한다.

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="out",
    padding_free=True,
    model_init_kwargs={
        "attn_implementation": "kernels-community/flash-attn2"
    },
)
```

API와 kernel 이름은 release마다 바뀔 수 있으므로 설치된 문서를 확인한다.

### 13.5 completion-only loss

instruction과 user prompt까지 모두 loss에 포함하면 모델이 입력을 복제하도록 학습될 수 있고, 불필요한 logit·gradient가 생긴다. 응답 token만 학습하는 task라면 label mask를 적용한다.

```text
system: ignore
user:   ignore
assistant answer: train
```

continued pretraining이나 language modeling에서는 전체 token loss가 목적이므로 같은 mask를 사용하지 않는다.

### 13.6 chunked cross-entropy

큰 vocabulary에서 전체 `[B, S, V]` logits를 한 번에 materialize하지 않고 token chunk별로 LM head와 cross-entropy를 계산할 수 있다.

```text
peak logits memory
standard: B × S × V
chunked:  chunk_size × V
```

TRL 문서의 release별 compatibility 표는 달라질 수 있으며, 현재 SFT 문서는 `chunked_nll`과 Liger의 비호환을 명시한다. PEFT·VLM에서는 설치 버전과 trainer 구현을 확인하고, 지원되지 않으면 fused linear-cross-entropy 또는 framework별 tiled loss를 검토한다.

### 13.7 length bucketing

같은 batch에 비슷한 길이 sample을 모으면 padding과 peak 변동을 줄인다.

```text
0–512
513–1024
1025–2048
2049–4096
4097–8192
```

각 bucket의 batch size를 다르게 설정하는 token-budget sampler가 효율적이다.

```text
짧은 sequence: batch 8
중간 sequence: batch 4
긴 sequence: batch 1
```

### 13.8 token 기준 batch

example 수가 아니라 총 token 수를 고정한다.

```text
max_tokens_per_micro_batch = 8192
```

이미지·오디오에서는 text token 외에 visual token·audio frame을 포함한 비용 함수를 사용해야 한다.

### 13.9 데이터 중복

중복 sample은 메모리보다 compute를 낭비하고 overfitting을 높인다.

- exact hash dedup
- MinHash·SimHash near-dedup
- template boilerplate 제거
- 동일 conversation의 변형 누수 검사
- train·validation contamination 검사


---

## 14. RAM·VRAM별 30초 모델 선택

이 장의 범위는 text decoder 모델을 기준으로 한 출발점이다. VLM·이미지·오디오는 24–27장의 modality별 추가 메모리를 반영한다.

### 14.1 단일 accelerator

| 실사용 가능 메모리 | BF16 LoRA 시작점 | QLoRA 시작점 | Full FT 후보 | 권장 context 시작점 | 비고 |
| ---: | --- | --- | --- | ---: | --- |
| 6 GB | 0.5–1B | 1–1.5B | 100–300M | 512 | laptop GPU·display 여유 확인 |
| 8 GB | 1–1.5B | 1.5–3B | 300–500M | 512–1K | 3B는 architecture별 편차 큼 |
| 10–12 GB | 1.5–3B | 3B, 7B 실험 | 0.5B | 1K | 7B QLoRA는 checkpointing 필수에 가까움 |
| 16 GB | 3B | 7–8B | 0.5–1B | 1–2K | official 3B LoRA·7B QLoRA 사례 존재 |
| 20 GB | 7–8B | 7–14B | 1B | 2K | 긴 context는 batch 1 |
| 24 GB | 7–8B, 일부 14B | 14B, 27B 실험 | 1–1.5B | 2–4K | 소비자 GPU의 일반적 sweet spot |
| 32 GB | 14B | 27–32B | 1–2B | 2–4K | Apple은 OS 여유를 더 크게 확보 |
| 40 GB | 14–27B | 32B | 2–3B | 4K | data loader·checkpoint RAM 확인 |
| 48 GB | 27B | 32B, 65–70B 엄격 설정 | 3B 후보 | 2–4K | 70B는 논문 수준 최적화와 실무 안정성 구분 |
| 64 GB | 32B | 70B | 3B | 4K | activation 여유를 확보한 70B QLoRA |
| 80 GB | 32B, 일부 70B | 70B | 3–7B 구현별 | 4–8K | online RL은 rollout engine을 별도 계산 |
| 96 GB | 70B LoRA 실험 | 70–120B | 7B 낮은 state 구성 | 4–8K | base BF16 70B만 약 130GiB이므로 일반 LoRA는 부족 |
| 128 GB | 70B BF16 LoRA에 근접 | 120B | 7B full FT 후보 | 4–8K | unified memory는 OS 여유 필수 |
| 192 GB | 70B LoRA | 235B 실험 | 7–14B | 8K | 분산·server-class 권장 |

표에서 BF16 LoRA 70B는 base weights만 약 130GiB이므로 96GB에 실제로 들어가지 않는다. “실험” 표시는 CPU offload·layer streaming·sharding 같은 추가 기법을 전제로 한다.

### 14.2 시스템 RAM

| 시스템 RAM | 권장 역할 | 가능한 offload 범위 | 주의점 |
| ---: | --- | --- | --- |
| 16 GB | 소형 dataset·1–3B single GPU | 최소 | browser·IDE와 경쟁 |
| 32 GB | 7–8B LoRA·QLoRA host | adapter optimizer·dataset | 7B BF16 checkpoint를 동시에 여러 copy 보관하지 않음 |
| 64 GB | 14–32B QLoRA host | activation 일부·optimizer | pinned memory·checkpoint staging 계산 |
| 128 GB | 70B QLoRA host | optimizer·parameter 일부 | raw BF16 70B 전체 변환은 여유가 적음 |
| 256 GB | 70–120B offload·FSDP host | 큰 optimizer·checkpoint merge | NUMA와 memory bandwidth 중요 |
| 512 GB | 235B급 연구·ZeRO-Offload | model state 상당 부분 | CPU compute가 병목일 수 있음 |
| 1 TB 이상 | 대형 full FT shard·NVMe staging | server-scale | disk throughput·network fabric까지 설계 |

CPU RAM은 VRAM의 저속 확장이 아니다. offload가 많아질수록 step time이 PCIe·CPU·NVMe에 제한된다.

### 14.3 Apple 통합 메모리

| 통합 메모리 | 현실적인 시작점 | 공격적인 실험 | 운영 메모 |
| ---: | --- | --- | --- |
| 16 GB | 1–3B QLoRA | 7B 짧은 sequence | 다른 앱 종료, swap 감시 |
| 24 GB | 3–7B QLoRA | 7B LoRA 일부 layer | batch 1 |
| 32 GB | 7B QLoRA·부분 LoRA | 7B LoRA, 14B QLoRA | MLX 공식 7B 부분 layer 예제 존재 |
| 48 GB | 7–14B LoRA·QLoRA | 27B QLoRA | long context보다 throughput 우선 |
| 64 GB | 14B LoRA, 27–32B QLoRA | 32B LoRA 일부 | OS·display 여유 12GB 이상 권장 |
| 96 GB | 32B LoRA·QLoRA | 70B QLoRA | swap 없이 peak 확인 |
| 128 GB | 70B QLoRA | 70B LoRA offload·부분 layer | 지속 부하의 thermal·memory pressure 측정 |
| 192 GB 이상 | 70B LoRA·대형 QLoRA | 120B급 | Mac-compatible architecture·kernel 확인 |

Apple Silicon에서는 단일 memory pool 덕분에 명시적 CPU offload 경계가 덜 보일 수 있지만, memory bandwidth와 GPU compute가 충분하다는 뜻은 아니다.

### 14.4 선택 공식

```text
usable_memory
= installed_memory
- OS·display·other_process
- runtime_context
- safety_margin

fit 조건
usable_memory
> model_state
+ representative_activation_peak
+ optimizer_step_peak
+ checkpoint_or_gather_peak
```

### 14.5 같은 메모리에서 무엇을 우선할 것인가

| 선택 | 일반적 품질·운영 경향 |
| --- | --- |
| 큰 모델 QLoRA + 1K context | 지식·추론 능력은 높을 수 있으나 긴 task 학습 제한 |
| 작은 모델 LoRA + 4K context | task context를 더 충실히 학습, 빠른 iteration |
| 더 작은 모델 full FT | 깊은 적응 가능, general capability 손실 위험 |
| 높은 rank + 짧은 context | adapter capacity 증가, 실제 task 정보 손실 가능 |
| 낮은 rank + 좋은 데이터 | 종종 더 안정적이고 재현 가능 |

항상 “한 단계 작은 모델의 BF16 LoRA”를 “큰 모델의 극단적 QLoRA”와 비교한다.

---

## 15. 파라미터 규모별 모델 상태 용량표

아래 표는 **activation·temporary·분산 buffer를 제외한 model state 근사치**다. GiB 기준이며, 반올림 때문에 합이 다를 수 있다.

- QLoRA base: `0.62 B/P` 계획값
- BF16 + 8-bit optimizer lower plan: `6 B/P`
- native BF16 Adam lower plan: `12 B/P`
- classic mixed-precision AdamW: `18 B/P`

| 파라미터 | BF16 weights only | QLoRA base estimate | BF16 + 8-bit opt lower | native BF16 Adam lower | classic mixed AdamW |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | 1.9 GiB | 0.6 GiB | 5.6 GiB | 11.2 GiB | 16.8 GiB |
| 3B | 5.6 GiB | 1.7 GiB | 16.8 GiB | 33.5 GiB | 50.3 GiB |
| 7B | 13.0 GiB | 4.0 GiB | 39.1 GiB | 78.2 GiB | 117.3 GiB |
| 8B | 14.9 GiB | 4.6 GiB | 44.7 GiB | 89.4 GiB | 134.1 GiB |
| 14B | 26.1 GiB | 8.1 GiB | 78.2 GiB | 156.5 GiB | 234.7 GiB |
| 27B | 50.3 GiB | 15.6 GiB | 150.9 GiB | 301.7 GiB | 452.6 GiB |
| 32B | 59.6 GiB | 18.5 GiB | 178.8 GiB | 357.6 GiB | 536.4 GiB |
| 70B | 130.4 GiB | 40.4 GiB | 391.2 GiB | 782.3 GiB | 1,173.5 GiB |
| 120B | 223.5 GiB | 69.3 GiB | 670.6 GiB | 1,341.1 GiB | 2,011.7 GiB |
| 235B | 437.7 GiB | 135.7 GiB | 1,313.2 GiB | 2,626.3 GiB | 3,939.5 GiB |
| 405B | 754.4 GiB | 233.9 GiB | 2,263.1 GiB | 4,526.2 GiB | 6,789.3 GiB |

### 15.1 해석 예시

#### 7B

```text
BF16 base weights:              약 13.0 GiB
QLoRA frozen base 계획값:       약 4.0 GiB
full FT native BF16 lower:      약 78.2 GiB
full FT classic mixed AdamW:    약 117.3 GiB
```

QLoRA 7B가 4GiB로 끝나는 것은 아니다. adapter train state, activation, dequant workspace와 CUDA context를 더하면 10–16GB급 장비가 일반적인 출발점이 된다.

#### 70B

```text
BF16 base weights:              약 130.4 GiB
QLoRA frozen base 계획값:       약 40.4 GiB
full FT native BF16 lower:      약 782.3 GiB
full FT classic mixed AdamW:    약 1.15 TiB
```

원 QLoRA 논문의 단일 48GB 65B 결과가 가능한 이유는 frozen base 4-bit와 adapter 학습을 조합했기 때문이다. full fine-tuning 70B의 메모리와 혼동하면 안 된다.

### 15.2 adapter 비율별 model-state 배수

다음 표는 trainable adapter state를 `14 B/trainable parameter`로 가정한 예시다. 실제 optimizer·dtype에 따라 달라진다.

| adapter / base 비율 | BF16 LoRA 총 B/P | QLoRA 총 B/P·base 0.62 가정 | 의미 |
| ---: | ---: | ---: | --- |
| 0.25% | 2.035 | 0.655 | 최소 attention target |
| 0.50% | 2.070 | 0.690 | 일반적인 낮은 rank |
| 1.00% | 2.140 | 0.760 | 넓은 target·중간 rank |
| 2.00% | 2.280 | 0.900 | all-linear·높은 rank 후보 |
| 5.00% | 2.700 | 1.320 | PEFT 이점이 줄어드는 구간 |

base 대비 adapter 비율이 작아도 activation은 거의 줄지 않는다.

### 15.3 activation 예산을 역산하기

```text
activation_budget
= usable_accelerator_memory
- model_state
- runtime·temporary reserve
```

예:

```text
24 GiB GPU
- 7B BF16 LoRA model state 약 13~15 GiB
- runtime·temporary 3 GiB
= activation에 약 6~8 GiB
```

실제 adapter state와 base file 형식에 따라 달라지므로 profiler 수치로 교체한다.

### 15.4 `accelerate estimate-memory`

Hugging Face Accelerate는 meta device에서 model loading과 Adam training state를 추정한다.

```bash
accelerate estimate-memory OWNER/MODEL

accelerate estimate-memory OWNER/MODEL \
  --library_name transformers \
  --dtypes float32 float16 int8 int4
```

이 도구는 model config 기반의 중요한 1차 필터지만, 공식 문서가 설명하듯 순수 loading 중심 추정이다. 실제 activation, custom architecture, multimodal processor, quantization kernel과 temporary peak는 별도로 측정한다.

### 15.5 자체 계산 manifest

```yaml
memory_plan:
  model:
    repo: OWNER/MODEL
    revision: <commit-sha>
    parameters_total: 7000000000
    parameters_trainable: 80000000
  method:
    name: qlora
    base_bytes_per_parameter: 0.62
    trainable_state_bytes_per_parameter: 14
  workload:
    micro_batch: 1
    max_length: 2048
    gradient_accumulation: 16
    attention_backend: flash_attention_2
  measured:
    idle_vram_gib: null
    forward_peak_gib: null
    backward_peak_gib: null
    optimizer_peak_gib: null
    checkpoint_peak_ram_gib: null
```

---

## 16. 단일 GPU 권장 구성

### 16.1 8GB

```text
모델: 0.5–1.5B LoRA 또는 1–3B QLoRA
micro-batch: 1
max_length: 512–1024
rank: 8–16
optimizer: 8-bit AdamW 또는 일반 AdamW adapter-only
checkpointing: 켬
packing: 짧은 데이터에서 켬
```

3B QLoRA가 들어가더라도 vocabulary가 크거나 context 2K 이상이면 OOM이 날 수 있다. 먼저 512 token에서 optimizer step까지 통과시킨다.

### 16.2 12GB

```text
모델: 3B LoRA·QLoRA, 7–8B QLoRA 실험
max_length: 1K
rank: 8–16
attention: SDPA·FlashAttention
activation: checkpointing
```

7B QLoRA가 목표라면 dataset p95 길이를 1K 이하로 정리하고, `modules_to_save`가 큰 LM head를 포함하지 않는지 확인한다.

### 16.3 16GB

```text
모델: 3B BF16 LoRA, 7–8B QLoRA
max_length: 1–2K
rank: 16
micro-batch: 1–2
accumulation: 8–32
```

Torchtune 공식 문서는 3B LoRA를 16GB 미만에서 실행하는 workflow와 7B QLoRA를 10GB 미만에서 실행하는 tutorial을 제공한다. 해당 결과는 좋은 sanity check지만, 다른 model·dataset에 그대로 보장되지 않는다.

### 16.4 24GB

```text
안정적: 7–8B LoRA·QLoRA, 14B QLoRA
공격적: 14B LoRA·27B QLoRA short context
max_length: 2–4K
rank: 16–32
```

24GB는 다음 실험을 병렬로 비교하기 좋은 구간이다.

```text
7–8B BF16 LoRA r16 4K
vs 14B QLoRA r16 2K
vs 3B LoRA r32 8K
```

### 16.5 32–48GB

```text
32GB:
  14B LoRA·QLoRA
  27–32B QLoRA

48GB:
  27–32B LoRA 또는 QLoRA
  65–70B QLoRA의 엄격한 single-GPU 실험
```

70B는 base와 adapter가 들어가도 긴 context activation과 checkpoint save가 실패할 수 있다. CPU RAM, disk와 loading 방식까지 함께 설계한다.

### 16.6 64–80GB

```text
70B QLoRA가 현실적인 주력 후보
32B BF16 LoRA에 긴 context
3B급 full fine-tuning
온라인 preference/RL은 generation worker 분리 검토
```

80GB에서 70B BF16 LoRA는 base weights만 약 130GiB이므로 single-GPU resident 구성은 불가능하다. QLoRA 또는 sharding·offload가 필요하다.

### 16.7 공통 최소 설정

```yaml
training:
  precision: bf16
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  max_length: 2048
  gradient_checkpointing: true
  use_cache: false
  attention_backend: sdpa_or_flash
  optimizer: adamw_8bit
  max_grad_norm: 1.0

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

logging:
  log_peak_memory: true
  eval_steps: 100
  save_steps: 500
  save_total_limit: 2
```

`target_modules`, attention backend와 config key는 framework·architecture에 맞게 수정한다.

### 16.8 실험 순서

```text
run A: r=8, short context, data 5%
run B: r=16, same settings
run C: target module 확대
run D: full dataset
run E: context 확대
run F: preference stage
```

한 번에 model size, rank, data와 context를 모두 바꾸지 않는다.

---

## 17. Apple Silicon과 MLX-LM

MLX-LM은 Apple Silicon에서 LoRA, QLoRA, DoRA와 full fine-tuning 경로를 제공한다. quantized model을 입력하면 QLoRA로 동작하고, 고정밀 model이면 일반 LoRA로 동작하는 현재 공식 workflow를 사용할 수 있다.

### 17.1 설치와 기본 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install "mlx-lm[train]"

mlx_lm.lora --help
```

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --train \
  --data ./data \
  --iters 600 \
  --batch-size 1 \
  --grad-accumulation-steps 16 \
  --grad-checkpoint
```

현재 MLX-LM 문서는 `lora` 기본값, `dora`, `full`을 지원하는 `--fine-tune-type` 옵션을 설명한다.

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --fine-tune-type dora \
  --train \
  --data ./data
```

### 17.2 QLoRA

MLX quantized model을 지정하면 QLoRA로 학습한다.

```bash
mlx_lm.convert \
  --hf-path OWNER/MODEL \
  --mlx-path ./mlx-model-4bit \
  -q

mlx_lm.lora \
  --model ./mlx-model-4bit \
  --train \
  --data ./data \
  --batch-size 1 \
  --grad-checkpoint
```

정확한 conversion option은 설치된 MLX-LM release의 `--help`를 우선한다.

### 17.3 메모리 절감 순서

MLX-LM 공식 LoRA 문서가 권장하는 방향과 일치하는 순서는 다음과 같다.

```text
quantized model로 QLoRA
→ batch size 1
→ gradient accumulation
→ fine-tune layer 수 축소
→ sequence 축소
→ --grad-checkpoint
```

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --train \
  --batch-size 1 \
  --num-layers 4 \
  --grad-checkpoint \
  --data ./data
```

### 17.4 통합 메모리 측정

- Activity Monitor의 memory pressure와 swap 확인
- `powermetrics`·system profiler를 관리자 정책에 맞게 사용
- Python process RSS만 보지 말고 전체 wired/compressed memory 확인
- first step, validation, fuse에서 각각 peak 기록
- 장시간 run의 thermal throttling과 tokens/s 변화 기록

### 17.5 prompt masking

chat·completion dataset에서 assistant completion만 학습하려면 현재 MLX-LM의 `--mask-prompt`를 사용할 수 있다.

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --train \
  --data ./data \
  --mask-prompt
```

### 17.6 adapter 평가와 fuse

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --adapter-path adapters \
  --data ./data \
  --test

mlx_lm.generate \
  --model OWNER/MODEL \
  --adapter-path adapters \
  --prompt "테스트 프롬프트"

mlx_lm.fuse \
  --model OWNER/MODEL
```

fuse에는 base와 adapter를 동시에 적재하고 새 weight를 저장할 메모리·disk가 필요하다.

### 17.7 Apple에서 피할 것

- installed unified memory 전체를 model budget으로 계산
- swap이 발생해도 “OOM이 아니므로 성공”으로 판단
- 긴 sequence를 한 번에 늘림
- quantized base를 fuse한 뒤 원본·adapter provenance 삭제
- fanless 또는 제한된 cooling 장비에서 짧은 benchmark만 보고 장시간 throughput 추정

---

## 18. CPU RAM·NVMe offload

Offload는 더 큰 모델을 실행하게 하지만, 학습량을 줄이는 기술은 아니다. GPU가 기다리는 동안 CPU·PCIe·NVMe가 일을 대신한다.

### 18.1 offload 유형

| 유형 | 이동 대상 | 장점 | 주요 비용 |
| --- | --- | --- | --- |
| optimizer offload | Adam moments·update | full FT VRAM 크게 감소 | CPU compute·PCIe |
| parameter offload | frozen/trainable weights | model state 감소 | layer별 transfer |
| gradient offload | gradients | backward peak 감소 | transfer·synchronization |
| activation offload | saved activations | long context 가능 | forward/backward transfer |
| NVMe offload | parameter·optimizer shard | RAM보다 큰 state | IOPS·latency·SSD 수명 |
| checkpoint offload | save serialization | GPU peak 방지 | host RAM·disk staging |

### 18.2 RAM 예산

```text
host RAM
> offloaded state
+ pinned transfer buffers
+ dataset·dataloader
+ checkpoint staging
+ OS reserve
```

NUMA server에서는 GPU와 가까운 CPU socket의 memory binding이 중요하다. 잘못된 NUMA placement는 QPI/UPI를 통해 traffic을 보내 throughput을 낮춘다.

### 18.3 NVMe 요구

- 고성능 local NVMe 우선
- network filesystem에 optimizer offload하지 않음
- 충분한 queue depth와 async I/O
- checkpoint와 offload scratch를 다른 disk에 분리 고려
- SSD endurance·온도·filesystem free space 감시
- encrypted disk가 필요한 data인지 검토

### 18.4 ZeRO-Offload 예시

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": 16,
  "train_micro_batch_size_per_gpu": 1
}
```

ZeRO-3 parameter offload는 다음과 같은 형태다.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

세부 key와 권장 bucket은 DeepSpeed release·model에 맞게 조정한다.

### 18.5 offload를 중단할 신호

- GPU utilization이 지속적으로 낮음
- PCIe RX/TX가 포화
- CPU optimizer step이 전체 시간 대부분
- swap 발생
- NVMe read/write latency spike
- checkpoint와 offload traffic이 충돌
- GPU를 추가하는 편이 비용·시간 면에서 유리

---

## 19. 다중 GPU: DDP·FSDP2·ZeRO

### 19.1 DDP

DistributedDataParallel은 GPU마다 전체 model·gradient·optimizer state를 복제하고 gradient를 동기화한다.

```text
per-GPU memory ≈ single-GPU memory
aggregate throughput ↑
model fit capacity는 거의 증가하지 않음
```

따라서 model이 한 GPU에 들어가고 batch throughput을 늘릴 때 사용한다.

### 19.2 FSDP2

FSDP2는 parameter, gradient와 optimizer state를 shard한다. layer compute 전에 parameter를 all-gather하고 backward에서 gradient를 reduce-scatter한다.

```text
steady sharded state per rank
≈ total model state / data-parallel world size

peak per rank
≈ sharded state
+ current FSDP unit all-gather
+ activation
+ prefetch·communication buffer
```

단순히 전체 state를 GPU 수로 나눈 값보다 peak가 크다.

### 19.3 ZeRO 단계

| 방식 | optimizer state | gradient | parameter | DDP/FSDP 대응 |
| --- | --- | --- | --- | --- |
| ZeRO-0 | 복제 | 복제 | 복제 | DDP |
| ZeRO-1 | shard | 복제 | 복제 | optimizer state sharding |
| ZeRO-2 | shard | shard | 복제 | FSDP `SHARD_GRAD_OP` 유사 |
| ZeRO-3 | shard | shard | shard | FSDP `FULL_SHARD` 유사 |

DeepSpeed ZeRO-3는 CPU·NVMe offload와 결합할 수 있다.

### 19.4 무엇을 선택할 것인가

| 조건 | 우선 선택 |
| --- | --- |
| model이 GPU 하나에 들어가고 throughput만 필요 | DDP |
| PyTorch-native composition·FSDP2 checkpoint 필요 | FSDP2 |
| CPU/NVMe offload와 DeepSpeed 생태계 필요 | ZeRO-2/3 |
| LoRA·partial freezing | FSDP2 또는 `use_orig_params` 지원 확인 |
| 4-bit QLoRA multi-GPU | framework 공식 FSDP+QLoRA recipe |
| 복잡한 TP·PP·CP 조합 | TorchTitan·Megatron·NeMo 계열 검토 |

### 19.5 FSDP2 Accelerate 설정 예시

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 4
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_offload_params: false
  fsdp_use_orig_params: true
```

```bash
accelerate config --config_file accelerate-fsdp.yaml
accelerate launch --config_file accelerate-fsdp.yaml train.py
```

FSDP2를 직접 사용하는 최신 code는 `fully_shard()`와 DTensor 기반 API를 따른다. framework abstraction을 사용할 때 해당 abstraction이 FSDP1인지 FSDP2인지 확인한다.

### 19.6 wrap unit 크기

너무 큰 FSDP unit:

- all-gather peak 증가
- communication overlap 감소 가능

너무 작은 unit:

- collective 수 증가
- metadata·launch overhead 증가

transformer block 단위가 일반적인 시작점이다. shared embedding과 tied weight가 다른 unit으로 갈라지지 않도록 한다.

### 19.7 prefetch

forward/backward prefetch는 communication과 compute를 겹치지만 다음 layer parameter를 미리 가져오므로 peak memory가 증가할 수 있다. OOM이 발생하면 prefetch policy를 조정한다.

### 19.8 checkpoint

분산학습은 sharded checkpoint를 기본으로 한다.

```text
각 rank가 자기 shard 저장
→ resume 시 동일·다른 world size로 reshard
→ 배포용 full model merge는 별도 job
```

Hugging Face Accelerate는 `SHARDED_STATE_DICT`와 `accelerator.save_state()` 경로를 권장하고, PyTorch Distributed Checkpoint는 parallel save/load와 load-time resharding을 지원한다.

### 19.9 launch 전 확인

```text
[ ] GPU topology·NVLink·PCIe
[ ] NCCL·ROCm collectives
[ ] rank별 visible device
[ ] model loading 중 CPU RAM peak
[ ] tokenizer·dataset cache 동시 접근
[ ] checkpoint shared filesystem throughput
[ ] failure 시 partial shard 정리
[ ] elastic restart·RNG state
```

### 19.10 FSDP가 해결하지 않는 것

- activation을 자동으로 world size만큼 줄이지 않음
- 한 layer가 단일 GPU에 펼쳐질 때의 peak
- 아주 긴 context
- policy·reference·reward model 복수 적재
- VLM vision encoder activation
- checkpoint full gather OOM

---

## 20. TP·PP·CP·SP·EP

FSDP·ZeRO만으로 한 layer의 compute peak가 들어가지 않거나, long context·MoE를 확장해야 할 때 다른 parallelism을 조합한다.

### 20.1 Tensor Parallelism·TP

weight matrix와 연산을 GPU 여러 장에 나눈다.

```text
장점:
  단일 layer weight·activation 일부 감소
  큰 hidden size model 학습 가능

비용:
  layer마다 collective
  빠른 NVLink·NVSwitch·fabric 필요
```

TP degree가 커질수록 communication이 늘고 작은 batch에서는 효율이 떨어질 수 있다.

### 20.2 Pipeline Parallelism·PP

layer block을 stage별 GPU에 배치하고 micro-batch를 pipeline으로 흘린다.

```text
장점: stage별 model state 감소
비용: pipeline bubble·schedule·activation transfer
```

micro-batch 수가 충분하지 않으면 bubble이 커진다. tied layer·skip connection·multimodal branch가 partition을 복잡하게 한다.

### 20.3 Context Parallelism·CP

sequence dimension을 여러 GPU에 나눠 긴 context activation과 attention을 분산한다. long-context training에서 model state보다 activation이 병목일 때 사용한다.

### 20.4 Sequence Parallelism·SP

일부 layer norm·dropout·activation을 sequence 축으로 shard한다. TP와 결합해 replicated activation을 줄이는 구현이 많다.

### 20.5 Expert Parallelism·EP

MoE expert를 GPU 그룹에 분산한다.

- expert weights 분산
- token routing all-to-all
- load balancing 중요
- capacity factor와 token drop 확인
- router·shared expert는 복제될 수 있음

### 20.6 조합 예시

```text
8 GPUs:
  TP=2 × FSDP=4

16 GPUs, long context:
  TP=2 × CP=2 × FSDP=4

32 GPUs, MoE:
  TP=2 × EP=4 × FSDP=4
```

degree 곱이 world size와 맞아야 하며, framework가 해당 조합을 지원해야 한다.

### 20.7 어떤 병목을 해결하는지

| 병목 | 우선 parallelism |
| --- | --- |
| 전체 model state | FSDP2·ZeRO-3 |
| 한 layer가 GPU에 안 들어감 | TP |
| layer 수가 매우 많음 | PP |
| sequence activation | CP·SP |
| MoE expert weights | EP |
| throughput | DDP·data parallel |

### 20.8 topology

- TP·EP는 가장 빠른 intra-node link 안에 배치
- FSDP·DP는 상대적으로 느린 inter-node link에 배치 가능
- PP stage 간 traffic과 compute 균형
- topology-aware mesh 구성
- heterogeneous GPU를 같은 TP group에 섞지 않음

TorchTitan은 최신 PyTorch-native training stack으로 FSDP2, TP, PP, CP, activation checkpointing, distributed checkpoint와 low-precision training 조합을 제공한다. model 지원과 config는 현재 repository를 확인한다.

---

## 21. Dense와 MoE 모델

### 21.1 추론의 active parameters와 학습 메모리

MoE model 카드에는 `total parameters`와 `active parameters per token`이 함께 표시된다. 학습 메모리는 일반적으로 **전체 expert weights**를 기준으로 계산한다.

```text
compute per token ≈ active experts
resident model state ≈ total experts
optimizer state ≈ trainable experts 전체
```

“30B total, 3B active” 모델을 3B dense처럼 계산하면 안 된다.

### 21.2 MoE full FT

full FT에서는 모든 expert에 optimizer state가 생긴다. 일부 token이 특정 expert를 사용하지 않아도 학습 run 전체에서 다른 batch가 사용할 수 있다.

```text
full MoE state
≈ total parameters × 12~18 B/P
+ router·expert activation
```

### 21.3 MoE LoRA·QLoRA

선택지:

- attention만 LoRA
- shared expert만 LoRA
- 모든 expert linear에 같은 rank LoRA
- expert별 adapter
- router를 full 또는 frozen

모든 expert에 LoRA를 붙이면 trainable parameter가 expert 수에 비례해 증가한다. adapter checkpoint가 dense model보다 커질 수 있다.

### 21.4 expert parallel

EP는 expert weight를 분산하지만 token all-to-all이 발생한다.

메모리 예산:

```text
local experts
+ shared layers
+ router
+ dispatched token buffers
+ all-to-all workspace
+ activation
```

### 21.5 load balancing

expert utilization 불균형은 일부 rank의 activation·communication peak를 높인다.

- router auxiliary loss
- capacity factor
- token drop 또는 dropless routing
- expert histogram
- rank별 peak memory
- straggler time

### 21.6 adapter merge

MoE adapter merge는 expert tensor 이름과 sharding layout이 복잡하다. serving runtime이 merged MoE architecture와 quantization을 지원하는지 확인하고, adapter-only 배포를 유지할 대안도 검토한다.


---

## 22. SFT·continued pretraining

### 22.1 SFT 메모리

Supervised fine-tuning은 보통 policy model 하나를 학습하므로 preference·RL보다 단순하다.

```text
SFT peak
≈ trainable model state
+ prompt·completion activation
+ logits·loss
+ optimizer workspace
```

chat SFT에서는 loss mask와 template가 올바른지가 메모리보다 먼저다. 잘못된 template로 학습하면 같은 compute를 쓰고도 품질이 악화된다.

### 22.2 SFT 데이터 형식

대표 형식:

```json
{"messages":[
  {"role":"system","content":"규칙"},
  {"role":"user","content":"질문"},
  {"role":"assistant","content":"정답"}
]}
```

```json
{"prompt":"질문", "completion":"정답"}
```

```json
{"text":"전체 language-modeling 문장"}
```

model tokenizer의 공식 chat template를 사용하고, 이미 template가 적용된 text에 다시 template를 적용하지 않는다.

### 22.3 completion-only vs full-sequence loss

| 목적 | loss 대상 |
| --- | --- |
| assistant 응답 행동 | assistant token만 |
| tool-call 생성 | assistant tool-call·arguments |
| language modeling | 전체 token |
| continued pretraining | 전체 token |
| input reconstruction task | 정의된 target token |

### 22.4 continued pretraining 메모리

같은 model·sequence라면 한 step의 기본 activation은 SFT와 유사하지만, 다음 때문에 전체 자원 요구가 커지는 경우가 많다.

- 더 많은 token
- 더 긴 document packing
- 높은 throughput 목표
- full fine-tuning 사용 빈도
- 많은 checkpoint·resume
- validation perplexity 계산
- 대규모 shuffle·dataset streaming

### 22.5 domain mixture

```yaml
mixture:
  general_replay: 0.30
  target_domain: 0.55
  instruction_replay: 0.10
  safety_and_policy: 0.05
```

비율은 예시다. target domain만 100% 사용하면 general capability와 instruction-following이 손상될 수 있다.

### 22.6 sequence packing

continued pretraining에서는 document boundary를 보존할지 결정한다.

- EOS로 document 구분
- cross-document attention 허용 여부
- position reset 여부
- long document split overlap
- code file·notebook·paper section 경계

### 22.7 tokenizer 변경

새 tokenizer나 token 추가는 embedding·LM head 수정과 checkpoint 호환 문제를 만든다.

```text
새 token 수 × hidden size × weight bytes
+ trainable embedding optimizer state
+ LM head가 untied이면 추가 state
```

새 token이 적으면 기존 tokenizer로 표현했을 때의 token efficiency와 비교한다.

### 22.8 학습량

```text
training tokens
= dataset tokens × epochs
```

sample 수보다 token 수와 unique token 비율을 기록한다. duplicate-heavy corpus는 epoch를 늘려도 유효 학습량이 적다.

### 22.9 SFT에서 preference 단계로 넘어갈 조건

- SFT validation이 안정됨
- prompt template와 loss mask 검증 완료
- chosen/rejected 품질 기준이 명확
- reference baseline 저장
- task automatic evaluator 존재
- preference stage가 개선해야 할 metric 정의

---

## 23. DPO·KTO·ORPO·GRPO·PPO

Preference·reinforcement 단계는 “LoRA이므로 SFT와 같은 메모리”가 아니다. model 복제, chosen/rejected pair, rollout과 generation KV cache가 추가된다.

### 23.1 구성요소 비교

| 방식 | 학습 입력 | 일반적 resident 구성 | 주요 추가 메모리 |
| --- | --- | --- | --- |
| DPO | prompt + chosen + rejected | policy + reference 논리 | 두 completion forward·reference log-prob |
| KTO | desirable·undesirable examples | policy + reference 논리 | reference log-prob·batch balance |
| ORPO | SFT + preference objective | policy 중심, reference-free 계열 | chosen/rejected activation |
| Reward model | prompt + ranked response | classifier/reward model | 여러 response pair |
| GRPO | prompt에서 여러 completion 생성 | policy + reference·reward functions | group rollout·KV cache·generation engine |
| RLOO | online rollout | policy + reference | rollout·reward·baseline statistics |
| PPO | online rollout | policy + reference + reward + value/critic | 가장 많은 model state·rollout buffer |

구현마다 reference model을 명시적으로 복제하거나 adapter disable·precomputed log-prob로 대체할 수 있다.

### 23.2 DPO 메모리

DPO는 같은 prompt에 chosen과 rejected를 함께 처리한다.

```text
DPO token cost per example
≈ prompt + chosen
+ prompt + rejected
```

prefix 공유 최적화가 없다면 prompt를 두 번 계산할 수 있다.

메모리 절감:

- policy는 LoRA·QLoRA
- reference log-prob 사전 계산
- 같은 base에서 reference adapter를 비활성화하는 framework path
- chosen/rejected 길이 제한
- padding-free
- micro-batch 1
- reference model CPU 또는 별도 GPU 배치

사전 계산된 reference log-prob는 GPU 메모리를 줄이지만 dataset storage와 preprocessing time을 늘리고, reference revision·tokenizer가 바뀌면 다시 계산해야 한다.

### 23.3 DPO 설정 예시

```python
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

args = DPOConfig(
    output_dir="out-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_length=2048,
    gradient_checkpointing=True,
    bf16=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model="OWNER/MODEL",
    ref_model=None,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
```

`ref_model=None`의 실제 의미와 adapter reference 처리 방식은 설치된 TRL·PEFT 조합을 확인한다.

### 23.4 KTO·ORPO

KTO는 paired preference가 없어도 desirable·undesirable signal을 사용할 수 있지만 reference 계산과 class balance가 중요하다. ORPO 계열은 reference-free 구성이 가능해 resident model 수를 줄일 수 있으나, chosen/rejected activation과 objective-specific temporary는 남는다.

알고리즘 선택은 메모리만으로 결정하지 않는다. 같은 데이터에서 SFT·DPO·reference-free objective를 비교한다.

### 23.5 GRPO

GRPO는 prompt마다 여러 completion을 생성하고 group 내 reward를 정규화한다. critic을 별도로 두지 않아 PPO보다 state가 적을 수 있지만, rollout이 메모리를 지배할 수 있다.

```text
rollout memory
∝ prompts_per_batch
× num_generations
× completion_length
× KV_cache_per_token
```

예를 들어 `num_generations=8`은 동일 prompt에서 completion KV·token buffer를 여러 개 유지한다.

메모리 절감 순서:

```text
num_generations ↓
→ max_completion_length ↓
→ prompt batch ↓
→ generation engine 분리
→ vLLM sleep/offload
→ reward model 분리
→ reference log-prob 최적화
```

### 23.6 online method와 ZeRO-3

ZeRO-3에서 generation을 위해 shard weight를 한 GPU에 gather하면 OOM이 발생할 수 있다. TRL의 현재 memory guide는 online method에서 다음 옵션을 제공한다.

```python
from trl import GRPOConfig

args = GRPOConfig(
    output_dir="out-grpo",
    ds3_gather_for_generation=False,
)
```

이 옵션은 gather OOM을 피하는 대신 generation을 느리게 할 수 있다.

### 23.7 vLLM colocate·server

| 방식 | 장점 | 단점 |
| --- | --- | --- |
| colocate | weight synchronization 단순, 한 node | training과 KV cache가 VRAM 경쟁 |
| 별도 server GPU | 메모리·failure domain 분리 | weight sync·network latency |
| sleep mode | optimize 중 weights·cache offload | wake latency·CPU RAM |

현재 TRL은 vLLM sleep mode를 지원하는 online trainer에서 다음과 같은 설정을 설명한다.

```python
args = GRPOConfig(
    output_dir="out-grpo",
    vllm_enable_sleep_mode=True,
)
```

### 23.8 PPO

PPO형 RLHF는 일반적으로 다음을 필요로 한다.

```text
policy model
reference model
reward model
value/critic model
rollout buffer
generation KV cache
optimizer states
```

LoRA를 적용해도 여러 frozen model의 inference memory가 남는다. 단일 GPU에서 큰 모델 PPO를 목표로 하기보다 model·role을 여러 GPU 또는 process로 분리한다.

### 23.9 reward function

가능하면 reward를 다음 우선순위로 분리한다.

1. deterministic verifier: test·compiler·math checker·schema validator
2. 작은 classifier·reward model
3. 큰 judge model
4. 사람 평가

큰 judge를 training GPU에 함께 올리지 않는다. 비동기 queue 또는 별도 worker를 사용한다.

### 23.10 rollout 저장

- prompt·completion token
- log-prob
- reward·advantage
- attention mask
- tool trace·environment state

긴 agent trajectory는 모델보다 rollout buffer가 커질 수 있다. 전체 text 대신 reproducible state와 필요한 token만 저장할 수 있는지 검토한다.

### 23.11 평가

online RL은 reward hacking을 반드시 확인한다.

- reward와 실제 task 성공 분리
- length bias
- format-only reward
- verifier exploit
- test leakage
- degenerate repetition
- safety regression
- unseen environment

---

## 24. VLM·OCR 파인튜닝

자세한 모델 선택은 [비전·OCR 가이드](../modalities/vision-ocr.md)를 참고한다. 이 장은 학습 메모리에 집중한다.

### 24.1 구성요소

```text
vision encoder
+ projector·resampler
+ language model
+ image preprocessing buffer
+ visual token activation
+ text token activation
+ multimodal loss
```

### 24.2 동결 순서

| 단계 | trainable | 메모리 | 용도 |
| --- | --- | --- | --- |
| 1 | projector만 | 가장 작음 | modality alignment·새 connector |
| 2 | projector + LLM LoRA | 중간 | 문서·OCR·UI task 적응 |
| 3 | vision encoder 마지막 blocks + LoRA | 증가 | 시각 domain shift |
| 4 | vision encoder·LLM full/partial | 매우 큼 | 대규모 연구 |

먼저 projector와 LLM LoRA로 task 성능을 측정한다.

### 24.3 visual token

visual token 수는 image resolution, patch size, dynamic tiling, crop 수와 model processor에 따라 달라진다.

```text
multimodal sequence
= text tokens
+ visual tokens per image × images
+ special tokens
```

고해상도 문서 4장과 일반 사진 1장은 같은 “batch 1”이 아니다.

### 24.4 resolution budget

```text
224/336/448 등 기본 resolution부터 시작
→ OCR 작은 글씨가 부족하면 tile·resolution 증가
→ visual token과 peak를 다시 측정
```

고해상도 문서는 전체 page와 crop을 동시에 넣는 대신 layout detector로 관심 영역을 잘라 학습할 수 있다.

### 24.5 image batching

- image 수 기준이 아니라 pixel·visual-token budget 사용
- aspect ratio bucket
- multi-image sample 별도 bucket
- video는 frame count bucket
- processor의 dynamic resize 결과 기록

### 24.6 QLoRA VLM

LLM body를 4-bit로 줄여도 vision encoder와 projector는 BF16·FP16으로 남을 수 있다.

```text
VLM QLoRA peak
≈ quantized LLM base
+ vision encoder weights
+ projector
+ visual activation
+ text activation
+ adapter state
```

projector까지 무조건 4-bit로 낮추면 OCR·small-object 품질이 악화될 수 있다. Q8/BF16 기준선을 유지한다.

### 24.7 OCR supervision

target이 긴 Markdown·HTML·LaTeX이면 output sequence가 매우 길다. 이미지보다 decoder logits가 병목일 수 있다.

- table cell 단위 curriculum
- page-level max output length
- formula sample 별도 bucket
- markup validity checker
- teacher forcing length 분석

### 24.8 dataset collator

```python
batch = processor(
    images=images,
    text=texts,
    return_tensors="pt",
    padding=True,
)
```

pixel tensor dtype, image normalization, text labels와 ignore index를 확인한다. processor가 CPU에서 고해상도 이미지를 여러 copy 만드는지도 profiler로 본다.

### 24.9 video

```text
activation ∝ frame count × visual tokens per frame
```

frame sampling, temporal crop, resolution을 함께 조정한다. 동일 video의 adjacent frames를 무작위 train·validation으로 나누면 누수가 발생한다.

### 24.10 보안

이미지 속 prompt injection, 개인정보, 얼굴·문서 원본과 OCR text의 중복 보존을 관리한다. training worker에서 외부 URL을 직접 fetch하지 않고 ingestion 단계에서 검증한다.

---

## 25. 이미지 생성 파인튜닝

자세한 모델 선택은 [이미지 생성 가이드](../modalities/image-generation.md)를 참고한다.

### 25.1 방식 비교

| 방식 | trainable 대상 | 메모리 | 적합한 목표 |
| --- | --- | --- | --- |
| Textual inversion | token embedding | 매우 작음 | 개념·스타일 token |
| LoRA | UNet·DiT attention/linear | 작음~중간 | 스타일·캐릭터·제품 |
| LoRA + text encoder | 생성 backbone + encoder | 증가 | 이름·문자·prompt alignment |
| DreamBooth full/partial | backbone 일부/전체 | 큼 | 강한 subject adaptation |
| ControlNet·adapter training | 별도 control network | 중간~큼 | pose·depth·edge 조건 |
| Full model FT | DiT/UNet 전체 | 매우 큼 | 대규모 domain 재학습 |

### 25.2 메모리 구성

```text
DiT·UNet weights
+ trainable LoRA/full states
+ text encoder
+ VAE
+ latent
+ noisy latent·scheduler state
+ attention activation
+ image augmentation buffer
```

VAE와 text encoder를 frozen해도 forward memory는 남는다.

### 25.3 resolution

latent diffusion activation은 resolution과 batch에 민감하다.

```text
512² → 768² → 1024²
```

pixel 수는 각각 선형 한 단계가 아니라 면적으로 증가한다. aspect-ratio bucket을 사용해 불필요한 square padding을 줄인다.

### 25.4 기본 절감 순서

```text
batch 1
→ resolution 축소
→ gradient checkpointing
→ memory-efficient attention
→ 8-bit optimizer
→ cache latents
→ text encoder 동결·embedding cache
→ LoRA rank 축소
→ CPU offload
```

### 25.5 latent cache

VAE가 frozen이고 augmentation이 latent 변환 후에도 유효하다면 latents를 미리 계산할 수 있다.

장점:

- VAE forward 제거
- VRAM·step time 감소

주의:

- random crop·color augmentation을 cache 전에 고정
- disk 사용 증가
- VAE revision·scaling factor 고정
- 개인정보가 latent에도 남을 수 있음

### 25.6 text embedding cache

text encoder가 frozen이고 caption dropout·token augmentation을 사용하지 않으면 embedding을 cache할 수 있다. text encoder를 학습하거나 prompt augmentation을 수행하면 사용하지 않는다.

### 25.7 LoRA target

- attention projection
- MLP/linear blocks
- convolution layer 지원 여부
- text encoder adapter
- transformer block별 rank

모델마다 layer 이름과 LoRA 적용법이 다르므로 Diffusers official training script와 model repository를 기준으로 한다.

### 25.8 DreamBooth

DreamBooth는 작은 subject dataset에서 overfitting과 memorization 위험이 크다. prior preservation image 생성과 class image cache가 추가 storage·compute를 요구할 수 있다.

### 25.9 검증

- 고정 seed·prompt grid
- identity·style similarity
- prompt adherence
- text rendering
- diversity
- overfitting·training image reconstruction
- NSFW·safety regression
- watermark·provenance

### 25.10 배포

adapter를 base model과 함께 배포할 때 base license, adapter license, training image 권리와 상업 이용 조건을 모두 확인한다.

---

## 26. 오디오·음성 파인튜닝

자세한 모델 선택은 [오디오·음성 가이드](../modalities/audio-speech.md)를 참고한다.

### 26.1 메모리 단위는 초·frame

ASR·TTS에서는 example 수보다 총 audio duration과 acoustic frame 수가 중요하다.

```text
batch_audio_seconds
= Σ sample_duration_seconds
```

frame rate가 50Hz라면 30초 audio는 약 1,500 frame이며, encoder subsampling 전에 더 많을 수 있다.

### 26.2 ASR 구성

```text
waveform decode·resample
+ feature extractor
+ acoustic encoder
+ CTC/RNNT/decoder
+ text tokenizer
+ alignment state
```

긴 audio는 chunking·streaming training을 사용하고, duration bucket으로 padding을 줄인다.

### 26.3 TTS 구성

```text
text encoder
+ duration·prosody model
+ acoustic model
+ speech tokenizer·codec
+ vocoder
+ speaker encoder
```

보이스 클로닝에서는 speaker encoder와 vocoder를 frozen하고 acoustic model adapter만 학습하는 구성이 메모리와 데이터 요구를 낮춘다.

### 26.4 duration bucket

```text
0–3 sec
3–8 sec
8–15 sec
15–30 sec
30 sec 이상 별도 처리
```

batch마다 총 audio seconds가 비슷하도록 sampler를 구성한다.

### 26.5 feature cache

frozen feature extractor·codec output을 미리 저장할 수 있다.

- resampling 설정 고정
- sample rate·channel 기록
- augmentation을 cache 전후 어디서 적용하는지 정의
- codec revision 고정
- disk와 개인정보 보호

### 26.6 augmentation

speed perturbation, noise, reverb는 CPU RAM·worker time을 늘린다. worker가 waveform 여러 copy를 만들지 확인한다.

### 26.7 streaming model

streaming ASR은 chunk state와 left context를 학습해야 한다.

```text
training sequence
= current chunk
+ cached left context
+ lookahead
```

offline WER만 평가하지 말고 latency·endpointing·state memory도 기록한다.

### 26.8 TTS speaker data

- 명시적 동의
- 허용된 발화 목적
- 원본 audio와 speaker embedding 접근 통제
- adapter에 화자 정보가 남는다는 점 인지
- 철회·삭제 절차
- 사칭·사회공학 방지 정책

### 26.9 평가

ASR:

- WER·CER
- timestamp·diarization
- noise·accent·code-switch
- real-time factor

TTS:

- intelligibility
- speaker similarity
- prosody·emotion
- artifact
- streaming latency
- unseen text·language

---

## 27. 임베딩·reranker·분류 모델

작은 encoder model은 full fine-tuning이 현실적인 경우가 많지만, contrastive batch와 negative 수가 activation을 크게 만든다.

### 27.1 임베딩 학습

```text
encoder activation
+ embedding vectors
+ similarity matrix
+ negatives
```

in-batch negative의 similarity matrix는 대략 `[batch, batch]`다. distributed all-gather로 global negatives를 사용하면 embedding buffer와 communication이 증가한다.

### 27.2 batch가 중요한 이유

contrastive learning은 큰 batch가 품질에 유리할 수 있다. 단순 gradient accumulation은 이전 micro-batch embedding을 현재 similarity matrix에 포함하지 않으므로, “global negative batch”와 같은 효과가 아닐 수 있다.

대안:

- cross-device negatives
- memory bank·queue
- cached negatives
- gradient-cache 계열
- hard-negative mining

### 27.3 reranker

cross-encoder reranker는 query-document pair마다 전체 encoder forward를 수행한다.

```text
pairs per batch
= queries × candidates_per_query
```

candidate 수를 늘리면 batch activation이 빠르게 증가한다. length bucket과 negative sampling을 사용한다.

### 27.4 분류·추출 모델

0.1–1B encoder는 16–24GB GPU에서 full FT가 가능한 경우가 많다. 그러나 long document classification, token classification과 CRF head는 sequence activation이 지배할 수 있다.

### 27.5 LoRA 사용 시점

- multi-task adapter
- 1B 이상 encoder
- base 보존·rollback 필요
- 작은 dataset
- multilingual model의 일부 domain 적응

작은 100M급 model에서는 QLoRA dequant overhead가 절감 이득보다 클 수 있다.

### 27.6 distillation

teacher embedding·score를 online으로 계산하면 teacher inference memory가 추가된다. 가능하면 teacher output을 사전 계산하고 다음을 고정한다.

- teacher revision
- tokenizer
- normalization
- score temperature
- negative set
- output hash

### 27.7 평가

Embedding:

- Recall@k
- nDCG@k
- MRR
- cross-language retrieval
- cosine distribution drift

Reranker:

- nDCG@10
- MRR@10
- pair latency
- candidates/sec

분류:

- class imbalance
- calibration
- AUROC·AUPRC
- abstention

---

## 28. 프레임워크 선택과 실행 예제

### 28.1 선택표

| 프레임워크 | 강점 | 메모리 기능 | 적합한 사용자 |
| --- | --- | --- | --- |
| Transformers + PEFT + TRL | 가장 넓은 model·trainer 생태계 | LoRA·QLoRA·checkpointing·packing·DPO·GRPO | 직접 Python 제어 |
| torchtune | PyTorch-native recipe·교육성 | LoRA·QLoRA·full FT·activation 최적화 | 단일·다중 GPU 연구 |
| Accelerate | launch·FSDP·DeepSpeed abstraction | FSDP2·ZeRO·FP8·checkpoint | custom script 분산화 |
| DeepSpeed | ZeRO·offload·pipeline | CPU/NVMe offload·ZeRO-1/2/3 | 대형 model·cluster |
| TorchTitan | 최신 PyTorch 대규모 학습 stack | FSDP2·TP·PP·CP·DCP·low precision | server-scale 연구 |
| Axolotl | YAML 중심 다양한 SFT·preference | QLoRA·FSDP·packing·sequence parallel | 빠른 실험 구성 |
| LLaMA-Factory | CLI·Web UI·다수 model | LoRA·QLoRA·ZeRO | 반복 실험·교육 |
| Unsloth | optimized single/multi-GPU 경로 | LoRA·QLoRA·full·RL | 지원 model에서 빠른 iteration |
| MLX-LM | Apple Silicon native | LoRA·QLoRA·DoRA·full | Mac 로컬 학습 |
| NVIDIA NeMo | 대규모 NVIDIA stack·멀티모달·음성 | PEFT·Megatron parallelism | NVIDIA cluster |
| Diffusers | 이미지 생성 공식 training scripts | LoRA·DreamBooth·checkpointing | diffusion·DiT |

vendor·project가 제시하는 속도·VRAM 절감 수치는 해당 benchmark 조건의 결과다. 공통 baseline으로 간주하지 않는다.

### 28.2 환경 고정

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install torch transformers datasets accelerate peft trl bitsandbytes
pip freeze > requirements-lock.txt
```

production·research 재현성에는 `uv.lock`, Conda lock, container digest 또는 Nix 같은 더 강한 lock을 사용한다.

### 28.3 Hugging Face artifact 사전 확인

```bash
hf download OWNER/MODEL --dry-run
hf download OWNER/MODEL \
  --revision <commit-sha> \
  --local-dir ./models/MODEL
```

모델과 dataset은 branch 이름보다 immutable commit SHA를 사용한다.

### 28.4 TRL SFT + QLoRA 예제

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

model_id = "OWNER/MODEL"
dataset = load_dataset("json", data_files="train.jsonl", split="train")

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="out-sft",
    max_length=2048,
    packing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    model_init_kwargs={
        "dtype": torch.bfloat16,
    },
)

trainer = SFTTrainer(
    model=model_id,
    args=args,
    train_dataset=dataset,
    peft_config=lora,
    quantization_config=quant,
)
trainer.train()
trainer.save_model("out-sft/final-adapter")
```

현재 TRL API는 빠르게 변하므로 installed version의 `SFTConfig` signature를 확인한다.

### 28.5 Accelerate launch

```bash
accelerate config
accelerate launch train.py
```

명시적 config 파일을 repository에 보존한다.

```bash
accelerate launch \
  --config_file configs/accelerate-fsdp.yaml \
  train.py \
  --config configs/train.yaml
```

### 28.6 DeepSpeed launch

```bash
deepseed --num_gpus 4 train.py \
  --deepspeed configs/zero3.json
```

또는 Transformers·Accelerate가 DeepSpeed config를 전달하도록 한다. launch 방식이 중복되지 않도록 한다.

### 28.7 Axolotl 예시 구조

```yaml
base_model: OWNER/MODEL
load_in_4bit: true
adapter: qlora
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 16
sample_packing: true
gradient_checkpointing: true
bf16: true

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true

optimizer: paged_adamw_8bit
output_dir: ./out
```

```bash
axolotl train configs/train.yml
```

config key는 Axolotl release와 model template에 맞게 확인한다.

### 28.8 LLaMA-Factory

```bash
llamafactory-cli train configs/train.yaml
```

Web UI는 편리하지만 최종 run의 전체 config를 version control에 저장한다. UI screenshot만으로 재현성을 남기지 않는다.

### 28.9 torchtune

```bash
tune ls
# 설치된 release에서 지원 recipe와 config 이름 확인
tune cp <recipe_name> ./train_recipe.py
tune run ./train_recipe.py --config ./config.yaml
```

공식 recipe를 복사한 후 model component, tokenizer, dataset와 checkpointer 설정을 고정한다.

### 28.10 MLX-LM

```bash
mlx_lm.lora \
  --model OWNER/MODEL \
  --train \
  --data ./data \
  --batch-size 1 \
  --grad-accumulation-steps 16 \
  --grad-checkpoint
```

### 28.11 custom trainer를 선택할 때

- loss·mask를 완전히 통제해야 함
- unusual multimodal architecture
- custom verifier·environment
- new parallelism
- research optimizer
- memory lifetime 최적화

custom code는 unit test, tiny model overfit test와 reference implementation 비교가 필요하다.

### 28.12 tiny overfit test

전체 run 전에 16–64개 sample에 overfit되는지 확인한다.

```text
loss가 내려가지 않음:
  label mask·template·optimizer·frozen param 문제

loss는 내려가지만 output이 이상함:
  generation template·EOS·data quality 문제
```

---

## 29. 측정·벤치마크·OOM 문제 해결

### 29.1 반드시 기록할 값

```text
idle allocated·reserved
after model load
forward peak
backward peak
optimizer-step peak
validation peak
checkpoint save peak
host RAM peak
step time
train tokens/s
samples/s
GPU utilization
PCIe·NVLink traffic
```

### 29.2 PyTorch CUDA 측정

```python
import torch


def gib(x: int) -> float:
    return x / 1024**3


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# model load 후
print("allocated", gib(torch.cuda.memory_allocated()))
print("reserved ", gib(torch.cuda.memory_reserved()))

# representative train step 후
print("peak allocated", gib(torch.cuda.max_memory_allocated()))
print("peak reserved ", gib(torch.cuda.max_memory_reserved()))
```

`empty_cache()`는 살아 있는 tensor를 해제하지 않는다. Python reference와 autograd graph가 남아 있으면 효과가 없다.

### 29.3 단계별 측정

```python
optimizer.zero_grad(set_to_none=True)

torch.cuda.reset_peak_memory_stats()
out = model(**batch)
print("after forward", gib(torch.cuda.max_memory_allocated()))

out.loss.backward()
print("after backward", gib(torch.cuda.max_memory_allocated()))

optimizer.step()
print("after step", gib(torch.cuda.max_memory_allocated()))
optimizer.zero_grad(set_to_none=True)
```

CUDA op는 asynchronous이므로 정밀한 구간 timing에는 `torch.cuda.synchronize()`를 사용한다.

### 29.4 profiler

- PyTorch Profiler memory timeline
- `torch.cuda.memory_snapshot()`
- Nsight Systems·Compute
- `nvidia-smi dmon`
- ROCm profiler·`rocm-smi`
- Apple Activity Monitor·Instruments
- DeepSpeed communication logging

profiler 자체가 메모리와 속도에 영향을 주므로 짧은 대표 run에서 사용한다.

### 29.5 benchmark protocol

```yaml
benchmark:
  warmup_steps: 5
  measured_steps: 20
  fixed_batches: true
  longest_batch_test: true
  checkpoint_test: true
  validation_test: true
  report:
    - peak_accelerator_memory
    - peak_host_memory
    - median_step_time
    - p95_step_time
    - tokens_per_second
    - samples_per_second
    - loss
```

### 29.6 binary search

max token budget을 탐색한다.

```text
512 성공
1024 성공
2048 실패
1536 성공
1792 실패
1664 성공
```

마지막 성공값에서 10–20% 여유를 남긴다.

### 29.7 OOM 시 결정 트리

```text
모델 load 전에 OOM
  → artifact 중복·CPU RAM·device_map·FSDP loading 확인

model load에서 OOM
  → QLoRA·smaller model·sharding

forward에서 OOM
  → micro-batch·length·resolution·attention backend

backward에서 OOM
  → checkpointing·trainable modules·activation offload

optimizer.step에서 OOM
  → 8-bit optimizer·sharding·optimizer-in-backward

validation에서 OOM
  → eval batch·generation·KV·prediction accumulation

checkpoint에서 OOM
  → sharded state dict·CPU offload·별도 merge

몇 step 후 OOM
  → tensor reference leak·variable length·logging buffer·fragmentation
```

### 29.8 첫 step만 큰 경우

- lazy kernel initialization
- compile
- optimizer state 첫 생성
- CUDA graph capture
- quantization conversion

warmup 이후 steady peak와 first-step peak를 모두 기록한다. production run은 first-step peak도 통과해야 한다.

### 29.9 validation OOM

training보다 validation에서 OOM이 날 수 있다.

- `predict_with_generate`가 KV cache 사용
- eval batch가 더 큼
- 모든 logits를 CPU 전송 전 GPU에 유지
- metric가 prediction 전체 저장
- FSDP full parameter gather

대응:

- eval batch 1
- generation length 제한
- `eval_accumulation_steps`
- streaming metric
- adapter-only evaluation
- 별도 inference job

### 29.10 memory leak 검사

```python
loss_value = float(loss.detach())
# loss tensor 자체를 list에 누적하지 않음
history.append(loss_value)
```

흔한 leak:

- `losses.append(loss)`
- hidden states·logits를 detach 없이 저장
- callback이 batch tensor 보유
- exception traceback이 tensor reference 유지
- closure·hook 미제거
- generation output 전체 누적

### 29.11 rank별 편차

분산 run에서 rank 0만 logging·validation·checkpoint를 수행해 peak가 더 높을 수 있다. 모든 rank의 memory를 기록하고 max를 사용한다.

### 29.12 속도가 지나치게 느린 경우

- QLoRA dequant kernel fallback
- CPU offload 과다
- dataloader starvation
- sequence 길이 변동으로 compile graph break
- TP·FSDP communication 병목
- small micro-batch로 tensor core 활용 부족
- gradient checkpointing 과다
- frequent evaluation·save
- storage throughput

### 29.13 품질이 내려간 경우

메모리 옵션만 탓하지 말고 다음 순서로 본다.

```text
chat template
→ label mask
→ data correctness
→ learning rate
→ effective batch tokens
→ truncation
→ quantization
→ rank·target modules
→ optimizer·precision
```

---

## 30. 데이터·보안·라이선스·재현성

### 30.1 데이터 provenance

각 dataset에 다음을 기록한다.

```yaml
dataset:
  name: internal-sft-v1
  source_revision: <sha>
  license: <license-or-contract>
  collected_at: 2026-07-21
  pii_reviewed: true
  deduplicated: true
  train_eval_decontaminated: true
  preprocessing_revision: <sha>
  sha256: <hash>
```

### 30.2 개인정보

- 이름·전화번호·주소·계정·secret 제거
- code dataset의 API key·private key scan
- document image와 OCR text 모두 검사
- audio 원본과 transcript 모두 검사
- face·voice 생체정보에 별도 동의
- adapter·checkpoint도 민감 artifact로 취급

파인튜닝 후 원본 데이터를 삭제해도 model이 정보를 기억하지 않는다고 보장할 수 없다.

### 30.3 data poisoning

공개 web·issue·PR·forum data에는 다음이 포함될 수 있다.

- 악성 instruction
- backdoor trigger
- 취약한 코드 패턴
- license header 제거
- benchmark 정답 누수
- 의도적 misinformation

source trust score, sandboxed preprocessing, anomaly detection과 human review를 사용한다.

### 30.4 prompt injection과 tool trace

agent trajectory를 학습할 때 문서 내부 instruction과 실제 system policy를 구분한다. 공격자가 작성한 tool output을 정답 행동으로 학습하지 않는다.

```text
untrusted document content
≠ trainer instruction
≠ tool authorization
```

### 30.5 custom code와 model supply chain

- `trust_remote_code=True` 최소화
- model repository commit SHA 고정
- custom CUDA extension source review
- build container에 production secret 금지
- `safetensors` 우선
- pickle·arbitrary Python dataset artifact 경계
- dependencies SBOM·vulnerability scan
- gated token은 read-only 최소 권한

### 30.6 training isolation

```text
training container
  → 기본 network 차단 또는 allowlist
  → read-only model·dataset mount
  → output directory만 writable
  → cloud metadata 차단
  → no production credentials
  → CPU·RAM·disk·GPU quota
```

### 30.7 라이선스 체인

최종 adapter·merged model의 조건은 다음을 모두 따른다.

```text
base model license
+ dataset license·contract
+ teacher model output policy
+ code·framework license
+ image·audio rights
+ adapter redistribution 조건
```

LoRA 파일이 작다고 원본 model license에서 독립되는 것은 아니다.

### 30.8 모델 카드

최소 포함 정보:

- base model·revision
- fine-tuning method
- dataset description·license
- hyperparameters
- hardware·runtime
- memory·throughput
- evaluation
- known limitations
- intended·prohibited use
- safety·privacy review

### 30.9 재현성 manifest

```yaml
run:
  id: 2026-07-21-qwen-domain-sft-r16
  git_commit: <sha>
  container_digest: sha256:<digest>
  seed: 42

model:
  repo: OWNER/MODEL
  revision: <sha>
  tokenizer_revision: <sha>
  trust_remote_code: false

method:
  type: qlora
  base_quant: nf4
  double_quant: true
  compute_dtype: bf16
  lora_rank: 16
  lora_alpha: 32
  target_modules: all-linear

training:
  max_length: 2048
  micro_batch: 1
  grad_accumulation: 16
  global_batch_tokens: <value>
  optimizer: paged_adamw_8bit
  learning_rate: 0.0002
  gradient_checkpointing: true
  attention_backend: <name>

hardware:
  gpu: <model>
  gpu_count: 1
  vram_gib: 24
  system_ram_gib: 64
  driver: <version>
  os: <version>

artifacts:
  adapter_sha256: <hash>
  optimizer_checkpoint_sha256: <hash>
  eval_report_sha256: <hash>
```

### 30.10 평가 재현성

- prompt·chat template 고정
- decoding parameters 고정
- deterministic evaluator version
- test image·audio preprocessing 고정
- execution sandbox image digest
- code test dependency lock
- judge model·revision·prompt 기록
- 사람 평가 rubric·blind setup 기록

### 30.11 checkpoint 보존 정책

```text
last resume checkpoint
best validation checkpoint
final adapter
merged deployment artifact
evaluation report
manifest·hash
```

모든 intermediate checkpoint를 무기한 보관하지 않는다. 민감 데이터가 포함될 수 있으므로 retention과 삭제 절차를 정의한다.

### 30.12 rollback

- base + adapter 조합을 immutable ID로 배포
- canary traffic
- 이전 adapter 즉시 복구
- merged artifact와 adapter artifact 매핑
- tokenizer·template도 함께 rollback
- production feedback를 자동 학습 데이터로 바로 편입하지 않음

### 30.13 기여 체크리스트

이 repository에 파인튜닝 memory 결과를 추가할 때 다음을 포함한다.

```text
[ ] base repository·revision
[ ] parameter 수·architecture
[ ] full/LoRA/QLoRA/DoRA 방식
[ ] rank·target module·trainable parameter 수
[ ] optimizer·precision
[ ] max length·micro batch·accumulation
[ ] attention·checkpointing·packing
[ ] GPU·RAM·OS·driver
[ ] forward·backward·step·save peak
[ ] tokens/s와 step time
[ ] dataset 길이 분포
[ ] quality baseline
[ ] artifact hash·license
```


---

## 31. 주요 출처와 저장소

이 장은 구현과 메모리 계산을 확인할 때 우선할 공식 문서·원 논문·원 저장소를 정리한다. API와 지원 model은 release마다 바뀔 수 있으므로 링크의 최신 문서를 다시 확인한다.

### 31.1 학습 메모리·Transformers

- [Hugging Face Transformers: GPU memory usage](https://huggingface.co/docs/transformers/model_memory_anatomy)
- [Hugging Face Transformers: Gradient accumulation](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation)
- [Hugging Face Transformers: Gradient checkpointing](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing)
- [Hugging Face Accelerate: Model memory estimator](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator)
- [Hugging Face Accelerate: Profiler](https://huggingface.co/docs/accelerate/usage_guides/profiler)
- [Hugging Face Accelerate: Gradient accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [Hugging Face Accelerate: Low precision training](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training)

### 31.2 LoRA·QLoRA·PEFT

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PEFT FSDP guide](https://huggingface.co/docs/peft/accelerate/fsdp)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [bitsandbytes repository](https://github.com/bitsandbytes-foundation/bitsandbytes)

### 31.3 TRL·SFT·Preference·RL

- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL reducing memory usage](https://huggingface.co/docs/trl/reducing_memory_usage)
- [TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL KTO Trainer](https://huggingface.co/docs/trl/kto_trainer)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer)
- [TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration)
- [TRL repository](https://github.com/huggingface/trl)

### 31.4 torchtune·torchao

- [torchtune documentation](https://docs.pytorch.org/torchtune/)
- [torchtune memory optimization overview](https://docs.pytorch.org/torchtune/stable/tutorials/memory_optimizations.html)
- [torchtune LoRA tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/lora_finetune.html)
- [torchtune QLoRA tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/qlora_finetune.html)
- [torchtune end-to-end workflow](https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html)
- [torchao fine-tuning with QAT·QLoRA·float8](https://docs.pytorch.org/ao/stable/eager_tutorials/finetuning.html)
- [torchao documentation](https://docs.pytorch.org/ao/stable/)

### 31.5 PyTorch distributed·checkpoint·profiler

- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch optimizer step in backward](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)

### 31.6 Accelerate·DeepSpeed

- [Accelerate FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
- [Accelerate FSDP1 vs FSDP2](https://huggingface.co/docs/accelerate/concept_guides/fsdp1_vs_fsdp2)
- [Accelerate DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [Accelerate FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
- [DeepSpeed ZeRO-Infinity paper](https://arxiv.org/abs/2104.07857)
- [DeepSpeed pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
- [DeepSpeed Ulysses·long-sequence training](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)
- [DeepSpeed repository](https://github.com/deepspeedai/DeepSpeed)

### 31.7 실전 training framework

- [Axolotl documentation](https://docs.axolotl.ai/)
- [Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl)
- [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/)
- [LLaMA-Factory distributed training](https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html)
- [LLaMA-Factory repository](https://github.com/hiyouga/LLaMA-Factory)
- [Unsloth repository](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/trainer)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

### 31.8 Apple Silicon

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM LoRA·QLoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-LM examples](https://github.com/ml-explore/mlx-lm/tree/main/examples)

### 31.9 NVIDIA 대규모 학습

- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NeMo PEFT](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/peft_nemo2.html)
- [NeMo parallelisms](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### 31.10 비전·이미지

- [Diffusers training overview](https://huggingface.co/docs/diffusers/training/overview)
- [Diffusers LoRA training](https://huggingface.co/docs/diffusers/training/lora)
- [Diffusers DreamBooth](https://huggingface.co/docs/diffusers/training/dreambooth)
- [Diffusers textual inversion](https://huggingface.co/docs/diffusers/training/text_inversion)
- [Diffusers memory optimization](https://huggingface.co/docs/diffusers/optimization/memory)
- [PEFT image classification LoRA](https://huggingface.co/docs/peft/task_guides/image_classification_lora)

### 31.11 오디오·음성

- [Transformers Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Transformers Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [Transformers Wav2Vec2-BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert)
- [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)

### 31.12 임베딩·reranker

- [Sentence Transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [Sentence Transformers distributed training](https://sbert.net/docs/sentence_transformer/training/distributed.html)
- [Sentence Transformers loss overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
- [Transformers sequence classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Transformers token classification](https://huggingface.co/docs/transformers/tasks/token_classification)

### 31.13 관련 레포지토리 문서

- [생산성·문서·RAG](../domains/productivity-rag.md)
- [데이터 분석](../domains/data-analysis.md)
- [비전·OCR](../modalities/vision-ocr.md)
- [이미지 생성](../modalities/image-generation.md)
- [오디오·음성](../modalities/audio-speech.md)
- [양자화](./quantization.md)
- [서빙·동시성](./serving-concurrency.md) (예정)
- [런타임·하드웨어](./runtime-hardware.md) (예정)

---

## 32. 최종 권장안과 갱신 주의

### 32.1 대부분의 text SFT

```text
BF16 base가 여유 있게 들어감
  → LoRA r=8~16, attention 또는 all-linear

BF16 base가 들어가지 않음
  → QLoRA NF4 + double quant + BF16 compute

첫 backward OOM
  → max_length·micro-batch부터 줄임
  → checkpointing·packing·loss 최적화

model state 자체가 안 들어감
  → FSDP2·ZeRO-3 또는 작은 model
```

### 32.2 메모리별 한 줄 권장

| 실사용 accelerator memory | 기본 권장 |
| ---: | --- |
| 8 GB | 1B LoRA 또는 1–3B QLoRA, 512–1K |
| 12 GB | 3B LoRA·QLoRA, 7B QLoRA는 짧게 검증 |
| 16 GB | 3B BF16 LoRA 또는 7–8B QLoRA |
| 24 GB | 7–8B LoRA, 14B QLoRA |
| 32 GB | 14B LoRA·QLoRA, 27–32B QLoRA |
| 48 GB | 27–32B QLoRA, 65–70B는 엄격한 실험 |
| 64–80 GB | 70B QLoRA |
| 96–192 GB | 70–235B QLoRA·분산 full FT 연구 |

### 32.3 Full fine-tuning

```text
classic mixed-precision AdamW model state
≈ 18 B/P + activation

native BF16 lower planning bound
≈ 12 B/P + activation
```

full FT는 작은 model, 큰 domain shift와 충분한 cluster가 있을 때 선택한다. PEFT와 같은 parameter 수로 비교하지 않는다.

### 32.4 긴 context

```text
긴 context가 필요하면
rank보다 max_length에 메모리를 우선 배정

activation이 병목이면
Q4를 Q3로 낮추는 것보다
checkpointing·packing·CP를 먼저 검토
```

### 32.5 Preference·RL

```text
SFT 기준선 완성
→ DPO·KTO·ORPO
→ online GRPO·PPO
```

online method에서는 model weight보다 rollout KV cache와 generation engine이 peak를 만들 수 있다. `num_generations`, completion length와 reward worker 배치를 먼저 설계한다.

### 32.6 멀티모달

```text
VLM:
  projector + LLM LoRA부터
  visual token·resolution을 별도 예산

이미지:
  DiT/UNet LoRA
  VAE·text encoder는 우선 frozen

오디오:
  duration bucket
  encoder·codec·vocoder 부분 동결
```

### 32.7 분산학습

```text
model이 GPU 하나에 들어감
  → DDP

전체 state가 안 들어감
  → FSDP2·ZeRO-3

한 layer가 안 들어감
  → TP

sequence activation이 안 들어감
  → CP·SP

MoE expert가 안 들어감
  → EP
```

### 32.8 최소 검증 세트

```text
[ ] 16–64 sample tiny overfit
[ ] representative longest batch one-step
[ ] forward·backward·optimizer peak
[ ] validation·generation peak
[ ] checkpoint save·resume
[ ] base vs adapter task metric
[ ] general capability regression
[ ] PII·license·poisoning review
[ ] model·dataset·code revision과 hash
```

### 32.9 한 문장 기준

> **파인튜닝 가능 여부는 모델 파일 크기가 아니라, 학습할 parameter state와 실제 최장 batch의 activation이 optimizer step·checkpoint peak까지 포함해 메모리에 들어가는지로 판단한다.**

### 32.10 갱신 주의

이 문서는 2026-07-21 KST 기준으로 공식 문서와 원 저장소를 확인해 작성했다. 다음 항목은 학습 직전에 다시 검증한다.

- model architecture와 remote code
- PEFT·TRL·Transformers·PyTorch API
- bitsandbytes·quantized training 지원 GPU
- FSDP2·ZeRO·TP 조합
- attention·fused kernel 지원
- model·tokenizer·processor revision
- dataset license·개인정보
- checkpoint format·resume compatibility
- adapter merge·serving runtime 호환
- security advisory

표의 메모리는 장비 구매나 학습 완료를 보장하는 수치가 아니다. 실제 장비에서 다음 값을 PR·issue 또는 benchmark report로 남기는 것이 이 repository의 가장 유용한 기여다.

```yaml
required_measurements:
  model_state_gib: null
  forward_peak_gib: null
  backward_peak_gib: null
  optimizer_peak_gib: null
  checkpoint_peak_ram_gib: null
  max_sequence_or_modality_size: null
  train_tokens_per_second: null
  validation_metric: null
```

---

**문서 종료**
