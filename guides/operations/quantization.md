# 로컬 AI 모델 양자화 가이드
> Q2·Q3·Q4·INT4·INT8·FP8·FP4·GGUF·AWQ·GPTQ·NF4·EXL3·MLX를 RAM·VRAM 기준으로 선택·변환·검증하는 운영 가이드

[← 메인 README](../../README.md) · [생산성·문서·RAG](../domains/productivity-rag.md) · [데이터 분석](../domains/data-analysis.md) · [비전·OCR](../modalities/vision-ocr.md) · [이미지 생성](../modalities/image-generation.md) · [오디오·음성](../modalities/audio-speech.md)

> **최종 검증일:** 2026-07-21 (KST)
> **주요 형식·도구:** GGUF·`llama.cpp`, AWQ, GPTQModel, `compressed-tensors`·LLM Compressor, AutoRound, bitsandbytes NF4/INT8, torchao, FP8·MXFP4·NVFP4, EXL3, MLX, ONNX Runtime, OpenVINO·NNCF, Core ML
> **범위:** 추론용 PTQ·QAT·native low-bit, 가중치·활성화·KV 캐시 양자화, 멀티모달 구성요소, 변환·캘리브레이션·벤치마크·재현성·보안
> **관련 문서:** [파인튜닝 메모리](./fine-tuning-memory.md) (예정) · [서빙·동시성](./serving-concurrency.md) (예정) · [런타임·하드웨어](./runtime-hardware.md) (예정)

이 문서는 “4비트면 메모리가 정확히 4분의 1이 되는가?” 또는 “Q3가 Q4보다 항상 빠른가?” 같은 오해를 피하면서, 보유한 **시스템 RAM**, **GPU VRAM**, **Apple Silicon 통합 메모리**에 맞는 양자화 형식과 런타임을 선택하기 위한 실전 가이드다.

양자화는 모델의 숫자를 낮은 정밀도로 표현해 저장 공간과 메모리 대역폭을 줄이는 기술이다. 그러나 실제 배포에서 중요한 것은 비트 수 하나가 아니다. 다음 요소를 함께 봐야 한다.

```text
양자화 알고리즘
  + 저장 형식과 tensor packing
  + 실제 실행 kernel
  + 대상 하드웨어의 native 연산 지원
  + 비양자화 tensor와 scale·zero-point metadata
  + activation·KV cache·멀티모달 encoder
  + context length·batch·동시 요청
  + 모델 구조(Dense·MoE·GQA·멀티모달)
```

예를 들어 같은 “4비트”라도 GGUF `Q4_K_M`, AWQ W4A16, GPTQ 4-bit, bitsandbytes NF4, MLX 4-bit, NVFP4는 서로 같은 파일이 아니며 런타임·품질·속도·하드웨어 조건도 다르다. 파일이 작아져도 해당 장치에 최적화된 kernel이 없으면 오히려 고정밀 모델보다 느릴 수 있다.

또한 모델 파일만 메모리를 사용하는 것이 아니다.

```text
실제 peak memory
≈ quantized weights
+ scale·zero-point·codebook·metadata
+ 비양자화 embedding·output·norm tensor
+ activation·workspace·graph buffer
+ KV cache × context × 동시 슬롯
+ tokenizer·sampler·runtime allocator
+ vision/audio encoder·projector·VAE·codec
+ OS·디스플레이·IDE·브라우저 여유
```

> **핵심 원칙:** 런타임과 하드웨어를 먼저 정하고, 그 런타임이 빠르게 실행하는 형식 중에서 가장 높은 정밀도를 선택한다. 특별한 이유가 없다면 범용 GGUF는 `Q4_K_M`, CUDA W4A16은 검증된 AWQ·GPTQ·compressed-tensors, QLoRA는 NF4, Apple Silicon은 MLX 4-bit 또는 GGUF Q4를 시작점으로 삼는다. Q2·Q3는 “더 큰 모델을 넣기 위한 비용”이므로 실제 작업셋에서 반드시 검증한다.

모델·런타임·kernel 지원은 빠르게 변한다. 이 문서의 지원표는 2026-07-21 기준이며, 배포 직전 공식 문서에서 **현재 버전, 지원 GPU 세대, 모델 아키텍처, quantization config, 파일 revision과 라이선스**를 다시 확인한다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [양자화가 줄이는 것과 줄이지 않는 것](#2-양자화가-줄이는-것과-줄이지-않는-것)
3. [메모리 계산식과 effective bits per weight](#3-메모리-계산식과-effective-bits-per-weight)
4. [형식·알고리즘·런타임 이름 읽기](#4-형식알고리즘런타임-이름-읽기)
5. [Q2·Q3·Q4·Q5·Q6·Q8 선택법](#5-q2q3q4q5q6q8-선택법)
6. [GGUF 양자화 계열](#6-gguf-양자화-계열)
7. [`llama.cpp` 변환과 양자화](#7-llamacpp-변환과-양자화)
8. [캘리브레이션과 importance matrix](#8-캘리브레이션과-importance-matrix)
9. [AWQ·GPTQModel·AutoRound](#9-awqgptqmodelautoround)
10. [LLM Compressor와 compressed-tensors](#10-llm-compressor와-compressed-tensors)
11. [bitsandbytes·NF4·QLoRA](#11-bitsandbytesnf4qlora)
12. [torchao·Quanto·HQQ](#12-torchaoquantohqq)
13. [FP8 양자화](#13-fp8-양자화)
14. [FP4·MXFP4·NVFP4와 native low-bit](#14-fp4mxfp4nvfp4와-native-low-bit)
15. [EXL3와 EXL2 레거시](#15-exl3와-exl2-레거시)
16. [Apple Silicon과 MLX](#16-apple-silicon과-mlx)
17. [KV 캐시 양자화](#17-kv-캐시-양자화)
18. [MoE 모델 양자화](#18-moe-모델-양자화)
19. [비전·이미지·오디오·임베딩 모델](#19-비전이미지오디오임베딩-모델)
20. [CPU·엣지·ONNX·OpenVINO·Core ML](#20-cpu엣지onnxopenvinocore-ml)
21. [런타임·하드웨어 지원 매트릭스](#21-런타임하드웨어-지원-매트릭스)
22. [파라미터와 RAM·VRAM별 용량표](#22-파라미터와-ramvram별-용량표)
23. [Hugging Face 다운로드·검사·고정](#23-hugging-face-다운로드검사고정)
24. [품질·속도·메모리 벤치마크](#24-품질속도메모리-벤치마크)
25. [목적별 권장 워크플로](#25-목적별-권장-워크플로)
26. [문제 해결](#26-문제-해결)
27. [보안·공급망·재현성](#27-보안공급망재현성)
28. [주요 출처와 저장소](#28-주요-출처와-저장소)
29. [최종 권장안](#29-최종-권장안)
30. [갱신 및 사용상 주의](#30-갱신-및-사용상-주의)

---

## 1. 30초 선택표

아래 표는 “어떤 양자화가 이론상 가장 좋나”가 아니라, **현재 장비와 런타임에서 실패 확률이 낮은 첫 선택**을 정리한 것이다. “W4A16”은 가중치 4비트·활성화 16비트, “W8A8”은 가중치와 활성화 8비트를 뜻한다.

| 환경·목적 | 우선 시작점 | 대안 | 피해야 할 기본 가정 | 확인 항목 |
| --- | --- | --- | --- | --- |
| CPU·범용 PC | GGUF `Q4_K_M` | `Q5_K_M`, `Q6_K`, 메모리 부족 시 `Q3_K_M` | Q2가 가장 빠르다고 가정 | SIMD·BLAS backend, RAM 대역폭, thread·NUMA |
| NVIDIA 소비자 GPU·단일 사용자 | GGUF Q4/Q5 전량 GPU offload 또는 EXL3 | GPTQModel·AWQ W4A16 | 파일만 VRAM에 들어가면 된다고 가정 | 지원 architecture, kernel, context·KV |
| NVIDIA CUDA·고처리량 서버 | vLLM + `compressed-tensors` W4A16/FP8 | GPTQModel·AWQ·AutoRound export | 모든 4-bit가 동일 throughput을 낸다고 가정 | GPU compute capability, tensor parallel, batch |
| Ada·Hopper | 검증된 FP8 W8A8 또는 W4A16 | BF16 기준선 | FP8이면 항상 정확도 손실이 없다고 가정 | static/dynamic scale, outlier, kernel |
| Blackwell | NVFP4·MXFP4 native 경로 또는 FP8 | W4A16·BF16 | 명칭이 FP4인 모든 파일이 native kernel을 사용한다고 가정 | CC 10.0급 경로, runtime version, calibration |
| AMD GPU | GGUF ROCm/Vulkan 또는 AMD Quark 지원 형식 | vLLM의 현재 지원 형식 | CUDA용 AWQ/GPTQ 파일이 그대로 최적이라고 가정 | ROCm·Quark·kernel 지원표 |
| Apple Silicon | MLX 4-bit 또는 GGUF `Q4_K_M` | MLX 8-bit·mixed recipe, GGUF Q5/Q6 | 통합 메모리 전체를 모델에 사용 가능하다고 가정 | macOS·display 여유, wired memory, context |
| Intel CPU·GPU | OpenVINO INT4/INT8 또는 GGUF | AutoRound export·ONNX Runtime | NVIDIA kernel 성능표를 그대로 적용 | ISA, iGPU/NPU 지원, graph coverage |
| QLoRA 파인튜닝 | bitsandbytes NF4 + BF16 compute + double quant | torchao QAT·LLM Compressor QAT | 추론용 GGUF Q4를 그대로 학습한다고 가정 | optimizer·activation·gradient memory |
| VLM·OCR | LLM 본체 Q4, projector·vision encoder Q8/BF16 | 검증된 mixed quant | 모든 구성요소를 같은 Q2/Q3로 일괄 변환 | 별도 `mmproj`, visual token, OCR 정확도 |
| 이미지 생성 | DiT/UNet FP8·INT8·NF4 또는 검증된 GGUF, VAE 고정밀 | CPU offload·sequential offload | LLM Q4 품질 규칙을 VAE에 그대로 적용 | text encoder·VAE·ControlNet별 peak |
| ASR·TTS | Whisper Q5/Q8·CTranslate2 INT8·검증된 FP16/FP8 | ONNX INT8·MLX | audio encoder·codec·vocoder를 무조건 Q4 | WER·speaker similarity·artifact·실시간성 |
| 임베딩·reranker | FP16/BF16 또는 INT8/Q8 | 4-bit는 자체 retrieval 평가 후 | 작은 모델도 Q2가 이득이라고 가정 | Recall@k·nDCG·cosine drift |


### 1.1 메모리만 알고 있을 때

아래는 운영체제와 KV 캐시 여유를 남기는 보수적인 출발점이다. 모델 구조와 context에 따라 달라지므로 22장의 계산표를 함께 본다.

| 모델에 실질적으로 할당 가능한 메모리 | 범용 첫 선택 | 용량 우선 대안 | 품질 우선 대안 | 운영 메모 |
| --- | --- | --- | --- | --- |
| 4 GB | 3B Q4 | 7B Q2·Q3 실험 | 1–2B Q8 | 4K context·단일 슬롯부터 시작 |
| 8 GB | 7–8B Q4 | 14B Q2·Q3 | 3–4B Q8 | 전용 GPU는 display VRAM을 제외 |
| 12 GB | 14B Q4 또는 8B Q6/Q8 | 27B Q2 | 7–8B Q8 | VLM projector·KV 여유 확인 |
| 16 GB | 14B Q5/Q6 또는 27B Q3 | 32B Q2·Q3 | 8B BF16 | 32K context는 KV가 수 GB 추가될 수 있음 |
| 24 GB | 27–32B Q4 | 70B Q2 | 14B Q8 또는 27B Q5 | 24GB GPU는 20–22GB 이하 weights가 안전 |
| 32 GB | 32B Q5/Q6 | 70B Q3 | 14B BF16·27B Q8 | Apple은 OS·앱과 통합 메모리 공유 |
| 48 GB | 70B Q4 | 120B Q2·Q3 | 32B Q8 | 긴 context·다중 슬롯이면 64GB급 권장 |
| 64 GB | 70B Q5/Q6 | 120B Q3 | 32B BF16 | MoE는 active가 아니라 total weights로 계산 |
| 96 GB | 120B Q4/Q5 | 235B Q3 | 70B Q8 | 서버는 tensor parallel 통신까지 측정 |
| 128 GB | 120B Q6/Q8 또는 235B Q3 | 405B Q2 실험 | 70B BF16에 근접 | 실제 128GB 장치에서는 OS 여유 필수 |
| 192 GB | 235B Q4/Q5 | 405B Q3 | 120B Q8 | 멀티 GPU는 단순 합산과 다름 |
| 256–320 GB | 405B Q4 | 초대형 Q3 | 235B Q8 | 405B Q4는 256GB에 간신히 넣지 않는 편이 안전 |


### 1.2 즉시 판단 규칙

```text
Q4가 weights + KV + 15~25% 여유와 함께 들어감
  → Q4로 시작

Q4가 5~15% 정도 초과
  → context·KV cache·동시 슬롯을 먼저 줄임
  → 그래도 초과하면 Q3

Q3도 간신히 들어감
  → 한 단계 작은 모델의 Q4/Q5와 반드시 비교

Q2만 들어감
  → 기능 확인·초대형 모델 탐색용
  → 코드·수학·구조화 출력·도구 호출은 자동 검증 필수

FP8/INT4 파일이 있음
  → 현재 GPU·runtime에 해당 kernel이 있는지 먼저 확인
  → 없으면 작은 파일일 뿐 빠른 파일이 아닐 수 있음
```

---

## 2. 양자화가 줄이는 것과 줄이지 않는 것

### 2.1 양자화 대상

| 대상 | 기호 예 | 메모리 효과 | 속도 효과 | 대표 위험 |
| --- | --- | --- | --- | --- |
| 가중치 | W4, W8 | 상시 resident memory를 크게 줄임 | memory-bound 추론에서 유리할 수 있음 | 출력 품질·outlier·kernel 비호환 |
| 활성화 | A8, A4 | batch·prefill workspace 감소 가능 | native tensor core가 있으면 큼 | 동적 범위·outlier·calibration 민감 |
| KV 캐시 | KV8, KV4 | context·동시 슬롯 메모리 감소 | 메모리 대역폭 감소 가능 | 긴 문맥 회상·attention 품질 저하 |
| optimizer state | 8-bit optimizer | 파인튜닝 메모리 감소 | 학습 단계에 따라 다름 | 수렴·지원 optimizer 제한 |
| gradient | 저정밀 gradient | 학습 메모리 감소 | 하드웨어 지원 시 유리 | 수치 불안정 |
| embedding·output head | mixed precision | 모델에 따라 수백 MB~수 GB 감소 | 대체로 영향 작음 | 어휘 분포·logit 품질 민감 |
| vision/audio encoder·projector | Q8/BF16 유지가 흔함 | 부가 컴포넌트 메모리 감소 | 지원 kernel이 있을 때만 | OCR·미세 시각·음성 정보 손실 |
| VAE·vocoder·codec | FP16/BF16 우선 | 일부 감소 | 일관되지 않음 | 색 번짐·banding·음성 artifact |


### 2.2 줄어들지 않거나 별도로 늘어나는 메모리

가중치를 4비트로 만들어도 다음 항목은 자동으로 4분의 1이 되지 않는다.

- 토크나이저와 CPU-side metadata
- CUDA·ROCm·Metal graph와 workspace
- prefill activation
- dequantization buffer와 fused kernel scratch
- KV 캐시
- `bitsandbytes` CPU offload 시 FP32로 유지되는 CPU-side weights
- 모델 변환 중 동시에 존재하는 원본 BF16과 출력 quant 파일
- 멀티모달 이미지·오디오 입력 buffer
- batch·parallel slot마다 늘어나는 상태
- 운영체제·디스플레이·다른 프로세스 점유량

### 2.3 PTQ·QAT·native low-bit 구분

| 방식 | 언제 적용 | 대표 예 | 장점 | 주의점 |
| --- | --- | --- | --- | --- |
| PTQ·round-to-nearest | 학습 완료 후 | GGUF 기본 quant, RTN | 빠르고 단순 | 2–3bit에서 품질 손실이 커질 수 있음 |
| PTQ·캘리브레이션 | 학습 완료 후 representative data 사용 | AWQ, GPTQ, AutoRound, imatrix, SmoothQuant | 낮은 bit에서 품질 보존 개선 | 데이터 편향·누수·시간·RAM 요구 |
| QAT | 학습 중 fake quant·scale 학습 | torchao QAT, vendor QAT | 저비트에서 품질 회복 가능 | 학습 비용·구현 복잡도 |
| QLoRA | frozen 4-bit base + LoRA 학습 | NF4 + double quant | 적은 VRAM으로 adapter 학습 | full-weight QAT와 다름 |
| native low-bit | pretraining·QAT 단계부터 저비트 구조 | gpt-oss MXFP4, BitNet 계열 | 형식에 맞는 효율 가능 | 임의 BF16 모델의 Q4와 동등 비교 불가 |


### 2.4 “작은 파일”과 “빠른 추론”은 별개다

속도는 대략 다음의 최소값에 제한된다.

```text
throughput ≈ min(
  메모리에서 weights·KV를 읽는 속도,
  dequant + matrix multiply kernel 속도,
  CPU/GPU 연산량,
  PCIe·interconnect 전송,
  scheduler·batching 효율
)
```

CPU의 memory-bound token generation에서는 더 작은 GGUF가 빠른 경우가 많다. 반대로 GPU에서 해당 3-bit kernel이 덜 최적화되어 있으면 Q3보다 Q4가 빠를 수 있다. FP8·FP4도 장치가 native 경로를 지원하지 않으면 dequantization overhead 때문에 기대만큼 빨라지지 않는다.

---

## 3. 메모리 계산식과 effective bits per weight

### 3.1 원시 가중치 계산

파라미터 수가 `P`, 저장 bit 수가 `b`라면 이상적인 raw weight 크기는 다음과 같다.

```text
raw_weight_bytes = P × b / 8
raw_weight_GiB   = P × b / 8 / 1024³
```

예를 들어 8B 파라미터를 정확히 4bit로만 저장하면 이론상 약 3.73 GiB다. 그러나 실제 `Q4_K_M` 파일은 scale, block metadata, 고정밀 tensor 때문에 이보다 크다.

### 3.2 effective bits per weight

실제 파일은 nominal bit가 아니라 **effective bits per weight, bpw**로 비교하는 편이 정확하다.

```text
effective_bpw
≈ (quantized tensor bytes × 8) / quantized parameter count
```

전체 파일 기준 bpw는 tokenizer·metadata·alignment와 비양자화 tensor까지 포함하므로 모델별로 달라진다. 같은 `Q4_K_M`이라도 embedding 크기, vocabulary, tied weights, MoE expert 수와 converter 정책에 따라 파일 크기가 다를 수 있다.

### 3.3 총 추론 메모리

```text
M_total ≈ M_quant_weights
        + M_nonquantized_tensors
        + M_scales_and_metadata
        + M_KV_cache
        + M_activation_and_workspace
        + M_runtime
        + M_modality_components
        + M_parallel_slots
        + M_OS_headroom
```

실무용 빠른 추정치는 다음과 같다.

```text
CPU·Apple 단일 슬롯:
  M_practical ≈ model_file × 1.15~1.35 + KV + OS 여유

전용 GPU 전량 offload:
  VRAM_practical ≈ GPU-resident tensors + KV + workspace + 10~20% 여유
  system_RAM     ≈ model download·load staging + runtime + OS

부분 offload:
  system_RAM과 VRAM에 tensor가 나뉘며
  load 과정에서 일시적으로 중복 copy가 생길 수 있음
```

위 배수는 표준이 아니라 초기 예산을 위한 보수적 경험칙이다. 최종 값은 실행 중 peak RSS·VRAM으로 측정한다.

### 3.4 KV 캐시 계산

일반적인 decoder-only transformer의 KV cache는 다음처럼 근사할 수 있다.

```text
KV bytes
≈ slots
× context_tokens
× layers
× 2                  # key + value
× n_kv_heads
× head_dim
× bytes_per_element
```

GQA·MQA 모델은 `n_kv_heads`가 attention head 수보다 작아 KV가 줄어든다. MLA·hybrid attention·sliding window·state-space 계열은 위 식과 다르므로 모델 config와 runtime 로그를 우선한다.

Llama 3 8B와 유사하게 `32 layers × 8 KV heads × head_dim 128`인 예시에서 단일 슬롯의 이론적 KV는 다음과 같다.

| context | FP16/BF16 KV | Q8 KV 근사 | Q4 KV 근사 | 비고 |
| ---: | ---: | ---: | ---: | --- |
| 4,096 | 0.50 GiB | 0.25 GiB + overhead | 0.12 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |
| 8,192 | 1.00 GiB | 0.50 GiB + overhead | 0.25 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |
| 16,384 | 2.00 GiB | 1.00 GiB + overhead | 0.50 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |
| 32,768 | 4.00 GiB | 2.00 GiB + overhead | 1.00 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |
| 65,536 | 8.00 GiB | 4.00 GiB + overhead | 2.00 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |
| 131,072 | 16.00 GiB | 8.00 GiB + overhead | 4.00 GiB + overhead | 슬롯 수만큼 거의 선형 증가 |


scale·block metadata, alignment, runtime scratch가 추가되므로 실제 값은 더 크다. KV를 줄였더라도 긴 context의 prefill activation과 attention workspace는 별도로 증가할 수 있다.

### 3.5 변환 작업의 메모리

직접 quantize할 때는 최종 파일 크기가 아니라 원본 모델을 읽는 메모리를 계산한다.

```text
M_conversion_peak
≈ BF16/FP16 source weights
 + converter tensors
 + calibration activations 또는 Hessian/importance statistics
 + output buffer
 + Python·framework overhead
```

`llama.cpp` 공식 변환 예시도 source 모델을 고정밀 GGUF로 변환한 뒤 `llama-quantize`를 실행한다. 큰 모델은 변환용 RAM과 임시 디스크가 추론 장비보다 훨씬 많이 필요할 수 있다.

---

## 4. 형식·알고리즘·런타임 이름 읽기

양자화 관련 이름은 서로 다른 층위를 섞어 사용한다. 먼저 분류한다.

| 이름 | 정체 | 예 | 잘못된 해석 |
| --- | --- | --- | --- |
| 숫자 형식·dtype | 값 표현 방식 | INT8, FP8 E4M3, BF16, NVFP4 | dtype만 알면 calibration·packing·kernel까지 안다고 생각 |
| 양자화 scheme | W/A/KV bit와 granularity 조합 | W4A16, W8A8, per-group 128 | 같은 W4A16이면 파일·속도가 동일하다고 생각 |
| 알고리즘 | scale·rounding·중요도 결정법 | AWQ, GPTQ, SmoothQuant, AutoRound | 알고리즘 이름만으로 runtime 호환을 보장 |
| 컨테이너·저장 형식 | tensor와 metadata 저장 | GGUF, safetensors | `safetensors` 자체가 양자화라고 생각 |
| 배포 format | runtime이 읽는 packing/config | GPTQModel, compressed-tensors, EXL3, MLX | 다른 runtime이 그대로 읽는다고 생각 |
| 런타임·kernel | 실제 연산 구현 | llama.cpp, vLLM, ExLlamaV3, MLX | 파일이 지원되면 모든 GPU에서 같은 속도라고 생각 |
| provider recipe label | 배포자가 정한 mixed recipe | `UD-Q4_K_XL` 등 | 모든 저장소에서 동일한 표준이라고 생각 |


### 4.1 W4A16·W8A8 읽기

```text
W4A16 = weights 4-bit, activations usually FP16/BF16
W8A8  = weights 8-bit, activations 8-bit
W4A8  = weights 4-bit, activations 8-bit
KV8   = key/value cache 8-bit
```

실제 accumulators와 특정 layer의 정밀도는 별도일 수 있다. W4A16이 모든 tensor를 4bit로 저장한다는 뜻도 아니다.

### 4.2 granularity

- **per-tensor:** tensor 전체에 하나 또는 소수의 scale. 단순하지만 outlier에 민감하다.
- **per-channel:** output/input channel별 scale. 정확도는 좋아질 수 있지만 metadata가 늘어난다.
- **per-group:** 일정한 weight 묶음마다 scale. 4-bit LLM의 흔한 절충이다.
- **per-token·dynamic:** 실행 중 token·activation 범위에 맞춰 scale을 계산한다.
- **per-block·microscaling:** 작은 block마다 공유 exponent·scale을 사용한다.

### 4.3 symmetric·asymmetric

대칭 양자화는 zero-point를 보통 0으로 두고 양·음 범위를 대칭으로 표현한다. 비대칭 양자화는 zero-point를 사용해 한쪽으로 치우친 분포를 더 잘 표현할 수 있지만 kernel·format 지원이 달라질 수 있다.

일반적인 affine 복원식은 다음과 같다.

```text
x_real ≈ scale × (x_quantized - zero_point)
```

### 4.4 파일명만으로 추정하지 말 것

다음 항목을 model card와 `quantization_config`에서 확인한다.

- 정확한 quant method와 bit
- group size·block size
- symmetric/asymmetric·zero point
- activation quant 여부
- 제외된 module
- calibration dataset·sequence length
- tokenizer·chat template revision
- 권장 runtime·최소 version
- tensor parallel·multimodal 지원

---

## 5. Q2·Q3·Q4·Q5·Q6·Q8 선택법

이 절의 Q 표기는 주로 GGUF 계열을 설명한다. AWQ·GPTQ·NF4·FP8에 같은 규칙을 기계적으로 적용하지 않는다.

| 수준 | 대표 위치 | 권장 용도 | 취약해지기 쉬운 작업 | 일반 권고 |
| --- | --- | --- | --- | --- |
| Q2·IQ2 | 약 2.4–3.2 effective bpw | 초대형 모델 탐색, 제한된 RAM, 기능 확인 | 코드 수정, 수학, JSON schema, 한국어 미세 표현, 장문 회상 | 실험용. 작은 Q4/Q5와 비교 |
| Q3·IQ3 | 약 3.3–4.3 bpw | Q4가 들어가지 않을 때, model capacity 우선 | 정밀 도구 호출, 인용, 어려운 reasoning | 용량 절충. 자동 검증 필수 |
| Q4·IQ4 | 약 4.5–4.9 bpw | 대부분의 로컬 추론 기본값 | 극도로 정밀한 수치·희귀 언어는 차이 가능 | `Q4_K_M` 또는 검증된 mixed Q4부터 |
| Q5 | 약 5.6–5.7 bpw | 코딩·수학·RAG·구조화 출력 품질 여유 | 메모리·대역폭 증가 | Q4 대비 실제 이득을 평가 |
| Q6 | 약 6.6 bpw | 고품질 로컬 기준선, 메모리 여유 | 파일이 커져 GPU offload 경계 넘음 | Q8보다 효율적인 고정밀 절충 |
| Q8 | 약 8.5 bpw | 고정밀 추론·양자화 기준선 | 메모리 절약이 작고 항상 빠르지 않음 | BF16과 비교하는 near-baseline |
| BF16/FP16 | 16 bpw 전후 | 품질 기준선, 파인튜닝, calibration source | 큰 메모리·대역폭 | 양자화 손실을 판단할 기준으로 보존 |


### 5.1 Q2보다 작은 모델 Q4가 나은 경우

- instruction following과 tool schema가 중요할 때
- 정확한 코드 patch와 테스트 통과가 목표일 때
- 수학·과학 계산에서 작은 오류가 치명적일 때
- 한국어·희귀 언어·전문 용어의 표현이 중요할 때
- 반복 가능한 production 결과가 필요할 때
- 모델 구조가 저비트에 특히 민감할 때

### 5.2 더 큰 Q3가 유리할 수 있는 경우

- 모델 capacity와 world knowledge가 양자화 손실보다 중요한 open-ended 작업
- context와 KV를 줄여도 더 큰 architecture가 task에 큰 차이를 만들 때
- 출력에 검색·계산·테스트 같은 외부 검증기를 붙였을 때
- 동일 계열의 작은 Q4와 실제 평가셋에서 우위가 확인됐을 때

### 5.3 Q4 family 안에서도 다르다

`Q4_0`, `Q4_K_S`, `Q4_K_M`, `IQ4_XS`, `IQ4_NL`은 명목상 4bit 계열이지만 packing, codebook, 중요 tensor 처리, effective bpw가 다르다. 런타임 지원과 CPU/GPU kernel 속도도 다르므로 파일 크기 하나로 순위를 정하지 않는다.

### 5.4 품질 우선 순서

일반적인 비교 순서는 다음처럼 잡을 수 있다.

```text
BF16/FP16 기준선
  → Q8
  → Q6
  → Q5
  → Q4
  → Q3
  → Q2
```

하지만 각 단계의 품질이 단조롭게 감소한다고 보장할 수 없다. quantizer, calibration, tensor별 mixed precision과 모델 구조가 다르면 더 작은 파일이 특정 task에서 우연히 더 나은 결과를 낼 수도 있다. 여러 seed와 충분한 평가 표본으로 비교한다.

---

## 6. GGUF 양자화 계열

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)는 모델 tensor와 metadata를 담는 컨테이너이며, [`llama.cpp`](https://github.com/ggml-org/llama.cpp)를 비롯한 로컬 런타임에서 널리 사용된다. GGUF는 “하나의 양자화 알고리즘”이 아니라 여러 quantized tensor type을 담을 수 있는 형식이다.

### 6.1 대표 family

| family | 대표 태그 | 특징 | 권장 위치 |
| --- | --- | --- | --- |
| legacy quant | `Q4_0`, `Q5_0`, `Q8_0` | 단순하고 광범위한 kernel 지원 | 호환성·특정 backend 기준선 |
| K-quants | `Q2_K`~`Q6_K`, `Q4_K_M` | block-wise scale과 mixed tensor policy | 범용 CPU·GPU offload 기본 |
| IQ quants | `IQ2_XS`, `IQ3_M`, `IQ4_XS`, `IQ4_NL` | importance-aware·codebook 계열의 저비트 선택지 | 저용량·품질 효율 비교 |
| TQ·ternary 계열 | runtime별 태그 | 극저비트·ternary 실험 | 지원 model·backend가 명확할 때 |
| 고정밀 | `F16`, `BF16`, `F32` | 원본·기준선·requantization source | 변환·평가·고정밀 추론 |
| provider mixed recipe | `UD-Q3_K_XL`, `UD-Q4_K_XL` 등 | 배포자가 tensor별 정밀도를 조합한 label | model card의 정확한 recipe를 확인한 후 |


### 6.2 `S`·`M`·`L`과 provider label

GGUF K-quant의 `S`, `M`, `L`은 일반적으로 파일 크기와 tensor별 정밀도 정책이 다른 변형을 가리킨다. 그러나 모든 provider가 같은 추가 label을 쓰는 것은 아니다. 특히 `UD-*` 같은 이름은 GGUF 사양의 단일 원자적 tensor type이라기보다 배포자의 mixed-quant recipe인 경우가 많다.

따라서 다음처럼 다운로드한다.

```text
잘못된 방식:
  “Q4” 문자열만 보고 아무 파일이나 선택

권장 방식:
  model card의 quant table
  + 실제 shard 합계
  + quantization notes
  + 권장 llama.cpp commit
  + calibration/imatrix 여부
  확인 후 선택
```

### 6.3 공식 Llama 3.1 8B 예시의 effective bpw

`llama.cpp`의 공식 quantize 문서가 제시한 Llama 3.1 8B 예시는 같은 모델에서 nominal tag와 실제 bpw 차이를 보여준다. 아래 값은 **그 모델과 당시 변환 조건의 예시**이며 모든 모델의 고정 비율이 아니다.

| quant | effective bpw | 예시 파일 크기 |
| --- | ---: | ---: |
| IQ1_S | 2.0042 | 1.87 GiB |
| IQ1_M | 2.1460 | 2.01 GiB |
| IQ2_XXS | 2.3824 | 2.23 GiB |
| IQ2_XS | 2.5882 | 2.42 GiB |
| IQ2_S | 2.7403 | 2.56 GiB |
| IQ2_M | 2.9294 | 2.74 GiB |
| Q2_K_S | 2.9697 | 2.78 GiB |
| Q2_K | 3.1593 | 2.95 GiB |
| IQ3_XXS | 3.2548 | 3.04 GiB |
| IQ3_XS | 3.4977 | 3.27 GiB |
| IQ3_S | 3.6606 | 3.42 GiB |
| IQ3_M | 3.7628 | 3.52 GiB |
| Q3_K_S | 3.6429 | 3.41 GiB |
| Q3_K_M | 3.9960 | 3.74 GiB |
| Q3_K_L | 4.2979 | 4.02 GiB |
| IQ4_XS | 4.4597 | 4.17 GiB |
| IQ4_NL | 4.6818 | 4.38 GiB |
| Q4_K_S | 4.6672 | 4.36 GiB |
| Q4_K_M | 4.8944 | 4.58 GiB |
| Q5_K_S | 5.5704 | 5.21 GiB |
| Q5_K_M | 5.7036 | 5.33 GiB |
| Q6_K | 6.5633 | 6.14 GiB |
| Q8_0 | 8.5008 | 7.95 GiB |
| F16 | 16.0005 | 14.96 GiB |


### 6.4 GGUF 장점

- CPU·CUDA·Metal·ROCm·Vulkan 등 다양한 backend의 로컬 배포
- 단일 파일 또는 shard로 가중치와 metadata 관리
- CPU/GPU layer offload 조절
- 다양한 Q2–Q8·IQ quant
- KV cache quant와 긴 context 조절
- CLI·server·benchmark 도구가 같은 생태계에 있음

### 6.5 GGUF 한계

- vLLM·Transformers의 native safetensors 경로와 kernel이 다르다.
- GGUF를 Transformers에서 학습용으로 불러오면 고정밀로 dequantize되어 메모리 이점이 사라질 수 있다.
- 새 architecture·multimodal projector 지원은 converter와 runtime version에 의존한다.
- 모델 provider가 변환한 GGUF의 tokenizer·chat template·special token이 원본과 다를 수 있다.
- Q2/Q3의 품질은 quantizer와 importance data에 크게 좌우된다.

---

## 7. `llama.cpp` 변환과 양자화

공식 절차는 Hugging Face 원본을 고정밀 GGUF로 변환한 뒤 `llama-quantize`를 실행하는 두 단계다. 원본 revision과 converter commit을 고정한다.

### 7.1 빌드

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

git checkout <verified-commit>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CPU 기본 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# NVIDIA CUDA 빌드는 configure 단계에 다음 옵션 사용
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

Apple Metal·AMD ROCm·Vulkan 옵션은 현재 공식 build 문서를 따른다. 이미 배포된 binary를 사용할 때도 버전과 commit을 기록한다.

### 7.2 원본을 BF16 GGUF로 변환

공식 문서의 현재 원격 변환 예시는 다음 형태다.

```bash
python convert_hf_to_gguf.py \
  --outfile model-bf16.gguf \
  --outtype bf16 \
  --remote OWNER/MODEL
```

로컬에 revision을 고정해 받은 모델은 경로를 직접 지정한다.

```bash
python convert_hf_to_gguf.py \
  /models/OWNER--MODEL/snapshots/<commit> \
  --outfile model-bf16.gguf \
  --outtype bf16
```

### 7.3 Q4·Q5·Q8 생성

```bash
./build/bin/llama-quantize \
  model-bf16.gguf \
  model-Q4_K_M.gguf \
  Q4_K_M

./build/bin/llama-quantize \
  model-bf16.gguf \
  model-Q5_K_M.gguf \
  Q5_K_M

./build/bin/llama-quantize \
  model-bf16.gguf \
  model-Q8_0.gguf \
  Q8_0
```

지원 type과 현재 옵션은 항상 확인한다.

```bash
./build/bin/llama-quantize --help
```

### 7.4 대형 모델과 shard

대형 모델은 입출력 파일 시스템과 RAM이 병목이 된다.

- source와 output을 서로 다른 빠른 NVMe에 둘 수 있다.
- 충분한 여유 디스크를 확보한다. BF16 source와 여러 quant가 동시에 존재한다.
- shard를 유지해야 하면 현재 `--keep-split` 지원과 runtime 호환을 확인한다.
- 네트워크 파일 시스템보다 로컬 scratch가 안정적일 수 있다.
- 변환 노드와 추론 노드를 분리할 수 있다.

`llama.cpp` 공식 문서의 Llama 3.1 예시에서는 8B·70B·405B 원본에서 Q4_K_M으로 변환할 때 source와 output 모두 상당한 디스크가 필요함을 보여준다. 직접 quantize할 계획이라면 **최종 Q4 파일 크기만큼의 RAM**이 아니라 **BF16 source 크기와 변환 overhead**를 기준으로 노드를 선택한다.

### 7.5 vision/audio projector

멀티모달 모델은 `--mmproj` 변환이 별도일 수 있다.

```bash
python convert_hf_to_gguf.py \
  --mmproj \
  --outfile mmproj-model-Q8_0.gguf \
  --outtype q8_0 \
  --remote OWNER/MODEL
```

일반적으로 projector·encoder는 본체보다 작고 품질 민감도가 높으므로 Q8 또는 BF16을 우선한다. 본체 Q4 + projector Q8/BF16 조합을 먼저 평가한다.

### 7.6 requantization 금지에 가까운 원칙

이미 Q4인 파일을 다시 Q3·Q2로 양자화하면 원본 BF16에서 직접 만든 Q3·Q2보다 오차가 누적될 수 있다. `llama-quantize`의 `--allow-requantize`는 가능하다는 뜻이지 권장된다는 뜻이 아니다.

```text
권장:
  BF16/FP16 source → Q8/Q6/Q5/Q4/Q3/Q2 각각 직접 생성

비권장:
  Q8 → Q4
  Q4 → Q3
  Q3 → Q2
```

원본 source가 사라졌거나 긴급 호환성 테스트인 경우에만 requantize하고, 결과에 그 사실을 명시한다.

### 7.7 output·embedding tensor 처리

일부 모델은 output head·token embedding이 양자화 품질에 민감하다. 현재 `llama-quantize`에는 output tensor를 남기거나 특정 tensor type을 지정하는 옵션이 있다. 무조건 더 작게 만들기보다 다음을 평가한다.

- rare token·한국어·코드 토큰 logit
- structured output의 정확성
- perplexity와 downstream task
- tied embedding 여부
- output tensor가 전체 파일에서 차지하는 비율

---

## 8. 캘리브레이션과 importance matrix

양자화 품질은 “몇 bit인가”뿐 아니라 어떤 데이터로 scale·rounding·중요도를 결정했는지에 좌우된다.

### 8.1 representative dataset 구성

| 평가 목적 | 캘리브레이션에 포함할 것 | 피할 것 |
| --- | --- | --- |
| 한국어 범용 | 한국어 설명·대화·문서·코드 혼합 | 영어 위키만으로 구성 |
| 프로그래밍 | 여러 언어의 코드·diff·테스트·도구 JSON | 정답 benchmark를 그대로 포함 |
| 수학·과학 | 수식·LaTeX·단위·표·긴 reasoning 형식 | 평가 정답 누수 |
| RAG·문서 | 인용·근거·긴 passage·구조화 출력 | 개인·기밀 원문 |
| VLM | 지원될 경우 이미지·OCR·표·UI의 대표 분포 | 텍스트 module만 quantized인데 전체 VLM calibrated라고 주장 |
| MoE | 다양한 도메인과 prompt로 가능한 많은 expert 활성화 | 짧고 단일 주제의 소량 데이터 |


캘리브레이션 세트와 평가 세트는 분리한다. 공개 배포용 quant를 만들 때 회사 문서·고객 데이터·비공개 코드가 statistics나 artifact에 포함되지 않도록 한다.

### 8.2 `llama-imatrix`

`llama.cpp`의 importance matrix 도구는 calibration text에서 tensor 중요도 통계를 생성하고, 일부 IQ·K quant의 품질을 개선하는 데 사용할 수 있다.

```bash
./build/bin/llama-imatrix \
  -m model-bf16.gguf \
  -f calibration-data.txt \
  -ngl 99 \
  --output-frequency 10 \
  -o imatrix.gguf
```

생성한 matrix로 quantize한다.

```bash
./build/bin/llama-quantize \
  --imatrix imatrix.gguf \
  model-bf16.gguf \
  model-IQ3_M.gguf \
  IQ3_M
```

통계를 확인한다.

```bash
./build/bin/llama-imatrix \
  --in-file imatrix.gguf \
  --show-statistics
```

여러 실행 결과를 합칠 때는 현재 도구의 multiple `--in-file` 사용법을 공식 README에서 확인한다. chat template·special token이 중요한 모델은 `--parse-special` 같은 현재 옵션을 검토한다.

### 8.3 calibration 길이와 분포

하나의 권장 샘플 수가 모든 모델에 맞지는 않는다. 다음 기준으로 충분성을 판단한다.

- 자주 쓰는 언어·도메인·출력 형식이 모두 등장하는가
- 긴 문맥의 위치 분포가 포함되는가
- tool call·JSON·코드 fence 같은 특수 token이 포함되는가
- MoE의 expert가 충분히 활성화되는가
- scale·clipping statistics가 샘플 추가 후 안정되는가
- 별도 평가셋의 metric이 더 이상 의미 있게 개선되지 않는가

### 8.4 dynamic quant와 static quant

- **dynamic activation quant:** 실행 시 activation scale을 계산한다. calibration 부담이 작지만 runtime overhead가 있다.
- **static activation quant:** calibration에서 scale을 고정한다. 빠른 kernel에 유리할 수 있지만 domain shift에 민감하다.
- **weight-only:** activation은 FP16/BF16로 유지하므로 대체로 안정적이고 배포가 단순하다.

### 8.5 캘리브레이션 manifest

```yaml
quantization:
  method: llama.cpp-imatrix
  source_model: OWNER/MODEL
  source_revision: <commit-sha>
  quantizer_commit: <commit-sha>
  output_type: Q4_K_M
  calibration:
    dataset_id: internal-quant-cal-v3
    dataset_sha256: <sha256>
    languages: [ko, en]
    domains: [chat, code, math, rag]
    samples: <count>
    max_tokens: <value>
    contains_private_data: false
  evaluation:
    suite_revision: <git-sha>
```

---

## 9. AWQ·GPTQModel·AutoRound

이 세 계열은 주로 CUDA GPU에서 weight-only 4-bit 배포에 사용된다. 같은 4-bit라도 calibration, packing, kernel과 지원 architecture가 다르다.

### 9.1 AWQ

[AWQ](https://huggingface.co/docs/transformers/quantization/awq)는 activation 통계를 이용해 중요한 weight를 보호하는 weight-only PTQ 계열이다. 흔한 설정은 W4A16, group size 128이다.

```json
{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 128,
  "zero_point": true,
  "version": "gemm"
}
```

장점:

- 4-bit weight-only에서 검증된 생태계
- Transformers·vLLM 등 여러 runtime 경로
- activation을 FP16/BF16로 유지해 배포가 비교적 안정적

주의:

- 모든 architecture와 multimodal module이 지원되는 것은 아니다.
- group size와 packing이 kernel 기대와 맞아야 한다.
- 과거 AutoAWQ 프로젝트는 보관·유지보수 상태가 바뀌었으므로 신규 파이프라인은 현재 유지되는 공식 runtime·quantizer 경로를 우선한다.
- AWQ 파일을 단순히 다른 4-bit loader로 읽을 수 있다고 가정하지 않는다.

### 9.2 GPTQModel

[Transformers GPTQ 문서](https://huggingface.co/docs/transformers/quantization/gptq)는 현재 GPT-QModel을 사용하며, AutoGPTQ는 더 이상 Transformers의 기본 지원 경로가 아니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "OWNER/MODEL"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quant_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
)
```

실무에서는 generic dataset 문자열 대신 task와 언어를 반영한 calibration dataset을 사용한다.

장점:

- row-wise error reconstruction 기반의 성숙한 4-bit PTQ
- GPTQModel과 Marlin 등 현재 kernel 경로
- 여러 bit·group 설정 비교 가능

주의:

- asymmetric·symmetric 설정과 legacy AutoGPTQ packing의 호환성이 다를 수 있다.
- Marlin은 실행 kernel이며 quantizer 자체가 아니다.
- calibration 시간이 길고 GPU/RAM 요구가 크다.
- 최신 architecture의 지원은 GPTQModel·runtime 버전을 확인한다.

### 9.3 AutoRound

[Intel AutoRound](https://github.com/intel/auto-round)는 sign-gradient 기반 rounding 최적화로 LLM·VLM의 2–4bit 및 여러 export format을 지원하는 활발한 프로젝트다. 현재 저장소의 CLI와 recipe를 기준으로 사용한다.

```bash
python -m pip install -U auto-round

auto-round \
  --model Qwen/Qwen3-0.6B \
  --scheme W4A16 \
  --format auto_round \
  --output_dir ./Qwen3-0.6B-W4A16
```

현재 도구는 version에 따라 `auto_round`, AWQ·GPTQ 호환, `llm_compressor`, GGUF 등의 export를 제공한다. 정확한 `--scheme`·`--format` 목록은 설치된 버전의 `--help`로 확인한다.

```bash
auto-round --help
```

장점:

- 여러 deployment format으로 export 가능한 통합 workflow
- 2·3·4·8bit와 FP8·FP4 계열의 확장
- CPU·Intel·CUDA 등 다양한 quantization 환경을 고려
- VLM text module과 최신 model 지원이 빠르게 추가됨

주의:

- mixed scheme quantization은 BF16 모델 크기의 약 1배를 넘는 변환 RAM을 요구할 수 있다.
- VLM의 non-text module quant는 version별로 제한적일 수 있다.
- export format이 같아도 quantizer recipe가 다르므로 model card에 기록한다.

### 9.4 비교

| 항목 | AWQ | GPTQModel | AutoRound |
| --- | --- | --- | --- |
| 대표 목적 | W4A16 inference | W4A16 inference | 다중 scheme·다중 export |
| calibration | activation-aware | error reconstruction | rounding optimization |
| 장점 | 성숙·빠른 kernel | 정밀한 PTQ·광범위 생태계 | 유연한 format·최신 연구 반영 |
| 첫 선택 | 이미 runtime 공식 지원 모델이 있을 때 | GPTQModel/Marlin 경로가 검증됐을 때 | 직접 quantize·여러 runtime export가 필요할 때 |
| 주요 위험 | legacy tool 유지보수·packing | 시간·호환성·legacy AutoGPTQ | 버전 변화·변환 자원·실험 기능 |


### 9.5 선택 규칙

```text
배포 runtime에 공식 quantized checkpoint가 있음
  → 그 format을 우선 사용

직접 4-bit CUDA checkpoint를 만들고 vLLM/Transformers로 배포
  → LLM Compressor 또는 AutoRound를 우선 검토
  → architecture별 GPTQModel/AWQ 지원 비교

단일 소비자 NVIDIA GPU에서 최대한 모델을 넣음
  → EXL3·GPTQ/Marlin·AWQ·GGUF를 같은 task와 context로 비교
```

---

## 10. LLM Compressor와 compressed-tensors

[vLLM Project의 LLM Compressor](https://github.com/vllm-project/llm-compressor)는 quantization·sparsity recipe를 적용하고 [`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors) 형식으로 저장해 vLLM에서 실행하는 주요 경로다.

### 10.1 주요 scheme

현재 공식 문서가 다루는 대표 scheme은 다음과 같다. 지원 compute capability와 kernel은 버전별로 확인한다.

| scheme | 의미 | 대표 hardware 시작점 | 용도 |
| --- | --- | --- | --- |
| W4A16 | 4-bit weights·16-bit activations | Ampere급 이상 경로가 일반적 | 메모리 절감·CUDA inference |
| W8A8 INT8 | 8-bit weights·activations | Turing급 이상 경로 | 높은 처리량·상대적 안정성 |
| W8A8 FP8 | FP8 weights·activations | Ada·Hopper·일부 AMD | native FP8 tensor core 활용 |
| W4A8 | 4-bit weights·8-bit activations | 지원 kernel 필요 | weight와 activation 동시 절감 |
| W4AFP8 | 4-bit weights·FP8 activations | Ada/Hopper 계열 | 저메모리와 FP8 연산 결합 |
| NVFP4·MXFP4 | microscaling 4-bit floating formats | Blackwell급 native 경로 | 최신 FP4 배포 |
| FP8 KV cache | attention K/V cache FP8 | runtime·model 지원 필요 | 긴 context·동시성 절감 |


### 10.2 지원 알고리즘

프로젝트는 version에 따라 다음 계열을 제공한다.

- round-to-nearest
- GPTQ
- AWQ
- SmoothQuant
- SparseGPT
- SpinQuant
- QuIP 계열
- AutoRound 통합
- FP8 KV cache

모든 조합이 모든 모델·GPU에서 동작하는 것은 아니다. recipe 문서와 vLLM 지원표를 함께 확인한다.

### 10.3 recipe 개념

```yaml
# 개념 예시. 실제 schema는 설치한 버전의 공식 recipe를 사용한다.
quant_stage:
  quant_method: GPTQ
  targets: Linear
  scheme: W4A16
  group_size: 128
  ignore:
    - lm_head
```

### 10.4 장점

- quantization config가 model artifact에 포함됨
- vLLM의 고처리량 serving과 연결
- weight-only·activation·KV·sparsity recipe를 한 계열에서 관리
- Hugging Face safetensors와 현대적인 배포 workflow

### 10.5 주의

- vLLM의 quantization 지원표는 GPU 세대·CPU·AMD·Intel별로 다르다.
- attention·KV의 per-head quant 같은 기능은 experimental일 수 있다.
- activation quant는 calibration 분포에 민감하다.
- tensor parallel에서 quantized shard·scale 배치가 지원되는지 확인한다.
- quantized checkpoint 생성이 성공해도 실제 target kernel로 실행되는지 로그와 profiler로 확인한다.

### 10.6 production checklist

```text
[ ] vLLM·LLM Compressor·compressed-tensors version 고정
[ ] GPU compute capability 확인
[ ] target architecture 공식 지원 확인
[ ] calibration dataset hash 기록
[ ] quantization_config 저장
[ ] BF16 baseline과 정확도 비교
[ ] 실제 kernel 이름·fallback 여부 확인
[ ] batch 1과 production batch 모두 benchmark
[ ] tensor parallel·KV cache·LoRA 동시 사용 검증
```

---

## 11. bitsandbytes·NF4·QLoRA

[bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)는 Transformers에서 8-bit `LLM.int8()`과 4-bit QLoRA loading을 제공한다. 추론에도 사용할 수 있지만 특히 adapter 파인튜닝에서 널리 쓰인다.

### 11.1 4-bit NF4 loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "OWNER/MODEL"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)

print(model.get_memory_footprint())
```

### 11.2 NF4·FP4

- **NF4:** 정규분포를 가정한 4-bit codebook으로, 4-bit base model의 학습에 권장되는 선택이다.
- **FP4:** 일반 4-bit floating representation. 모델·kernel·학습 조건에 따라 비교한다.
- **double quant:** quantization constant 자체를 다시 quantize해 약 0.4 bit/parameter를 추가로 절약할 수 있다.

### 11.3 QLoRA와 추론 quant의 차이

```text
QLoRA:
  frozen 4-bit base weights
  + BF16/FP16 compute
  + trainable LoRA adapters
  + optimizer·gradient·activation

GGUF Q4 inference:
  quantized weights를 inference kernel로 실행
  학습 graph·optimizer가 없음
```

GGUF Q4 파일이 5GB라는 이유로 QLoRA도 5GB VRAM에서 된다고 계산하면 안 된다. 학습에는 activation, gradient checkpointing, optimizer, LoRA states와 batch가 추가된다.

### 11.4 8-bit `LLM.int8()`

8-bit loading은 weight memory를 대략 절반 수준으로 줄이는 출발점이 될 수 있으며 outlier handling을 포함한다. 4-bit보다 품질 위험이 낮지만 메모리 절감도 작다.

### 11.5 CPU offload 주의

bitsandbytes의 CPU offload는 CPU-side weights를 FP32로 유지할 수 있다. GPU VRAM은 줄어도 시스템 RAM 요구가 예상보다 커질 수 있다.

```text
GPU VRAM만 확인 → OOM 또는 swap
시스템 RAM + PCIe throughput + pinned memory까지 확인
```

### 11.6 hardware

공식 문서의 현재 지원은 NVIDIA CUDA, Intel XPU, Intel Gaudi, CPU 등으로 확장되어 있다. 세부 최소 GPU 세대와 backend 상태가 바뀌므로 설치 버전의 지원표를 따른다. NVIDIA에서는 NF4/FP4와 `LLM.int8()`의 최소 세대 조건이 다르다.

### 11.7 언제 선택할까

- LoRA·QLoRA 학습이 목적일 때
- Transformers API에서 빠르게 4/8-bit를 시험할 때
- GGUF 변환 없이 원본 safetensors를 loading할 때
- 모델 전체를 하나의 inference artifact로 최적 배포하기보다 연구·개발 편의가 중요할 때

고처리량 serving에서는 vLLM·Marlin·compressed-tensors·TensorRT 계열과 별도로 benchmark한다.

---

## 12. torchao·Quanto·HQQ

이 절의 도구는 PyTorch·Transformers 생태계에서 유연하게 quantization을 적용할 때 유용하다. 지원 matrix가 빠르게 변하므로 version과 architecture를 고정한다.

### 12.1 torchao

[torchao](https://docs.pytorch.org/ao/stable/)는 PyTorch-native quantization·sparsity 도구로 다음 범주를 다룬다.

- INT4·INT8 weight-only
- dynamic activation + weight quantization
- static/export quantization
- FP8·microscaling·FP4 계열
- QAT
- low-bit optimizer

개념적 선택:

```text
가장 단순한 메모리 절감
  → weight-only INT4/INT8

native FP8 hardware의 throughput
  → dynamic/static FP8 W8A8

모바일·export graph
  → export quantization workflow

정확도 회복이 필요한 저비트
  → QAT
```

현재 group size, supported module, CUDA/ROCm generation은 release별로 다르므로 API 예제를 복사하기 전에 해당 version 문서를 확인한다.

### 12.2 Optimum Quanto

[Quanto](https://huggingface.co/docs/transformers/quantization/quanto)는 `torch.nn.Linear` 기반 여러 modality 모델에 weight quantization을 적용할 수 있다. Transformers integration은 주로 weight quantization에 초점을 두며 activation quant·calibration·QAT는 Quanto 라이브러리 직접 사용이 필요할 수 있다.

```python
from transformers import AutoModelForCausalLM, QuantoConfig

quant_config = QuantoConfig(weights="int4")
model = AutoModelForCausalLM.from_pretrained(
    "OWNER/MODEL",
    device_map="auto",
    quantization_config=quant_config,
)
```

### 12.3 HQQ

[HQQ](https://huggingface.co/docs/transformers/quantization/hqq)는 calibration 없이도 빠르게 low-bit weights를 생성하는 연구·배포 옵션이다. kernel과 supported backend, serialization 형식을 확인하며, production 기본값으로 채택하기 전에 GPTQ/AWQ/GGUF와 task-level 비교한다.

### 12.4 언제 이 계열을 선택할까

- PyTorch module을 직접 제어해야 할 때
- 새 architecture의 빠른 prototype이 필요할 때
- QAT·sparsity·low-bit optimizer를 같은 연구 코드에서 다룰 때
- text 외 modality의 `Linear` module을 선택적으로 quantize할 때

### 12.5 공통 주의

- Python에서 quantized layer가 생성됐다고 target GPU의 fused kernel을 쓴다는 뜻은 아니다.
- `torch.compile`·FlashAttention·tensor parallel·LoRA와 조합 호환성을 따로 확인한다.
- 저장한 artifact를 어떤 runtime이 다시 읽을 수 있는지 명시한다.
- eager fallback이 생기면 메모리는 줄어도 latency가 악화될 수 있다.

---

## 13. FP8 양자화

FP8은 대표적으로 E4M3·E5M2 형식을 사용하며, INT8과 달리 exponent를 갖는다. weight-only FP8과 W8A8 FP8은 서로 다르다.

### 13.1 E4M3·E5M2

| 형식 | 일반 성격 | 대표 위치 | 주의 |
| --- | --- | --- | --- |
| E4M3 | 정밀도 비중이 높고 범위가 상대적으로 좁음 | forward weights·activations | outlier·scale 관리 필요 |
| E5M2 | 표현 범위가 넓고 mantissa가 짧음 | gradient·범위가 큰 값 | 정밀도 손실 가능 |


실제 runtime은 tensor별 또는 block별 scale을 사용한다. 파일에 FP8이라고 적혀 있어도 scale granularity와 accumulation dtype이 다르면 품질·속도가 달라진다.

### 13.2 FP8이 유리한 환경

- Ada·Hopper·후속 GPU에서 native FP8 tensor core를 활용할 때
- large batch·prefill·throughput이 중요할 때
- BF16보다 weight와 activation memory를 줄이고 싶을 때
- vLLM·TensorRT·LLM Compressor·torchao 등 target runtime이 해당 scheme을 공식 지원할 때

### 13.3 weight-only FP8과 W8A8 FP8

```text
FP8 weight-only:
  weights만 8-bit float
  activations는 BF16/FP16
  메모리 절감은 있으나 activation bandwidth는 유지

FP8 W8A8:
  weights와 activations 모두 FP8
  native kernel이 있으면 throughput 이점이 큼
  calibration·dynamic scale·outlier 처리 중요
```

### 13.4 static·dynamic FP8

- **static:** calibration에서 scale을 고정. 안정된 production 분포와 빠른 kernel에 적합할 수 있다.
- **dynamic:** 실행 중 scale을 계산. domain 변화에 유연하지만 scale 계산 overhead가 있다.

### 13.5 정확도 검증

FP8은 4-bit보다 보통 안전하다고 알려져 있어도 다음 task를 별도 평가한다.

- 긴 prefill과 긴 generation
- code·math exactness
- multilingual·한국어
- logit 기반 ranking·reranking
- VLM OCR·작은 글자
- MoE router·expert activation
- tool-call JSON과 constrained decoding

### 13.6 fallback 확인

GPU가 FP8 storage를 읽을 수 있어도 모든 연산이 native FP8은 아닐 수 있다. profiler·runtime log에서 다음을 확인한다.

- 실제 kernel dtype
- BF16 dequant/cast 여부
- unsupported layer fallback
- tensor parallel communication dtype
- KV cache dtype

---

## 14. FP4·MXFP4·NVFP4와 native low-bit

4-bit floating format은 단순 INT4와 다르다. 작은 block마다 scale·shared exponent를 사용하는 microscaling 형식이 포함된다.

### 14.1 용어

| 형식 | 핵심 | 대표 hardware·runtime | 주의 |
| --- | --- | --- | --- |
| MXFP4 | OCP microscaling 계열 FP4 | Blackwell native 경로·일부 software fallback | 모든 GPU에서 native FP4 속도를 내지 않음 |
| NVFP4 | NVIDIA Blackwell 최적화 FP4 scheme | Blackwell·TensorRT/LLM Compressor/FP-Quant 계열 | calibration·block scale·kernel version 의존 |
| FP4 일반 | 4-bit floating representation | bitsandbytes·research format 등 | 동일 명칭이어도 codebook·packing이 다름 |
| INT4 | integer 4-bit + scale/zero point | AWQ·GPTQ·W4A16·OpenVINO | FP4와 동일 형식이 아님 |


### 14.2 gpt-oss의 native MXFP4

OpenAI의 [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)와 [`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)는 원래부터 MXFP4 weights로 배포되는 native low-bit 모델이다. 공식 안내의 대표 메모리 범위는 20B가 약 16GB급, 120B가 80GB급 장치다.

이 모델을 일반 BF16 모델의 “Q4 변환본”처럼 해석하면 안 된다.

```text
native MXFP4 checkpoint
  ≠ BF16 모델을 임의로 GGUF Q4로 PTQ
  ≠ GPTQ 4-bit
  ≠ NF4 QLoRA base
```

모델 architecture, 일부 고정밀 layer, runtime kernel과 전체 active parameter 구조가 다르다.

### 14.3 FP-Quant·Blackwell

[Transformers FP-Quant](https://huggingface.co/docs/transformers/quantization/fp_quant)는 MXFP4·NVFP4를 위한 PTQ·QAT 계열을 다룬다. Blackwell-native 성능을 목표로 할 때 다음을 확인한다.

- GPU compute capability
- PyTorch·CUDA·Triton·runtime 최소 version
- model architecture 지원
- calibration recipe
- tensor parallel 지원
- KV cache 형식
- BF16 baseline과 정확도

### 14.4 native low-bit 모델의 원칙

- 모델 카드가 지정한 runtime과 dtype을 우선한다.
- 임의 dequant→requant로 다른 형식으로 바꾸지 않는다.
- nominal bit만으로 일반 Q4 모델과 품질을 비교하지 않는다.
- native checkpoint의 non-MXFP4 layer와 runtime overhead를 포함해 메모리를 측정한다.

### 14.5 BitNet

[Transformers BitNet](https://huggingface.co/docs/transformers/quantization/bitnet)과 [Microsoft BitNet](https://github.com/microsoft/BitNet)은 ternary weight `{-1, 0, 1}`와 저비트 activation을 사용하는 native 저비트 계열을 지원한다.

중요한 차이:

```text
BitNet 모델:
  학습·QAT 단계부터 ternary 구조

일반 BF16 LLM:
  실행 시 임의로 BitNet으로 바꾸는 것이 아님
```

BitNet runtime·checkpoint가 지원하는 architecture와 tokenizer를 그대로 사용한다. 1.58-bit라는 숫자를 GGUF IQ2와 직접 동일시하지 않는다.

---

## 15. EXL3와 EXL2 레거시

[ExLlamaV3](https://github.com/turboderp-org/exllamav3)는 소비자 NVIDIA GPU에서 로컬 LLM을 실행하기 위한 현재 세대의 runtime이며, QTIP 계열의 EXL3 format을 제공한다.

### 15.1 EXL3 특징

- target bitrate 기반 2–8bit급 flexible quantization
- consumer CUDA GPU 최적화
- tensor parallel·expert parallel
- continuous dynamic batching
- 2–8bit KV cache
- multimodal·LoRA 지원
- OpenAI-compatible server는 TabbyAPI 경로

### 15.2 변환 개념

현재 공식 저장소의 기본 흐름은 working directory를 두고 target bitrate를 지정하는 방식이다.

```bash
python convert.py \
  -i /path/to/hf-model \
  -o /path/to/exl3-model \
  -w /path/to/workdir \
  -b 4.0
```

중단된 작업을 이어가는 기능은 현재 CLI 도움말과 문서를 확인한다.

```bash
python convert.py --help
```

### 15.3 EXL2 상태

ExLlamaV2 저장소는 보관 상태이며 개발은 V3로 이어졌다. 기존 EXL2 artifact와 runtime은 당장 사라지는 것은 아니지만, 신규 모델·신규 자동화의 기본값은 EXL3 지원 여부를 먼저 확인한다.

### 15.4 EXL3가 적합한 경우

- 단일 또는 소수의 NVIDIA 소비자 GPU
- VRAM을 촘촘하게 채우기 위한 target bpw 선택
- 높은 token-generation throughput이 중요
- tensor/expert parallel을 consumer hardware에서 구성
- GGUF보다 CUDA 전용 runtime 최적화가 중요

### 15.5 주의

- CUDA 중심이며 AMD·Apple의 범용 format이 아니다.
- target bitrate가 낮다고 모든 모델이 coherent한 것은 아니다.
- 새 architecture·multimodal·MoE 지원 상태를 확인한다.
- EXL3 파일은 vLLM·llama.cpp가 자동으로 읽는 범용 파일이 아니다.
- calibration·expert coverage와 converter version을 기록한다.

### 15.6 GGUF와 비교

| 항목 | EXL3 | GGUF |
| --- | --- | --- |
| 주요 장치 | NVIDIA CUDA 소비자 GPU | CPU·CUDA·Metal·ROCm·Vulkan 등 |
| 용량 선택 | target bitrate 세밀 조정 | 정해진 quant type·mixed recipe |
| CPU offload | runtime 전략에 의존 | layer/tensor offload 생태계 성숙 |
| 배포 범용성 | ExLlamaV3 계열 | 다수 로컬 runtime |
| 서빙 | TabbyAPI·native batching | llama-server·여러 wrapper |
| 첫 선택 | NVIDIA-only 최적화 | 다중 하드웨어·간단 배포 |


---

## 16. Apple Silicon과 MLX

[MLX-LM](https://github.com/ml-explore/mlx-lm)은 Apple Silicon 통합 메모리와 Metal을 활용하는 로컬 LLM runtime·도구다. GGUF와 별도 format이며 Apple 환경에서 4-bit·8-bit·mixed quant를 비교할 가치가 있다.

### 16.1 기본 변환

```bash
python -m pip install -U mlx-lm

mlx_lm.convert \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  -q
```

공식 기본 동작에서는 변환 결과를 `mlx_model`에 저장한다. 현재 CLI의 출력 위치·bit·group·mode·mixed recipe·Hub 업로드 옵션은 버전에 따라 확장될 수 있으므로 도움말을 확인한다.

```bash
mlx_lm.convert --help
```

### 16.2 직접 실행

```bash
mlx_lm.generate \
  --model ./mlx_model \
  --prompt "양자화의 장단점을 설명해 주세요." \
  --max-tokens 256
```

server·batch·prompt cache 사용법은 현재 MLX-LM 문서를 따른다.

### 16.3 MLX 4-bit와 GGUF Q4

둘 다 Apple 통합 메모리를 줄이지만 다음이 다르다.

| 항목 | MLX 4-bit | GGUF Q4 |
| --- | --- | --- |
| 런타임 | MLX-LM·Apple-native Python/C++ stack | llama.cpp·Metal |
| 형식 | MLX weights·config | GGUF |
| 생태계 | MLX fine-tuning·prompt cache | 다중 플랫폼·CLI·server |
| 성능 | 모델·kernel·OS version별 | 모델·offload·Metal backend별 |
| 선택 | Apple-only workflow와 MLX tooling | 다른 장치와 artifact 공유 |


### 16.4 통합 메모리 예산

Apple Silicon은 CPU와 GPU가 같은 메모리를 쓰므로 모델 파일이 30GB이고 장치가 32GB라고 실행 가능한 것이 아니다.

```text
available_for_model
≈ physical_unified_memory
 - macOS wired/compressed memory
 - display
 - IDE·browser·terminal
 - runtime·KV·activation
 - safety margin
```

swap으로 실행될 수 있어도 token/s와 시스템 응답성이 크게 악화될 수 있다. Activity Monitor와 `memory_pressure`를 함께 본다.

### 16.5 KV와 prompt cache

MLX-LM의 rotating KV cache·prompt cache는 긴 context의 메모리를 줄이거나 재사용하는 데 유용하다. 그러나 cache 제한은 오래된 token의 attention 정보를 버릴 수 있으므로 긴 문서 QA·코드베이스 작업에서 품질을 검증한다.

### 16.6 Apple 권장 순서

```text
1. MLX 4-bit와 GGUF Q4_K_M를 같은 모델·prompt로 비교
2. 4K~8K context, batch 1로 peak unified memory 측정
3. Q5/Q6 또는 MLX 8-bit로 품질 기준선 확보
4. KV·prompt cache 조절
5. 앱·browser를 포함한 실제 일상 환경에서 재측정
```

---

## 17. KV 캐시 양자화

가중치를 Q4로 줄여도 긴 context와 다중 사용자에서는 KV cache가 메모리 병목이 된다.

### 17.1 언제 필요한가

- 32K·64K·128K 이상 context
- 여러 parallel slot
- continuous batching server
- prefix cache·prompt cache
- VLM의 많은 visual token
- long-running agent session

### 17.2 `llama.cpp`

현재 `llama.cpp`는 K·V cache type을 별도로 지정하는 옵션을 제공한다. 정확한 지원 type은 빌드와 model에 따라 다르므로 `llama-cli --help` 또는 `llama-server --help`를 확인한다.

개념 예시:

```bash
./build/bin/llama-server \
  -m model-Q4_K_M.gguf \
  -c 32768 \
  -np 2 \
  -ctk q8_0 \
  -ctv q8_0
```

더 낮은 Q4 계열 KV는 메모리를 더 줄일 수 있지만 model·backend별 품질과 kernel 지원을 확인한다.

### 17.3 vLLM FP8 KV

vLLM은 지원되는 환경에서 FP8 KV cache를 제공한다. scale 처리, model 지원, attention backend와 GPU 세대가 맞아야 한다. per-head·세부 quant 기능은 experimental일 수 있으므로 production 배포 전에 정확도와 fallback을 확인한다.

### 17.4 KV quant 품질 평가

다음은 짧은 benchmark로는 놓치기 쉽다.

- context 앞부분의 고유 정보 회상
- 여러 문서 사이의 entity 구분
- 긴 코드의 variable·function reference
- 장문의 수식 전제 유지
- multi-turn conversation의 정책·도구 state
- VLM의 초기 page·image 정보 회상

### 17.5 weight quant와 분리해 실험

```text
실험 A: BF16 weights + FP16 KV
실험 B: Q4 weights   + FP16 KV
실험 C: Q4 weights   + Q8 KV
실험 D: Q4 weights   + Q4 KV
```

A→B에서 weight quant 손실, B→C→D에서 KV quant 손실을 분리할 수 있다.

### 17.6 동시 슬롯 계산

KV는 대체로 슬롯 수에 비례한다.

```text
4 GiB KV per slot × 8 slots = 약 32 GiB
```

server의 `max_num_seqs`, parallel slot, batch token 한도를 모델 weights와 별도로 계산한다. prefix cache도 별도 메모리를 사용한다.

---

## 18. MoE 모델 양자화

Mixture-of-Experts 모델은 **활성 파라미터**가 작더라도 모든 expert weights를 메모리에 두는 배포가 일반적이다.

```text
weight memory ≈ total parameters 기준
token당 compute ≈ active parameters 기준
```

예를 들어 “120B total·6B active” 모델은 계산량이 6B와 비슷할 수 있어도 weight storage를 6B로 계산하면 안 된다.

### 18.1 MoE 메모리 항목

- shared attention·embedding·router weights
- 모든 expert weights
- expert parallel shard·replica
- routing buffer
- top-k expert activation
- all-to-all communication buffer
- KV cache

### 18.2 quantization 주의

- router·gate는 고정밀 유지가 유리할 수 있다.
- expert별 분포가 다르므로 하나의 짧은 calibration set이 부족할 수 있다.
- calibration에서 활성화되지 않은 expert의 scale이 부정확할 수 있다.
- tensor/expert parallel과 quant packing의 호환성이 중요하다.
- 일부 runtime은 attention은 quantize하지만 expert kernel은 fallback할 수 있다.

### 18.3 expert coverage

캘리브레이션 로그에서 다음을 기록한다.

```yaml
moe_calibration:
  total_experts: <N>
  experts_seen: <N_seen>
  min_tokens_per_expert: <value>
  median_tokens_per_expert: <value>
  max_tokens_per_expert: <value>
  domains: [ko, en, code, math, rag]
```

### 18.4 mixed precision

MoE에서는 다음 조합을 실험할 수 있다.

```text
router/gate: BF16 or FP16
shared attention: Q5/Q6 or FP8
experts: Q3/Q4
embedding/output: Q6/Q8/BF16
KV: Q8/FP8
```

단, provider recipe label만 보고 위 구성을 추정하지 말고 실제 tensor map과 model card를 확인한다.

### 18.5 multi-GPU

- tensor parallel: 각 layer의 tensor를 GPU에 분할
- expert parallel: expert를 GPU별 분배
- data parallel: 모델 replica를 여러 개 생성

양자화가 weight memory를 줄여도 all-to-all latency와 interconnect가 병목일 수 있다. PCIe-only consumer GPU와 NVLink·고속 fabric server는 같은 총 VRAM에서도 결과가 다르다.

---

## 19. 비전·이미지·오디오·임베딩 모델

양자화 문법은 모달리티별로 다르다. 텍스트 LLM의 Q4 규칙을 전체 pipeline에 일괄 적용하지 않는다.

### 19.1 VLM·OCR

일반적인 구성:

```text
vision encoder
  + projector / adapter
  + text decoder LLM
  + visual token KV
```

권장 출발점:

```text
text LLM body: Q4_K_M 또는 W4A16
projector: Q8/BF16
vision encoder: FP16/BF16 또는 공식 INT8/FP8
KV: FP16/Q8부터
```

평가 항목:

- 작은 글자 OCR character accuracy
- 표 row·column 구조
- 수식·기호·소수점
- UI coordinate·spatial relation
- chart legend·axis
- 다중 이미지 구분

자세한 모델별 내용은 [비전·OCR 가이드](../modalities/vision-ocr.md)를 본다.

### 19.2 Diffusion·이미지 생성

일반적인 구성:

```text
text encoder(s)
  + DiT / UNet / transformer
  + VAE
  + ControlNet / adapter / LoRA
  + latent·attention activations
```

권장 출발점:

- transformer/UNet: FP8·INT8·NF4 또는 공식 GGUF
- text encoder: Q8/INT8/FP8 또는 일부 Q4
- VAE: FP16/BF16 유지
- ControlNet·IP-Adapter: 개별 검증

[Diffusers GGUF](https://huggingface.co/docs/diffusers/quantization/gguf)는 지원 모델 component를 `from_single_file` 경로로 loading한다. low-bit weights가 forward에서 compute dtype으로 dequantize될 수 있으므로 작은 파일이 곧 native low-bit matrix multiply를 의미하지 않는다.

평가 항목:

- 손·얼굴·미세 질감
- 타이포그래피·로고
- 색 banding·VAE artifact
- 동일 seed의 prompt adherence
- edit identity preservation
- ControlNet 구조 정합

자세한 내용은 [이미지 생성 가이드](../modalities/image-generation.md)를 본다.

### 19.3 오디오·음성

ASR·TTS는 다음 형식이 흔하다.

- Whisper GGML Q5/Q8
- CTranslate2 INT8·INT8-float16
- ONNX INT8
- TensorRT·NeMo FP16/FP8
- MLX 4/8-bit
- LLM backbone만 GGUF Q4

`audio encoder`, `speaker encoder`, `speech tokenizer`, `codec`, `vocoder`는 품질 민감도가 다르다.

평가 항목:

- ASR WER/CER·숫자·고유명사
- timestamp drift
- TTS speaker similarity·prosody
- 노이즈·metallic artifact
- real-time factor·first-audio latency
- streaming state 안정성

자세한 내용은 [오디오·음성 가이드](../modalities/audio-speech.md)를 본다.

### 19.4 임베딩·reranker

작은 embedding 모델은 Q2/Q3로 줄여 얻는 메모리 이득보다 retrieval 품질 손실이 더 클 수 있다.

권장 순서:

```text
BF16/FP16 baseline
  → INT8/Q8
  → 4-bit 실험
```

평가:

- Recall@k
- MRR·nDCG
- cosine similarity drift
- multilingual·cross-lingual retrieval
- hard negative 순위
- reranker top-k stability

### 19.5 structured prediction·tabular·timeseries

작은 encoder·foundation model은 weight memory보다 input batch·table row·context window가 더 클 수 있다. INT8·FP16을 먼저 평가하고, unsupported 4-bit conversion으로 수치 calibration을 손상시키지 않는다.

---

## 20. CPU·엣지·ONNX·OpenVINO·Core ML

### 20.1 ONNX Runtime

[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)은 dynamic·static INT8 quantization API와 model debugging 도구를 제공한다.

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model.int8.onnx",
    weight_type=QuantType.QInt8,
)
```

- dynamic quant: activation scale을 실행 중 계산
- static quant: representative calibration data 필요
- FP16 conversion: GPU·provider가 지원할 때 별도 선택
- graph optimization 순서와 quant debugging을 공식 문서에 맞춘다.

### 20.2 OpenVINO·NNCF

[OpenVINO weight compression](https://docs.openvino.ai/)과 [NNCF](https://github.com/openvinotoolkit/nncf)는 Intel CPU·GPU·NPU에서 INT8·INT4·FP8·microscaling 계열을 다룬다.

권장 사용:

- Intel CPU 서버의 weight-only INT4
- static INT8 activation quant
- LLM·VLM·diffusion weight compression
- accuracy-aware quantization
- edge·NPU deployment

주의:

- 모든 graph node가 low-bit kernel로 내려가는지 확인한다.
- unsupported operation이 FP32/FP16 fallback할 수 있다.
- group size·symmetric/asymmetric·ratio를 기록한다.
- OpenVINO version과 target device plugin을 고정한다.

### 20.3 Core ML

[Core ML Tools optimization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)은 Apple 장치 배포를 위한 weight 8/4-bit, activation INT8, per-tensor·per-channel·per-block quantization을 제공한다.

- iPhone·iPad·Mac의 Neural Engine·GPU·CPU target을 명시한다.
- deployment target에 따라 사용 가능한 operator와 dtype이 다르다.
- A17 Pro·M4 같은 최신 장치의 optimized path와 구형 장치를 분리한다.
- model package 크기뿐 아니라 load time·peak memory·battery·thermal을 측정한다.

### 20.4 AMD Quark

[AMD Quark](https://quark.docs.amd.com/)는 AMD CPU·GPU를 주 대상으로 INT8·INT4·INT2, FP8·FP6·FP4, OCP MX, SmoothQuant·AWQ·GPTQ·회전 변환 등 다양한 PTQ/QAT 기능을 제공한다.

[Transformers Quark integration](https://huggingface.co/docs/transformers/quantization/quark)은 artifact loading·evaluation 경로로 사용할 수 있지만, 실제 deployment는 Quark와 target runtime의 공식 지원표를 확인한다.

### 20.5 CPU format 선택

```text
범용 로컬 LLM·여러 OS
  → GGUF

Intel CPU·iGPU·NPU 최적화
  → OpenVINO/NNCF

기존 ONNX graph·다양한 execution provider
  → ONNX Runtime INT8/FP16

Apple 앱 배포
  → Core ML

Apple 개발·로컬 LLM
  → MLX 또는 GGUF
```

### 20.6 엣지 장치의 추가 제약

- thermal throttling
- battery drain
- storage I/O·model load time
- unified/shared memory
- NPU operator coverage
- app sandbox·model encryption
- OTA update 크기
- 개인정보와 offline processing

---

## 21. 런타임·하드웨어 지원 매트릭스

아래 표는 2026-07-21의 일반적인 선택 방향이다. 세부 지원은 release마다 바뀌므로 최종 근거는 각 runtime의 현재 compatibility table이다.

| 형식·도구 | CPU | NVIDIA | AMD | Intel GPU/NPU | Apple Silicon | 대표 강점 | 주요 제한 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GGUF·llama.cpp | 강함 | CUDA offload | ROCm·Vulkan | SYCL/Vulkan 등 build별 | Metal | 가장 범용적인 로컬 artifact | 새 architecture·kernel version 의존 |
| AWQ | runtime별 제한 | Turing~Hopper 중심 | 제한적·version별 | 일부 runtime | 기본 경로 아님 | W4A16 CUDA 생태계 | packing·kernel·model 지원 |
| GPTQModel·Marlin | 일부 CPU path | Volta~Hopper·Marlin은 세대별 | 제한적 | 일부 Intel GPU | 기본 경로 아님 | GPTQ·Marlin CUDA 생태계 | CUDA 최적화·legacy format 차이 |
| compressed-tensors·vLLM | 일부 scheme | 주요 서버 경로 | FP8·일부 scheme | 지원표 확인 | 기본 경로 아님 | 고처리량·다중 quant | compute capability·model support |
| bitsandbytes | CPU backend | CUDA | 진행 중·제한적 | XPU·Gaudi | 기본 경로 아님 | QLoRA·빠른 loading | offload RAM·serving kernel |
| AutoRound | quantization·일부 inference | export·inference | format별 | Intel 최적화 | export별 | 다중 scheme·format | 버전 변화·변환 자원 |
| torchao | PyTorch path | CUDA | ROCm 일부 | export/backend별 | MPS 지원 범위 확인 | PyTorch-native·QAT | prototype dtype·kernel 범위 |
| EXL3 | 아님 | 주 대상 | 현재 기본 아님 | 아님 | 아님 | 소비자 CUDA·세밀한 bpw | 전용 runtime·architecture 지원 |
| MLX | Apple CPU/GPU 통합 | 아님 | 아님 | 아님 | 주 대상 | Apple-native 통합 메모리 | Apple 전용 |
| OpenVINO·NNCF | Intel 중심 | 일부 path | 기본 아님 | 주 대상 | 기본 아님 | Intel edge·server 최적화 | graph coverage·plugin version |
| ONNX Runtime | 강함 | CUDA·TensorRT EP | ROCm EP 상태 확인 | OpenVINO/DML 등 | CoreML EP 등 | 다양한 graph·provider | operator·quant coverage |
| Core ML | Apple device | 아님 | 아님 | 아님 | 주 대상 | Apple 앱·ANE | deployment target·operator 제한 |
| AMD Quark | AMD CPU | 기본 아님 | 주 대상 | 기본 아님 | 기본 아님 | AMD PTQ/QAT·다양한 dtype | target runtime·hardware 의존 |


### 21.1 vLLM 형식 선택

vLLM의 공식 quantization 문서는 AWQ, GPTQModel, bitsandbytes, LLM Compressor, NVIDIA ModelOpt, AMD Quark, TorchAO, GGUF, quantized KV cache 등 여러 경로를 제공한다. 그러나 하드웨어 matrix는 형식마다 다르다.

공식 compatibility table의 현재 요약은 다음과 같다. 이 표는 **vLLM 구현에 한정**되며, 같은 형식이 다른 런타임에서 지원되는지와는 별개다.

| vLLM 구현 | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AWQ | — | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — |
| GPTQ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — |
| Marlin (GPTQ·AWQ·FP8·FP4) | — | ✓* | ✓ | ✓ | ✓ | — | — | — | — |
| LLM Compressor INT8 W8A8 | — | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| LLM Compressor INT8 W4A8 | — | — | — | — | — | — | — | — | ✓ |
| LLM Compressor FP8 W8A8 | — | — | — | ✓ | ✓ | ✓ | — | — | — |
| bitsandbytes | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| DeepSpeedFP | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| GGUF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |

`*` 현재 표에서 Turing의 Marlin은 MXFP4를 지원하지 않는다. Volta·Turing·Ampere·Ada·Hopper는 각각 대략 SM 7.0·7.5·8.0/8.6·8.9·9.0을 가리킨다. 표는 릴리스에 따라 바뀌므로 배포 시점의 공식 문서를 다시 확인한다.

따라서 “vLLM이 GGUF를 지원한다”와 “vLLM CPU에서 GGUF가 최적이다”를 혼동하지 않는다. 같은 이유로 AWQ·GPTQ·FP8이라는 이름만 보고 AMD·Intel·Arm까지 동일하게 실행된다고 가정해서도 안 된다.

### 21.2 GPU 세대별 빠른 기준

```text
Pascal·구형:
  GGUF CUDA offload·bitsandbytes 일부·FP16 비교

Turing:
  INT8·W4A16 AWQ/GPTQ·Marlin 지원 범위 확인

Ampere:
  W4A16·INT8·GGUF·EXL3의 성숙한 선택지

Ada:
  FP8 + W4A16 비교

Hopper:
  FP8 고처리량·대형 serving

Blackwell:
  NVFP4/MXFP4 native + FP8 + W4A16 비교
```

### 21.3 실제 kernel 확인

다음 지표가 artifact 이름보다 중요하다.

- runtime startup log의 selected kernel
- profiler의 matmul dtype·operator
- GPU utilization·memory bandwidth
- prompt processing과 token generation 각각의 throughput
- fallback layer 수
- tensor parallel communication overhead

---

## 22. 파라미터와 RAM·VRAM별 용량표

### 22.1 raw weight 추정표

아래 표는 6.3절의 Llama 3.1 8B 공식 예시에서 관측된 effective bpw를 다른 파라미터 수에 단순 적용한 **설명용 raw-weight 추정치**다. 실제 파일 크기를 보장하지 않는다.

사용 bpw:

- IQ2_XS 2.5882
- Q2_K 3.1593
- Q3_K_M 3.9960
- Q4_K_M 4.8944
- Q5_K_M 5.7036
- Q6_K 6.5633
- Q8_0 8.5008
- BF16/F16 16.0005

| params | IQ2_XS GiB | Q2_K GiB | Q3_K_M GiB | Q4_K_M GiB | Q5_K_M GiB | Q6_K GiB | Q8_0 GiB | BF16/F16 GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 | 1.9 |
| 3B | 0.9 | 1.1 | 1.4 | 1.7 | 2.0 | 2.3 | 3.0 | 5.6 |
| 7B | 2.1 | 2.6 | 3.3 | 4.0 | 4.6 | 5.3 | 6.9 | 13.0 |
| 8B | 2.4 | 2.9 | 3.7 | 4.6 | 5.3 | 6.1 | 7.9 | 14.9 |
| 14B | 4.2 | 5.1 | 6.5 | 8.0 | 9.3 | 10.7 | 13.9 | 26.1 |
| 27B | 8.1 | 9.9 | 12.6 | 15.4 | 17.9 | 20.6 | 26.7 | 50.3 |
| 32B | 9.6 | 11.8 | 14.9 | 18.2 | 21.2 | 24.5 | 31.7 | 59.6 |
| 70B | 21.1 | 25.7 | 32.6 | 39.9 | 46.5 | 53.5 | 69.3 | 130.4 |
| 120B | 36.2 | 44.1 | 55.8 | 68.4 | 79.7 | 91.7 | 118.8 | 223.5 |
| 235B | 70.8 | 86.4 | 109.3 | 133.9 | 156.0 | 179.6 | 232.6 | 437.7 |
| 405B | 122.0 | 149.0 | 188.4 | 230.8 | 268.9 | 309.4 | 400.8 | 754.4 |


실제 차이 원인:

- vocabulary·embedding·output head
- nonquantized tensor
- tied/untied weights
- MoE total parameters
- quantizer의 mixed precision
- metadata·alignment·shard
- multimodal encoder·projector

### 22.2 실용 메모리 권장선

아래는 단일 슬롯·4K–8K context를 가정한 보수적인 시작점이다. “최소”가 아니라 **운영 여유를 남긴 권장 구간**이다.

| 모델 규모 | Q2·IQ2 | Q3 | Q4 | Q5·Q6 | Q8 | BF16 | 주의 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–3B | 4 GB 이하 | 4 GB | 4–6 GB | 6–8 GB | 8–12 GB | 8–16 GB | runtime overhead가 weights보다 크게 보일 수 있음 |
| 7–8B | 4–6 GB | 6–8 GB | 8 GB | 10–12 GB | 12–16 GB | 20–24 GB | 32K KV는 정밀도에 따라 1–4GB+ |
| 14B | 8 GB | 10–12 GB | 12–16 GB | 16–20 GB | 20–24 GB | 32–40 GB | VLM이면 projector·visual token 추가 |
| 27–32B | 12–16 GB | 16–20 GB | 24 GB | 28–32 GB | 40–48 GB | 64–80 GB | 24GB GPU는 Q4 weights 크기 여유 확인 |
| 70B | 24–32 GB | 40–48 GB | 48–64 GB | 64–80 GB | 80–96 GB | 160 GB+ | CPU는 메모리 대역폭이 핵심 |
| 120B | 48–64 GB | 64–80 GB | 80–96 GB | 96–128 GB | 128–160 GB | 256 GB+ | MoE라도 total weights 기준 |
| 235B | 80–96 GB | 128 GB | 160–192 GB | 192–256 GB | 256–320 GB | 512 GB+ | multi-GPU interconnect 확인 |
| 405B | 160 GB | 224–256 GB | 320 GB 권장 | 384 GB+ | 512 GB+ | 1 TB급 | 256GB Q4는 headroom이 부족할 수 있음 |


### 22.3 전용 GPU와 시스템 RAM

전용 GPU에서 model을 전량 offload해도 시스템 RAM이 필요하다.

```text
최소 system RAM 후보
≈ model artifact 또는 load staging
 + runtime
 + OS·application
 + CPU-side cache
```

실무 권장:

- VRAM 8–16GB: 시스템 RAM 16–32GB 이상
- VRAM 24GB: 시스템 RAM 32–64GB 이상
- VRAM 48GB: 시스템 RAM 64–128GB 이상
- VRAM 80GB+: 시스템 RAM 128GB 이상

대형 shard를 memory-map하는 runtime은 실제 RSS와 virtual mapping이 다를 수 있다. swap을 안전 여유로 계산하지 않는다.

### 22.4 Apple 통합 메모리

모델에 할당 가능한 통합 메모리는 보통 물리 메모리보다 작다.

```text
32GB Mac → 모델 파일 30GB는 비현실적
64GB Mac → 48~52GB 내외의 전체 AI working set부터 보수적으로 시작
128GB Mac → 96~110GB 내외부터 시작
```

위 값은 앱 사용량과 OS version에 따라 달라지는 보수적 예시다. 실제 `memory_pressure`, swap, token/s를 측정한다.

### 22.5 context와 concurrency 추가

```text
최종 장치 용량
≥ weights
 + KV_per_slot × slots
 + prefill workspace
 + modality components
 + 10~25% headroom
```

Q4 70B weights가 48GB GPU에 들어가도 32K context·여러 slot이면 OOM이 발생할 수 있다. 이때 Q3로 내리기 전에 context·KV dtype·slot을 조절하면 품질을 더 잘 보존할 수 있다.

---

## 23. Hugging Face 다운로드·검사·고정

### 23.1 전체 크기 사전 확인

```bash
python -m pip install -U huggingface_hub

hf download OWNER/REPO --dry-run
```

저장소에 BF16, 여러 quant, tokenizer, duplicate format이 함께 있으면 전체 저장소를 무조건 clone하지 않는다.

### 23.2 원하는 quant만 받기

```bash
hf download OWNER/REPO \
  --include '*Q4_K_M*.gguf' \
  --local-dir ./models/NAME-Q4_K_M
```

shard 파일이면 모든 shard가 포함되는지 dry-run 결과를 확인한다.

```bash
hf download OWNER/REPO \
  --include 'NAME-Q4_K_M-*.gguf' \
  --local-dir ./models/NAME-Q4_K_M
```

### 23.3 revision 고정

```bash
hf download OWNER/REPO \
  --revision <commit-sha> \
  --include '*.gguf' \
  --local-dir ./models/NAME
```

branch name `main`보다 immutable commit SHA가 재현성에 유리하다.

### 23.4 checksum

Linux:

```bash
sha256sum model.gguf > SHA256SUMS
sha256sum -c SHA256SUMS
```

macOS:

```bash
shasum -a 256 model.gguf > SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

### 23.5 config 검사

safetensors quant artifact는 다음 파일을 함께 확인한다.

```text
config.json
quantization_config
model.safetensors.index.json
*.safetensors
special_tokens_map.json
tokenizer.json
tokenizer_config.json
generation_config.json
chat_template
processor_config.json
preprocessor_config.json
```

### 23.6 GGUF 검사

- file name과 실제 metadata의 model architecture가 맞는지
- tokenizer·BOS/EOS·special token
- context length
- quant type별 tensor 수
- split/shard 순서
- multimodal `mmproj`가 별도인지
- converter name·version

`llama.cpp`의 load log는 architecture·tensor type·metadata를 출력하므로 첫 실행 로그를 artifact와 함께 저장한다.

### 23.7 gated·license

- access token은 최소 권한으로 사용한다.
- gated model의 quant 재배포 가능 여부를 원본 license에서 확인한다.
- commercial restriction·acceptable use·attribution을 기록한다.
- 원본 모델의 license가 quantized derivative에도 적용될 수 있다.

### 23.8 다운로드 manifest

```yaml
artifact:
  source_repo: OWNER/REPO
  source_revision: <commit-sha>
  files:
    - path: model-Q4_K_M.gguf
      size_bytes: <bytes>
      sha256: <sha256>
  downloaded_at: 2026-07-21T00:00:00+09:00
  license: <spdx-or-text>
  runtime:
    name: llama.cpp
    revision: <commit-sha>
```

---

## 24. 품질·속도·메모리 벤치마크

양자화는 최소 세 축으로 평가한다.

```text
품질
+ 성능
+ 메모리·운영 안정성
```

### 24.1 동일 조건 고정

- 원본 model revision
- tokenizer·chat template
- prompt와 system message
- context length
- sampling parameters
- seed
- runtime·kernel version
- GPU driver·CUDA/ROCm·OS
- batch·parallel slot
- KV cache dtype
- GPU offload layer

### 24.2 품질 평가

| 작업 | 최소 metric | 추가 검증 |
| --- | --- | --- |
| 일반 대화 | pairwise preference·instruction success | 반복·환각·언어 혼합 |
| 프로그래밍 | unit test·compile·patch apply | repository-level regression |
| 수학 | exact answer·symbolic check | 중간 식·단위·수치 오차 |
| RAG | faithfulness·citation correctness | JSON schema·근거 누락 |
| tool calling | schema parse·argument exactness | 도구 선택·retry loop |
| 한국어·다국어 | task accuracy·human rating | 고유명사·존댓말·code-switch |
| long context | needle recall·document QA | 초반·중반·후반 위치별 |
| VLM·OCR | CER·table F1·VQA accuracy | 작은 글자·수식·공간 관계 |
| ASR·TTS | WER/CER·speaker similarity | artifact·latency·timestamp |
| embedding | Recall@k·MRR·nDCG | vector drift·cross-lingual |


### 24.3 perplexity

perplexity는 quantization 손실을 감지하는 유용한 지표지만 production task를 대체하지 않는다.

```bash
./build/bin/llama-perplexity \
  -m model-Q4_K_M.gguf \
  -f eval-corpus.txt \
  -c 2048
```

동일 corpus·tokenizer·context로 BF16, Q8, Q6, Q5, Q4, Q3를 비교한다.

### 24.4 `llama-bench`

```bash
./build/bin/llama-bench \
  -m model-Q4_K_M.gguf \
  -p 512 \
  -n 128 \
  -ngl 999
```

측정할 항목:

- model load time
- prompt processing tokens/s
- generation tokens/s
- time to first token
- peak VRAM
- peak RSS·unified memory
- power·temperature

### 24.5 batch와 production 조건

batch 1에서 빠른 format이 large batch에서도 빠르다고 보장되지 않는다. 다음을 분리한다.

```text
interactive:
  batch 1
  low latency
  token generation 중심

throughput server:
  many sequences
  continuous batching
  prefill 비중 큼
  activation·KV·scheduler 중요
```

### 24.6 acceptance gate 예시

기준은 프로젝트가 정한다. 아래는 형식 예시다.

```yaml
acceptance:
  peak_vram_gib_max: 22.0
  ttft_p95_ms_max: 1200
  generation_tps_min: 25
  json_schema_success_min: 0.995
  code_test_pass_drop_max_pp: 1.0
  korean_eval_drop_max_pp: 1.5
  long_context_recall_drop_max_pp: 2.0
```

### 24.7 A/B 표

| 항목 | BF16 | Q8 | Q6 | Q5 | Q4 | Q3 | 선택 기준 |
|---|---:|---:|---:|---:|---:|---:|---|
| 파일 크기 |  |  |  |  |  |  |  |
| peak RAM/VRAM |  |  |  |  |  |  |  |
| prompt tok/s |  |  |  |  |  |  |  |
| generation tok/s |  |  |  |  |  |  |  |
| task accuracy |  |  |  |  |  |  |  |
| 한국어 score |  |  |  |  |  |  |  |
| JSON success |  |  |  |  |  |  |  |
| long-context score |  |  |  |  |  |  |  |

### 24.8 통계

- 한두 prompt가 아니라 대표 workload 사용
- warm-up 후 여러 반복
- 평균뿐 아니라 median·p95
- thermal state와 background process 기록
- stochastic generation은 여러 seed
- pairwise result는 blind 평가 가능하면 적용

---

## 25. 목적별 권장 워크플로

### 25.1 범용 CPU·Mac·여러 장치 공유

```text
1. 원본 BF16 revision 고정
2. 공식 또는 신뢰 가능한 GGUF Q4_K_M 확보
3. 4K context·단일 슬롯 실행
4. Q5_K_M 또는 Q6_K를 품질 기준선으로 비교
5. 필요 시 imatrix 기반 Q3·IQ3 실험
6. KV Q8은 긴 context에서 별도 평가
7. checksum·runtime commit 기록
```

### 25.2 NVIDIA 단일 GPU 개인용

```text
1. 모델·context·VRAM budget 계산
2. GGUF Q4 GPU full-offload를 기준선
3. EXL3 3~5 bpw와 GPTQ/AWQ W4A16 비교
4. prompt/generation throughput 분리 측정
5. 긴 context에서는 KV Q8/Q4 비교
6. 가장 빠른 format이 아니라 task gate를 통과한 format 선택
```

### 25.3 vLLM production serving

```text
1. target GPU generation 확인
2. BF16 baseline server 구축
3. LLM Compressor W4A16 또는 FP8 recipe 적용
4. selected kernel·fallback 확인
5. production batch·max_num_seqs·KV dtype으로 load test
6. tensor parallel·prefix cache·LoRA 조합 검증
7. canary 배포 후 quality telemetry 확인
```

### 25.4 QLoRA

```text
1. base model BF16 source와 license 확인
2. bitsandbytes NF4 + BF16 compute + double quant
3. gradient checkpointing·sequence length·batch 예산
4. LoRA target module과 rank 고정
5. eval loss뿐 아니라 downstream task 비교
6. adapter와 base revision을 함께 배포
```

자세한 계산은 향후 [파인튜닝 메모리 가이드](./fine-tuning-memory.md)를 본다.

### 25.5 Apple Silicon

```text
1. MLX 4-bit와 GGUF Q4 비교
2. unified memory pressure·swap 측정
3. Q5/Q6 또는 MLX 8-bit 품질 기준선
4. context·rotating KV·prompt cache 조절
5. 일상 앱을 켠 상태에서 재측정
```

### 25.6 VLM·OCR

```text
1. text body Q4
2. mmproj·vision encoder Q8/BF16
3. image resolution·visual token 고정
4. OCR CER·table·formula·spatial benchmark
5. text Q3를 시도하기 전에 image count·context 조절
6. projector 저비트는 별도 A/B
```

### 25.7 이미지 생성

```text
1. DiT/UNet FP16/BF16 baseline
2. FP8·INT8·NF4·GGUF 중 runtime-native 선택
3. text encoder와 VAE를 독립적으로 quantize
4. 동일 seed·prompt·scheduler·steps 비교
5. 얼굴·손·텍스트·색 artifact 검사
6. ControlNet·LoRA 추가 후 peak 재측정
```

### 25.8 오디오

```text
1. Whisper Q5/Q8 또는 ASR FP16 baseline
2. INT8/CTranslate2/ONNX 비교
3. audio encoder·decoder를 분리 평가
4. TTS codec·vocoder는 고정밀 유지부터
5. WER/CER·speaker similarity·RTF 측정
6. streaming cache와 동시 세션 추가
```

### 25.9 초대형 MoE

```text
1. total parameter로 weight memory 계산
2. provider native quant와 runtime 확인
3. expert coverage calibration
4. router·shared layer 고정밀 mixed recipe 검토
5. expert/tensor parallel topology benchmark
6. active parameter 수를 VRAM 계산에 사용하지 않음
```

---

## 26. 문제 해결

### 26.1 파일은 들어가는데 OOM

원인 후보:

- KV cache·context가 큼
- parallel slot·batch가 큼
- model load 중 temporary duplicate
- vision/audio component 추가
- CUDA graph·workspace·allocator fragmentation
- CPU offload가 FP32 copy를 생성
- display·다른 process가 VRAM 점유

해결 순서:

```text
1. batch·parallel slot을 1로
2. context를 4K~8K로
3. KV dtype을 Q8/FP8로
4. GPU offload·tensor split 조정
5. modality component 순차 load
6. workspace·graph option 조정
7. 마지막에 weight quant를 한 단계 낮춤
```

### 26.2 Q3가 Q4보다 느림

가능한 원인:

- Q3 kernel 최적화 부족
- dequantization overhead
- GPU에서 일부 tensor fallback
- Q4가 tensor core·SIMD에 더 잘 맞음
- memory bandwidth가 이미 병목이 아님
- prompt processing과 generation을 섞어 측정

해결:

- `llama-bench`에서 prompt와 generation 분리
- GPU layer offload 동일화
- runtime update·kernel log 확인
- Q4_K_M·IQ4_XS·Q3_K_M를 같은 설정으로 비교

### 26.3 출력이 깨짐·무한 반복·이상한 언어

- tokenizer·chat template mismatch
- wrong architecture conversion
- corrupt/missing shard
- unsupported quant tensor
- requantization 누적 손실
- 너무 낮은 Q2/Q3
- sampler·BOS/EOS 설정
- model revision과 tokenizer revision 불일치

먼저 BF16/FP16 또는 공식 Q8에서 재현되는지 확인한다.

### 26.4 코드·JSON만 실패

낮은 bit는 일반 대화보다 구조화 출력에서 먼저 차이가 날 수 있다.

- Q4→Q5/Q6 비교
- output head·embedding 고정밀 recipe
- constrained decoding·JSON schema validator
- temperature 0·deterministic seed
- tool parser retry를 숨기지 말고 metric으로 기록

### 26.5 FP8/FP4인데 느림

- GPU에 native dtype 연산이 없음
- runtime가 BF16로 dequantize
- unsupported layer fallback
- batch가 너무 작아 kernel 이점을 못 씀
- quantized load만 되고 matmul은 고정밀
- tensor parallel communication 병목

profiler에서 실제 operator dtype을 확인한다.

### 26.6 변환 중 RAM 부족

- BF16 source를 full load
- calibration activation·Hessian statistics
- output quant와 source 동시 존재
- Python copy·CPU staging
- shard merge

대응:

- 더 큰 RAM의 변환 노드 사용
- local NVMe scratch
- converter의 low-memory·shard 옵션 확인
- `--keep-split` 검토
- calibration batch·sequence 조절
- source를 FP16/BF16로 직접 사용하고 F32를 피함

### 26.7 VLM이 이미지를 무시

- `mmproj` 누락·잘못된 revision
- projector format 불일치
- image token/chat template 오류
- vision encoder가 지원되지 않음
- 본체만 quantized checkpoint로 바꾸면서 processor 누락

### 26.8 품질이 benchmark와 다름

- 다른 source model revision
- 다른 calibration data
- 다른 prompt template·sampling
- KV quant·context 차이
- runtime kernel bug·fallback
- provider가 mixed recipe 세부를 공개하지 않음
- benchmark contamination·평가 표본 부족

### 26.9 Apple에서 swap이 심함

- 모델·KV·앱이 통합 메모리를 초과
- memory-mapped file과 resident memory 혼동
- browser·IDE·Docker가 메모리 사용
- context·prompt cache가 큼

모델 bit를 낮추기 전에 앱 종료, context·cache·batch를 줄이고 `memory_pressure`를 확인한다.

### 26.10 CPU가 매우 느림

- 모델이 RAM에는 들어가지만 memory bandwidth 부족
- NUMA node를 넘는 memory access
- thread oversubscription
- unsupported SIMD·잘못된 binary
- 일부 GPU offload가 PCIe 왕복을 늘림

CPU benchmark에서는 thread 수, NUMA binding, memory channel, huge page 여부를 함께 기록한다.

---

## 27. 보안·공급망·재현성

양자화 artifact도 실행 가능한 software supply chain의 일부다.

### 27.1 신뢰 경계

- 원본 model provider
- quantization uploader
- converter·runtime source
- custom CUDA/Triton kernel
- Python package·wheel
- model card의 `trust_remote_code`
- tokenizer·processor·chat template
- calibration data

### 27.2 안전한 파일 형식

`safetensors`는 pickle보다 임의 코드 실행 위험을 줄이지만, 모델 repository의 custom Python·install script·custom operator는 별도 위험이다. GGUF도 parser 취약점·비정상 metadata·거대한 allocation을 고려한다.

### 27.3 `trust_remote_code`

가능하면 검토된 revision에만 사용한다.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="<commit-sha>",
    trust_remote_code=True,
)
```

- repository code를 먼저 검토
- network·secret 없는 container에서 실행
- dependency lock과 hash 사용
- model token과 production secret 분리

### 27.4 quantizer 환경 격리

```text
untrusted model files
  → rootless container
  → read-only source mount
  → network disabled after download
  → no cloud credential
  → output-only writable directory
  → CPU/RAM/disk/time limit
```

custom CUDA extension build는 compiler toolchain·arbitrary build script 위험을 포함한다.

### 27.5 calibration data 보호

- 고객·회사·개인정보를 공개 quant에 사용하지 않는다.
- 필요하면 synthetic·licensed data를 사용한다.
- 내부 data를 사용한 statistics와 output artifact의 정보 누출 가능성을 검토한다.
- dataset hash·접근 권한·retention을 기록한다.
- 평가 benchmark와 calibration data를 분리한다.

### 27.6 artifact provenance

```yaml
provenance:
  original_model:
    repo: OWNER/MODEL
    revision: <sha>
    license: <license>
  quantizer:
    name: auto-round
    version: <version>
    source_revision: <sha>
  runtime_target:
    name: vllm
    version: <version>
    hardware: <gpu>
  calibration:
    dataset_sha256: <sha>
    private: false
  output:
    format: compressed-tensors
    scheme: W4A16
    files_sha256:
      model-00001-of-00004.safetensors: <sha>
  evaluation:
    suite_revision: <sha>
    report_sha256: <sha>
```

### 27.7 모델 교체와 rollback

- immutable revision으로 배포
- quant별 별도 artifact ID
- canary traffic
- quality·latency·OOM telemetry
- BF16/Q8 fallback artifact 유지
- tokenizer·processor도 함께 rollback

### 27.8 라이선스

양자화는 원본 모델의 license를 제거하지 않는다. 다음을 확인한다.

- 파생 가중치 재배포 허용
- 상업 이용
- attribution·notice
- acceptable use
- gated access 조건
- 연구 전용 제한
- 모델 output 정책

### 27.9 재현성 체크리스트

```text
[ ] 원본 repository와 commit SHA
[ ] 다운로드 파일 SHA-256
[ ] quantizer 이름·version·commit
[ ] 정확한 command line
[ ] quant scheme·group size·excluded module
[ ] calibration dataset hash
[ ] tokenizer·chat template
[ ] runtime·driver·kernel version
[ ] hardware·OS
[ ] benchmark prompt·seed·context·KV dtype
[ ] peak RAM·VRAM·latency·quality report
[ ] license·redistribution 검토
```

---

## 28. 주요 출처와 저장소

### 28.1 GGUF·llama.cpp

- [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`llama.cpp` quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [`llama.cpp` importance matrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Transformers GGUF guide](https://huggingface.co/docs/transformers/gguf)

### 28.2 Transformers quantization

- [Quantization overview](https://huggingface.co/docs/transformers/quantization/overview)
- [bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [AWQ](https://huggingface.co/docs/transformers/quantization/awq)
- [GPTQ](https://huggingface.co/docs/transformers/quantization/gptq)
- [Quanto](https://huggingface.co/docs/transformers/quantization/quanto)
- [HQQ](https://huggingface.co/docs/transformers/quantization/hqq)
- [AMD Quark](https://huggingface.co/docs/transformers/quantization/quark)
- [BitNet](https://huggingface.co/docs/transformers/quantization/bitnet)
- [FP-Quant](https://huggingface.co/docs/transformers/quantization/fp_quant)

### 28.3 서버·GPU quantization

- [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor)
- [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- [Intel AutoRound](https://github.com/intel/auto-round)
- [torchao](https://docs.pytorch.org/ao/stable/)
- [ExLlamaV3·EXL3](https://github.com/turboderp-org/exllamav3)

### 28.4 Apple·AMD·엣지

- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [AMD Quark](https://quark.docs.amd.com/)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [OpenVINO documentation](https://docs.openvino.ai/)
- [NNCF](https://github.com/openvinotoolkit/nncf)
- [Core ML Tools optimization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)

### 28.5 멀티모달·native low-bit

- [Diffusers GGUF quantization](https://huggingface.co/docs/diffusers/quantization/gguf)
- [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
- [`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)
- [OpenAI gpt-oss local inference guide](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)
- [Microsoft BitNet](https://github.com/microsoft/BitNet)

### 28.6 관련 레포지토리 문서

- [생산성·문서·RAG](../domains/productivity-rag.md)
- [데이터 분석](../domains/data-analysis.md)
- [비전·OCR](../modalities/vision-ocr.md)
- [이미지 생성](../modalities/image-generation.md)
- [오디오·음성](../modalities/audio-speech.md)
- [파인튜닝 메모리](./fine-tuning-memory.md) (예정)
- [서빙·동시성](./serving-concurrency.md) (예정)
- [런타임·하드웨어](./runtime-hardware.md) (예정)

---

## 29. 최종 권장안

### 29.1 대부분의 사용자

```text
범용 첫 선택: GGUF Q4_K_M
품질 비교:    Q5_K_M 또는 Q6_K
메모리 부족:  Q3_K_M 또는 검증된 IQ3
극한 용량:    Q2·IQ2 — 작은 Q4와 반드시 비교
긴 context:   weight bit보다 KV·slot부터 조절
```

### 29.2 CUDA production

```text
Turing·Ampere:
  W4A16 GPTQ/AWQ/compressed-tensors + BF16 baseline

Ada·Hopper:
  FP8 W8A8 + W4A16 비교

Blackwell:
  NVFP4/MXFP4 native + FP8 + W4A16 비교

공통:
  vLLM 지원표와 실제 selected kernel 확인
```

### 29.3 Apple Silicon

```text
MLX 4-bit와 GGUF Q4_K_M를 같은 장비에서 비교
통합 메모리 전체를 모델 예산으로 잡지 않음
Q5/Q6 또는 8-bit를 품질 기준선으로 유지
swap보다 context·KV·앱 점유량을 먼저 줄임
```

### 29.4 파인튜닝

```text
QLoRA: NF4 + BF16 compute + double quant
추론용 GGUF Q4의 파일 크기로 학습 VRAM을 계산하지 않음
QAT는 별도 학습 workflow와 평가 필요
```

### 29.5 멀티모달

```text
VLM: text body Q4, projector·encoder Q8/BF16
이미지: DiT/UNet만 먼저 quantize, VAE 고정밀
오디오: ASR INT8/Q5/Q8, codec·vocoder 고정밀 우선
embedding: INT8/Q8부터, 4-bit는 retrieval metric 통과 후
```

### 29.6 한 문장 기준

> **메모리에 간신히 들어가는 더 큰 Q2보다, 런타임이 제대로 가속하고 KV·운영 여유를 남긴 한 단계 작은 Q4/Q5가 production에서는 더 좋은 선택인 경우가 많다.**

---

## 30. 갱신 및 사용상 주의

양자화 생태계는 모델 architecture, GPU generation, runtime kernel과 함께 빠르게 변한다. 이 문서는 2026-07-21 KST 기준으로 공식 문서와 원 저장소를 확인해 작성했지만, 다음 항목은 다운로드·배포 직전에 다시 검증해야 한다.

- runtime의 최신 지원 architecture
- GPU compute capability와 kernel
- quantizer·converter version
- model·tokenizer·processor revision
- 정확한 shard와 파일 크기
- quantization config·group size·excluded layer
- calibration dataset과 공개 가능성
- context·KV dtype·parallel slot
- 모델 license와 파생 배포 조건
- security advisory·custom code

이 문서의 메모리 값은 장비 구매나 서비스 SLA를 보장하는 수치가 아니다. 동일한 parameter 수라도 architecture와 runtime에 따라 peak가 달라진다. 실제 도입 전에는 target hardware에서 다음을 기록한다.

```text
model artifact size
peak system RAM
peak VRAM 또는 unified memory
prompt processing throughput
generation throughput
time to first token
context length
KV cache dtype
parallel slots
quality metrics
runtime·driver·OS revision
```

새 quant format을 추가할 때는 단순히 “몇 bit·몇 GB”만 적지 말고, 다음 정보를 함께 기여한다.

```yaml
required:
  - original_model_repo
  - original_revision
  - quant_method
  - format
  - weight_activation_kv_scheme
  - group_or_block_size
  - calibration_description
  - quantizer_version
  - target_runtime
  - target_hardware
  - artifact_sizes_and_hashes
  - peak_memory
  - throughput
  - task_quality_comparison
  - license
```

---

**문서 종료**
