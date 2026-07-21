# 이미지 생성·편집용 로컬 AI 모델 가이드
> RAM·VRAM·Apple 통합 메모리별 텍스트→이미지·이미지 편집·ControlNet·LoRA 선택표

[← 메인 README](../../README.md) · [비전·OCR](./vision-ocr.md) · [생산성·문서·RAG](../domains/productivity-rag.md) · [데이터 분석](../domains/data-analysis.md)

> **최종 검증일:** 2026-07-21 (KST)  
> **주요 실행 형식:** Diffusers, ComfyUI, 단일 `safetensors`, FP16/BF16, FP8, INT8, NF4, GGUF + ComfyUI-GGUF  
> **범위:** 텍스트→이미지, 이미지→이미지, 인페인팅·아웃페인팅, 지시 기반 편집, 다중 참조, ControlNet·IP-Adapter, LoRA, 업스케일·복원, 로컬 서비스 운영  
> **관련 문서:** [양자화](../operations/quantization.md) (예정) · [파인튜닝 메모리](../operations/fine-tuning-memory.md) (예정) · [런타임·하드웨어](../operations/runtime-hardware.md) (예정) · [오디오·음성](./audio-speech.md) (예정)

이 문서는 보유한 **시스템 RAM**, **GPU VRAM**, 또는 **Apple Silicon 통합 메모리**를 기준으로 로컬 이미지 생성·편집 모델을 선택하기 위한 실전 가이드다. 최신 대형 DiT뿐 아니라 낮은 메모리에서 여전히 강력한 Stable Diffusion 생태계, 이미지 편집·다중 참조 모델, ControlNet·LoRA·업스케일러까지 하나의 메모리 예산으로 계산한다.

이미지 생성 모델은 텍스트 LLM보다 구성요소가 많다. 모델 저장소 전체 또는 GGUF transformer 파일 하나만 보고 실행 가능 여부를 판단하면 안 된다. 실제 파이프라인은 보통 다음을 함께 로드한다.

```text
텍스트 인코더 1개 이상
  + UNet 또는 Diffusion Transformer(DiT)
  + VAE / pixel decoder
  + scheduler와 activation buffer
  + 입력 이미지·mask·reference encoder
  + ControlNet / IP-Adapter / LoRA
  + 업스케일러·복원 모델
  + 출력 이미지와 UI cache
```

특히 **GGUF Q2·Q3·Q4는 이미지 모델 전체가 아니라 DiT/UNet만 양자화한 파일인 경우가 많다.** 예를 들어 Qwen-Image GGUF를 실행하려면 별도의 Qwen2.5-VL 텍스트 인코더, projector와 VAE가 필요하다. FLUX 계열도 T5/CLIP 또는 해당 세대의 텍스트 인코더와 VAE를 추가로 로드한다.

모델 카드·가중치·라이선스·런타임 지원은 계속 바뀐다. 아래 값은 2026-07-21에 확인한 대표값이며, 다운로드 직전 Hugging Face의 **정확한 파일명, 총 다운로드 크기, gated access, 라이선스, base model revision, 권장 runtime**을 다시 확인한다.

> **핵심 원칙:** 낮은 메모리에서는 해상도·batch·동시성을 먼저 줄이고, 그다음 VAE tiling·CPU offload를 적용하며, 마지막 수단으로 Q2/Q3를 사용한다. 이미지 내 글자, 손·얼굴, 미세 질감, 동일 인물 편집은 저비트 양자화의 영향을 크게 받으므로 Q4와 Q5/Q6 또는 FP8을 반드시 같은 seed로 비교한다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [이미지 생성 전체 메모리 계산](#2-이미지-생성-전체-메모리-계산)
3. [RAM·VRAM·Apple 통합 메모리 해석](#3-ramvramapple-통합-메모리-해석)
4. [BF16·FP8·NF4·Q2·Q3·Q4 선택법](#4-bf16fp8nf4q2q3q4-선택법)
5. [작업 유형과 모델 계열 선택](#5-작업-유형과-모델-계열-선택)
6. [최신 범용 이미지 생성 모델](#6-최신-범용-이미지-생성-모델)
7. [저사양·성숙 생태계 모델](#7-저사양성숙-생태계-모델)
8. [이미지 편집·인페인팅·다중 참조](#8-이미지-편집인페인팅다중-참조)
9. [타이포그래피·포스터·로고·UI](#9-타이포그래피포스터로고ui)
10. [ControlNet·구조 제어·IP-Adapter](#10-controlnet구조-제어ip-adapter)
11. [LoRA·스타일·캐릭터 일관성](#11-lora스타일캐릭터-일관성)
12. [메모리별 완성형 이미지 생성 스택](#12-메모리별-완성형-이미지-생성-스택)
13. [해상도·종횡비·batch·steps·sampler](#13-해상도종횡비batchstepssampler)
14. [텍스트 인코더·VAE·업스케일러 메모리](#14-텍스트-인코더vae업스케일러-메모리)
15. [Hugging Face 직접 다운로드](#15-hugging-face-직접-다운로드)
16. [ComfyUI 구성](#16-comfyui-구성)
17. [Diffusers 실행](#17-diffusers-실행)
18. [`stable-diffusion.cpp`와 GGUF](#18-stable-diffusioncpp와-gguf)
19. [Nunchaku·SVDQuant·FP4·FP8](#19-nunchakusvdquantfp4fp8)
20. [NVIDIA·AMD·Apple·CPU 운영](#20-nvidiaamdapplecpu-운영)
21. [파인튜닝·LoRA 메모리 빠른 기준](#21-파인튜닝lora-메모리-빠른-기준)
22. [라이선스·보안·개인정보·출처표시](#22-라이선스보안개인정보출처표시)
23. [평가·재현성·운영 체크리스트](#23-평가재현성운영-체크리스트)
24. [문제 해결](#24-문제-해결)
25. [주요 출처와 저장소](#25-주요-출처와-저장소)

---

## 1. 30초 선택표

아래 표의 “장착 메모리”는 한 장치에서 실제로 사용할 수 있는 **VRAM 또는 Apple 통합 메모리**를 우선 의미한다. CPU offload를 사용할 때는 별도의 시스템 RAM이 필요하다. 전용 GPU 환경에서는 최소 **VRAM의 1.5–2배 시스템 RAM**을 권장하며, 24 GB 이상 대형 모델은 64 GB RAM부터 운영이 편하다.

| 사용 가능 메모리 | 가장 안전한 시작점 | 권장 형식 | 현실적인 작업 | 시스템 RAM 권장 | 주의점 |
| ---: | --- | --- | --- | ---: | --- |
| **4 GB** | [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px) | FP16, low-VRAM | 512–768px 단일 이미지, LoRA 1개, 간단 img2img | 16 GB | SDXL은 실행 가능하더라도 매우 공격적인 offload와 tiled VAE가 필요하다. |
| **6 GB** | [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), SANA-Sprint, SD 1.5 + ControlNet | FP16/BF16 | 768–1024px, batch 1, inpaint·ControlNet 순차 실행 | 16–32 GB | refiner·upscaler·ControlNet을 동시에 GPU에 올리지 않는다. |
| **8 GB** | SDXL, [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) 저비트, [FLUX.1 dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf) Q4 + T5 offload | FP16, NF4/INT8, Q4 | 1024px 생성, SDXL LoRA·ControlNet, FLUX 시험 | 32 GB | Q4 DiT가 들어가도 T5·CLIP·VAE와 activation 때문에 full GPU load는 어렵다. |
| **12 GB** | [FLUX.2 Klein 4B GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) Q4/Q5 + encoder offload, [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 양자화, SD 3.5 Large Q4 | Q4–Q6, FP8/NF4 | 1024px 고품질 T2I, 제한적 편집·다중 참조 | 32–64 GB | FLUX.2 Klein 4B 공식 full pipeline 기준은 약 13 GB VRAM이므로 12 GB에서는 offload가 전제다. |
| **16 GB** | [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), Z-Image Turbo BF16, SD 3.5 Large/Turbo 양자화 | BF16 또는 Q4–Q8 | 1024px generation·editing, 소수 reference, SDXL 다중 ControlNet | 32–64 GB | FLUX.2 Klein 4B는 공식적으로 약 13 GB VRAM에 맞지만 UI와 고해상도 여유를 남긴다. |
| **24 GB** | [Ideogram 4 NF4](https://huggingface.co/ideogram-ai/ideogram-4-nf4), [HiDream O1](https://huggingface.co/HiDream-ai/HiDream-O1-Image), [HunyuanImage 2.1 FP8](https://huggingface.co/tencent/HunyuanImage-2.1), [Qwen-Image GGUF](https://huggingface.co/city96/Qwen-Image-gguf) Q4, FLUX.2 Klein 9B Q4 + offload | NF4/FP8/Q4–Q6 | 타이포그래피, 고품질 편집, 1K–2K 단일 이미지 | 64 GB | Qwen-Image는 Q4 본체만 13 GB 안팎이며 텍스트 인코더·VAE를 별도 계산한다. |
| **32 GB** | [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), Ideogram 4, HiDream O1, [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) quant/offload, Qwen-Image Q5/Q6 | FP8/NF4/Q5–Q8 | 2K 단일 생성, multi-reference, 고품질 text rendering | 64–96 GB | Klein 9B 공식 full pipeline은 약 29 GB라서 32 GB에서 batch 1과 제한된 reference가 안전하다. |
| **48 GB** | [FLUX.2 dev GGUF](https://huggingface.co/unsloth/FLUX.2-dev-GGUF) Q4/Q5, Qwen-Image Q8 + encoder, Krea 2 Raw/Turbo 저비트 | Q4–Q8, FP8 | 전문가용 단일 사용자 워크스테이션, 복잡한 editing·layout | 96 GB | full BF16 checkpoint가 아니라 transformer quant만 들어가는지 확인한다. |
| **64 GB** | FLUX.2 dev FP8/NVFP4 또는 Q8, [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw) BF16 + staged load, Qwen-Image BF16 + offload | FP8/Q8/BF16 | 고품질 2K, 여러 LoRA·condition model, 낮은 동시성 서버 | 128 GB | Krea 2 Raw 저장소 전체는 약 62 GB이므로 full pipeline 상시 GPU 상주는 별도 여유가 필요하다. |
| **96 GB** | [FLUX.2 dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) BF16/FP8, Krea 2 BF16, Qwen-Image BF16, 여러 모델 상주 | BF16/FP8 | 연구 기준선, 2K–4K tiled generation, 2–4 슬롯 | 128–192 GB | 대형 text encoder와 reference image activation이 동시성에 따라 급증한다. |
| **128 GB** | 32B급 DiT BF16 + 편집·업스케일 스택, 다중 모델 A/B 서버 | BF16/FP8 | 팀용 저동시성 서비스, batch 처리, 모델 교차평가 | 192–256 GB | 모델 복제 대신 request queue와 순차 component offload를 우선한다. |
| **192 GB VRAM 합계** | FLUX.2 dev·Qwen·Krea 다중 인스턴스, 대형 모델 A/B 서버 | BF16/FP8 | 32B급 multi-worker·고해상도 배치 | 256–512 GB | HunyuanImage 3.0 공식 Base 최소선인 3×80 GB에는 미달한다. |
| **240–320 GB VRAM 합계** | [HunyuanImage 3.0 Base](https://huggingface.co/tencent/HunyuanImage-3.0) 공식 다중 GPU 경로 | BF16·분산 | 서버급 text-to-image 연구 | 512 GB+ | 공식 권장 VRAM은 최소 3×80 GB다. 80B total/13B active라도 전체 weights를 저장한다. |
| **640 GB VRAM 합계** | HunyuanImage 3.0 Instruct·Instruct-Distill 공식 경로 | BF16·분산 | 생성·편집·prompt rewrite·reasoning 연구 | 1 TB+ | 공식 권장 VRAM은 최소 8×80 GB다. interconnect와 FlashAttention·FlashInfer를 함께 설계한다. |

### 1.1 메모리별 즉시 추천

- **4–6 GB:** 최신 초대형 모델을 억지로 offload하기보다 SD 1.5·SDXL·SANA-Sprint로 prompt, ControlNet, LoRA 작업을 완성한다.
- **8 GB:** SDXL이 가장 안정적이다. FLUX.1·SD 3.5는 Q4/NF4 transformer와 CPU text encoder로 시험한다.
- **12–16 GB:** FLUX.2 Klein 4B와 Z-Image Turbo가 최신 성능·속도·메모리의 중심이다. 16 GB가 Klein 4B의 현실적인 full-pipeline 기준이다.
- **24–32 GB:** Ideogram 4, HiDream O1, Qwen-Image, FLUX.2 Klein 9B, HunyuanImage 2.1을 실제 작업별로 비교한다.
- **48–96 GB:** FLUX.2 dev·Krea 2·Qwen-Image 고정밀과 여러 보조 모델을 함께 운용할 수 있다.
- **192 GB VRAM 합계:** 32B급 모델의 다중 인스턴스에는 유용하지만 HunyuanImage 3.0 공식 최소선에는 미달한다.
- **240 GB·640 GB VRAM 합계:** HunyuanImage 3.0 Base는 최소 3×80 GB, Instruct·Distilled는 최소 8×80 GB의 공식 경로를 전제로 한다.

### 1.2 목적별 첫 모델

| 목적 | 첫 모델 | 비교 모델 | 선택 이유 |
|---|---|---|---|
| 저사양 일러스트·LoRA 생태계 | SD 1.5 | SDXL | 작은 VRAM, 방대한 checkpoint·ControlNet·LoRA 자산 |
| 6–12 GB 범용 생성 | SDXL | SD 3.5 Medium, SANA-Sprint | 안정적인 툴 지원과 1024px 품질 |
| 16 GB 최신 빠른 생성·편집 | FLUX.2 Klein 4B | Z-Image Turbo | 4B는 generation·editing·multi-reference 통합, Z-Image는 8-step 고속 T2I |
| 영어·중국어 text rendering | Z-Image Turbo | Qwen-Image 2512, Ideogram 4 | bilingual 텍스트와 prompt adherence 비교 |
| 포스터·광고·레이아웃 | Ideogram 4 | Qwen-Image 2512 | JSON prompt, bbox·palette·2K typography |
| 인물·제품 identity 편집 | Qwen-Image-Edit-2511 | FLUX.2 Klein, HiDream O1 | character consistency·지시 기반 편집·multi-reference |
| 스타일 탐색·LoRA 학습 | Krea 2 Raw | SDXL, Z-Image base | Raw는 undistilled base, Turbo는 8-step 배포용 |
| 고품질 통합 생성·편집 | HiDream O1 | FLUX.2 dev | 별도 VAE·frozen text encoder 없는 unified pixel transformer |
| 24 GB 2K 생성 | HunyuanImage 2.1 FP8 | Ideogram 4 NF4 | 2K 생성과 텍스트·레이아웃 비교 |
| 서버급 native multimodal | HunyuanImage 3.0 | FLUX.2 dev | understanding·generation·editing을 통합한 대형 MoE |

### 1.3 선택 절차

```text
1. T2I, editing, inpaint, reference consistency 중 주 작업을 정한다.
2. 목표 해상도와 batch를 정한다. 처음에는 1024px, batch=1이다.
3. transformer/UNet + text encoder + VAE + 조건 모델의 파일 합계를 계산한다.
4. Q4와 FP8/NF4 또는 Q5/Q6를 같은 seed·prompt로 비교한다.
5. CPU offload·VAE tiling 적용 전후의 peak VRAM과 생성 시간을 기록한다.
6. LoRA·ControlNet·upscaler는 하나씩 추가하며 peak를 다시 측정한다.
7. 라이선스와 상업 이용 가능 범위를 배포 전에 별도로 검토한다.
```

---

## 2. 이미지 생성 전체 메모리 계산

### 2.1 기본식

```text
M_total ≈ M_OS_and_UI
        + M_text_encoder_1..N
        + M_UNet_or_DiT_weights
        + M_VAE_or_pixel_decoder
        + M_activations_and_attention
        + M_input_image_mask_reference
        + M_ControlNet_IPAdapter_LoRA
        + M_upscaler_restoration
        + M_output_preview_cache
        + M_allocator_fragmentation
        + M_headroom
```

가중치 메모리는 대략 `parameters × bytes per parameter`로 시작할 수 있지만 실제 peak는 activation, kernel workspace, attention backend, CFG branch, 해상도와 batch 때문에 달라진다.

| 형식 | 이론적 가중치 바이트/parameter | 10B 가중치 이론값 | 실제 해석 |
|---|---:|---:|---|
| FP32 | 4 | 40 GB | inference에는 비효율적, 일부 VAE·norm이 FP32일 수 있음 |
| BF16/FP16 | 2 | 20 GB | 일반적인 고품질 기준선 |
| FP8/INT8 | 약 1 | 10 GB | kernel·scale·미지원 layer 때문에 실제값은 더 큼 |
| NF4/INT4 | 약 0.5 + metadata | 5 GB+ | bitsandbytes·backend 지원과 compute dtype 확인 |
| GGUF Q4 | 모델별 약 4–5 bit | 5–7 GB | block metadata와 고정밀 layer 포함, transformer 파일만인 경우가 많음 |
| GGUF Q2/Q3 | 약 2–4 bit | 3–5 GB | 품질 저하와 upcast layer 때문에 단순 비트 계산과 다름 |

### 2.2 파일 크기와 실행 peak의 차이

```text
예: FLUX.2 Klein 4B Q4_K_M transformer     약 2.60 GB
  + 텍스트 인코더·tokenizer                 수 GB
  + VAE / decoder                           수백 MB~수 GB
  + 1024px activation·attention             가변
  + ComfyUI·PyTorch·CUDA context             1 GB 이상 가능
  = 공식 full pipeline 기준 약 13 GB VRAM
```

따라서 “Q4 파일이 2.6 GB이므로 4 GB GPU에서 실행된다”는 결론은 잘못이다. CPU offload로 실행 자체는 가능할 수 있지만, 시스템 RAM·전송 비용·latency가 크게 늘어난다.

### 2.3 대표 GGUF transformer 크기

아래는 **DiT/UNet 본체만의 대표 크기**다. text encoder·VAE는 별도다.

| 모델 | Q2 | Q3 | Q4 | Q5/Q6 | Q8 | F16/BF16 |
|---|---:|---:|---:|---:|---:|---:|
| [FLUX.2 Klein 4B GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) | 1.83 GB | 2.10–2.12 GB | 2.46–2.69 GB | 2.92–3.41 GB | 4.30 GB | 7.75 GB |
| [FLUX.2 Klein 9B GGUF](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) | 3.98 GB | 4.69–4.77 GB | 5.62–6.16 GB | 6.71–7.87 GB | 9.98 GB | 18.2 GB |
| [FLUX.1 dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf) | 4.03 GB | 5.23 GB | 6.79–7.53 GB | 8.27–9.86 GB | 12.7 GB | 23.8 GB |
| [Qwen-Image GGUF](https://huggingface.co/city96/Qwen-Image-gguf) | 7.06 GB | 8.95–9.68 GB | 11.9–13.1 GB | 14.1–16.8 GB | 21.8 GB | 40.9 GB BF16 |
| [Qwen-Image-Edit-2509 GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF) | 7.15 GB | 9.04–9.76 GB | 11.9–13.1 GB | 14.1–16.8 GB | 21.8 GB | 저장소별 확인 |

### 2.4 FLUX.1의 T5 encoder 추가 예산

[city96/t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)의 대표 크기:

| T5 XXL encoder | 크기 | 권장 용도 |
|---|---:|---|
| Q3_K_S / Q3_K_M / Q3_K_L | 2.10 / 2.30 / 2.46 GB | 메모리 제약 시험, prompt fidelity 평가 필요 |
| Q4_K_S / Q4_K_M | 2.74 / 2.90 GB | 최소 실용선 |
| Q5_K_S / Q5_K_M | 3.29 / 3.39 GB | 저장소가 권장하는 품질 중심 시작점 |
| Q6_K | 3.91 GB | text adherence 우선 |
| Q8_0 | 5.06 GB | 고정밀 양자화 |
| F16 | 9.53 GB | 품질 기준선 |

T5 encoder GGUF는 imatrix가 적용되지 않은 변환본일 수 있다. prompt 이해가 중요한 포스터·복합 장면에서는 DiT를 Q4로 낮추더라도 text encoder는 Q5 이상을 유지하는 구성이 유리할 수 있다.

### 2.5 해상도와 activation

latent diffusion은 픽셀 전체를 직접 처리하지 않지만, 해상도가 증가하면 latent token 수와 attention activation이 증가한다.

```text
512 × 512  → 기준 1배 픽셀
1024 × 1024 → 4배 픽셀
1536 × 1536 → 9배 픽셀
2048 × 2048 → 16배 픽셀
```

메모리가 정확히 4·9·16배가 되는 것은 아니지만, attention·VAE decode·reference image encoder가 급격히 커질 수 있다. 2K 모델이라도 2K를 full-frame로 디코딩하지 않고 tiled VAE를 쓰는 이유다.

### 2.6 CFG가 만드는 추가 비용

일반적인 classifier-free guidance는 conditional과 unconditional branch를 함께 계산한다. runtime에 따라 두 branch를 batch 차원으로 묶으면 activation이 크게 늘 수 있다.

- distilled Turbo/Schnell 모델은 `guidance_scale=0` 또는 1 근처로 동작하여 메모리와 시간을 줄이는 경우가 많다.
- base/Raw 모델은 CFG와 negative prompt를 지원하지만 더 많은 steps와 메모리를 요구한다.
- “steps가 적다”는 것은 주로 시간 절감이며, 한 step의 peak memory가 반드시 비례해 줄지는 않는다.

### 2.7 안전 여유

| 장착 메모리 | 최소 headroom | 운영 권장 |
|---:|---:|---|
| 4–8 GB | 25–35% | UI preview와 다른 GPU 앱 종료 |
| 12–24 GB | 20–30% | 1024px batch 1부터 측정 |
| 32–64 GB | 15–25% | reference·ControlNet·upscaler 별도 queue |
| 96 GB+ | 최소 15%, 서비스별 실측 | 동시성·allocator fragmentation·model clone 포함 |

---

## 3. RAM·VRAM·Apple 통합 메모리 해석

### 3.1 전용 NVIDIA GPU

- 가장 빠른 구성은 DiT/UNet, text encoder, VAE를 VRAM에 상주시킨다.
- 화면 출력에 같은 GPU를 쓰면 1–2 GB 이상을 비운다. 2K preview와 브라우저 WebGL도 VRAM을 쓴다.
- FP8·NVFP4는 GPU 세대와 kernel 지원에 따라 속도 이득이 없거나 fallback할 수 있다.
- bitsandbytes NF4는 일반적으로 CUDA 환경이 가장 성숙하지만 모든 파이프라인·layer가 동일하게 지원되지는 않는다.
- `torch.cuda.max_memory_allocated()`와 `nvidia-smi --query-compute-apps`를 함께 기록한다.
- CPU offload는 PCIe 전송이 반복되어 latency가 크게 늘 수 있다. NVMe 속도가 아니라 RAM↔VRAM 대역폭이 병목이 된다.

### 3.2 AMD GPU

- ROCm 지원 GPU와 OS 조합을 먼저 확인한다.
- 모델 카드에 “all hardware”라고 적혀 있어도 실제 runtime의 FP8·attention kernel이 AMD에서 동일하게 최적화된다는 뜻은 아니다.
- DirectML·ZLUDA·Vulkan backend는 지원 node와 quant 형식이 다르므로 ComfyUI workflow를 그대로 복사하지 않는다.
- AMD 최적화 ONNX/TensorRT 유사 배포본이 있는 모델은 원본 Diffusers와 출력·precision을 별도 검증한다.
- VRAM이 충분하더라도 unsupported op가 CPU로 fallback하면 처리량이 급락할 수 있다.

### 3.3 Apple Silicon 통합 메모리

- CPU·GPU·운영체제·앱이 하나의 메모리 풀을 공유한다.
- 16 GB Mac에서 13 GB 모델을 “로드 가능”하더라도 macOS·WindowServer·브라우저·preview가 같은 메모리를 사용하므로 안정적이지 않을 수 있다.
- MPS, Core ML, MLX, Draw Things 전용 형식은 같은 4-bit라도 GGUF·NF4와 메모리·품질·속도가 다르다.
- swap이 시작되면 생성은 계속되더라도 시간이 급격히 늘고 SSD 쓰기가 증가한다.
- Activity Monitor에서 Memory Pressure, Swap Used, Compressed Memory를 기록한다.
- 모델별 MPS 미지원 op와 BF16 지원 상태가 다르므로 공식 Diffusers 예제가 `mps`를 언급해도 실제 최신 pipeline을 테스트한다.

### 3.4 CPU 전용

- SD 1.5·SANA 0.6B·일부 quant model은 CPU에서도 실행할 수 있지만 1024px 대형 DiT는 지연 시간이 매우 길다.
- CPU에서는 RAM 용량뿐 아니라 메모리 대역폭, AVX2/AVX-512/AMX, 코어 수, NUMA가 중요하다.
- GGUF + `stable-diffusion.cpp` 또는 Vulkan backend가 PyTorch CPU보다 실용적인 경우가 있다.
- VAE decode와 이미지 업스케일은 CPU thread 수를 과도하게 늘리면 UI와 I/O를 방해할 수 있다.
- mmap 모델과 page cache를 위해 RAM을 꽉 채우지 않는다.

### 3.5 전용 GPU + 시스템 RAM 예산

| VRAM | 최소 RAM | 권장 RAM | 용도 |
|---:|---:|---:|---|
| 4–6 GB | 16 GB | 32 GB | SD 1.5/SDXL, aggressive offload |
| 8–12 GB | 32 GB | 64 GB | SDXL·SD3.5·FLUX Q4, text encoder offload |
| 16 GB | 32 GB | 64 GB | FLUX.2 Klein 4B, Z-Image, 여러 ControlNet |
| 24 GB | 64 GB | 96 GB | Ideogram NF4, Qwen Q4, 2K 단일 작업 |
| 32–48 GB | 64 GB | 128 GB | Klein 9B, Krea/Qwen/FLUX.2 dev quant |
| 64–96 GB | 128 GB | 192–256 GB | BF16 대형 모델, 다중 모델·동시성 |
| 192 GB VRAM 합계 | 256 GB | 512 GB+ | 32B급 다중 인스턴스·대형 serving; HunyuanImage 3.0 공식 최소선 미달 |
| 240–320 GB VRAM 합계 | 512 GB | 512 GB–1 TB | HunyuanImage 3.0 Base 공식 3×80 GB 경로 |
| 640 GB VRAM 합계 | 1 TB | 1 TB+ | HunyuanImage 3.0 Instruct·Distilled 공식 8×80 GB 경로 |

---

## 4. BF16·FP8·NF4·Q2·Q3·Q4 선택법

### 4.1 이미지 모델에서 사용하는 주요 정밀도

| 형식 | 장점 | 단점 | 권장 위치 |
|---|---|---|---|
| BF16 | 넓은 dynamic range, 최신 GPU에서 안정적 | 가중치 메모리 큼 | 품질 기준선, 학습·고급 inference |
| FP16 | 넓은 호환성 | overflow·VAE 색상 문제 가능 | SD 1.x/SDXL, 구형 GPU |
| FP8 | 약 절반 가중치, 최신 GPU에서 빠를 수 있음 | kernel·GPU 세대 의존, 일부 layer upcast | 9B–32B DiT 배포 |
| INT8 | 비교적 안정적인 절감 | backend별 속도 편차 | text encoder·DiT |
| NF4 | 4-bit로 큰 절감, Diffusers bitsandbytes | CUDA 중심, 일부 모델·학습 제약 | Ideogram 4 공식 NF4, 저메모리 DiT |
| GGUF Q8 | 품질 손실이 작고 파일 작음 | 전용 node/runtime 필요 | ComfyUI-GGUF 고품질 |
| GGUF Q5/Q6 | 품질·메모리 절충 | Q4보다 수 GB 증가 | text rendering·editing 권장 |
| GGUF Q4 | 실용적 최소선 | 미세 texture·text·identity 손실 가능 | 일반 T2I, 12–32 GB |
| GGUF Q3 | 더 작은 메모리 | anatomy·글자·detail 오류 증가 가능 | fit test, preview |
| GGUF Q2 | 최대 절감 | 모델별 품질 편차 큼 | 실행 가능성 확인, 최종 결과 비권장 |

### 4.2 Q2/Q3/Q4를 LLM과 똑같이 해석하면 안 되는 이유

이미지 모델의 출력은 단일 token 정답이 아니라 수십 step의 누적 denoising 결과다. 작은 quant error가 다음 step으로 전달될 수 있으며 다음 영역에서 체감된다.

- 작은 글자와 긴 문장
- 손가락·눈·치아·장신구
- 반복 패턴·직물·머리카락
- 제품 로고와 규칙적 geometry
- 입력 인물의 identity 유지
- mask 경계와 색상 gradient
- ControlNet의 정확한 edge·pose 추종

Q2·Q3가 항상 실패하는 것은 아니다. 일부 최신 dynamic quant는 중요한 first/last layer를 고정밀로 유지한다. 그러나 저장소의 샘플 한 장이 아니라 최소 20–50개 prompt와 seed로 평가한다.

### 4.3 권장 기본값

| 작업 | 최소 권장 | 품질 우선 | 이유 |
|---|---|---|---|
| 일반 일러스트 T2I | Q4 | Q5/Q6 또는 FP8 | 구성·색상은 저비트에 비교적 견고 |
| 포토리얼 인물 | Q5 | Q6/Q8/FP8 | 피부·눈·머리카락 detail |
| 이미지 내 글자 | Q5/Q6 | Q8/FP8/BF16 | 글자 획·간격·layout 민감 |
| identity editing | Q5/Q6 | Q8/FP8/BF16 | 참조 인물 특징 보존 |
| ControlNet pose/edge | Q4 | Q5/Q6 | condition fidelity |
| preview·draft | Q2/Q3 | Q4 | 반복 탐색 속도·fit 우선 |
| benchmark | FP8 또는 BF16 | BF16 | quant 영향과 모델 능력 분리 |

### 4.4 transformer와 text encoder를 다르게 양자화

메모리가 부족하면 모든 구성요소를 같은 비트로 낮추지 않는다.

```text
권장 예시 A: prompt adherence 우선
  DiT Q4 + text encoder Q6/Q8 + VAE FP16

권장 예시 B: texture 품질 우선
  DiT Q6 + text encoder Q4/Q5 + VAE FP16

권장 예시 C: 최대 절감
  DiT Q3 + text encoder Q4 + VAE tiled FP16
```

복합 prompt·타이포그래피는 text encoder 정밀도의 영향을 받을 수 있고, 미세 시각 품질은 DiT 정밀도의 영향을 더 받을 수 있다. 실제 모델마다 다르므로 A/B한다.

### 4.5 VAE는 저비트보다 tiling 우선

VAE는 최종 색상·미세 detail·decode 안정성에 직접 관여한다. VAE를 과도하게 양자화하기보다 다음을 먼저 사용한다.

- `enable_vae_tiling()`
- `enable_vae_slicing()`
- FP16 VAE fix 또는 모델 권장 VAE
- decode를 CPU로 이동
- latent upscale 후 tiled decode

### 4.6 저장소 이름의 함정

- `Q4_K_M`은 GGUF block quant 이름이다. NF4·INT4·FP4와 동일하지 않다.
- `FP8`은 E4M3, E5M2, scaled FP8 등 구체 형식이 다를 수 있다.
- `NVFP4`는 NVIDIA 전용 kernel이 필요한 배포 형식일 수 있다.
- `4bit` checkpoint라도 text encoder는 BF16로 포함될 수 있다.
- 전체 repo 크기는 샘플 이미지·중복 checkpoint·optimizer 파일을 포함할 수 있다.
- “model size 9B”와 HF metadata의 tensor count가 다르게 보일 수 있다. 공식 architecture 설명을 우선한다.

---

## 5. 작업 유형과 모델 계열 선택

### 5.1 주요 계열

| 계열 | 강점 | 약점 | 적합 작업 |
|---|---|---|---|
| SD 1.5 | 최소 메모리, 방대한 LoRA·ControlNet | 낮은 native 해상도, 최신 prompt 이해보다 약함 | 4–6 GB, 스타일 LoRA, inpaint |
| SDXL | 성숙한 1024px 생태계 | 최신 typography·complex prompt보다 약함 | 6–12 GB, 범용 창작·ControlNet |
| SD 3.5 | typography·prompt 이해 개선 | gated·Stability license, T5 encoder 부담 | 8–24 GB, 상업 범위 확인 필요 |
| SANA-Sprint | 0.6B/1.6B, 1–4 step, 빠름 | 복잡한 글자·포토리얼 한계 | 저메모리 preview·interactive |
| FLUX.1 | 고품질·성숙한 ComfyUI 자산 | T5 XXL 부담, dev 비상업 license | 8–24 GB, 기존 workflow |
| FLUX.2 Klein | 4B/9B, generation+editing+multi-reference | 9B 비상업, full pipeline 메모리 큼 | 16–32 GB 최신 통합 작업 |
| FLUX.2 dev | 32B, 고품질 generation/editing | 48–96 GB+, 비상업 open weights | 전문가·연구·서버 |
| Z-Image | 6B, bilingual text, Turbo 8-step | Turbo는 negative prompt·CFG 제약 | 12–24 GB 고속 T2I |
| Qwen-Image | 20B, 중국어·영어 text·editing | 큰 text encoder·VAE, 24 GB+ | 포스터·문자·정밀 편집 |
| Ideogram 4 | JSON, bbox, palette, native 2K typography | gated·비상업, Qwen3-VL-8B encoder | 디자인·광고·layout |
| HiDream O1 | unified pixel transformer, generation/editing | 새 runtime, 높은 activation | 24–48 GB 통합 생성·편집 |
| Krea 2 | 스타일·미학, Raw/Turbo 연계 LoRA | 62 GB repo, community license | 스타일 연구·fine-tuning |
| HunyuanImage 2.1 | 17B, 2K, FP8 24 GB 경로 | 큰 pipeline | 24–48 GB 고해상도 |
| HunyuanImage 3.0 | 80B total/13B active MoE, native multimodal | Base ≥3×80 GB, Instruct·Distilled ≥8×80 GB | 240–640 GB VRAM 합계의 서버급 연구 |

### 5.2 아키텍처별 특징

#### UNet latent diffusion

대표: SD 1.5, SDXL.

- 상대적으로 작은 가중치와 성숙한 최적화
- ControlNet·inpaint·LoRA ecosystem가 가장 넓음
- 4–12 GB VRAM에서 실용적
- 복합 문장·정확한 글자·세계 지식은 최신 DiT보다 약할 수 있음

#### MMDiT·single-stream DiT

대표: SD 3.5, Qwen-Image, Ideogram 4, Z-Image, Krea 2.

- text token과 image token의 깊은 상호작용
- prompt adherence·spatial layout·typography 향상
- attention activation과 text encoder 메모리가 큼
- 최신 quant/backend 지원 여부가 중요

#### unified generation/editing model

대표: FLUX.2 Klein/dev, HiDream O1, HunyuanImage 3.0.

- 별도 edit model 없이 T2I·I2I·reference를 하나의 checkpoint로 처리할 수 있음
- 입력 이미지 수와 해상도가 늘면 activation이 크게 증가
- single-reference와 multi-reference를 별도 benchmark해야 함

#### distilled few-step model

대표: SANA-Sprint, FLUX Schnell/Klein, Z-Image Turbo, SD 3.5 Large Turbo, Krea 2 Turbo.

- 1–8 step으로 빠른 반복
- CFG·negative prompt가 제한되거나 권장값이 고정될 수 있음
- base model보다 diversity·fine-tunability가 낮을 수 있음

### 5.3 작업별 모델 우선순위

| 작업 | 1순위 | 2순위 | 메모리 전략 |
|---|---|---|---|
| 빠른 콘셉트 draft | SANA-Sprint, Z-Image Turbo | FLUX.2 Klein 4B | 1–8 step, batch 1–4 |
| 포토리얼 인물 | FLUX.2 dev/Klein, Qwen-Image 2512 | Ideogram 4, HiDream O1 | Q5+·FP8, face crop 평가 |
| 포스터·메뉴·간판 | Ideogram 4 | Qwen-Image 2512, Z-Image | text encoder 고정밀, 2K tile |
| 제품 사진 | FLUX.2, Ideogram 4 | Krea 2, Qwen-Image | reference·mask·색상 일관성 |
| 캐릭터 consistency | Qwen-Image-Edit-2511 | FLUX.2 Klein/dev | multi-reference, identity score |
| 스타일 LoRA | SDXL, Krea 2 Raw | Z-Image base, FLUX.1 dev | 학습은 base/Raw, 추론은 Turbo |
| pose·depth 제어 | SDXL + ControlNet | FLUX/Qwen 전용 control adapter | condition model 순차 로드 |
| low-VRAM inpaint | SD 1.5/SDXL inpaint | SD 3.5 quant | crop-and-stitch, tiled VAE |
| 서버급 multimodal | HunyuanImage 3.0 | FLUX.2 dev | tensor parallel·queue |

---


## 6. 최신 범용 이미지 생성 모델

### 6.1 핵심 비교표

| 계열 | 공개 시점·세대 | 구조·규모 | 주요 작업 | 공식·대표 저정밀 배포 | 실전 시작 메모리 | 라이선스 핵심 | 다운로드 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| **FLUX.2 klein 4B** | 2026 | 4B rectified-flow transformer | T2I, 지시 편집, 다중 참조 | 공식 FP8 약 4.07 GB | **16 GB VRAM** | Apache 2.0 | [FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) · [BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| **FLUX.2 klein 9B** | 2026 | 9B rectified-flow transformer | T2I, 고품질 편집·다중 참조 | 공식 FP8 약 9.43 GB | **32 GB VRAM** | FLUX Non-Commercial, gated | [FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) · [BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| **Krea 2 Raw** | 2026-06 | 12B dense DiT | 다양성 높은 T2I, LoRA·후학습 기준선 | community FP8·INT8·NVFP4 | **32–48 GB VRAM** | Krea 2 Community License | [Raw](https://huggingface.co/krea/Krea-2-Raw) |
| **Krea 2 Turbo** | 2026-06 | 12B dense DiT, distilled | 8-step 고속 T2I, 1K–2K | community FP8·INT8·NVFP4 | **32–48 GB VRAM** | Krea 2 Community License | [Turbo](https://huggingface.co/krea/Krea-2-Turbo) |
| **HiDream-O1-Image** | 2026-05 | 8B pixel-level unified transformer | T2I, 편집, subject personalization, 2K | BF16, community FP8 경로 | **48 GB VRAM/통합 메모리** | MIT | [Full](https://huggingface.co/HiDream-ai/HiDream-O1-Image) · [Dev](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604) |
| **Z-Image-Turbo** | 2025–2026 계열 | 6B single-stream DiT | 8-NFE 고속 T2I, 영·중문 텍스트 | GGUF Q2–Q8, BF16 | **8–16 GB VRAM** | Apache 2.0 | [공식](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF) |
| **Z-Image Base** | 2026 계열 | 6B foundation model | 다양성·negative prompt·fine-tuning | GGUF Q2–Q8, BF16 | **12–24 GB VRAM** | Apache 2.0 | [공식](https://huggingface.co/Tongyi-MAI/Z-Image) · [GGUF](https://huggingface.co/unsloth/Z-Image-GGUF) |
| **Qwen-Image-2512** | 2025-12 최신 T2I revision | 약 20B급 denoiser + Qwen2.5-VL encoder | 사진·인물·자연물·텍스트 | GGUF Q2–Q8, BF16 | **24 GB Q4 / 64 GB BF16** | Apache 2.0 | [공식](https://huggingface.co/Qwen/Qwen-Image-2512) · [GGUF](https://huggingface.co/city96/Qwen-Image-gguf) |
| **Qwen-Image-Edit-2511** | 2025-11 최신 Edit revision | Qwen-Image 편집 계열 | 인물·제품·텍스트 편집, 다중 이미지 | GGUF Q2–Q8 community | **24 GB Q4** | Apache 2.0 | [공식](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) |
| **Ovis-Image 7B** | 2025-11 | 7B generator + 2B text/VLM component | 포스터·배너·로고·UI·긴 텍스트 | Intel AutoRound int4 | **12–24 GB VRAM** | Apache 2.0 | [공식](https://huggingface.co/ATH-MaaS/Ovis-Image-7B) · [int4](https://huggingface.co/Intel/Ovis-Image-7B-int4-AutoRound) |

> **“실전 시작 메모리”는 공식 최소 요구량과 동일하지 않다.** 모델 파일 외 구성요소, 1024² batch 1, UI·driver 여유를 포함한 보수적 권장치다. offload·tiling·더 낮은 해상도로 하향 실행할 수 있다.

### 6.2 FLUX.2 klein 4B와 9B

[FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)는 생성과 다중 참조 편집을 함께 지원하는 4B rectified-flow transformer다. 공식 FP8 저장소는 Apache 2.0이며, 모델 카드가 약 **13 GB VRAM**에 들어간다고 명시한다. 따라서 16 GB GPU가 가장 단순한 시작점이다.

[FLUX.2 klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)는 동일 계열의 더 큰 모델이다. 공식 FP8이라도 전체 파이프라인은 약 29 GB급으로 보는 것이 안전해 32 GB GPU를 권장한다. 9B는 **FLUX Non-Commercial License**와 gated access가 적용되므로, 상업 프로젝트에서는 4B Apache 2.0과 혼동하면 안 된다.

| 비교 | 4B | 9B |
| --- | --- | --- |
| 첫 대상 GPU | RTX 4070 Ti SUPER·4080 16 GB급 | 32 GB workstation GPU급 |
| 강점 | 속도·메모리·상업 라이선스 단순성 | 편집 품질·복잡한 참조·구도 여유 |
| 공식 FP8 파일 | 약 4.07 GB | 약 9.43 GB |
| 전체 실행 권장 | 16 GB VRAM | 32 GB VRAM |
| 상업 사용 | Apache 2.0 조건 검토 | 비상업 라이선스 조건 확인 |
| 선택 기준 | 일반 로컬 제작·제품 프로토타입 | 비상업 연구·품질 우선 편집 |

FLUX.2 사용 시에는 다음을 고정한다.

- `text-to-image`와 `image-editing` pipeline의 입력 형식
- 참조 이미지 수·크기·crop 정책
- 모델 revision과 single-file checkpoint revision
- guidance·steps·scheduler
- 결과 이미지의 metadata·C2PA·watermark 처리 여부

### 6.3 Krea 2 Raw와 Turbo

[Krea 2](https://huggingface.co/krea/Krea-2-Turbo)는 2026-06-22 공개된 12B dense DiT 계열이다. 두 checkpoint의 역할이 명확히 다르다.

| 항목 | Krea 2 Raw | Krea 2 Turbo |
| --- | --- | --- |
| 목적 | base checkpoint, LoRA·fine-tuning·후학습 | distilled inference checkpoint |
| 기본 steps | 비교적 많은 steps로 품질 탐색 | **8 steps** 중심 |
| CFG | 일반 생성 설정을 모델 카드에 맞춰 조정 | 공식 예제 `guidance_scale=0.0` |
| 다양성 | 높고 조정 가능성이 큼 | 속도·즉시 품질 우선 |
| 추천 사용자 | 모델 연구자·LoRA 제작자 | 제작자·interactive workflow |
| checkpoint 선택 | `krea/Krea-2-Raw` | `krea/Krea-2-Turbo` |

공식 Turbo 예시의 핵심 설정은 다음과 같다.

```text
steps = 8
guidance / CFG = 0.0
mu ≈ 1.15  # 공식 코드 경로에서 사용하는 값
목표 해상도 = 1024 또는 2048, 메모리에 따라 시작
```

Krea 2 저장소에는 Diffusers 형식과 단일 checkpoint가 함께 있어 저장소 총용량이 실제 필요한 파일보다 크게 보일 수 있다. `hf download --dry-run`과 `--include`로 한 형식만 선택한다. community FP8·NVFP4·INT8 변환은 원본 라이선스가 그대로 승계되는지, transformer뿐 아니라 text encoder와 VAE가 어떤 정밀도인지 확인한다.

### 6.4 HiDream-O1-Image

[HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image)는 raw pixel, text, task condition을 하나의 shared token space로 처리하는 **Pixel-level Unified Transformer**다. 외부 VAE와 분리된 text encoder가 없고, 한 모델에서 다음을 지원한다.

- 텍스트→이미지
- 지시 기반 이미지 편집
- 다중 참조 subject personalization
- skeleton·layout conditioning
- 긴 텍스트와 다중 영역 layout
- 최대 2048×2048 출력

일반 latent diffusion처럼 “denoiser Q4 + VAE FP16 + text encoder FP8”로 분해해 생각하기 어렵다. 체크포인트 파일이 약 35 GB여도 raw-pixel activation과 runtime workspace가 추가되므로 **48 GB급**을 안정적 시작점으로 본다. 공식 프로젝트는 편집에는 full 모델 사용을 권장하며, Dev 모델은 T2I 지향 distilled 변형으로 구분한다.

| 선택 | 권장 사용 |
| --- | --- |
| Full | 편집, subject personalization, 통합 기능 기준선 |
| Dev-2604 | 텍스트→이미지, distilled 실험, 비교적 빠른 샘플링 |
| Prompt Agent 사용 | 복잡한 layout·긴 문구를 구조화할 때 |
| Prompt Agent 미사용 | 재현 가능한 정적 prompt benchmark |

Prompt Agent가 별도 대형 언어 모델을 사용하면 총메모리가 크게 늘 수 있다. 생성 모델과 prompt agent를 동시에 GPU에 상주시킬 필요가 없다면 순차 실행하거나 API·CPU·다른 GPU로 분리한다.

### 6.5 Z-Image 계열

[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)는 6B single-stream diffusion transformer 계열이다. 공식 모델 카드 기준 Turbo는 **8 NFEs**로 동작하며 16 GB consumer GPU에 들어가는 것을 목표로 한다. Base는 다양성, negative prompt, fine-tuning과 downstream 개발에 더 적합하다.

| 변형 | 권장 steps·NFE | CFG·negative prompt | 주 용도 | 저메모리 선택 |
| --- | ---: | --- | --- | --- |
| Turbo | 8 NFE | 모델 카드 설정 우선 | 빠른 일반 생성, 사진, 영·중문 텍스트 | GGUF Q3/Q4 |
| Base | 약 50-step 계열 | 지원 | 다양성·fine-tuning·스타일 | GGUF Q4/Q5 |
| Omni-Base | 공개 상태 확인 | 생성+편집 foundation | community fine-tuning | 공식 공개 revision 확인 |
| Edit | 편집 pipeline 설정 | 입력 이미지 필요 | 지시 편집 | 지원 runtime 확인 |

Z-Image GGUF는 denoiser만 포함하는 경우가 일반적이다. 별도 **Qwen3 4B text encoder와 VAE**를 함께 다운로드해야 한다. 4 GB VRAM 이하 실행 사례는 CPU offload·quantized encoder·낮은 해상도를 포함할 수 있으므로, “Q4 파일 3.86 GB가 4 GB GPU에 완전히 상주한다”는 의미로 해석하지 않는다.

### 6.6 Qwen-Image-2512와 Qwen-Image-Edit-2511

[Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)는 Qwen-Image T2I 계열의 2025년 12월 revision으로, 공식 카드가 인물 realism, 자연물 세부, 텍스트 렌더링 개선을 강조한다. 한국어를 포함한 실제 다국어 철자 정확도는 사용자의 문구·font·layout 데이터로 따로 평가한다.

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)는 편집 전용 revision이다. 생성 checkpoint와 Edit checkpoint의 workflow·prompt template·입력 image processor를 섞지 않는다.

| 모델 | 권장 작업 | GGUF Q4_K_M denoiser | 추가 구성 | 실전 시작점 |
| --- | --- | ---: | --- | ---: |
| Qwen-Image-2512 | T2I, 인물·자연물·텍스트 | 약 13.1 GB | Qwen2.5-VL encoder, VAE | 24 GB VRAM |
| Qwen-Image-Edit-2511 | 지시 편집, 인물·제품·텍스트 수정 | 약 13.1 GB급 | mmproj/encoder, VAE, 입력 이미지 | 24 GB VRAM |
| BF16 전체 | 최고정밀 기준선 | denoiser 약 40.9 GB | 전체 repo 약 57 GB급 | 64 GB+ |

Qwen-Image 계열은 text encoder가 커서 denoiser Q4만 선택해도 16 GB GPU에서 offload가 필요할 수 있다. 16 GB에서는 Q2/Q3 또는 encoder CPU offload를 사용하고, 24 GB에서는 Q4를 기본점으로 삼는다.

### 6.7 Ovis-Image 7B

[Ovis-Image 7B](https://huggingface.co/ATH-MaaS/Ovis-Image-7B)는 텍스트 렌더링을 우선한 모델이다. 포스터, 배너, 로고, UI mockup, infographic처럼 **글자와 레이아웃이 이미지 품질의 핵심인 작업**에서 먼저 비교할 가치가 있다.

- 공식 BF16은 generator와 text/VLM component를 합쳐 24 GB보다 큰 메모리를 요구할 수 있다.
- [Intel AutoRound int4](https://huggingface.co/Intel/Ovis-Image-7B-int4-AutoRound)는 약 7.5 GB repo이지만, 지원 backend와 나머지 구성요소 정밀도를 확인해야 한다.
- 공식 pipeline은 `diffusers>=0.36.0`의 `OvisImagePipeline`을 사용한다.
- Apache 2.0이지만 생성물의 상표·폰트·저작권·퍼블리시티권은 별도 문제다.

타이포그래피 모델의 성능은 benchmark 한 점수보다 다음 실무 테스트가 중요하다.

```text
한글 2–4행
영문 대문자·소문자 혼합
숫자·통화·날짜
브랜드명과 제품 코드
좌우 정렬·중앙 정렬
작은 footer와 큰 headline 동시 배치
세로쓰기·곡선 텍스트
동일 문구 20 seed 반복
```

---

## 7. 저사양·성숙 생태계 모델

최신 모델이 항상 저사양 시스템의 최선은 아니다. SD 1.5·SDXL은 adapter, LoRA, ControlNet, 인페인팅, 업스케일, custom node와 학습 도구가 가장 넓고, 문제 해결 자료가 많다. SANA·PRX처럼 효율을 목표로 설계된 소형 모델도 별도 후보로 둔다.

### 7.1 계열 비교

| 계열 | 규모·기본 해상도 | 권장 VRAM | 장점 | 한계 | 다운로드 |
| --- | --- | ---: | --- | --- | --- |
| **Stable Diffusion 1.5** | 약 860M UNet, 512 px | 4–6 GB | 가장 넓은 LoRA·ControlNet·checkpoint 생태계 | 최신 prompt adherence·텍스트·해부학 한계 | [공식](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| **SDXL 1.0** | 약 2.6B UNet, 1024 px | 8–12 GB | 성숙한 1K 생태계, 다양한 finetune·inpaint | 두 text encoder와 VAE로 peak 증가 | [Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) · [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) |
| **SD 3.5 Medium** | 2.5B MMDiT급, 1K | 8–12 GB | prompt following·텍스트 개선, NF4 예제 | ecosystem이 SDXL보다 작고 encoder 구성 복잡 | [공식](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| **SD 3.5 Large** | 8B급 | 16–24 GB | 더 큰 모델의 품질·구도 | 단일 파일 약 16.5 GB, 전체 pipeline 더 큼 | [공식](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| **FLUX.1 schnell/dev** | 12B rectified-flow | 8–24 GB | 강한 prompt adherence, 방대한 community LoRA | T5 encoder 메모리, dev 라이선스 확인 | [schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) · [dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) · [GGUF](https://huggingface.co/leejet/FLUX.1-dev-gguf) |
| **SANA 0.6B/1.6B/4.8B** | linear DiT, 1K–4K 연구 | 4–12 GB | 효율적 attention·DC-AE, 공식 4-bit 경로 | workflow·LoRA ecosystem이 SDXL보다 작음 | [모델 모음](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e) · [GitHub](https://github.com/NVlabs/Sana) |
| **SANA-Sprint** | one/few-step | 4–8 GB | 매우 빠른 1K 생성 | distilled 설정·스타일 범위 검증 필요 | [프로젝트](https://nvlabs.github.io/Sana/Sprint/) |
| **PRX 1.3B 계열** | 경량 pixel-space generation 계열 | 4–8 GB | 낮은 메모리 연구, 빠른 반복 | checkpoint별 해상도·runtime·학습 목적 확인 | [공식 collection](https://huggingface.co/collections/Photoroom/prx) |
| **PRXPixel 7B** | pixel-space, VAE 없음 | 16–24 GB | VAE artifact 없는 pixel-space 연구 | Qwen3-VL text encoder·raw-pixel activation 고려 | [공식](https://huggingface.co/Photoroom/prxpixel-t2i) |

### 7.2 Stable Diffusion 1.5를 유지할 이유

SD 1.5는 순수 T2I 품질만 보면 오래된 계열이지만 다음 작업에서는 여전히 효율적이다.

- 4 GB GPU의 실시간 초안·thumbnail
- 수천 개의 특화 LoRA·embedding 활용
- 여러 ControlNet을 낮은 해상도로 조합
- 512 px 인페인팅·mask workflow
- 게임 asset·sprite·특정 일러스트 checkpoint
- 저비용 LoRA 학습과 개념 검증

단, community checkpoint는 training data·라이선스·악성 pickle 위험이 다르다. 가능한 경우 `safetensors`를 사용하고, 모델 카드와 base license를 확인한다.

### 7.3 SDXL을 기본 운영 모델로 둘 경우

SDXL은 8–12 GB에서 다음 균형이 좋다.

```text
1024×1024
batch 1
FP16 UNet
VAE tiled decode 필요 시 활성화
xFormers 또는 PyTorch SDPA
ControlNet 1개부터 시작
LoRA 1–3개, scale 기록
```

8 GB에서는 refiner까지 동시에 상주시키지 말고 base 생성 후 unload한다. 여러 ControlNet·IP-Adapter·upscaler를 함께 쓰면 12–16 GB가 더 안정적이다.

### 7.4 SD 3.5 Medium의 NF4 경로

SD 3.5 Medium은 bitsandbytes `NF4`를 적용할 수 있지만, 다음을 구분한다.

- transformer만 NF4인지
- T5·CLIP encoder도 양자화하는지
- VAE는 FP16/BF16인지
- `device_map`이 어떤 구성요소를 CPU로 보냈는지
- `torch_dtype`과 quant storage dtype

NF4는 파일을 GGUF Q4로 변환하는 것과 다르다. CUDA bitsandbytes 지원이 없는 환경에서는 동일 코드를 사용할 수 없다.

### 7.5 FLUX.1의 역할

FLUX.2가 공개되었어도 FLUX.1은 community LoRA·ControlNet·Nunchaku·GGUF 지원이 더 성숙할 수 있다.

| 선택 | 적합한 상황 |
| --- | --- |
| `FLUX.1-schnell` | 적은 steps, Apache 2.0 경로, 초안·고속 생성 |
| `FLUX.1-dev` | 품질·LoRA 생태계, 비상업 라이선스 조건을 수용하는 연구 |
| GGUF Q4/Q5 | 12–16 GB GPU 또는 CPU/Metal/Vulkan offload |
| Nunchaku SVDQuant | 지원 NVIDIA GPU에서 4-bit W4A4 고속 추론 |
| FP8 | 16–24 GB에서 Diffusers·ComfyUI 구성 |

### 7.6 SANA를 저메모리 후보에 넣는 이유

SANA는 linear attention과 높은 압축률의 DC-AE를 사용해 고해상도 효율을 목표로 한다. 공식 프로젝트는 0.6B·1.6B·4.8B 계열, 4-bit inference, ControlNet, LoRA와 2K/4K 경로를 제공한다. 4-bit가 8 GB 미만 GPU에서 실행 가능하다는 프로젝트 설명은 **해당 공식 workflow의 조건**으로 이해해야 하며, 모든 4K prompt가 8 GB에서 동일하게 동작한다는 보장은 아니다.

---

## 8. 이미지 편집·인페인팅·다중 참조

### 8.1 편집 작업별 선택

| 편집 작업 | 1순위 후보 | 저메모리 대안 | 추가 메모리 요인 | 실패 기준 |
| --- | --- | --- | --- | --- |
| 자연어 지시 편집 | Qwen-Image-Edit-2511, FLUX.2 klein | SDXL instruct/edit finetune | 입력 image encoder·conditioning | 비편집 영역 drift |
| 텍스트 교체 | Qwen Edit, Ovis 계열 workflow | SDXL inpaint + 후처리 | 고해상도 crop·OCR 확인 | 철자·font·원근 불일치 |
| 인물 의상·배경 변경 | FLUX.2, Qwen Edit, HiDream-O1 | SDXL inpaint + IP-Adapter | identity encoder·mask | 얼굴·체형·피부 변화 |
| 제품 합성 | FLUX.2 multi-ref, Qwen Edit | SDXL + IP-Adapter | 참조 2–4장·고해상도 | 로고·제품 형상 drift |
| 인페인팅 | SDXL inpaint, Qwen Edit | SD 1.5 inpaint | mask blur·crop·VAE | seam·조명 불일치 |
| 아웃페인팅 | Qwen Edit, FLUX 계열 | SDXL outpaint workflow | canvas가 커져 latent 증가 | 반복 texture·perspective 붕괴 |
| 다중 참조 | FLUX.2 9B/KV, HiDream-O1, Qwen Edit | IP-Adapter | 참조별 encoder embedding | 참조 혼합·identity collapse |
| skeleton·layout | HiDream-O1 | ControlNet OpenPose·depth | detector + condition network | 포즈는 맞지만 의미 불일치 |

### 8.2 입력 이미지 메모리

편집 모델은 생성 모델과 달리 입력 이미지를 encode하고 참조 특징을 유지한다. 다음이 추가된다.

```text
M_edit_extra ≈ M_input_decode
             + M_vision_encoder
             + M_reference_embeddings
             + M_mask_and_control
             + M_edit_attention
```

- 4장의 2048 px 참조 이미지를 그대로 넣지 말고 모델이 요구하는 최대 입력 크기와 crop 정책을 확인한다.
- EXIF orientation을 적용한 뒤 resize한다. 그렇지 않으면 mask와 원본 좌표가 어긋날 수 있다.
- 얼굴·제품 참조는 resize와 center crop이 identity 특징을 훼손할 수 있으므로 letterbox와 subject crop을 비교한다.
- 다중 참조에서 같은 객체를 중복 설명하면 prompt와 visual condition이 충돌할 수 있다.

### 8.3 비편집 영역 보존 평가

편집 결과는 “보기 좋음”만으로 평가하지 않는다.

| 지표 | 측정 방법 |
| --- | --- |
| 비편집 영역 보존 | mask 외부 LPIPS·SSIM·pixel diff |
| identity | 얼굴·제품 embedding cosine, 사람 검토 |
| 지시 충실도 | 편집 요구사항 checklist, VLM judge + 사람 검토 |
| 텍스트 정확도 | OCR CER/WER, 정확한 문자열 비교 |
| 구조 보존 | keypoint·segmentation·edge overlap |
| 색상 보존 | 지정 영역 ΔE, histogram 비교 |
| artifact | seam, halo, 반복 패턴, 손·눈 오류 태깅 |

### 8.4 mask 운영 원칙

```text
1. 원본과 mask의 width·height·orientation을 동일하게 만든다.
2. 흰색/검은색 의미를 workflow마다 확인한다.
3. 4–32 px blur/feather를 여러 값으로 비교한다.
4. 전체 canvas 대신 masked crop + context padding을 고려한다.
5. 결과를 원본 크기로 합성하고 mask 외부를 원본으로 복원할지 결정한다.
6. 원본·mask·prompt·seed·model·scheduler를 함께 보존한다.
```

### 8.5 다중 참조의 메모리 절감

- 참조 이미지를 사전 encode하고 embedding을 cache할 수 있는지 확인한다.
- FLUX.2 klein 9B-KV 같은 KV caching 변형은 editing 반복에서 유리할 수 있으나, cache 자체가 RAM·VRAM을 사용한다.
- 동일 참조로 여러 prompt를 생성할 때만 cache가 이득이다.
- reference encoder를 CPU에 두면 초기 encode는 느리지만 반복 denoise VRAM을 확보할 수 있다.
- 참조 이미지를 1장씩 추가하며 품질과 peak 메모리를 기록한다.

---

## 9. 타이포그래피·포스터·로고·UI

### 9.1 권장 모델

| 우선순위 | 모델 | 강점 | 메모리 기준 |
| ---: | --- | --- | ---: |
| 1 | Ovis-Image 7B | 긴 텍스트·포스터·배너·UI에 특화 | int4 12–16 GB, BF16 24–48 GB |
| 2 | Qwen-Image-2512 | 사진과 텍스트 조합, 다국어 가능성 | Q4 24 GB |
| 3 | Qwen-Image-Edit-2511 | 기존 포스터의 문구·요소 수정 | Q4 24 GB |
| 4 | HiDream-O1 | 긴 텍스트·다중 영역 layout, 2K | 48 GB |
| 5 | Z-Image-Turbo | 영·중문 텍스트와 빠른 반복 | Q4 8–12 GB |
| 6 | FLUX.2 klein | 일반 이미지와 텍스트·편집의 균형 | 4B FP8 16 GB |
| 7 | SDXL + ControlNet/inpaint | 정밀 layout을 외부 도구로 강제 | 8–16 GB |

### 9.2 생성 모델만으로 로고를 확정하지 않는다

이미지 생성 모델은 다음을 보장하지 않는다.

- 상표 비침해
- 기존 로고와의 유사성 없음
- 벡터 path의 기하학적 정확성
- font 라이선스
- 작은 크기에서의 가독성
- CMYK·spot color·인쇄 bleed

실무에서는 모델을 concept generator로 사용하고, 최종 로고·UI는 Figma·Illustrator·Inkscape 등에서 벡터로 재작성한다. 생성 결과에 포함된 텍스트를 그대로 배포하지 말고 OCR과 사람 검수를 통과시킨다.

### 9.3 텍스트 prompt 구조

```text
[매체] A2 vertical poster / mobile app onboarding screen
[정확한 문자열] headline: "RAM FOR LOCAL AI"
[언어] Korean subtitle: "메모리에 맞는 로컬 모델 선택"
[레이아웃] headline top center, subtitle below, footer at bottom-right
[타이포] bold geometric sans, high contrast, large tracking
[그래픽] abstract memory chips and neural network diagram
[색상] two-color palette, white background, black and cobalt elements
[금지] no extra letters, no watermark, no misspelled text
```

정확한 문자열은 따옴표와 줄바꿈을 명시하고, prompt에 불필요한 다른 문구를 넣지 않는다. 모델이 작은 footer를 반복해서 틀리면 footer는 후처리로 합성한다.

### 9.4 평가용 문자열 세트

```text
RAM-for-Local-AI
2026-07-21
₩19,900 / $14.99
GPU VRAM 16 GB
서울 · Busan · 東京 · Zürich
Q2 / Q3 / Q4 / FP8
support@example.com
Model ID: FLUX.2-klein-4B
```

20개 seed에서 exact-match OCR 비율을 계산한다. 한 장의 성공 사례보다 seed 간 안정성이 운영에 중요하다.

---

## 10. ControlNet·구조 제어·IP-Adapter

### 10.1 구성요소

| 구성요소 | 입력 | 제어 대상 | 추가 메모리 |
| --- | --- | --- | ---: |
| Canny ControlNet | edge map | 윤곽·구도 | 모델 크기 + feature maps |
| Depth ControlNet | depth map | 공간·원근 | depth detector + ControlNet |
| OpenPose | keypoints | 인체 포즈 | pose detector + ControlNet |
| Lineart/SoftEdge | 선화 | 일러스트·제품 윤곽 | detector + ControlNet |
| Segmentation | class mask | 영역·객체 배치 | segmenter + condition model |
| Tile ControlNet | low-res image | 업스케일 세부 | 큰 canvas와 tile overlap |
| IP-Adapter | reference image | 스타일·identity | vision encoder + adapter |
| InstantID/PhotoMaker류 | 얼굴 reference | identity | face detector·encoder·adapter |

### 10.2 메모리 계획

```text
M_controlled ≈ M_base_pipeline
             + Σ M_control_models
             + M_preprocessors
             + M_control_feature_maps
             + M_reference_encoder
```

- 8 GB: SD 1.5/SDXL + ControlNet 1개부터 시작한다.
- 12–16 GB: SDXL + ControlNet 1–2개, 또는 FLUX 저비트 + adapter 1개.
- 24 GB: Qwen·FLUX Q4 + 참조 adapter·upscale를 단계적으로 적재.
- 48 GB+: 대형 BF16 + 여러 control을 동시 상주시킬 수 있지만 peak를 실측한다.

### 10.3 preprocessor 분리

OpenPose·depth·segmentation detector를 생성 GPU에 계속 상주시키지 않는다.

```text
원본 이미지
  → CPU 또는 별도 GPU에서 pose/depth/edge 생성
  → control map을 PNG/NPY로 저장
  → detector unload
  → 생성 pipeline 로드
```

이 방식은 VRAM을 절약하고 control map을 재현 가능한 artifact로 남긴다. detector의 model revision과 resize 설정도 manifest에 기록한다.

### 10.4 control scale과 충돌

ControlNet scale이 너무 높으면 prompt가 무시되고, 너무 낮으면 구조가 흐려진다. 여러 control을 사용할 때 각 scale을 단순 합산하지 않는다.

| 증상 | 우선 조정 |
| --- | --- |
| 포즈는 정확하지만 스타일이 밋밋함 | pose scale 낮춤, start/end 범위 조정 |
| edge가 과도하게 선으로 남음 | canny threshold·scale·ending step 조정 |
| 얼굴 identity가 깨짐 | IP-Adapter scale·crop·face embedding 확인 |
| depth와 pose가 충돌 | 한 control씩 ablation, preprocessing 좌표 확인 |
| VRAM OOM | detector unload, control 1개 제거, 해상도·batch 감소 |

---

## 11. LoRA·스타일·캐릭터 일관성

### 11.1 LoRA 메모리의 두 부분

LoRA 파일 자체는 수십 MB–수 GB지만 실행 시에는 adapter tensor, merge buffer, attention activation이 추가된다.

```text
M_LoRA_runtime ≈ Σ M_adapter_weights
               + M_loader_or_merge_buffer
               + M_extra_activation
```

여러 LoRA를 적용하면 파일 합보다 peak가 커질 수 있다. runtime이 flyweight adapter를 유지하는지, base weight에 merge하는지 확인한다.

### 11.2 생태계별 특징

| base 계열 | LoRA 생태계 | 낮은 메모리 학습 | 실행 호환성 | 권장 용도 |
| --- | --- | --- | --- | --- |
| SD 1.5 | 매우 큼 | 가장 쉬움 | 거의 모든 UI | 스타일·캐릭터·저사양 |
| SDXL | 매우 큼 | 12–24 GB에서 가능 | ComfyUI·A1111·Diffusers | 1K 상업 제작 workflow |
| FLUX.1 | 큼 | 16–24 GB 이상 또는 quantized training | ComfyUI·Diffusers·Nunchaku 일부 | 사진·스타일·제품 |
| FLUX.2 | 성장 중 | architecture 지원 확인 | 최신 ComfyUI/Diffusers | 생성+편집 최신 실험 |
| Krea 2 Raw | 신규, 학습 기준선 지향 | 48–96 GB급 recipe부터 검증 | 공식·community trainer | 고품질 creative LoRA |
| Krea 2 Turbo | 추론용 | Raw에서 학습 후 전이 검증 | 8-step inference | 빠른 LoRA serving |
| Z-Image Base | 성장 중 | 16–48 GB 범위 recipe 검증 | Diffusers·sd.cpp 일부 | 다양성·fine-tuning |
| Qwen-Image | 성장 중 | 큰 encoder로 메모리 큼 | Diffusers·ComfyUI | 텍스트·인물·편집 |
| SANA | 공식 training 지원 | 효율 지향 | 공식 repo·Diffusers | 연구·저메모리 실험 |

### 11.3 LoRA 조합 규칙

```text
1. base model revision을 정확히 맞춘다.
2. 한 개의 LoRA로 baseline을 만든다.
3. scale 0.4 / 0.7 / 1.0을 비교한다.
4. 두 번째 LoRA를 추가하고 교차 grid를 만든다.
5. trigger token과 prompt 순서를 고정한다.
6. seed·sampler·steps·CFG를 고정한다.
7. merge 전 원본 checksum을 보존한다.
```

### 11.4 캐릭터 일관성

단일 캐릭터 LoRA만으로 다음이 모두 해결되지는 않는다.

- 얼굴 identity
- 의상·소품
- 신체 비율
- 여러 시점
- 감정·표정
- 동일 캐릭터 2명 이상 배치

일관성이 중요한 경우 LoRA + reference adapter + pose/layout control을 조합한다. 단, 세 조건을 동시에 강하게 걸면 prompt 자유도가 줄어든다. “정체성”, “구도”, “스타일”의 우선순위를 정하고 각 condition scale을 ablation한다.

### 11.5 quantized base와 LoRA

- GGUF Q4 base가 모든 LoRA를 지원하는 것은 아니다.
- Nunchaku quantized transformer는 지원되는 LoRA 형식·rank·module 범위를 확인한다.
- NF4로 로드한 Diffusers transformer에 adapter를 적용할 때 merge·unmerge가 제한될 수 있다.
- quantization 전에 LoRA를 merge한 checkpoint와 runtime adapter 적용 결과는 다를 수 있다.
- 품질 기준선은 BF16/FP16 base + LoRA로 만든다.

---

## 12. 메모리별 완성형 이미지 생성 스택

### 12.1 4 GB VRAM + 16 GB RAM

| 구성 | 선택 |
| --- | --- |
| base | SD 1.5 FP16 또는 pruned safetensors |
| 해상도 | 512×512, 필요 시 768 portrait |
| batch | 1 |
| control | ControlNet 1개, detector 후 unload |
| LoRA | 1–2개 |
| VAE | tiled decode 또는 경량 VAE |
| upscaler | 생성 후 별도 단계, CPU/VRAM unload |
| 대안 실험 | Z-Image-Turbo GGUF Q2 + CPU offload |

권장 workflow:

```text
prompt → SD 1.5 512 px → 2x latent/pixel upscale → 선택 영역 inpaint
```

4 GB에서 최신 대형 모델을 억지로 돌리는 것보다 SD 1.5를 빠르게 반복하고 후처리하는 편이 대개 생산적이다.

### 12.2 6–8 GB VRAM + 24–32 GB RAM

| 역할 | 기본 선택 | 품질 도전 |
| --- | --- | --- |
| T2I | SDXL FP16 low-VRAM | Z-Image Q3/Q4, SD3.5 Medium NF4 |
| 편집 | SDXL inpaint | Qwen Edit Q2 offload |
| control | SDXL ControlNet 1개 | 2개는 순차·낮은 해상도 |
| upscale | tiled ESRGAN/SwinIR | SD upscale workflow |
| UI | ComfyUI | sd.cpp GGUF 병행 |

운영 설정:

```text
1024×1024, batch 1
VAE tiled
preview 낮춤 또는 비활성화
control preprocessor unload
모델 전환 시 cache 정리
```

### 12.3 10–12 GB VRAM + 32–64 GB RAM

| 작업 | 권장 스택 |
| --- | --- |
| 범용 제작 | SDXL + ControlNet + LoRA + tiled upscale |
| 최신 품질 | Z-Image-Turbo Q4/Q8 또는 FLUX.1 Q4 |
| 타이포그래피 | Ovis int4 또는 Qwen-Image Q3 offload |
| 빠른 serving | SANA-Sprint·Z-Image-Turbo |
| 스타일 학습 | SDXL LoRA |

이 구간은 최신 대형 모델과 성숙한 SDXL workflow 사이의 선택이 중요하다. 한 장 품질이 아니라 **분당 유효 결과 수**를 비교한다.

### 12.4 16 GB VRAM + 32–64 GB RAM

권장 기본:

```text
FLUX.2 klein 4B FP8
1024×1024
batch 1
13 GB 전후 pipeline + 2–3 GB headroom
편집 참조는 1장부터 시작
```

대안:

- Z-Image-Turbo BF16 또는 Q8
- FLUX.1 dev Nunchaku/SVDQuant
- Qwen-Image Q3 + text encoder offload
- Ovis-Image int4
- SDXL + 여러 ControlNet·upscaler

16 GB에서 브라우저·게임·다른 CUDA 프로세스를 함께 실행하지 않는다. display VRAM이 많이 점유되면 1024²에서도 OOM이 날 수 있다.

### 12.5 20–24 GB VRAM + 64 GB RAM

| 역할 | 선택 |
| --- | --- |
| 최신 T2I·편집 | FLUX.2 klein 4B BF16/FP8 |
| 사진·텍스트 | Qwen-Image Q4/Q5 |
| 고속 T2I | Z-Image BF16/Turbo |
| 포스터 | Ovis BF16 경계 또는 int4 |
| LoRA ecosystem | FLUX.1·SDXL |
| 대형 도전 | Krea 2·HiDream FP8/CPU offload |

24 GB에서는 모델 하나를 안정적으로 실행하는 것이 목적이면 Q4/Q8 선택 폭이 넓다. 생성 모델·upscaler·VLM captioner를 모두 GPU에 상주시킬 필요는 없다.

### 12.6 32 GB VRAM + 64–128 GB RAM

```text
FLUX.2 klein 9B FP8
또는
Krea 2 Turbo BF16 + component offload
또는
Qwen-Image Q8 / Q4 + encoder 상주
```

- 1536²·다중 참조는 batch 1부터 측정한다.
- Krea 2 Turbo는 8-step 설정을 유지한다.
- 9B FLUX 비상업 라이선스를 배포 환경에서 확인한다.
- 다중 사용자는 single worker queue부터 시작한다.

### 12.7 48 GB VRAM + 96–128 GB RAM

| 워크로드 | 권장 구성 |
| --- | --- |
| Krea 2 | Raw/Turbo BF16, 1K–2K batch 1 |
| HiDream-O1 | Full BF16, 편집·personalization |
| Qwen-Image | BF16 denoiser + 일부 component offload 또는 Q8 전체 상주 |
| 모델 비교 | 두 모델을 순차 preload하거나 한 모델씩 worker 분리 |
| LoRA 연구 | Krea 2 Raw·FLUX·Qwen 학습 recipe 검증 |

48 GB는 대형 단일 모델에는 유용하지만, Python process 두 개가 각각 모델을 복제하면 즉시 부족해진다. 공유 server 또는 request queue를 사용한다.

### 12.8 64–96 GB VRAM + 128–192 GB RAM

- Qwen-Image-2512 BF16 전체 pipeline
- HiDream-O1·Krea 2의 생성·편집 worker 분리
- 대형 ControlNet·reference encoder·upscaler 동시 상주
- 2K batch 1–2 실험
- 여러 LoRA의 A/B grid 생성
- latency·throughput serving benchmark

64 GB 이상에서는 가중치보다 해상도·batch·동시성 activation이 병목이 되기 쉽다. 2K batch 2를 바로 적용하지 말고 1K batch 1 기준선에서 한 축씩 올린다.

### 12.9 Apple 16–32 GB

| 통합 메모리 | 권장 앱·runtime | 모델 |
| ---: | --- | --- |
| 16 GB | Draw Things·DiffusionBee·ComfyUI MPS | SD 1.5·SDXL·Z-Image 저비트 |
| 24 GB | ComfyUI MPS·MLX community runtime | SDXL, FLUX.1 Q4, Z-Image Q4/Q8 |
| 32 GB | ComfyUI/MLX | FLUX.2 4B FP8, Qwen Q2/Q3 |

- 브라우저 탭과 IDE를 줄인다.
- Activity Monitor의 memory pressure와 swap을 함께 본다.
- 첫 실행 compilation·shader cache 시간을 정상 inference 시간과 분리한다.
- MPS fallback이 CPU로 발생하는지 로그를 확인한다.

### 12.10 Apple 48–128 GB

| 통합 메모리 | 실전 범위 |
| ---: | --- |
| 48 GB | FLUX.2 9B, Qwen Q4, Krea/HiDream offload |
| 64 GB | Krea 2·HiDream, Qwen Q8, 다중 단계 workflow |
| 96 GB | 대형 BF16 한 모델 + 후처리, LoRA 실험 |
| 128 GB | Qwen BF16, 여러 모델 순차 비교, 로컬 연구 서버 |

통합 메모리가 충분해도 discrete high-end GPU와 연산 성능·대역폭이 같지는 않다. 모델이 “들어감”과 acceptable latency를 별도로 측정한다.

---


## 13. 해상도·종횡비·batch·steps·sampler

### 13.1 해상도는 총픽셀로 비교한다

같은 “1K”라도 종횡비에 따라 메모리가 다르다. 모델이 권장하는 resolution bucket과 8·16·32·64의 배수를 따른다.

| 해상도 | 총픽셀 | 1024² 대비 | 실전 용도 |
| ---: | ---: | ---: | --- |
| 512×512 | 0.26M | 0.25× | SD 1.5, thumbnail |
| 768×768 | 0.59M | 0.56× | 저VRAM 초안 |
| 832×1216 | 1.01M | 0.97× | portrait 1K bucket |
| 1024×1024 | 1.05M | 1.00× | 일반 기준선 |
| 1152×896 | 1.03M | 0.98× | landscape 1K bucket |
| 1344×768 | 1.03M | 0.98× | 16:9에 가까운 와이드 |
| 1536×1024 | 1.57M | 1.50× | 고품질 와이드 |
| 1536×1536 | 2.36M | 2.25× | 1.5K square |
| 2048×2048 | 4.19M | 4.00× | 2K square, tiled VAE 고려 |

가로·세로가 모델 patch size나 VAE downsample factor의 배수가 아니면 내부 padding·crop이 생길 수 있다. workflow가 자동 반올림한 실제 해상도를 metadata에 기록한다.

### 13.2 batch와 queue

`batch_size=4`와 이미지 1장을 네 번 순차 생성하는 것은 다르다.

| 방식 | VRAM | throughput | latency | 재현성 |
| --- | ---: | --- | --- | --- |
| batch 1 × 4회 | 낮음 | 낮거나 보통 | 첫 결과 빠름 | seed 관리 단순 |
| batch 4 × 1회 | 높음 | GPU 활용 증가 가능 | 첫 결과 늦음 | batch 내 RNG 확인 |
| server dynamic batch | 변동 | 동시 요청에서 유리 | queue 영향 | 요청별 seed·order 기록 |

소비자 GPU에서는 batch 1을 기본으로 하고, throughput이 필요할 때만 2→4로 올린다. 이미지 생성은 activation이 커서 batch가 거의 선형 이상으로 VRAM을 늘릴 수 있다.

### 13.3 steps는 모델 계열마다 의미가 다르다

| 계열 | 대표 steps/NFE | 주의 |
| --- | ---: | --- |
| SD 1.5·SDXL 일반 | 20–40 | sampler와 CFG에 따라 수렴 차이 |
| SDXL Turbo·Lightning·LCM | 1–8 | distilled scheduler 필요 |
| FLUX.1 schnell | 약 4 | `guidance`·scheduler를 공식 예제에 맞춤 |
| FLUX.1 dev | 약 20–30 | CFG 구현과 guidance 구분 |
| Z-Image-Turbo | **8 NFE** | 일반 30–50 steps를 강제하지 않음 |
| Z-Image Base | 약 50 계열 | negative prompt·CFG 지원 |
| Krea 2 Turbo | **8** | 공식 `guidance_scale=0.0` |
| Krea 2 Raw | 더 긴 sampling 탐색 | Turbo 설정을 그대로 복사하지 않음 |
| HiDream-O1 Full/Dev | 공식 preset 우선 | 편집은 Full 권장 |
| Qwen-Image Lightning 변형 | 4/8 등 변형별 | base와 Lightning scheduler 구분 |

steps를 두 배로 늘려도 품질이 단조롭게 좋아지지 않는다. distilled 모델은 과도한 steps에서 과포화·artifact·색상 붕괴가 나타날 수 있다.

### 13.4 CFG·guidance·negative prompt

- `guidance_scale=0` 또는 1을 쓰는 모델이 모두 같은 의미는 아니다.
- Diffusers parameter 이름과 ComfyUI node의 CFG가 내부적으로 동일한지 확인한다.
- distilled CFG-free 모델에 negative prompt를 넣어도 무시될 수 있다.
- Base 모델의 negative prompt 지원을 Turbo에 그대로 기대하지 않는다.
- prompt adherence가 약할 때 CFG만 올리지 말고 prompt 길이·text encoder truncation·scheduler를 확인한다.

### 13.5 sampler·scheduler 기록

최소 manifest:

```yaml
sampler:
  name: euler
  scheduler: simple
  steps: 28
  cfg: 3.5
  guidance: null
  denoise_strength: 1.0
  flow_shift: null
  seed: 123456789
```

ComfyUI의 `KSampler`, Diffusers scheduler, `stable-diffusion.cpp`의 `--sampling-method`·`--scheduler` 이름은 일대일 대응하지 않을 수 있다. 다른 runtime 비교에서는 “이름이 같음”보다 실제 algorithm·sigma schedule을 확인한다.

### 13.6 attention backend

| backend | 장점 | 주의 |
| --- | --- | --- |
| PyTorch SDPA | 기본 지원·설치 단순 | GPU·dtype별 kernel 선택이 다름 |
| xFormers | 일부 모델에서 메모리·속도 개선 | 최신 architecture 호환성 확인 |
| FlashAttention | 대형 DiT에서 유리 가능 | 설치·GPU compute capability·shape 제한 |
| SageAttention류 | community 고속 경로 | 출력 품질·수치 안정성 회귀 테스트 |
| split/cross attention | 저VRAM | 속도 저하 가능 |
| stable-diffusion.cpp `--diffusion-fa` | 지원 모델에서 메모리·속도 개선 | 모델·backend별 bug가 있을 수 있음 |

attention backend를 바꿀 때 같은 seed 결과가 bit-identical하지 않을 수 있다. 품질 회귀와 NaN·검은 이미지 여부를 확인한다.

### 13.7 권장 튜닝 순서

```text
모델·revision 고정
  → 1024²·batch 1·공식 steps
  → sampler/scheduler 고정
  → quant/precision 비교
  → 해상도 확대
  → ControlNet·LoRA 추가
  → batch 또는 동시성 확대
```

한 번에 두 축 이상 바꾸면 OOM과 품질 변화의 원인을 추적하기 어렵다.

---

## 14. 텍스트 인코더·VAE·업스케일러 메모리

### 14.1 텍스트 인코더

| encoder 계열 | 사용 예 | 메모리 특성 | 절감 방법 |
| --- | --- | --- | --- |
| CLIP-L/G | SD 1.x·SDXL | 비교적 작고 빠름 | encode 후 CPU offload |
| T5-XXL | FLUX.1·SD3.5 일부 | 수 GB–10 GB 이상, prompt encode가 큼 | FP8·int8·4-bit, CPU cache |
| Qwen2.5-VL | Qwen-Image | 매우 큼, vision/text 구성 확인 | quantized encoder·offload |
| Qwen3 4B | Z-Image | denoiser Q4와 별개로 수 GB | GGUF Q4/Q8·`--clip-on-cpu` |
| Qwen3-VL 계열 | Krea 2 등 최신 계열 | transformer 외 메모리 비중 큼 | component offload·prompt embedding cache |
| 통합 encoder 없음 | HiDream-O1 | 외부 encoder 파일은 없지만 unified model이 큼 | 모델 전체 quant·offload 지원 확인 |

prompt가 반복된다면 embedding을 cache할 수 있다. 단, 다음이 달라지면 cache key를 분리한다.

```text
model revision
text encoder revision·dtype
prompt·negative prompt
max sequence length
clip skip·prompt weighting
LoRA가 text encoder에 적용되는지
```

### 14.2 text encoder를 CPU에 둘 때

| 상황 | 효과 |
| --- | --- |
| 단일 prompt 1장 | prompt encoding 지연이 체감될 수 있음 |
| 같은 prompt로 여러 seed | 초기 지연 후 큰 영향이 작음 |
| 긴 prompt·다중 참조 | CPU RAM·latency 증가 |
| server 동시 요청 | CPU bottleneck·queue 발생 가능 |
| PCIe 3.0 이하 | component 이동 비용이 큼 |

GPU에 denoiser를 완전히 적재할 수 있게 되는 경우, text encoder CPU offload의 총시간이 오히려 더 짧을 수 있다.

### 14.3 VAE와 decoder

VAE 가중치는 비교적 작아 보여도 고해상도 decode의 activation이 크다.

| 기법 | 메모리 | 속도·품질 영향 |
| --- | ---: | --- |
| VAE slicing | batch 방향 절약 | batch가 클 때 유리 |
| VAE tiling | 고해상도 절약 | tile seam·시간 증가 가능 |
| FP16 VAE | 절약 | 일부 VAE에서 NaN·색상 이상 |
| BF16 VAE | 안정적일 수 있음 | GPU 지원 확인 |
| CPU VAE | VRAM 크게 절약 | decode가 느림 |
| TAESD preview | preview 경량화 | 최종 decode와 품질 다름 |
| tiled pixel upscale | 고해상도 절약 | overlap·seam 관리 |

검은 이미지·NaN이 발생하면 VAE를 FP32로 올리거나 force-upcast 설정을 확인한다. VAE만 바꾸면 색감·contrast·미세 질감이 바뀔 수 있으므로 checkpoint와 권장 VAE를 함께 기록한다.

### 14.4 업스케일러

| 종류 | 예 | 장점 | 메모리 전략 |
| --- | --- | --- | --- |
| pixel SR | Real-ESRGAN·SwinIR·4x-UltraSharp류 | 빠르고 prompt 불필요 | tile 128–512, overlap |
| latent upscale | SD/SDXL latent upscale | 원본 의미 유지 | denoise strength 낮게 시작 |
| diffusion upscale | SD x2/x4·ControlNet Tile | 새로운 세부 생성 | base+upscaler 순차 로드 |
| face restoration | CodeFormer·GFPGAN류 | 작은 얼굴 개선 | 얼굴 crop만 처리 |
| model-native 2K | HiDream·Krea·Qwen 등 | 한 단계 생성 | 높은 activation·시간 |

업스케일러를 base pipeline과 동시에 GPU에 올려 둘 필요가 없다.

```text
base 생성
  → 결과와 metadata 저장
  → base unload
  → upscaler load
  → tile upscale
  → 선택 영역만 inpaint
```

### 14.5 1K→2K와 native 2K 비교

| 방법 | 장점 | 단점 |
| --- | --- | --- |
| native 2K | 전체 구도와 텍스트를 한 번에 모델링 | VRAM·시간 큼, 모델이 실제 2K에 강해야 함 |
| 1K + pixel SR | 빠르고 원본 보존 | 새로운 구조적 세부가 적음 |
| 1K + diffusion upscale | 세부 개선 가능 | identity·텍스트 drift |
| 1K + tile ControlNet | 구조와 세부 균형 | workflow 복잡·seam 가능 |

텍스트가 있는 포스터는 native 2K가 항상 유리하지 않다. 1K에서 큰 headline을 정확히 만든 뒤 벡터 텍스트로 후처리하는 편이 더 안정적일 수 있다.

### 14.6 보조 모델까지 합산

실제 제작 pipeline에는 다음이 추가될 수 있다.

- prompt refiner LLM
- image captioning VLM
- NSFW·안전 classifier
- watermark detector·embedder
- OCR verifier
- face detector·embedding model
- background remover·segmentation model
- color transfer·upscaler

모든 보조 모델을 한 GPU에 상주시킬 필요가 없다. 작업 큐를 단계별 worker로 분리하거나 CPU·두 번째 GPU에 배치한다.

---

## 15. Hugging Face 직접 다운로드

### 15.1 CLI 설치와 로그인

```bash
python -m pip install -U "huggingface_hub[cli]"

# gated 모델 또는 private token이 필요한 경우
hf auth login
```

토큰은 shell history, workflow JSON, screenshot, repository에 넣지 않는다. 최소 권한 read token을 사용한다.

### 15.2 먼저 `--dry-run`

```bash
hf download black-forest-labs/FLUX.2-klein-4b-fp8 --dry-run
hf download krea/Krea-2-Turbo --dry-run
hf download HiDream-ai/HiDream-O1-Image --dry-run
hf download Qwen/Qwen-Image-2512 --dry-run
```

`--dry-run` 출력에서 다음을 확인한다.

- 총 다운로드 크기
- 같은 가중치의 Diffusers shard와 single-file 중복
- 필요한 text encoder·VAE 포함 여부
- LFS/Xet 파일명
- gated access 오류
- local cache에 이미 있는 파일

### 15.3 공식 single-file 다운로드

```bash
mkdir -p models/flux2

hf download black-forest-labs/FLUX.2-klein-4b-fp8 \
  flux-2-klein-4b-fp8.safetensors \
  --local-dir models/flux2/klein-4b-fp8

# 9B는 웹에서 라이선스 조건에 동의한 뒤 로그인 필요
hf download black-forest-labs/FLUX.2-klein-9b-fp8 \
  flux-2-klein-9b-fp8.safetensors \
  --local-dir models/flux2/klein-9b-fp8
```

single-file만 받아도 runtime이 별도 config·encoder·VAE를 요구할 수 있다. 사용하는 ComfyUI template나 sd.cpp 문서의 companion 파일 목록을 확인한다.

### 15.4 Krea 2

공식 코드 경로에서 단일 checkpoint만 받을 경우:

```bash
mkdir -p models/krea2

hf download krea/Krea-2-Turbo \
  turbo.safetensors \
  --local-dir models/krea2/turbo-single

hf download krea/Krea-2-Raw \
  raw.safetensors \
  --local-dir models/krea2/raw-single
```

Diffusers pipeline 전체를 받을 경우 먼저 파일 트리를 확인한 뒤 한 형식만 선택한다.

```bash
hf download krea/Krea-2-Turbo \
  --include "model_index.json" \
            "scheduler/*" \
            "tokenizer*/*" \
            "text_encoder*/*.json" \
            "text_encoder*/*.safetensors" \
            "transformer/*.json" \
            "transformer/*.safetensors" \
            "vae/*.json" \
            "vae/*.safetensors" \
  --local-dir models/krea2/turbo-diffusers
```

저장소 revision에 따라 디렉터리 이름이 달라질 수 있다. `--dry-run`에서 glob이 실제 파일을 선택하는지 확인한다.

### 15.5 HiDream-O1

```bash
hf download HiDream-ai/HiDream-O1-Image \
  --local-dir models/hidream/o1-full

hf download HiDream-ai/HiDream-O1-Image-Dev-2604 \
  --local-dir models/hidream/o1-dev-2604
```

전체가 크므로 local cache와 `--local-dir`에 중복 저장되는 방식을 이해한다. 운영에서는 revision SHA를 고정한다.

```bash
hf download HiDream-ai/HiDream-O1-Image \
  --revision <COMMIT_SHA> \
  --local-dir models/hidream/o1-full-pinned
```

### 15.6 Z-Image GGUF

```bash
mkdir -p models/z-image/denoiser

hf download leejet/Z-Image-Turbo-GGUF \
  z_image_turbo-Q4_K.gguf \
  --local-dir models/z-image/denoiser

hf download unsloth/Z-Image-GGUF \
  z-image-Q4_K_M.gguf \
  --local-dir models/z-image/denoiser
```

별도 text encoder와 VAE도 필요하다. 정확한 저장소와 파일은 [stable-diffusion.cpp Z-Image 문서](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md)의 현재 목록을 따른다.

### 15.7 FLUX.1 GGUF

파일명을 먼저 확인한다.

```bash
hf download leejet/FLUX.1-dev-gguf --dry-run
```

그 후 필요한 quant만 내려받는다.

```bash
hf download leejet/FLUX.1-dev-gguf \
  --include "*Q4_0*.gguf" \
  --local-dir models/flux1/dev-q4
```

저장소 파일명이 바뀔 수 있으므로 glob 결과를 dry-run에서 확인한다. 원본 FLUX.1-dev 라이선스가 quant에도 적용된다.

### 15.8 Qwen-Image GGUF

```bash
hf download city96/Qwen-Image-gguf --dry-run
hf download QuantStack/Qwen-Image-Edit-GGUF --dry-run
```

Q4 denoiser만 선택하는 예:

```bash
hf download city96/Qwen-Image-gguf \
  --include "*Q4_K_M*.gguf" \
  --local-dir models/qwen-image/t2i-q4

hf download QuantStack/Qwen-Image-Edit-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir models/qwen-image/edit-q4
```

Qwen-Image 계열은 encoder·mmproj·VAE companion 파일이 분리될 수 있다. 다운로드 후 다음처럼 manifest를 만든다.

```yaml
pipeline: qwen-image-2512
base_model: Qwen/Qwen-Image-2512
quant_repo: city96/Qwen-Image-gguf
files:
  denoiser: <EXACT_Q4_FILENAME>
  text_encoder: <EXACT_FILENAME>
  projector: <EXACT_FILENAME_OR_NULL>
  vae: <EXACT_FILENAME>
revisions:
  base: <SHA>
  quant: <SHA>
```

### 15.9 전체 snapshot을 받을 때

```bash
hf download Qwen/Qwen-Image-2512 \
  --local-dir models/qwen-image/2512-bf16

hf download ATH-MaaS/Ovis-Image-7B \
  --local-dir models/ovis/7b-bf16
```

대형 repo는 수십 GB가 필요하다. 다운로드 중 임시 파일과 Hugging Face cache까지 고려해 **최종 크기의 1.5–2배 디스크 여유**를 확보한다.

### 15.10 checksum과 파일 목록

```bash
find models -type f -print0 | sort -z | xargs -0 sha256sum > models/SHA256SUMS
find models -type f -printf '%s\t%p\n' | sort -n > models/FILES.tsv
```

Hugging Face revision SHA, local SHA-256, 원본·quant 저장소 URL, 라이선스를 함께 보존한다.

---

## 16. ComfyUI 구성

### 16.1 기본 디렉터리

| 모델 종류 | 기본 경로 |
| --- | --- |
| 통합 checkpoint | `ComfyUI/models/checkpoints/` |
| 분리 diffusion model·UNet·DiT | `ComfyUI/models/diffusion_models/` |
| text encoder·CLIP·T5 | `ComfyUI/models/text_encoders/` |
| VAE | `ComfyUI/models/vae/` |
| LoRA | `ComfyUI/models/loras/` |
| ControlNet | `ComfyUI/models/controlnet/` |
| CLIP Vision | `ComfyUI/models/clip_vision/` |
| IP-Adapter | custom node 설명에 따른 경로 |
| upscale model | `ComfyUI/models/upscale_models/` |
| style model | `ComfyUI/models/style_models/` |
| embedding | `ComfyUI/models/embeddings/` |

`unet/`은 legacy 경로로 남아 있을 수 있지만 새 구성에서는 `diffusion_models/`를 우선한다. custom node는 자체 경로를 사용할 수 있다.

### 16.2 공유 모델 저장소

`extra_model_paths.yaml.example`을 복사해 모델을 한 곳에서 관리할 수 있다.

```yaml
ram_for_local_ai:
  base_path: /mnt/ai-models
  is_default: true
  checkpoints: checkpoints/
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  clip_vision: clip_vision/
  vae: vae/
  loras: loras/
  controlnet: controlnet/
  upscale_models: upscale_models/
```

Windows에서는 YAML path quoting과 backslash를 주의한다. 시작 로그에서 실제 추가된 경로를 확인한다.

### 16.3 최소 workflow 구성

```text
Load Diffusion Model / Checkpoint
  + Load Text Encoder
  + Load VAE
  → Text Encode
  → Empty Latent / model-specific latent
  → Sampler
  → VAE Decode
  → Save Image
```

최신 모델은 일반 `CheckpointLoaderSimple`로 로드되지 않을 수 있다. 공식 workflow template의 loader node를 사용한다.

### 16.4 메모리 절약 순서

1. batch를 1로 내린다.
2. 해상도를 줄인다.
3. preview를 줄이거나 TAESD preview를 사용한다.
4. VAE tiled decode를 사용한다.
5. text encoder를 CPU offload한다.
6. FP8·Q4 diffusion model을 선택한다.
7. ControlNet·upscaler를 순차 로드한다.
8. 다른 workflow·model cache를 unload한다.

`--lowvram` 같은 CLI flag는 버전별 동작이 달라질 수 있으므로 현재 `python main.py --help`를 확인한다. model-specific custom node의 offload 옵션이 core flag보다 우선할 수 있다.

### 16.5 FLUX 계열 배치 예

```text
models/diffusion_models/flux1-dev.safetensors
models/text_encoders/clip_l.safetensors
models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
models/vae/ae.safetensors
models/loras/<flux-lora>.safetensors
```

ComfyUI 공식 FLUX 예시는 RAM이 충분하면 T5 FP16을, 메모리가 부족하면 FP8 T5를 사용하도록 안내한다. FP8 text encoder와 FP8 denoiser를 구분한다.

### 16.6 GGUF custom node

ComfyUI-GGUF 계열 custom node를 사용할 때:

- ComfyUI core와 node revision을 함께 고정한다.
- 지원 architecture 목록에서 FLUX·Qwen·Z-Image를 확인한다.
- denoiser GGUF만 있고 encoder loader가 다른 경우 workflow를 분리한다.
- K-quant 이름을 일반 checkpoint loader의 dtype 옵션과 혼동하지 않는다.
- quantized model + LoRA 호환성을 작은 test workflow로 검증한다.

### 16.7 custom node 보안

custom node는 단순 JSON plugin이 아니라 **로컬 Python 코드**다.

```text
검토할 항목
- install.py / requirements.txt / prestartup_script.py
- subprocess·shell 실행
- 외부 URL 다운로드
- telemetry
- credential·environment variable 접근
- output·input 디렉터리 외 파일 접근
- pickle·torch.load 사용
```

운영 서버에서는 allowlist된 node만 설치하고 Git commit SHA를 고정한다. ComfyUI Manager의 자동 업데이트를 production에서 무조건 실행하지 않는다.

### 16.8 workflow provenance

ComfyUI PNG에는 workflow metadata가 포함될 수 있다. 공유 전에 다음을 확인한다.

- 로컬 절대경로
- 사용자명·프로젝트명
- prompt의 개인정보
- custom API key
- 내부 제품명·미공개 asset
- 모델·LoRA 이름에 포함된 민감 정보

공개용 이미지는 metadata를 제거한 export를 따로 만든다. 재현용 내부 artifact는 원본 workflow JSON과 checksum을 보존한다.

---

## 17. Diffusers 실행

### 17.1 공통 환경

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -U pip
python -m pip install -U torch torchvision
python -m pip install -U diffusers transformers accelerate safetensors pillow
```

최신 모델이 아직 release 버전에 없으면 해당 모델 카드가 요구하는 Git revision을 별도 virtual environment에 설치한다.

```bash
python -m pip install -U git+https://github.com/huggingface/diffusers.git
```

production에서는 설치 후 commit과 package lock을 고정한다.

### 17.2 재현 가능한 공통 함수

```python
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def save_run(
    image,
    *,
    output: str,
    model_id: str,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    manifest = {
        "model_id": model_id,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "created_unix": time.time(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

### 17.3 Krea 2 Turbo

```python
from __future__ import annotations

import torch
from diffusers import Krea2Pipeline

MODEL_ID = "krea/Krea-2-Turbo"
SEED = 123456
PROMPT = "A clean editorial photograph of a red fox in fresh snow, soft overcast light"

pipe = Krea2Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = pipe(
    PROMPT,
    width=1024,
    height=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator(device="cuda").manual_seed(SEED),
).images[0]
image.save("outputs/krea2-turbo.png")
```

OOM이면 `.to("cuda")` 대신 지원되는 offload method를 사용한다. 2048²는 1024² 기준선이 안정된 후 시도한다.

### 17.4 Z-Image-Turbo

```python
from __future__ import annotations

import torch
from diffusers import DiffusionPipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SEED = 123456

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A studio photograph of a translucent mechanical keyboard, bilingual label EN/中文",
    width=1024,
    height=1024,
    num_inference_steps=9,  # pipeline의 NFE 계산과 공식 예제를 확인
    guidance_scale=0.0,
    generator=torch.Generator(device="cpu").manual_seed(SEED),
).images[0]
image.save("outputs/z-image-turbo.png")
```

공식 예제와 현재 Diffusers scheduler가 요구하는 steps를 우선한다. “8 NFE”와 API의 `num_inference_steps`가 항상 같은 정수 표현인지 runtime 문서를 확인한다.

### 17.5 Qwen-Image-2512

```python
from __future__ import annotations

import torch
from diffusers import DiffusionPipeline

MODEL_ID = "Qwen/Qwen-Image-2512"
SEED = 123456

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = (
    'A minimal Korean technology poster. Exact headline: "RAM FOR LOCAL AI". '
    'Exact subtitle: "메모리에 맞는 모델 선택". White background, precise grid layout.'
)

image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    generator=torch.Generator(device="cpu").manual_seed(SEED),
).images[0]
image.save("outputs/qwen-image-2512.png")
```

대형 모델의 offload에서는 generator device를 pipeline 요구사항에 맞춘다. 현재 모델 카드의 권장 sampling parameter를 manifest에 추가한다.

### 17.6 Ovis-Image 7B

```python
from __future__ import annotations

import torch
from diffusers import OvisImagePipeline

MODEL_ID = "ATH-MaaS/Ovis-Image-7B"

pipe = OvisImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt=(
        'A modern conference poster. Exact headline: "LOCAL AI 2026". '
        'Exact Korean subtitle: "이미지 생성 워크숍". No extra letters.'
    ),
    width=1024,
    height=1024,
).images[0]
image.save("outputs/ovis-poster.png")
```

Ovis 지원은 `diffusers>=0.36.0`을 확인한다. int4 AutoRound 모델은 공식 BF16 pipeline과 loader가 다를 수 있다.

### 17.7 SD 3.5 Medium NF4

```python
from __future__ import annotations

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import BitsAndBytesConfig

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    "A scientific illustration of a memory hierarchy, clean labels, white background",
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=4.5,
).images[0]
image.save("outputs/sd35-medium-nf4.png")
```

bitsandbytes·GPU·PyTorch 조합을 지원하는지 확인한다. text encoder까지 NF4인 예제가 아니므로 전체 pipeline이 4-bit라는 뜻은 아니다.

### 17.8 group offloading과 layerwise casting

Diffusers는 component·group offloading과 layerwise casting을 제공한다. API는 버전에 따라 변할 수 있다.

```python
import torch

# 모델이 지원하는 경우에만 사용
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2,
)

pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```

주의:

- normalization·modulation 계층은 FP8 저장에서 제외될 수 있다.
- PEFT/LoRA와 layerwise casting 조합은 모델별로 깨질 수 있다.
- group offload와 disk offload는 NVMe I/O를 병목으로 만들 수 있다.
- 같은 모델이라도 custom forward가 내부 dtype cast를 하면 지원되지 않을 수 있다.

### 17.9 peak VRAM 측정

```python
import time
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

start = time.perf_counter()
result = pipe(prompt="A red cube on a white table").images[0]

if torch.cuda.is_available():
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    print({"peak_allocated_gib": peak_gib, "peak_reserved_gib": reserved_gib})

print({"latency_s": time.perf_counter() - start})
```

첫 실행은 kernel compilation·model warm-up을 포함할 수 있다. warm-up 1회 후 3–10회 median을 기록한다.

### 17.10 오류 처리

```python
try:
    image = pipe(prompt="test", width=1024, height=1024).images[0]
except torch.OutOfMemoryError:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise RuntimeError(
        "CUDA OOM: batch=1, 768px, VAE tiling, encoder offload 순서로 낮추십시오."
    )
```

OOM 후 같은 process를 계속 사용하면 fragmentation·부분 로드 상태가 남을 수 있다. production worker는 요청 실패 후 안전하게 재시작하는 정책을 둔다.

---

## 18. `stable-diffusion.cpp`와 GGUF

### 18.1 적합한 경우

- Python 의존성을 줄이고 싶은 경우
- CPU·CUDA·ROCm/HIP·Metal·Vulkan·SYCL 등 여러 backend를 시험하는 경우
- GGUF Q2/Q3/Q4/Q5/Q6/Q8 diffusion model을 쓰는 경우
- mmap·CPU/GPU placement를 세밀하게 조절하는 경우
- FLUX·Qwen-Image·Z-Image·Krea 2·HiDream 등 최신 지원을 단일 CLI로 실험하는 경우

프로젝트는 활발히 개발되며 CLI가 바뀔 수 있다. **release 또는 commit SHA를 고정**한다.

### 18.2 빌드

```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
cd stable-diffusion.cpp
PINNED_COMMIT_OR_TAG="<commit-or-tag>"
git checkout "$PINNED_COMMIT_OR_TAG"

# NVIDIA CUDA
cmake -B build -DSD_CUDA=ON
cmake --build build --config Release -j

# Apple Metal
cmake -B build-metal -DSD_METAL=ON
cmake --build build-metal --config Release -j

# Vulkan
cmake -B build-vulkan -DSD_VULKAN=ON
cmake --build build-vulkan --config Release -j

# AMD ROCm/HIP
cmake -B build-hip -DSD_HIPBLAS=ON
cmake --build build-hip --config Release -j
```

현재 CMake option은 repository의 `CMakeLists.txt`와 build guide를 재확인한다.

### 18.3 SD 1.5 최소 실행

```bash
./build/bin/sd-cli \
  -m models/sd15/model.safetensors \
  -p "a small robot reading a book" \
  -W 512 -H 512 \
  --steps 24 \
  --seed 42 \
  -o outputs/sd15.png
```

### 18.4 Z-Image-Turbo GGUF

```bash
./build/bin/sd-cli \
  --diffusion-model models/z-image/z_image_turbo-Q4_K.gguf \
  --llm models/z-image/Qwen3-4B-Instruct-2507-Q8_0.gguf \
  --vae models/z-image/ae.safetensors \
  --clip-on-cpu \
  --vae-tiling \
  --vae-conv-direct \
  --diffusion-fa \
  --cfg-scale 1.0 \
  --steps 9 \
  --seed 366737809 \
  -W 1024 -H 1024 \
  -p "A minimalist hardware poster with the exact text RAM 16 GB" \
  -o outputs/z-image-q4.png
```

파일명·steps·sampler는 현재 [Z-Image guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md)를 따른다. 낮은 VRAM에서는 `--clip-on-cpu`, `--vae-tiling`, `--offload-to-cpu`의 지원 상태를 하나씩 검증한다.

### 18.5 runtime quantization과 사전 GGUF

기존 checkpoint를 load 시 quantize:

```bash
./build/bin/sd-cli \
  -m models/sd15/model.safetensors \
  --type q4_0 \
  -p "a red cube" \
  -W 512 -H 512 \
  -o outputs/q4-runtime.png
```

사전 변환:

```bash
./build/bin/sd-cli \
  -M convert \
  -m models/sd15/model.safetensors \
  -o models/sd15/model-q8_0.gguf \
  --type q8_0 \
  -v
```

일반 변환 문서는 `q4_0/q4_1/q5_0/q5_1/q8_0`을 설명하지만, 특정 최신 모델용 repo는 Q2_K·Q3_K·Q4_K를 제공할 수 있다. runtime이 해당 architecture와 quant type을 모두 지원하는지 확인한다.

### 18.6 CPU/GPU placement

| 옵션·전략 | 목적 | 주의 |
| --- | --- | --- |
| `--clip-on-cpu` | 큰 text encoder를 RAM에 배치 | prompt encoding 지연 |
| `--vae-tiling` | 고해상도 decode peak 감소 | seam·시간 |
| `--vae-conv-direct` | VAE 메모리 최적화 | 모델별 출력 검증 |
| `--offload-to-cpu` | 일부 weight·component CPU 배치 | backend별 안정성·PCIe 병목 |
| `--diffusion-fa` | attention 최적화 | 일부 revision에서 artifact·bug 가능 |
| mmap | 로드·RAM page 활용 | 느린 디스크·network FS 주의 |

옵션은 조합마다 테스트한다. 여러 최적화를 한 번에 켠 뒤 오류가 나면 원인을 찾기 어렵다.

### 18.7 GGUF 품질 평가

```text
BF16/F16 기준선
Q8_0
Q6_K
Q4_K_M 또는 Q4_K
Q3_K_M
Q2_K
```

각 quant마다 같은 30–100 prompt × 같은 seed를 생성하고 다음을 비교한다.

- prompt adherence
- 얼굴·손·작은 객체
- 긴 직선·기하 패턴
- 정확한 텍스트
- 미세 texture와 색 gradient
- image-editing에서 비편집 영역
- latency, peak VRAM, peak RAM

### 18.8 안정적 실행 wrapper

```bash
#!/usr/bin/env bash
set -euo pipefail

SD_CLI=${SD_CLI:-./build/bin/sd-cli}
MODEL=${MODEL:?MODEL is required}
PROMPT=${PROMPT:-"a red cube on a white background"}
SEED=${SEED:-42}
OUT=${OUT:-outputs/run-${SEED}.png}

mkdir -p "$(dirname "$OUT")"

"$SD_CLI" \
  --diffusion-model "$MODEL" \
  --seed "$SEED" \
  -W 1024 -H 1024 \
  --steps 20 \
  -p "$PROMPT" \
  -o "$OUT"

sha256sum "$OUT" > "${OUT}.sha256"
```

Z-Image·Qwen·FLUX처럼 companion 파일이 필요한 모델은 wrapper에 필수 environment variable 검사를 추가한다.

---

## 19. Nunchaku·SVDQuant·FP4·FP8

### 19.1 용어 구분

| 기술 | weight | activation | 주 runtime·hardware | 핵심 목적 |
| --- | --- | --- | --- | --- |
| BF16/FP16 | 16-bit | 16-bit | 범용 GPU | 품질 기준선 |
| FP8 weight | 8-bit 저장 또는 연산 | 8/16-bit 혼합 | 최신 CUDA·Diffusers·ComfyUI | 메모리 약 절반 수준 목표 |
| bitsandbytes NF4 | 4-bit weight | 16-bit activation 흔함 | CUDA | 손쉬운 weight-only 절감 |
| GGUF Q4 | 4-bit급 block quant | runtime 혼합 | sd.cpp, CPU·GPU backend | 이식성·mmap |
| SVDQuant W4A4 | 4-bit | 4-bit | Nunchaku 지원 NVIDIA | 속도와 메모리 동시 절감 |
| NVFP4 | 4-bit floating | kernel별 | Blackwell 등 지원 GPU | 더 높은 4-bit 품질·속도 목표 |
| INT4 AutoRound | 4-bit | 8/16-bit 혼합 가능 | Intel·지원 runtime | post-training quantization |

### 19.2 SVDQuant와 Nunchaku

[Nunchaku](https://github.com/nunchaku-ai/nunchaku)는 SVDQuant 기반 4-bit diffusion inference engine이다. SVDQuant는 outlier를 low-rank component로 분리해 W4A4 품질을 유지하려는 방식이다.

프로젝트가 공개한 FLUX.1-dev 결과에서는 BF16 대비 3.6× memory reduction과 지원 장치에서 큰 속도 향상을 제시한다. 이 수치는 특정 모델·GPU·software revision의 결과이므로 사용자의 GPU에서 직접 benchmark한다.

### 19.3 선택 기준

| 상황 | 권장 |
| --- | --- |
| NVIDIA 16 GB에서 FLUX.1 품질·속도 | Nunchaku SVDQuant 우선 비교 |
| CPU·Metal·Vulkan 이식성 | GGUF/stable-diffusion.cpp |
| Diffusers 코드 최소 변경 | FP8·NF4·layerwise casting |
| Blackwell GPU | NVFP4 지원 checkpoint·kernel 비교 |
| LoRA 다중 적용 | 현재 Nunchaku native node의 지원 범위 확인 |
| ControlNet·PuLID | 지원되는 exact architecture만 사용 |
| 최신 미지원 모델 | BF16/FP8 기준선을 유지하고 공식 지원 대기 |

### 19.4 설치 시 확인

```text
GPU compute capability
CUDA toolkit·driver
PyTorch version
Nunchaku wheel의 CUDA/Python 조합
ComfyUI-Nunchaku node revision
모델 quant revision
attention backend
LoRA converter version
```

prebuilt wheel이 GPU architecture를 포함하지 않으면 source build가 필요할 수 있다. 비공식 wheel은 공급망 위험을 검토한다.

### 19.5 Nunchaku와 offload

Nunchaku는 4-bit text encoder, per-layer CPU offload, asynchronous offload 같은 기능을 제공해 극저VRAM 실행 범위를 넓힌다. 하지만 “3–4 GB에서 실행”은 다음 비용을 포함할 수 있다.

- 높은 시스템 RAM
- 긴 초기 로드와 prompt encode
- PCIe 전송
- 낮은 batch
- 제한된 해상도
- 특정 LoRA·ControlNet 비호환

VRAM 하나의 숫자만 기록하지 말고 peak RAM, latency와 image/s를 함께 기록한다.

### 19.6 FP8의 두 형태

```text
A. 저장만 FP8, 계산 시 BF16/FP16으로 upcast
B. FP8 Tensor Core에서 일부 matmul도 FP8 수행
```

동일한 `.safetensors` FP8 파일이라도 GPU와 kernel에 따라 A 또는 B에 가까운 동작을 할 수 있다. `nvidia-smi`의 VRAM 감소만 보고 FP8 compute가 활성화됐다고 단정하지 않는다.

### 19.7 NVFP4

NVFP4는 일반 INT4·GGUF Q4와 다른 floating-point format과 scaling 체계를 사용한다. 주로 최신 NVIDIA GPU·Tensor Core·전용 kernel에서 이점을 얻는다.

- RTX 50/Blackwell 지원 여부를 model/runtime 문서에서 확인한다.
- Ampere·Ada에서 fallback하거나 다른 INT4 path를 사용할 수 있다.
- checkpoint 이름에 `NVFP4`가 있어도 text encoder·VAE가 NVFP4라는 뜻은 아니다.
- 품질 비교는 BF16·FP8·INT4·NVFP4를 같은 prompt/seed로 수행한다.

### 19.8 quant 선택 매트릭스

| VRAM | 우선 | 다음 | 최후 수단 |
| ---: | --- | --- | --- |
| 4–6 GB | SD 1.5 FP16, SANA 4-bit | Z/FLUX GGUF Q2/Q3 | aggressive offload |
| 8 GB | SDXL FP16, Z Q4 | NF4·Nunchaku | Qwen Q2 offload |
| 12 GB | Z Q8, FLUX Q4/Nunchaku | Ovis int4 | 대형 Q3 offload |
| 16 GB | FLUX.2 4B FP8 | Qwen Q3/Q4 offload | HiDream community 4-bit 실험 |
| 24 GB | Qwen Q4/Q5, FLUX BF16/FP8 | Krea/HiDream FP8 offload | BF16 disk offload |
| 32 GB | FLUX.2 9B FP8, Qwen Q8 | Krea 2 offload | 대형 BF16 경계 |
| 48 GB+ | BF16 기준선 | FP8로 동시성 확보 | Q4는 throughput 목적 |

---


## 20. NVIDIA·AMD·Apple·CPU 운영

### 20.1 플랫폼 비교

| 플랫폼 | 우선 runtime | 장점 | 주요 제약 |
| --- | --- | --- | --- |
| NVIDIA CUDA | Diffusers, ComfyUI, TensorRT 계열 | 가장 넓은 kernel·FP8·quant 생태계 | VRAM이 분리되어 model+activation headroom 필요 |
| AMD ROCm | Diffusers, ComfyUI ROCm, PyTorch | 대용량 VRAM GPU 선택지 | GPU·OS·ROCm compatibility matrix 확인 필수 |
| AMD Windows/WSL | 지원 범위 내 ROCm·DirectML·Vulkan runtime | desktop 접근성 | 모델·dtype·custom node 지원 편차 큼 |
| Apple Silicon | Diffusers MPS, ComfyUI MPS, Draw Things·MLX 계열 | 통합 메모리로 큰 모델 offload 없이 접근 | GPU·CPU·OS가 같은 메모리를 공유, 일부 kernel 미지원 |
| Intel GPU | OpenVINO·IPEX·Diffusers 지원 범위 | iGPU·Arc 활용 | model별 backend·dtype 지원 확인 |
| CPU | stable-diffusion.cpp, Diffusers CPU, OpenVINO | GPU 없이 실행, 대용량 RAM 활용 | 고해상도·대형 DiT는 매우 느림 |

### 20.2 NVIDIA

#### 권장 사항

- driver와 CUDA wheel 조합을 고정한다.
- 8 GB 이하에서는 SDXL·SD1.5와 tiled/offload부터 시작한다.
- 12–16 GB에서는 FLUX.2 Klein 4B, Z-Image 등 compact 최신 모델을 우선한다.
- 24 GB에서는 NF4·FP8·Q4급 대형 모델과 encoder staged load를 쓴다.
- 여러 request를 process별 복제하지 말고 queue와 한 process 내 model cache를 사용한다.

상태 확인:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

PyTorch 확인:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

FP8·FlashAttention·fused kernel은 GPU compute capability와 package build에 따라 지원 여부가 다르다.

### 20.3 AMD ROCm

ROCm은 “AMD GPU면 모두 같은 방식으로 지원”되는 단일 환경이 아니다. [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)에서 GPU, 운영체제, kernel, ROCm과 PyTorch 조합을 확인한다.

```bash
rocminfo | head
rocm-smi
```

PyTorch는 AMD 공식 설치 문서의 현재 wheel·container를 사용한다. CUDA 전용 custom node, bitsandbytes build, FlashAttention kernel은 그대로 동작하지 않을 수 있다.

운영 원칙:

- 우선 pure PyTorch SDPA workflow를 검증한다.
- CUDA-only node를 하나씩 대체한다.
- FP8·BF16 지원을 GPU generation별로 확인한다.
- OOM 메시지와 실제 VRAM fragmentation을 구분한다.
- ROCm·PyTorch·ComfyUI가 검증된 container digest를 보존한다.

AMD Instinct용 ComfyUI 문서와 Radeon용 예제는 대상 hardware가 다를 수 있으므로 적용 범위를 읽는다.

### 20.4 Apple Silicon 통합 메모리

Mac의 “32 GB 통합 메모리”는 32 GB VRAM과 같지 않다. macOS, WindowServer, 브라우저, Python, model weights, activation이 같은 pool을 공유한다.

| 통합 메모리 | 현실적인 시작점 | 권장 여유 |
| ---: | --- | --- |
| 8 GB | SD 1.5·소형 SANA, 512px | 앱 종료 후 batch 1 |
| 16 GB | SDXL, compact quant model | 4–6 GB OS 여유 |
| 24 GB | FLUX.1/2 compact quant, Z-Image quant | staged load |
| 32 GB | FLUX.2 Klein 4B, Qwen Q2/Q3 실험 | 8 GB 이상 여유 |
| 48–64 GB | Klein 9B quant, Qwen Q4, high-res SDXL | 여러 model 동시 상주 금지 |
| 96–128 GB | 대형 Q8/BF16 일부, 연구·batch | thermal·memory pressure 모니터링 |

MPS 확인:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

실행 예시:

```python
pipe = pipe.to("mps")
```

지원되지 않는 operation이 CPU fallback되면 속도와 RAM 사용량이 크게 변할 수 있다. `PYTORCH_ENABLE_MPS_FALLBACK=1`은 편리하지만 성능 문제를 숨길 수 있으므로 benchmark 결과에 기록한다.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Apple 전용 앱·runtime은 Core ML·Metal·MLX로 최적화된 별도 model format을 사용할 수 있다. Hugging Face BF16 repository를 그대로 넣을 수 있다고 가정하지 않는다.

### 20.5 CPU와 system RAM

CPU-only는 다음 용도에 적합하다.

- SD 1.5·소형 model의 비실시간 batch
- GGUF·low-bit transformer의 기능 검증
- prompt embedding·VAE·upscaler 일부 분리
- GPU 서버가 없는 CI smoke test
- 대용량 RAM을 활용한 staged load

CPU 실행에서 중요한 것은 RAM뿐 아니라 memory bandwidth, vector instruction, core 수와 NVMe 속도다.

```bash
lscpu
free -h
numactl --hardware
```

dual-socket 서버에서는 NUMA node를 넘나드는 weight access가 느릴 수 있다. process와 memory binding을 benchmark한다.

### 20.6 multi-GPU

GPU가 두 장이라고 모든 runtime이 자동으로 VRAM을 합쳐 하나의 모델을 실행하지는 않는다.

| 방식 | 설명 | 적합한 상황 |
| --- | --- | --- |
| model sharding | layer·expert를 여러 GPU에 분할 | Hunyuan 80B급, 32B BF16 |
| pipeline parallel | component/stage별 GPU 분리 | encoder GPU0, DiT GPU1, VAE GPU0 |
| data parallel | 각 GPU에 모델 복제 | 여러 독립 요청, VRAM 충분할 때 |
| sequential workers | GPU마다 다른 모델 | editing·upscale service 분리 |

PCIe만 사용하는 desktop과 NVLink/NVSwitch 서버의 성능은 크게 다르다. active parameter가 작아도 전체 MoE weight를 여러 GPU에 저장해야 할 수 있다.

### 20.7 storage와 cache

이미지 모델 repository는 shard·encoder·VAE·여러 precision을 포함해 수십–수백 GB가 될 수 있다.

```text
OS·runtime SSD
HF cache NVMe
검증된 production model read-only volume
사용자 input/output encrypted volume
temporary tiles·latents 별도 scratch
```

모델을 여러 virtualenv·container에 복사하지 말고 content-addressed cache를 공유하되 접근권한을 분리한다.

---

## 21. 파인튜닝·LoRA 메모리 빠른 기준

상세 계산은 향후 [파인튜닝 메모리 가이드](../operations/fine-tuning-memory.md) **(예정)**에 분리한다. 아래 값은 **batch 1, gradient checkpointing, memory-efficient optimizer, 낮은 rank, 512–1024px, 단일 GPU LoRA**를 가정한 보수적 시작 범위다. 실제 요구량은 optimizer, trainable layer, text encoder 학습, resolution, aspect bucket과 cache 전략에 따라 달라진다.

### 21.1 추론 메모리로 학습 가능 여부를 판단하지 않는다

학습에는 다음이 추가된다.

```text
trainable parameter
+ gradient
+ optimizer state
+ activation for backward
+ data batch·augmentation
+ EMA 또는 checkpoint save buffer
```

Q4로 추론되는 모델이 같은 8 GB에서 LoRA 학습되는 것은 아니다.

### 21.2 빠른 기준표

| base 계열 | LoRA 권장 VRAM 시작점 | 편안한 범위 | 비고 |
| --- | ---: | ---: | --- |
| SD 1.5 | 6–8 GB | 12 GB | 512px, UNet LoRA, text encoder 고정 |
| SDXL | 12 GB | 16–24 GB | 1024px, bucket·checkpointing 필요 |
| SD 3.5 Medium 2B | 16 GB | 24–32 GB | text encoder 학습 여부가 큼 |
| SANA 0.6B | 8–12 GB | 16 GB | 공식 training recipe와 license 확인 |
| FLUX.1 dev | 24 GB | 48 GB | quantized/QLoRA recipe와 optimizer에 민감 |
| FLUX.2 Klein 4B | 24–32 GB | 48 GB | generation/editing dataset와 architecture support 확인 |
| Z-Image 6B | 32 GB | 48–80 GB | Base가 fine-tuning에 적합, Turbo 제한 확인 |
| Krea 2 Raw | 80–96 GB | 120 GB+ | Raw에서 학습 후 Turbo 적용 workflow 검증 |
| Qwen-Image 20B | 80 GB+ | 120–192 GB | DiT뿐 아니라 encoder·activation 고려 |
| 32B급 FLUX.2 dev | 80–120 GB+ | multi-GPU | rank·trainable layer·resolution에 따라 크게 변동 |

이 표는 하드웨어 구매 보장이 아니라 첫 실험의 capacity planning 기준이다. 공식 training recipe가 더 높은 최소치를 제시하면 공식 값을 우선한다.

### 21.3 메모리 절약 순서

1. text encoder를 고정하고 embedding을 cache한다.
2. dataset 이미지를 native training resolution·aspect bucket으로 전처리한다.
3. batch 1 + gradient accumulation을 사용한다.
4. gradient checkpointing을 켠다.
5. memory-efficient attention을 사용한다.
6. 8-bit optimizer 또는 검증된 low-memory optimizer를 사용한다.
7. mixed precision(BF16/FP16)을 사용한다.
8. base weight의 검증된 quantization/QLoRA를 적용한다.
9. optimizer·activation offload 또는 multi-GPU를 고려한다.

### 21.4 text encoder 학습

text encoder까지 학습하면 특정 이름·희귀 단어·스타일 token 학습이 개선될 수 있지만 메모리와 overfitting 위험이 늘어난다. 다음 순서로 비교한다.

```text
A. DiT/UNet LoRA only
B. + text encoder LoRA, 낮은 LR
C. + 더 높은 rank
```

동시에 여러 변수를 바꾸지 않는다.

### 21.5 dataset 저장과 RAM

이미지 전체를 RAM cache하면 대규모 dataset에서 system OOM이 발생한다. latent·embedding cache를 사용할 때 필요한 disk를 계산한다.

```text
cache disk ≈ sample 수 × sample당 latent/embedding byte × augmentation variant
```

random crop·flip·caption dropout을 매 epoch 적용하려면 완전한 latent cache가 학습 다양성을 줄일 수 있다.

### 21.6 데이터 품질

LoRA 품질은 model size보다 dataset 품질에 더 민감할 수 있다.

- 중복·near-duplicate 제거
- 저해상도·watermark·손상 이미지 제거
- caption의 주제·스타일·구도 분리
- 학습하지 말아야 할 개인정보·copyrighted asset 검토
- 얼굴·미성년자·민감속성 동의와 법적 근거
- train/validation identity leakage 방지
- trigger word 없이도 base prompt를 따르는지 평가

### 21.7 checkpoint 전략

```text
save every N steps
+ validation prompt grid
+ fixed seed
+ optimizer state는 필요한 checkpoint만
+ best와 last 분리
+ base revision·dataset manifest·code commit 저장
```

LoRA 파일명에 step·rank·alpha·base revision을 포함한다.

```text
product-style_flux2-klein4b_r16-a16_step2400_base-<shortsha>.safetensors
```

---

## 22. 라이선스·보안·개인정보·출처표시

### 22.1 대표 라이선스 빠른 표

라이선스는 모델 세대·크기·checkpoint마다 다르다. “같은 브랜드”라는 이유로 다른 모델의 조건을 적용하지 않는다.

| 모델 | 대표 weight 라이선스·접근 | 운영 전 확인할 점 |
| --- | --- | --- |
| FLUX.2 Klein 4B | Apache 2.0 | component별 제3자 라이선스, 사용 runtime |
| FLUX.2 Klein 9B | 비상업 조건 | 상업 서비스·내부 업무 범위 |
| FLUX.2 dev | 비상업 조건 | output·fine-tune·서비스 조건 |
| FLUX.1 schnell | Apache 2.0 | 해당 checkpoint와 component 확인 |
| FLUX.1 dev | FLUX 비상업 라이선스 | 상업 사용 금지·조건 확인 |
| Qwen-Image 계열 | Apache 2.0로 배포된 공식 checkpoint | variant·encoder 라이선스 확인 |
| Z-Image | Apache 2.0 | Base/Turbo별 model card |
| Ideogram 4 | gated, 비상업 라이선스 | 약관 동의, commercial restriction |
| Krea 2 | Krea community license | 상업·재배포·fine-tune 조건 |
| HiDream O1 | 공식 카드의 MIT 표기 확인 | dependency·dataset 제한 별도 검토 |
| Stable Diffusion 3.5 | Stability Community License | 연 매출 기준 등 commercial 조건 최신 약관 확인 |
| SDXL | OpenRAIL++ 계열 | prohibited use와 redistribution |
| SD 1.5 | OpenRAIL-M 계열 | derivative·distribution 조건 |
| SANA-Sprint | NVIDIA·Gemma 관련 조건 | research·상업 범위와 encoder 조건 |
| HunyuanImage | Tencent model license | variant별 상업·서비스 조건 |

레포지토리의 `license` metadata와 모델 카드 본문, linked license file이 충돌하면 법무 검토 후 보수적으로 적용한다.

### 22.2 모델 파일 공급망

이미지 생성 생태계는 custom node·workflow·LoRA·checkpoint 공유가 활발해 공격면이 넓다.

```text
위험
- pickle checkpoint의 arbitrary code execution
- malicious custom node와 install script
- dependency confusion·typosquatting
- Git branch가 바뀐 후 재설치
- workflow가 외부 URL·shell·filesystem node 호출
- 모델 저장소의 remote code
- 가짜 quant 파일·변조된 VAE·LoRA
```

통제:

- `safetensors`와 GGUF를 우선한다.
- repository commit과 SHA-256을 고정한다.
- custom node는 code review 후 allowlist한다.
- `trust_remote_code=True`는 기본 금지하고 필요한 경우 격리한다.
- package lockfile·container digest·SBOM을 보존한다.
- runtime user에서 cloud credential·SSH key를 제거한다.
- model volume은 read-only로 mount한다.
- 다운로드 host와 production host를 분리할 수 있다.

### 22.3 입력 이미지 검증

공격자는 작은 파일로 매우 큰 bitmap을 만들거나 parser 취약점을 노릴 수 있다.

```text
검사 항목
MIME magic bytes
지원 codec allowlist
compressed file size
width × height와 총 pixel
frame 수와 animation
ICC profile·EXIF·XMP 크기
alpha channel
decompression timeout
malformed image 처리
```

Pillow 예시:

```python
from __future__ import annotations

from io import BytesIO
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def validate_image(payload: bytes) -> tuple[int, int, str]:
    if not payload:
        raise ValueError("empty image")
    if len(payload) > MAX_BYTES:
        raise ValueError("compressed image exceeds limit")

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS

    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError(f"unsupported format: {image.format}")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise ValueError("pixel count exceeds limit")
            return width, height, str(image.format)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image") from exc
```

실제 서비스에서는 parser를 별도 low-privilege process로 격리한다.

### 22.4 EXIF와 위치정보

참조 사진의 EXIF에는 GPS, 촬영시간, 기기 정보가 포함될 수 있다. 모델 입력 전에 제거하더라도 감사 목적의 원본 보관이 필요한지 정책으로 정한다.

```python
from PIL import Image

with Image.open("input.jpg") as source:
    clean = source.convert("RGB")
    clean.save("clean.jpg", quality=95, exif=b"")
```

결과 파일에도 원본 EXIF를 무심코 복사하지 않는다.

### 22.5 prompt와 결과물 개인정보

- prompt에 고객명·주소·의료·법률·사내 프로젝트명을 넣을 수 있다.
- edit reference에는 얼굴·신분증·화면 캡처가 들어갈 수 있다.
- UI history·workflow JSON·console log가 원문을 보존할 수 있다.
- thumbnail·upscale tile·latent cache도 민감정보가 될 수 있다.

운영 원칙:

```text
최소 수집
암호화 저장
tenant 분리
TTL 자동 삭제
관리자 열람 감사
backup 포함 삭제 정책
prompt 본문 없는 metric log
```

### 22.6 합성 이미지 오용 방지

다음과 같은 사용은 법적·윤리적 위험이 크다.

- 동의 없는 성적 합성물
- 미성년자 성적 콘텐츠
- 사기·신원 도용·허위 증거
- 실제 인물의 명예를 훼손하는 조작
- 선거·재난·보도 맥락의 오인 유도
- 타인의 상표·저작물을 권리 없이 복제

조직용 서비스는 허용 사용 정책, 신고·차단 절차, 사람 검토, 법 집행 요청 처리, 모델별 safety filter와 audit를 마련한다. 로컬 실행이라는 이유로 책임이 사라지지 않는다.

### 22.7 provenance와 출처표시

프로젝트 요구에 따라 다음을 저장한다.

```yaml
is_ai_generated: true
model: owner/repo
revision: commit-sha
quantization: bf16_or_q4
workflow_sha256: ...
prompt_hash: ...
seed: 20260721
source_assets:
  - asset-id-or-consent-record
post_processing:
  - svg_text_overlay
  - tiled_upscale_2x
created_at_utc: "..."
operator: pseudonymous-id
```

C2PA·Content Credentials를 적용할 수 있지만 metadata는 제거될 수 있으므로 내부 provenance database와 함께 쓴다. watermark만으로 진위를 보장하지 않는다.

### 22.8 저작권·상표·초상권

모델 라이선스가 output의 권리를 자동 보장하지 않는다. prompt, reference image, LoRA dataset, logo, character, font와 최종 사용 지역의 법을 각각 검토한다. 상업 광고·제품 패키지·보도·의료·정치 콘텐츠는 사람 검수와 권리 clearance가 필요하다.

### 22.9 폰트

이미지에 글자를 후합성할 때 font license도 배포 범위에 포함된다.

- desktop license와 web/app embedding license를 구분한다.
- font 파일을 Git 저장소나 결과 패키지에 무단 포함하지 않는다.
- 결과 이미지에 rasterized glyph를 사용하는 조건을 확인한다.
- OFL font도 reserved font name과 redistribution 조건을 읽는다.

### 22.10 서비스 네트워크 격리

```text
frontend/API
  → schema validator
  → malware/image parser sandbox
  → GPU queue
  → allowlisted workflow worker (network disabled)
  → output sanitizer/provenance
  → object storage
```

GPU worker에 인터넷이 필요하지 않다면 egress를 차단한다. 모델 다운로드는 별도 build 단계에서 수행한다.

---

## 23. 평가·재현성·운영 체크리스트

### 23.1 “가장 좋은 모델” 대신 작업별 평가셋을 만든다

| 평가 subset | 예시 |
| --- | --- |
| 일반 photorealism | 인물·제품·실내·자연·야간 |
| prompt adherence | 개수·색·상대 위치·행동·재질 |
| typography | 한글·영문·숫자·URL·가격·긴 문장 |
| layout | 포스터·UI·인포그래픽·패키지 |
| editing | 색 변경·객체 추가/삭제·배경 교체 |
| identity | 같은 인물·제품·캐릭터 다중 장면 |
| structure control | edge·depth·pose·segmentation |
| safety | 실제 인물·민감속성·금지 요청·prompt injection |
| quantization | BF16/FP8/Q8/Q6/Q4/Q3/Q2 동일 seed 비교 |
| operations | load time·peak RAM/VRAM·images/min·failure rate |

### 23.2 양자화 A/B

같은 모델 family에서 다음을 고정한다.

```text
prompt·negative prompt
seed
resolution·aspect ratio
steps·sampler·scheduler
guidance
text encoder·VAE precision
reference images·mask
runtime commit
```

비교 항목:

- 주제와 개수 정확성
- 작은 물체·손·눈·치아
- 미세 text와 spelling
- 배경 반복·texture collapse
- 색상·dynamic range
- reference identity
- peak memory·시간
- crash·NaN·black image 빈도

Q2가 30% 작아도 실패율과 재생성 횟수가 늘면 실제 throughput은 더 나쁠 수 있다.

### 23.3 사람 평가

blind pairwise review를 권장한다.

```text
질문 예
A와 B 중 prompt를 더 정확히 따른 결과는?
A와 B 중 글자가 더 정확한 결과는?
A와 B 중 reference 인물을 더 잘 보존한 결과는?
A와 B 중 상업 결과물로 수정 비용이 적은 결과는?
```

모델명·quant를 숨기고 prompt별로 순서를 무작위화한다.

### 23.4 자동 metric의 한계

| metric | 용도 | 한계 |
| --- | --- | --- |
| CLIP similarity | prompt·image 의미 유사도 | 글자·개수·미세 관계를 놓침 |
| aesthetic predictor | 선호도 proxy | 학습 편향·스타일 편향 |
| FID/KID | dataset distribution 비교 | 작은 개인 평가셋과 단일 이미지에 부적합 |
| LPIPS | 편집 전후 perceptual distance | 바람직한 변화와 오류를 구분 못함 |
| OCR exact match | 화면 내 text 정확도 | stylized font·layout에서 OCR 자체 오류 |
| face embedding | identity 유사도 | 생체정보·인구집단 편향·오탐 |
| segmentation/edge IoU | 구조 보존 | 자연스러운 변화까지 penalty 가능 |

자동 metric과 사람 검수를 함께 사용한다.

### 23.5 typography 평가

```text
1. 생성된 text 영역 crop
2. OCR 2개 이상 또는 OCR + 수동 검수
3. Unicode normalization(NFC)
4. character exact match·word error rate
5. 줄바꿈·대소문자·구두점 별도 점수
6. extra text hallucination 검사
```

한글은 자모 분해 상태와 완성형을 normalization한 뒤 비교한다.

### 23.6 편집 평가

편집 목표와 보존 목표를 분리한다.

```yaml
edit_target:
  region: jacket
  instruction: change blue to red
  success_metric: color_delta_in_mask
preserve:
  - face_identity
  - background_pixels_outside_mask
  - logo_text
```

전체 LPIPS 하나로 평가하면 바꿔야 할 영역과 지켜야 할 영역이 섞인다.

### 23.7 성능 benchmark

최소 다음 두 상황을 측정한다.

- **cold:** process 시작 후 model load 포함 첫 이미지
- **warm:** model 상주 후 반복 생성

```text
load_seconds
time_to_first_image
warm_seconds_per_image
images_per_hour
peak_vram_gib
peak_ram_gib
NVMe read bytes
energy 또는 평균 power
failure/OOM rate
```

offload 환경에서는 PCIe·SSD throughput도 기록한다.

### 23.8 benchmark manifest 예시

```yaml
schema_version: 1
run_id: flux2-klein4b-q4-1024-20260721
created_at_utc: "2026-07-21T00:00:00Z"

hardware:
  gpu: "NVIDIA GeForce RTX 4090"
  vram_gib: 24
  system_ram_gib: 64
  cpu: "..."
  os: "Ubuntu 24.04"
  driver: "..."

runtime:
  name: "ComfyUI"
  revision: "<commit>"
  pytorch: "<version>"
  cuda_or_rocm: "<version>"
  custom_nodes:
    ComfyUI-GGUF: "<commit>"

model:
  repo: "unsloth/FLUX.2-klein-4B-GGUF"
  revision: "<commit>"
  file: "<exact-Q4-file>.gguf"
  sha256: "..."
  text_encoder_repo: "..."
  text_encoder_quant: "Q5_K_M"
  vae_repo: "..."

workflow:
  sha256: "..."
  width: 1024
  height: 1024
  batch_size: 1
  steps: 4
  sampler: "..."
  scheduler: "..."
  guidance: 1.0
  seed: 20260721

metrics:
  cold_seconds: null
  warm_seconds: null
  peak_vram_gib: null
  peak_ram_gib: null
  ocr_exact_match: null
  human_pairwise_win_rate: null
```

### 23.9 운영 acceptance gate

새 모델·quant·runtime을 production으로 올리기 전에 다음을 통과한다.

```text
[ ] model·encoder·VAE·LoRA revision 고정
[ ] license와 gated 약관 검토
[ ] checksum·malware·custom node 검토
[ ] 대표 해상도 peak VRAM/RAM 측정
[ ] 100회 이상 OOM·누수 soak test
[ ] typography·editing·identity benchmark
[ ] unsafe request·reference image policy test
[ ] tenant 간 cache·output 분리 test
[ ] timeout·cancel·worker recovery test
[ ] rollback image와 이전 model 보존
```

### 23.10 재현성의 현실적 한계

GPU architecture, kernel, mixed precision, attention backend와 PyTorch version이 달라지면 같은 seed라도 pixel-identical 결과가 나오지 않을 수 있다. 재현성 수준을 구분한다.

1. **artifact 재현:** 같은 model·workflow·prompt·seed를 확보한다.
2. **지각적 재현:** 구도와 내용이 실질적으로 같다.
3. **pixel 재현:** byte-level 또는 pixel-level 동일하다.

대부분의 연구·운영에서는 1과 2를 목표로 하고, 규제·forensics 환경에서는 3의 가능 여부를 별도 검증한다.

---

## 24. 문제 해결

### 24.1 CUDA out of memory

다음 순서로 하나씩 적용한다.

1. batch size를 1로 만든다.
2. reference image 수와 해상도를 줄인다.
3. output 해상도를 1024 또는 native resolution으로 내린다.
4. VAE tiling·slicing을 켠다.
5. preview와 intermediate cache를 끈다.
6. text/vision encoder를 prompt encoding 후 offload한다.
7. main transformer를 Q6/Q5/Q4 또는 FP8/NF4로 바꾼다.
8. model CPU/sequential offload를 켠다.
9. ControlNet·upscaler를 순차 실행한다.
10. process를 재시작해 fragmentation·누수를 제거한다.

### 24.2 model file은 들어가는데 실행 중 OOM

정상적인 현상일 수 있다. 파일은 압축·양자화 weight만 나타내며 runtime에는 다음이 추가된다.

```text
dequantization/workspace
attention activation
latent·image buffer
text encoder·VAE
reference encoder
UI preview·cache
CUDA/ROCm context
```

30초 표의 장착 메모리와 실제 peak benchmark를 따른다.

### 24.3 검은 이미지·NaN

가능한 원인:

- 잘못된 VAE 또는 scaling factor
- VAE FP16 overflow
- model과 text encoder version 불일치
- unsupported FP8/NF4 kernel
- 지나치게 높은 guidance
- 잘못된 scheduler·latent format
- quant loader bug

조치:

```text
공식 BF16/FP16 workflow로 baseline
→ VAE만 FP32
→ quant 제거
→ 공식 scheduler·steps 복원
→ custom node update/rollback
→ 같은 파일 checksum 확인
```

### 24.4 색상이 회색·과포화·밴딩

- VAE mismatch를 확인한다.
- preview가 아니라 저장된 원본 PNG를 본다.
- ICC profile과 브라우저 color management를 확인한다.
- quantized VAE 대신 원본 VAE를 사용한다.
- tiled VAE overlap을 늘린다.
- 후처리 pipeline의 8-bit 변환 시점을 확인한다.

### 24.5 prompt를 무시한다

- 모델의 prompt template·maximum sequence length를 확인한다.
- text encoder quant를 Q5/Q8 또는 BF16으로 높인다.
- 장문을 핵심 subject·relation·style로 구조화한다.
- Turbo 모델의 guidance 설정을 모델 카드대로 돌린다.
- LoRA weight를 낮추고 base only와 비교한다.
- reference image가 prompt보다 과도하게 강하지 않은지 확인한다.

### 24.6 글자가 틀린다

1. 문자열을 따옴표로 짧게 지정한다.
2. 한 이미지에 긴 문단을 넣지 않는다.
3. Ideogram 4·Qwen-Image·Z-Image 등 typography 강점 모델을 비교한다.
4. text encoder 정밀도를 높인다.
5. OCR 자동 검증과 재시도를 사용한다.
6. 최종 정확성이 필요하면 SVG·Pillow로 후합성한다.

### 24.7 얼굴·인물이 매번 달라진다

- reference crop을 정규화한다.
- native multi-reference 또는 IP-Adapter/FaceID를 사용한다.
- identity LoRA를 검증한다.
- denoise·reference strength를 낮춘다.
- face restoration을 끄고 원본 생성 결과와 비교한다.
- 인물의 특징을 prompt로 중복 설명하되 민감속성·동의 정책을 적용한다.

### 24.8 mask 밖이 바뀐다

- mask 방향과 blur를 확인한다.
- crop inpaint와 full-frame inpaint를 비교한다.
- denoise를 낮춘다.
- seed·scheduler를 고정한다.
- 편집 후 mask 외부를 원본과 alpha composite한다.
- 모델이 global relighting을 수행하는 native edit 모델인지 확인한다.

### 24.9 tile seam

- overlap을 늘린다.
- tile size를 늘릴 수 있으면 늘린다.
- per-tile random seed가 아니라 일관된 noise 전략을 쓴다.
- color correction을 global pass로 수행한다.
- 눈에 띄는 seam만 mask inpaint한다.

### 24.10 GGUF loader에 파일이 보이지 않는다

- 파일을 `models/unet` 또는 loader가 지정한 폴더에 둔다.
- ComfyUI를 재시작하거나 model list를 refresh한다.
- ComfyUI-GGUF commit과 requirements 설치를 확인한다.
- text encoder GGUF를 일반 checkpoint loader로 열지 않는다.
- split file이 모두 다운로드됐는지 확인한다.
- file extension만 바꾸지 않는다.

### 24.11 LoRA가 적용되지 않는다

- base architecture와 version을 확인한다.
- loader가 GGUF transformer에 LoRA를 지원하는지 확인한다.
- layer name mismatch warning을 본다.
- strength를 0.5–1.0 범위에서 sweep한다.
- trigger word·caption convention을 확인한다.
- merged checkpoint가 이미 같은 LoRA를 포함하는지 확인한다.

ComfyUI-GGUF의 LoRA 지원은 runtime·architecture별로 실험적일 수 있다.

### 24.12 두 번째 요청부터 느려지거나 OOM

- output·latent tensor가 global list에 남아 있는지 확인한다.
- UI history와 preview cache를 제한한다.
- `torch.inference_mode()`를 사용한다.
- Python garbage collection과 worker recycle을 측정한다.
- request cancel 시 background CUDA operation이 남는지 확인한다.
- model을 매 요청 reload하지 않는다.

### 24.13 CPU offload가 너무 느리다

- sequential 대신 model/group offload를 사용한다.
- text encoder embedding을 cache한다.
- system RAM이 swap 중인지 확인한다.
- NVMe가 아니라 HDD disk offload인지 확인한다.
- PCIe link width·speed를 확인한다.
- 더 작은 모델 또는 Q5/Q4 transformer를 선택한다.

Linux:

```bash
free -h
vmstat 1
lspci -vv | grep -A 20 -i 'VGA\|3D controller'
```

### 24.14 Apple MPS 오류

- 해당 operation의 MPS 지원 여부를 확인한다.
- FP16 대신 BF16/FP32가 필요한 component가 있는지 본다.
- CPU fallback 사용 여부와 성능을 기록한다.
- macOS·PyTorch update 전후 회귀를 비교한다.
- 통합 메모리 pressure와 swap을 Activity Monitor로 본다.
- Metal/Core ML용 변환 checkpoint가 필요한 앱인지 확인한다.

### 24.15 AMD에서 CUDA 전용 node 오류

- 해당 custom node의 ROCm 지원 issue를 확인한다.
- pure PyTorch·SDPA node로 교체한다.
- `xformers`, CUDA extension, bitsandbytes dependency를 분리한다.
- ROCm 공식 PyTorch wheel과 compatibility matrix를 사용한다.
- GPU architecture가 현재 ROCm release의 지원 대상인지 확인한다.

### 24.16 결과가 모델 카드 예제와 다르다

다음을 정확히 대조한다.

```text
repository revision
runtime version/commit
prompt와 prompt enhancer
seed generator device
width/height
steps
sampler/scheduler
CFG/guidance
negative prompt
dtype
VAE/text encoder
watermark·post-processing
```

모델 카드 screenshot은 외부 prompt enhancer나 비공개 후처리를 사용했을 수 있으므로 재현 조건을 읽는다.

### 24.17 다운로드가 중단되거나 cache가 손상됨

```bash
hf cache scan
hf download owner/repo --revision <sha> --dry-run
```

지원되는 current CLI 명령은 `hf --help`와 Hugging Face 문서를 확인한다. 수동으로 shard 일부를 삭제하기 전에 cache path와 hardlink/symlink 구조를 확인한다.

### 24.18 라이선스를 알 수 없음

- repository metadata만 보지 말고 `LICENSE`, model card와 linked terms를 모두 읽는다.
- base model·text encoder·VAE·LoRA·ControlNet의 라이선스를 각각 확인한다.
- 상업 사용·SaaS·재배포·fine-tune 배포를 구분한다.
- 불명확하면 production 배포를 보류하고 권리자 또는 법무에 문의한다.

---

## 25. 주요 출처와 저장소

아래 링크는 2026-07-21에 확인한 공식 모델 카드·공식 저장소·주요 runtime 문서다. 커뮤니티 GGUF는 파일 크기와 실제 다운로드 편의를 위해 포함했으며, 원본 모델의 라이선스와 revision을 함께 확인한다.

### 25.1 최신 범용·편집 모델

- [FLUX.2 Klein 4B — Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [FLUX.2 Klein 9B — Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [FLUX.2 dev — Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2 Klein 4B GGUF — Unsloth](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF)
- [FLUX.2 Klein 9B GGUF — Unsloth](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF)
- [FLUX.2 dev GGUF — Unsloth](https://huggingface.co/unsloth/FLUX.2-dev-GGUF)
- [Ideogram 4 NF4](https://huggingface.co/ideogram-ai/ideogram-4-nf4)
- [Ideogram 4 FP8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
- [HiDream O1 Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
- [HiDream O1 project](https://hidream.ai/hidream-o1)
- [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw)
- [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)
- [Krea 2 official code](https://github.com/krea-ai/krea-2)
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Z-Image Base](https://huggingface.co/Tongyi-MAI/Z-Image)
- [HunyuanImage 2.1](https://huggingface.co/tencent/HunyuanImage-2.1)
- [HunyuanImage 3.0](https://huggingface.co/tencent/HunyuanImage-3.0)

### 25.2 Stable Diffusion·SANA

- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
- [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px)

### 25.3 GGUF와 보조 component

- [FLUX.1 dev GGUF — city96](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [T5 v1.1 XXL Encoder GGUF — city96](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)
- [Qwen-Image GGUF — city96](https://huggingface.co/city96/Qwen-Image-gguf)
- [Qwen-Image-Edit-2509 GGUF — QuantStack](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

### 25.4 실행 도구

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [ComfyUI documentation](https://docs.comfy.org/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- [Diffusers memory optimization](https://huggingface.co/docs/diffusers/optimization/memory)
- [Diffusers GGUF loading](https://huggingface.co/docs/diffusers/quantization/gguf)
- [Diffusers bitsandbytes quantization](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [ROCm PyTorch installation](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [ComfyUI on ROCm](https://rocm.docs.amd.com/projects/comfyui/en/latest/)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

### 25.5 관련 RAM-for-Local-AI 문서

- [비전·OCR](./vision-ocr.md)
- [생산성·문서·RAG](../domains/productivity-rag.md)
- [데이터 분석](../domains/data-analysis.md)
- [양자화](../operations/quantization.md) — 예정
- [파인튜닝 메모리](../operations/fine-tuning-memory.md) — 예정
- [런타임·하드웨어](../operations/runtime-hardware.md) — 예정

---

## 최종 권장안

모델 하나만 고르기 어렵다면 다음 기준을 사용한다.

```text
4–6 GB    SD 1.5 / SANA-Sprint / SDXL low-VRAM
8 GB      SDXL 중심, FLUX.1 Q4 시험
12 GB     FLUX.2 Klein 4B Q4/Q5 + offload
16 GB     FLUX.2 Klein 4B 또는 Z-Image Turbo
24 GB     Ideogram 4 NF4 / HiDream O1 / Qwen-Image Q4 / Hunyuan 2.1 FP8
32 GB     FLUX.2 Klein 9B / Qwen Q5-Q6 / Krea 2 quant
48 GB     FLUX.2 dev Q4-Q5 / Qwen Q8 / Krea 2 staged
64 GB     FLUX.2 dev FP8-Q8 / Krea 2 / Qwen BF16 offload
96 GB+    BF16 연구 기준선과 저동시성 서버
192 GB     32B급 다중 인스턴스; HunyuanImage 3.0 공식 최소선 미달
240 GB+    HunyuanImage 3.0 Base 공식 3×80 GB 경로
640 GB+    HunyuanImage 3.0 Instruct·Distilled 공식 8×80 GB 경로
```

기본 선택은 다음과 같다.

- **저메모리 범용:** SDXL
- **16 GB 최신 균형:** FLUX.2 Klein 4B 또는 Z-Image Turbo
- **글자·포스터:** Ideogram 4 또는 Qwen-Image
- **통합 편집:** FLUX.2 Klein/dev, HiDream O1, Qwen-Image-Edit
- **스타일·LoRA 연구:** Krea 2 Raw 또는 SDXL 생태계
- **서버급 연구:** FLUX.2 dev BF16; HunyuanImage 3.0은 Base 3×80 GB, Instruct·Distilled 8×80 GB부터 검토

그리고 어떤 모델이든 다음 원칙을 지킨다.

1. 다운로드 파일 크기가 아니라 **전체 pipeline peak memory**를 본다.
2. Q2/Q3는 최후의 메모리 절약 수단으로 두고 Q4·FP8 기준선과 비교한다.
3. 정확한 글자는 이미지 모델에만 맡기지 않고 OCR 검증·후합성을 사용한다.
4. model·encoder·VAE·LoRA·workflow·runtime revision을 함께 고정한다.
5. custom node와 workflow를 실행 코드로 취급한다.
6. 참조 이미지·얼굴·prompt·출력의 개인정보와 권리를 관리한다.
7. 같은 seed·해상도·조건으로 품질과 peak RAM/VRAM을 직접 측정한다.


---

## 갱신 및 사용상 주의

- 이 문서는 **2026-07-21 KST** 기준 공개 모델 카드·저장소·runtime 문서를 바탕으로 작성했다.
- Hugging Face 파일명, quant tag, 모델 revision, gated access, API와 라이선스는 변경될 수 있다.
- 다운로드 직전 `hf download --dry-run`으로 실제 파일과 총용량을 확인한다.
- 공식 최소 VRAM은 특정 설정의 실행 가능 사례일 수 있으므로 이 문서의 보수적 장착 메모리와 다를 수 있다.
- 이미지 생성 결과의 사실성, 철자, 수치, identity, 저작권·상표·초상권을 자동으로 신뢰하지 않는다.
- 공개 benchmark보다 본인의 prompt·이미지·hardware에서 측정한 품질, peak RAM·VRAM, latency와 실패율을 우선한다.
