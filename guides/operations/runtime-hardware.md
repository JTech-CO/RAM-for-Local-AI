# 로컬 AI 런타임·하드웨어 가이드
> CPU·NVIDIA CUDA·AMD ROCm·Apple Silicon·Intel XPU/NPU·Vulkan·WebGPU 환경에서 RAM·VRAM·통합 메모리와 런타임을 선택·설치·벤치마크하는 운영 가이드

[← 메인 README](../../README.md) · [생산성·문서·RAG](../domains/productivity-rag.md) · [데이터 분석](../domains/data-analysis.md) · [비전·OCR](../modalities/vision-ocr.md) · [이미지 생성](../modalities/image-generation.md) · [오디오·음성](../modalities/audio-speech.md)

> **최종 검증일:** 2026-07-21 (KST)
> **주요 하드웨어:** x86-64·Arm CPU, NVIDIA CUDA GPU, AMD ROCm/HIP GPU·APU, Apple Silicon, Intel CPU·Arc·Data Center GPU·NPU, Vulkan·WebGPU 지원 장치, 엣지·모바일 장치
> **주요 런타임:** `llama.cpp`, Ollama, MLX-LM·MLX-VLM, vLLM, SGLang, TensorRT-LLM, OpenVINO GenAI·OVMS, PyTorch·Transformers, ONNX Runtime, MLC LLM·WebLLM, ExLlamaV3
> **관련 문서:** [양자화](./quantization.md) · [파인튜닝 메모리](./fine-tuning-memory.md) · [서빙·동시성](./serving-concurrency.md)

이 문서는 “VRAM이 몇 GB인가?”만으로 로컬 AI 장비와 런타임을 선택하지 않도록 돕는다. 같은 24GB GPU라도 메모리 대역폭, 연산 정밀도, 커널 지원, PCIe 연결, 운영체제와 드라이버 조합에 따라 실제 처리량과 안정성이 크게 달라진다. 반대로 전용 GPU가 없어도 대용량 시스템 RAM, Apple 통합 메모리, 고대역폭 서버 CPU와 적절한 GGUF 양자화를 조합하면 큰 모델을 실용적으로 운용할 수 있다.

실제 결과는 다음 계층의 교집합으로 결정된다.

```text
모델 아키텍처·양자화 형식
  × 런타임이 구현한 kernel
  × 하드웨어 ISA·compute capability·gfx target
  × 드라이버·CUDA·ROCm·Metal·oneAPI 버전
  × RAM·VRAM·통합 메모리 용량과 대역폭
  × PCIe·NVLink·Infinity Fabric·NUMA topology
  × context·batch·동시 요청·멀티모달 입력
  × 냉각·전력 제한·지속 부하 안정성
```

이 문서의 목적은 특정 제조사를 일률적으로 순위화하는 것이 아니다. 다음 질문에 재현 가능한 방식으로 답하는 것이다.

1. 보유 장비에서 어떤 런타임을 우선 시험해야 하는가?
2. 모델 가중치가 들어간 뒤 실제로 남는 메모리는 얼마인가?
3. CPU·GPU·통합 메모리·offload 중 어느 경로가 병목인가?
4. 같은 모델을 어떤 backend와 정밀도로 비교해야 하는가?
5. 멀티 GPU·다중 socket·컨테이너 환경에서 성능 손실이 어디서 발생하는가?
6. 모델·runtime·driver 업데이트 후 결과를 어떻게 재현하고 회귀 검증하는가?

> **핵심 원칙:** 먼저 하드웨어와 운영체제가 공식 지원하는 런타임을 고르고, 그 런타임에서 최적화된 양자화 형식을 선택한다. 모델 파일이 메모리에 들어간다는 사실만으로 실용적인 속도·동시성·안정성이 보장되지는 않는다.

지원 범위와 기본 버전은 빠르게 변한다. 이 문서의 버전 표기는 2026-07-21의 스냅샷이며, 실제 설치 직전에는 각 프로젝트의 **stable 문서, release notes, hardware matrix, security advisory**를 다시 확인한다. `latest`, `nightly`, `dev` 컨테이너 태그는 재현 가능한 배포에 사용하지 않는다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [하드웨어 선택 우선순위](#2-하드웨어-선택-우선순위)
3. [RAM·VRAM·통합 메모리 해석](#3-ramvram통합-메모리-해석)
4. [작업별 병목과 권장 장치](#4-작업별-병목과-권장-장치)
5. [런타임 선택 매트릭스](#5-런타임-선택-매트릭스)
6. [장비·드라이버·토폴로지 점검](#6-장비드라이버토폴로지-점검)
7. [CPU 추론](#7-cpu-추론)
8. [메모리 채널·NUMA·huge page](#8-메모리-채널numahuge-page)
9. [NVIDIA CUDA](#9-nvidia-cuda)
10. [AMD ROCm·HIP](#10-amd-rocmhip)
11. [Apple Silicon·Metal·MLX](#11-apple-siliconmetalmlx)
12. [Intel CPU·XPU·NPU](#12-intel-cpuxpunpu)
13. [Vulkan·SYCL·DirectML·WinML·WebGPU](#13-vulkansycldirectmlwinmlwebgpu)
14. [Arm·엣지·모바일·기타 가속기](#14-arm엣지모바일기타-가속기)
15. [`llama.cpp`](#15-llamacpp)
16. [Ollama](#16-ollama)
17. [MLX-LM·MLX-VLM](#17-mlx-lmmlx-vlm)
18. [vLLM](#18-vllm)
19. [SGLang](#19-sglang)
20. [TensorRT-LLM](#20-tensorrt-llm)
21. [OpenVINO GenAI·OVMS](#21-openvino-genaiovms)
22. [PyTorch·Transformers·Optimum](#22-pytorchtransformersoptimum)
23. [ONNX Runtime](#23-onnx-runtime)
24. [MLC LLM·WebLLM](#24-mlc-llmwebllm)
25. [ExLlamaV3와 특화 런타임](#25-exllamav3와-특화-런타임)
26. [비전·이미지·오디오 런타임](#26-비전이미지오디오-런타임)
27. [Windows·WSL·Linux·macOS](#27-windowswsllinuxmacos)
28. [컨테이너·가상화·Kubernetes](#28-컨테이너가상화kubernetes)
29. [다중 GPU·다중 socket·interconnect](#29-다중-gpu다중-socketinterconnect)
30. [스토리지·모델 캐시·I/O](#30-스토리지모델-캐시io)
31. [전력·열·소음·지속 부하](#31-전력열소음지속-부하)
32. [벤치마크 설계](#32-벤치마크-설계)
33. [모니터링·프로파일링](#33-모니터링프로파일링)
34. [하드웨어별 배포 레시피](#34-하드웨어별-배포-레시피)
35. [보안·공급망·권한](#35-보안공급망권한)
36. [문제 해결](#36-문제-해결)
37. [재현성 manifest와 기여 형식](#37-재현성-manifest와-기여-형식)
38. [주요 출처·최종 권장안](#38-주요-출처최종-권장안)

---

## 1. 30초 선택표

### 1.1 장비 유형별 첫 런타임

| 보유 장비 | 첫 선택 | 두 번째 선택 | 적합한 형식 | 주의점 |
| --- | --- | --- | --- | --- |
| 전용 GPU 없는 일반 PC | `llama.cpp` CPU | Ollama CPU | GGUF Q4/Q5 | 코어 수보다 메모리 대역폭·채널 구성이 중요 |
| NVIDIA GeForce·RTX PRO | `llama.cpp` CUDA 또는 vLLM | SGLang·ExLlamaV3 | GGUF, AWQ·GPTQ·compressed-tensors, EXL3 | compute capability와 runtime wheel의 CUDA 조합 확인 |
| NVIDIA 데이터센터 GPU | vLLM·SGLang | TensorRT-LLM | BF16·FP8·W4A16 | TP·EP·NVLink/NVSwitch·NCCL topology 확인 |
| AMD Radeon | `llama.cpp` HIP 또는 Vulkan | Ollama Vulkan·ROCm, vLLM 지원 범위 | GGUF 우선, 지원 시 FP8·AWQ 계열 | 공식 ROCm compatibility selector에서 GPU·OS를 먼저 확인 |
| AMD Instinct | vLLM·SGLang ROCm | `llama.cpp` HIP | BF16·FP8·W4A16·GGUF | ROCm·PyTorch·runtime 버전을 묶어 pin |
| Apple Silicon | MLX-LM·MLX-VLM | `llama.cpp` Metal, SGLang MLX | MLX 4/8-bit, GGUF Q4/Q5 | 통합 메모리는 OS·앱·GPU가 공유하므로 memory pressure를 확인 |
| Intel Xeon CPU | `llama.cpp`·OpenVINO GenAI | SGLang CPU | GGUF·OpenVINO INT4/INT8 | AMX·AVX-512·VNNI와 NUMA·DIMM 채널 확인 |
| Intel Arc·Data Center GPU | OpenVINO·native PyTorch XPU | vLLM XPU·SGLang XPU, `llama.cpp` SYCL/Vulkan | OpenVINO IR·INT4/INT8, 지원 HF 형식 | IPEX가 아닌 native PyTorch XPU/OpenVINO를 우선 |
| Windows 범용 GPU | `llama.cpp` Vulkan·CUDA·HIP | Ollama, WinML/ONNX Runtime | GGUF 또는 ONNX | 서버급 vLLM은 WSL2/Linux가 일반적 |
| 브라우저·클라이언트 앱 | WebLLM | `llama.cpp` WebGPU 실험 | MLC WebGPU artifact | 브라우저 메모리·스토리지 quota와 모델 다운로드 UX |
| ARM Linux·SBC | `llama.cpp` CPU/KleidiAI | MLC LLM | GGUF | 열 throttling·메모리 대역폭·swap 수명 |

### 1.2 사용 가능 메모리만 알고 있을 때

아래 표의 “실사용 가능 메모리”는 운영체제, display, runtime buffer, KV 캐시와 안전 여유를 제외한 뒤 모델에 할당할 수 있는 값이다. 모델별 실제 크기는 domain·modality 가이드와 [양자화 가이드](./quantization.md)를 함께 본다.

| 실사용 가능 메모리 | 범용 시작점 | 권장 runtime 유형 | 운영 메모 |
| ---: | --- | --- | --- |
| 4–6GB | 1–3B Q4, 소형 ASR·embedding | CPU·Metal·Vulkan 경량 runtime | 4K context·단일 요청부터 시작 |
| 8GB | 7–8B Q4 또는 3B Q8 | `llama.cpp`, Ollama, MLX | display GPU는 1–2GB 이상 여유 확보 |
| 12GB | 8B Q6/Q8 또는 14B Q4 | `llama.cpp`, MLX, 단일 GPU Transformers | VLM projector·VAE·KV를 별도 계산 |
| 16GB | 14B Q5 또는 27B Q3 | `llama.cpp`, Ollama, MLX | 긴 context보다 모델 품질을 우선 |
| 24GB | 27–32B Q4 또는 14B 고정밀 | CUDA/HIP `llama.cpp`, vLLM·SGLang | 20–22GB 이하 weight가 운영상 안전한 경우가 많음 |
| 32GB | 32B Q5 또는 70B Q3 | vLLM·SGLang·MLX·GGUF | batch·동시성 증가 시 KV가 빠르게 커짐 |
| 48GB | 70B Q4 또는 32B Q8 | 서버 runtime·대형 MLX/GGUF | 단일 48GB와 2×24GB는 동일하지 않음 |
| 64GB | 70B Q5/Q6 | 단일 대용량 장치 또는 빠른 interconnect | 멀티모달·다중 worker에 유리 |
| 80–96GB | 70B Q8/BF16 근접, 120B Q4 | 데이터센터 runtime | FP8·TP·prefix cache·continuous batching 검토 |
| 128GB | 120B Q5/Q6, 235B Q3 | Apple 통합 메모리·서버 GPU | OS와 cache 여유를 15–25% 남김 |
| 192–256GB | 235B Q4/Q5, 405B Q3 | 대형 unified memory·multi-GPU | topology와 총 대역폭이 용량만큼 중요 |
| 320GB 이상 | 405B Q4 이상, 대형 멀티모달 연구 | NVSwitch·고대역폭 서버 | 단일 사용자보다 serving·연구 cluster 설계 영역 |

### 1.3 목적별 결론

```text
가장 단순한 로컬 실행
  → Ollama 또는 llama.cpp

Apple Silicon에서 최고 수준의 통합
  → MLX-LM/MLX-VLM

NVIDIA·AMD 서버에서 높은 동시 처리량
  → vLLM 또는 SGLang

NVIDIA 지원 장비에서 고정된 최적화 engine
  → TensorRT-LLM

Intel CPU·iGPU·dGPU·NPU 통합
  → OpenVINO GenAI/OVMS

브라우저·모바일·다중 플랫폼 앱 배포
  → MLC LLM/WebLLM 또는 ONNX Runtime
```

### 1.4 피해야 할 판단

- “VRAM 총합이 같으므로 1×48GB와 2×24GB가 같다.”
- “FP8·INT4 파일이 있으므로 모든 GPU에서 빨라진다.”
- “Apple 통합 메모리 128GB를 모델에 128GB 전부 쓸 수 있다.”
- “ROCm 설치가 되므로 해당 Radeon이 모든 runtime에서 공식 지원된다.”
- “CUDA toolkit 버전만 맞으면 driver·PyTorch·runtime wheel도 자동 호환된다.”
- “CPU 코어가 많을수록 token/s가 선형으로 증가한다.”
- “모델 load가 성공했으므로 긴 context·동시 요청도 안전하다.”
- “nightly에서 동작했으므로 production에서도 재현된다.”

---

## 2. 하드웨어 선택 우선순위

### 2.1 용량보다 먼저 확인할 것

실무 우선순위는 다음과 같다.

```text
1. 필요한 모델·runtime의 공식 하드웨어 지원
2. 가중치 + KV + workspace + 안전 여유가 들어가는 용량
3. 지속 memory bandwidth와 실제 kernel 효율
4. 필요한 정밀도·tensor core·matrix instruction 지원
5. 단일 장치인지, PCIe/NVLink/NUMA를 건너는지
6. OS·driver·container·Python 조합의 유지보수성
7. 전력·냉각·소음·24시간 안정성
8. 구매비·운영비·개발 시간
```

### 2.2 추론의 두 단계

Autoregressive LLM 추론은 대체로 다음 두 단계로 나뉜다.

| 단계 | 주된 특성 | 중요한 하드웨어 요소 | 흔한 병목 |
| --- | --- | --- | --- |
| Prefill·prompt processing | 많은 token을 병렬 계산 | compute, matrix core, batch, HBM/VRAM | activation·workspace·attention |
| Decode·token generation | 매 token마다 weights·KV를 반복 접근 | memory bandwidth, cache, low-latency kernel | weight·KV memory traffic |

CPU GGUF에서 token generation은 메모리 대역폭의 영향을 크게 받는다. 반면 대규모 batch prefill과 이미지 생성은 연산량과 tensor/matrix core 활용률이 더 중요해질 수 있다. 한 개의 `tokens/s`만으로 하드웨어를 비교하지 않는다.

### 2.3 용량·대역폭·연산량의 관계

```text
모델을 적재할 수 없음
  → 용량 병목

적재되지만 GPU utilization이 낮고 PCIe traffic이 큼
  → offload·interconnect 병목

decode에서 bandwidth가 포화됨
  → memory-bound

prefill·image generation에서 compute utilization이 높음
  → compute-bound

짧은 burst 뒤 속도가 하락함
  → 열·전력 throttling 또는 memory pressure
```

### 2.4 하드웨어 비교 시 최소 지표

| 범주 | 기록할 값 |
| --- | --- |
| 메모리 | 물리 용량, 실사용 가능량, 대역폭, ECC 여부, 채널 수 |
| GPU | architecture, compute capability 또는 `gfx*`, VRAM, power limit |
| CPU | ISA, socket, core/thread, memory channel, NUMA node |
| 연결 | PCIe 세대·lane, NVLink/NVSwitch, P2P 가능 여부 |
| 소프트웨어 | OS kernel, driver, CUDA/ROCm/Metal, runtime commit·version |
| 모델 | repo, revision, quant, file hash, context, KV dtype |
| 성능 | prompt tokens/s, generation tokens/s, TTFT, peak memory, watts |
| 안정성 | 30–60분 지속 부하, throttling, error·reset, swap·OOM |

---

## 3. RAM·VRAM·통합 메모리 해석

### 3.1 메모리 종류

| 종류 | 위치·접근 | 장점 | 제약 |
| --- | --- | --- | --- |
| 시스템 RAM | CPU memory controller 뒤 | 대용량·비교적 저렴 | dGPU가 직접 쓰려면 PCIe 전송 또는 managed memory 필요 |
| GPU VRAM·HBM | GPU에 직접 연결 | 높은 대역폭·낮은 GPU 접근 지연 | 용량·가격, 다른 GPU와 기본적으로 분리 |
| Apple 통합 메모리 | CPU·GPU가 같은 물리 pool 공유 | 복사 감소·대용량 단일 address space | OS·display·앱과 공유, swap 시 급격한 지연 |
| iGPU 공유 메모리 | 시스템 RAM 일부를 GPU가 공유 | 저비용·큰 addressable pool 가능 | CPU와 대역폭 경쟁, BIOS·driver 예약 |
| pinned host memory | DMA용으로 고정한 RAM | GPU 전송 효율 | pageable RAM보다 시스템에 부담, 과도하면 전체 성능 저하 |
| managed/unified virtual memory | driver가 CPU·GPU page를 이동 | OOM 대신 oversubscription 가능 | page migration·PCIe로 속도 급락 가능 |
| swap·pagefile | SSD에 내려간 메모리 | crash 회피 | 추론용 active weights·KV에는 매우 느림, SSD 쓰기 증가 |

### 3.2 실사용 가능 메모리 계산

```text
usable_device_memory
≈ physical_device_memory
- display·desktop reserve
- driver·runtime context
- graph·kernel workspace
- safety margin
```

```text
peak_inference_memory
≈ resident weights
+ KV cache
+ activation·prefill workspace
+ quantization/dequantization scratch
+ multimodal encoder·VAE·codec
+ communication buffer
+ allocator fragmentation
```

보수적으로 다음 여유를 시작점으로 둔다.

| 환경 | 권장 초기 여유 | 이유 |
| --- | ---: | --- |
| display 겸용 8–16GB GPU | 15–25% | 데스크톱·브라우저·그래픽 앱 변동 |
| headless 단일 GPU | 10–15% | runtime graph·workspace·fragmentation |
| 서버 다중 GPU | 장치당 10–20% | NCCL/RCCL·CUDA graph·불균등 shard |
| Apple 통합 메모리 | 20–30% | macOS·wired·compressed memory·다른 앱 |
| CPU-only 서버 | 15–25% | page cache·worker·OS·NUMA 불균형 |

이 비율은 절대 법칙이 아니다. 실제 peak를 측정해 조정하되, production은 benchmark 최대치보다 낮게 운영한다.

### 3.3 GB와 GiB

Hugging Face와 제조사 표기는 decimal GB인 경우가 많고, 운영체제·runtime은 GiB를 쓰기도 한다.

```text
1 GB  = 1,000,000,000 bytes
1 GiB = 1,073,741,824 bytes
1 GB  ≈ 0.9313 GiB
```

24GB 파일은 약 22.35GiB다. 다운로드 크기, disk 크기, resident memory를 같은 단위로 맞춘다.

### 3.4 Apple 통합 메모리

Apple Silicon에서는 CPU와 GPU가 같은 통합 메모리를 사용하지만, 전체 물리 메모리가 모델 전용 VRAM은 아니다. Activity Monitor의 다음 값을 함께 본다.

- Memory Pressure
- Wired Memory
- Compressed Memory
- Swap Used
- 앱별 실제·compressed memory

Memory Pressure가 yellow/red로 바뀌고 swap이 지속 증가하면 “모델이 실행된다”와 “안정적으로 빠르다”는 다른 상태다. 장시간 작업은 GUI 앱을 줄이고, 모델·context·batch를 조정해 green pressure를 유지하는 편이 안전하다.

### 3.5 dGPU oversubscription

`llama.cpp`의 CPU+GPU hybrid offload, CUDA managed memory, ROCm UMA, framework CPU offload는 모델을 더 큰 주소 공간에 넣을 수 있게 한다. 그러나 active layer가 반복적으로 PCIe를 건너면 처리량이 크게 떨어진다.

```text
전량 VRAM 적재
  > 일부 layer CPU offload
  > 매 token마다 page migration
  > active weights가 swap에 위치
```

속도 우선이면 한 단계 작은 Q4/Q5 모델을 전량 accelerator에 적재하는 구성을 반드시 비교한다.

---

## 4. 작업별 병목과 권장 장치

### 4.1 작업 매트릭스

| 작업 | 주된 병목 | 우선 장치 | 메모리에서 별도 계산할 것 |
| --- | --- | --- | --- |
| 단일 사용자 LLM chat | decode bandwidth | GPU·Apple GPU·고대역폭 CPU | KV cache, context |
| 장문 논문·RAG prefill | compute+attention | GPU·Apple GPU·AMX CPU | prefill workspace, retrieval stack |
| 다중 사용자 API | KV+batch scheduler | 데이터센터 GPU | KV pool, CUDA graph, queue |
| 코드 agent | 긴 context+tool latency | 충분한 메모리의 단일 GPU/UMA | repo context, parallel tool calls |
| VLM·OCR | vision encoder+visual tokens | GPU·통합 메모리 | projector, image resolution, page queue |
| 이미지 생성 | DiT/UNet compute+activation | GPU | VAE, text encoder, ControlNet, resolution |
| ASR batch | encoder compute | GPU·CPU vector engine | audio decode, VAD, batch duration |
| 실시간 ASR/TTS | latency+stream state | GPU·NPU·저지연 CPU | session cache, codec/vocoder |
| embedding 대량 생성 | batch compute | GPU·NPU·AMX CPU | corpus buffer, index build |
| QLoRA | optimizer+activation | GPU·통합 메모리 | gradients, optimizer, checkpoint |
| full fine-tuning | model state+interconnect | multi-GPU server | FSDP/ZeRO buffer, checkpoint staging |

### 4.2 CPU가 합리적인 경우

- 1–8B Q4/Q5 모델의 개인용·저빈도 실행
- 시스템 RAM은 크지만 GPU VRAM이 작은 경우
- x86/Arm 서버에 이미 대용량 RAM과 높은 메모리 대역폭이 있는 경우
- 임베딩·reranker·ASR처럼 최적화된 CPU runtime이 있는 경우
- privacy appliance·edge 장비처럼 GPU driver 유지보수를 피해야 하는 경우
- 정확한 latency보다 비용·호환성·안정성이 중요한 batch 작업

### 4.3 GPU가 사실상 필요한 경우

- 고해상도 이미지 생성·편집
- 대규모 VLM·video input
- 긴 prompt의 낮은 TTFT
- 다중 사용자 continuous batching
- LoRA·QLoRA·full fine-tuning
- FP8·FP4·W4A16 등 특정 matrix kernel을 활용하려는 경우

### 4.4 NPU가 적합한 경우

NPU는 일반적으로 전력 효율이 높지만, 동적 shape·긴 KV cache·새로운 모델 architecture에 대한 runtime coverage가 GPU보다 좁을 수 있다. 다음에 우선한다.

- 고정 shape embedding·classification
- 지원되는 ASR·vision encoder
- 배터리 기반 always-on inference
- OpenVINO·Core ML·WinML이 완전히 compile하는 모델

LLM 전체를 NPU로 실행하기 전에 unsupported op의 CPU fallback과 실제 memory copy를 확인한다.

---

## 5. 런타임 선택 매트릭스

### 5.1 범용 비교

| 런타임 | 주력 환경 | 대표 형식 | 장점 | 주요 제약 |
| --- | --- | --- | --- | --- |
| `llama.cpp` | CPU·CUDA·HIP·Metal·Vulkan·SYCL·Arm | GGUF | 가장 넓은 장치 범위, hybrid offload, 단일 binary | 최고 동시 처리량은 서버 runtime보다 낮을 수 있음 |
| Ollama | 개인 PC·간단한 API | GGUF 기반 package | 설치·모델 관리·API가 간단 | 세밀한 kernel·scheduler 제어는 제한적 |
| MLX-LM | Apple Silicon | MLX safetensors 4/8-bit | 통합 메모리·Metal에 자연스러운 경로 | macOS·Apple Silicon 전용 |
| vLLM | GPU 서버·OpenAI API | HF, AWQ·GPTQ·FP8 등 | paged KV·continuous batching·넓은 serving 기능 | 버전·GPU별 kernel 호환성을 엄격히 맞춰야 함 |
| SGLang | GPU 서버·agent·structured generation | HF·여러 quant | radix cache·serving·다양한 플랫폼 | 플랫폼별 기능 차이가 큼 |
| TensorRT-LLM | NVIDIA 지원 장비 | engine·checkpoint recipe | NVIDIA에서 고성능·세밀한 최적화 | 공식 지원 hardware가 CUDA 전체보다 좁고 build 관리 필요 |
| OpenVINO GenAI | Intel CPU·GPU·NPU | OpenVINO IR·INT4/INT8 | Intel 장치 통합·LLM/VLM/ASR/T2I pipeline | 모델 export·지원 topology 확인 필요 |
| PyTorch·Transformers | 연구·새 모델·커스텀 코드 | safetensors·HF | 최신 모델 구현·유연성 | 범용 serving 효율·memory overhead |
| ONNX Runtime | 앱·edge·cross-platform | ONNX | 다양한 execution provider와 언어 binding | 생성형 모델은 graph/export·EP 지원 범위에 좌우 |
| MLC LLM·WebLLM | 브라우저·모바일·앱 | MLC compile artifact | WebGPU·iOS·Android·Vulkan | 모델 compile·artifact 관리 필요 |
| ExLlamaV3 | NVIDIA CUDA 단일/소수 GPU | EXL3 | 저비트 CUDA 추론에 특화 | CUDA 중심, 형식·architecture 지원 확인 |

### 5.2 운영 목적별 선택

| 목적 | 권장 순서 |
| --- | --- |
| 가장 빨리 로컬 chat 시작 | Ollama → `llama.cpp` |
| RAM에 맞춰 GGUF를 세밀하게 offload | `llama.cpp` |
| Apple 앱·개인 서버 | MLX-LM → `llama.cpp` Metal → SGLang MLX |
| NVIDIA 단일 GPU 연구 | Transformers → vLLM/SGLang → ExLlamaV3 비교 |
| NVIDIA production API | vLLM/SGLang → TensorRT-LLM A/B |
| AMD Instinct production | vLLM/SGLang ROCm → HIP `llama.cpp` |
| AMD Radeon 개인용 | HIP·Vulkan `llama.cpp` → Ollama |
| Intel 통합 장치 | OpenVINO GenAI → native PyTorch XPU → SYCL/Vulkan GGUF |
| 브라우저 완전 로컬 | WebLLM |
| 새 architecture bring-up | Transformers·PyTorch → 지원 runtime으로 이동 |

### 5.3 “지원”의 네 단계

```text
1. 설치됨
2. 모델 load 성공
3. 주요 op가 accelerator kernel에서 실행됨
4. 목표 품질·속도·메모리·안정성을 충족
```

단계 1이나 2만으로 hardware 지원을 판단하지 않는다. CPU fallback, dequant copy, unsupported attention, eager mode 전환을 log와 profiler로 확인한다.

---

## 6. 장비·드라이버·토폴로지 점검

### 6.1 Linux 공통

```bash
uname -a
cat /etc/os-release
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE,MAXMHZ
free -h
numactl --hardware 2>/dev/null || true
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS
lspci -nn | grep -Ei 'vga|3d|display|nvidia|amd|intel'
```

메모리 채널과 DIMM 배치를 확인하려면 권한이 있을 때 다음을 사용한다.

```bash
sudo dmidecode -t memory
sudo lshw -class memory -class display
```

### 6.2 NVIDIA

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,driver_version,power.limit,temperature.gpu --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi topo -p2p n
nvcc --version 2>/dev/null || true
```

`nvidia-smi`의 “CUDA Version” 표시는 driver가 지원하는 최대 CUDA 호환 수준이지, 현재 Python 환경의 PyTorch wheel이 실제로 사용하는 CUDA runtime과 동일하지 않을 수 있다.

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, p.total_memory, p.major, p.minor)
PY
```

### 6.3 AMD

```bash
rocminfo | grep -E 'Name:|gfx' | head -80
hipconfig --full 2>/dev/null || true
amd-smi static 2>/dev/null || amd-smi 2>/dev/null || true
amd-smi monitor 2>/dev/null || true
```

GPU target은 `gfx1100`, `gfx942`, `gfx950` 같은 식별자로 기록한다. 제품명만으로 runtime wheel을 고르지 않는다.

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('accelerator available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

PyTorch ROCm도 호환 API 때문에 `torch.cuda` namespace를 사용한다.

### 6.4 Apple Silicon

```bash
sw_vers
uname -m
system_profiler SPHardwareDataType SPDisplaysDataType
sysctl -n hw.memsize
vm_stat
memory_pressure
```

```bash
python - <<'PY'
try:
    import mlx.core as mx
    print(mx.default_device())
    print(mx.metal.get_active_memory())
    print(mx.metal.get_peak_memory())
except Exception as e:
    print('MLX unavailable or API changed:', e)
PY
```

MLX API는 버전에 따라 달라질 수 있으므로 `python -c "import mlx; print(mlx.__version__)"`와 현재 문서를 함께 기록한다.

### 6.5 Intel

```bash
lscpu | grep -E 'Model name|Flags|Socket|NUMA'
sycl-ls 2>/dev/null || true
xpu-smi discovery 2>/dev/null || true
```

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('xpu available:', hasattr(torch, 'xpu') and torch.xpu.is_available())
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(torch.xpu.get_device_name(0))
PY
```

OpenVINO 장치는 다음처럼 조회한다.

```bash
python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for d in core.available_devices:
    try:
        print(d, core.get_property(d, 'FULL_DEVICE_NAME'))
    except Exception:
        print(d)
PY
```

### 6.6 Vulkan

```bash
vulkaninfo --summary
```

여러 GPU가 있으면 device index와 driver를 기록한다. Mesa RADV, AMDVLK, vendor proprietary driver 등 Vulkan implementation이 달라지면 결과도 달라질 수 있다.

### 6.7 자동 inventory 예제

```bash
#!/usr/bin/env bash
set -u
OUT="hardware-inventory-$(date +%Y%m%d-%H%M%S).txt"
{
  echo '=== DATE ==='; date --iso-8601=seconds 2>/dev/null || date
  echo '=== OS ==='; uname -a; cat /etc/os-release 2>/dev/null || true
  echo '=== CPU ==='; lscpu 2>/dev/null || true
  echo '=== RAM ==='; free -h 2>/dev/null || true
  echo '=== NUMA ==='; numactl --hardware 2>/dev/null || true
  echo '=== PCI ==='; lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' || true
  echo '=== NVIDIA ==='; nvidia-smi 2>/dev/null || true
  echo '=== AMD ==='; amd-smi static 2>/dev/null || rocminfo 2>/dev/null | head -120 || true
  echo '=== VULKAN ==='; vulkaninfo --summary 2>/dev/null || true
} | tee "$OUT"
printf 'saved: %s\n' "$OUT"
```

공개 issue에 붙이기 전 UUID, serial, hostname, user path, IP와 조직 정보를 마스킹한다.

---

## 7. CPU 추론

### 7.1 CPU에서 중요한 순서

```text
RAM 용량
→ 지속 메모리 대역폭·채널 수
→ SIMD/matrix ISA
→ NUMA locality
→ runtime kernel
→ 코어 수
→ boost clock
```

작은 모델이나 prompt processing에서는 compute와 cache가 중요하지만, 큰 GGUF의 decode는 weights를 반복 읽으므로 메모리 대역폭이 병목이 되기 쉽다.

### 7.2 ISA별 확인 항목

| CPU 계열 | 확인할 기능 | 대표 경로 |
| --- | --- | --- |
| x86 일반 | AVX2, FMA | `llama.cpp` CPU |
| 최신 x86 | AVX-512, VNNI | `llama.cpp`, OpenVINO |
| 4세대 이상 Xeon 계열 | AMX BF16/INT8 | OpenVINO, SGLang CPU, PyTorch |
| AMD EPYC | AVX2/AVX-512 세대별 지원, ZenDNN | `llama.cpp` ZenDNN·기본 CPU |
| Arm64 | NEON, dotprod, i8mm, SVE/SME | `llama.cpp` KleidiAI, MLC |
| Apple CPU | ARM SIMD+Accelerate | Metal/MLX 우선, CPU fallback |

### 7.3 thread 수

물리 코어 수를 무조건 모두 사용하는 것이 최적은 아니다.

```bash
# 후보를 sweep한다.
for t in 4 8 12 16 24 32; do
  ./llama-bench -m model.gguf -t "$t" -p 512 -n 128
 done
```

- SMT thread가 decode를 개선하는지 실제 측정한다.
- dual-socket에서 thread가 remote memory를 읽지 않도록 NUMA binding을 시험한다.
- 다른 서비스와 core를 공유한다면 CPU affinity를 고정한다.
- thermal·power cap 아래에서 10초 결과가 아니라 30분 결과를 본다.

### 7.4 BLAS의 역할

`llama.cpp` 공식 build 문서는 OpenBLAS·oneMKL 등의 BLAS가 큰 batch의 prompt processing에 도움을 줄 수 있지만 generation 자체에는 영향을 주지 않는다고 설명한다. 따라서 BLAS on/off는 `pp`와 `tg`를 분리해 측정한다.

```bash
cmake -B build-openblas \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build-openblas --config Release -j
```

Intel oneMKL 예시는 다음과 같다.

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-mkl \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Intel10_64lp \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_NATIVE=ON
cmake --build build-mkl --config Release -j
```

### 7.5 Arm KleidiAI

최신 `llama.cpp`는 Arm CPU용 KleidiAI microkernel을 선택적으로 빌드할 수 있다.

```bash
cmake -B build-kleidiai -DGGML_CPU_KLEIDIAI=ON
cmake --build build-kleidiai --config Release -j
```

지원 CPU에서는 dotprod, int8mm, SVE, SME 등을 runtime detection으로 선택한다. 다른 GPU backend가 기본 선택되면 `--device none` 또는 compile-time 옵션으로 CPU 경로를 명시한다.

### 7.6 CPU-only 현실적 기대

| 상황 | 판단 |
| --- | --- |
| 1–8B Q4 개인 chat | 충분히 실용적일 수 있음 |
| 14–32B Q4 | RAM 대역폭과 허용 latency에 따라 실용적 |
| 70B Q4 | 대용량·다채널 RAM 서버에서 batch·offline 중심 |
| 이미지 생성 | 소형·저해상도 외에는 GPU 대비 느림 |
| ASR | Whisper·전용 CPU runtime은 실용적일 수 있음 |
| embedding | batch·INT8 최적화 시 CPU 서버가 경제적일 수 있음 |

---

## 8. 메모리 채널·NUMA·huge page

### 8.1 DIMM 채널

CPU inference는 RAM 용량뿐 아니라 채널을 모두 채웠는지가 중요하다. 같은 256GB라도 소수의 대용량 DIMM으로 일부 채널만 사용한 시스템은 모든 채널을 균형 있게 채운 시스템보다 대역폭이 낮을 수 있다.

확인 항목:

- socket당 memory channel 수
- 채널별 DIMM 균형
- memory clock과 downclock 조건
- ECC·registered DIMM 구성
- BIOS의 NUMA·interleaving 정책
- 실제 STREAM·메모리 benchmark

### 8.2 NUMA

NUMA에서는 각 socket이 local memory와 I/O를 가진다. remote memory 접근은 interconnect를 지나므로 지연과 대역폭 손실이 발생한다.

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE
```

단일 process를 한 node에 고정해 비교한다.

```bash
numactl --cpunodebind=0 --membind=0 \
  ./llama-bench -m /models/model.gguf -t 32 -p 512 -n 128
```

두 node에 interleave하는 경우도 비교한다.

```bash
numactl --interleave=all \
  ./llama-bench -m /models/model.gguf -t 64 -p 512 -n 128
```

모델 파일 page가 어느 node에 배치되는지는 최초 page fault를 일으킨 thread와 memory policy의 영향을 받는다. benchmark 전에 process binding을 적용한다.

### 8.3 GPU와 NUMA locality

PCIe GPU는 특정 CPU socket의 root complex에 연결된다. GPU process와 tokenizer·data worker를 해당 NUMA node에 가깝게 배치한다.

```bash
nvidia-smi topo -m
lspci -tv
numactl --hardware
```

멀티 GPU에서는 GPU별 CPU affinity와 NIC locality도 기록한다. 원격 socket의 RAM에서 GPU로 전송하면 QPI/UPI/Infinity Fabric과 PCIe를 모두 통과할 수 있다.

### 8.4 mmap·mlock·page cache

GGUF runtime은 모델 파일을 `mmap`해 OS page cache를 활용할 수 있다.

| 옵션·상태 | 효과 | 주의점 |
| --- | --- | --- |
| mmap | 빠른 startup·page cache 공유 | 첫 접근 page fault, 네트워크 FS 지연 |
| mlock | 모델 page를 RAM에 고정 | 권한·memlock limit, 다른 프로세스 압박 |
| page cache warm | 반복 load 단축 | benchmark cold/warm 구분 필요 |
| swap disabled | latency 예측 가능 | OOM killer 위험 증가 |
| swap enabled | crash 회피 | active model page가 swap되면 매우 느림 |

```bash
ulimit -l
cat /proc/meminfo | grep -E 'MemAvailable|Cached|Swap|Huge'
```

### 8.5 Transparent Huge Pages

Linux THP는 큰 working set의 TLB overhead를 줄일 수 있지만, allocation·compaction latency와 workload별 차이가 있다. 전역 설정을 무조건 바꾸지 말고 isolated benchmark로 비교한다.

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
```

production 변경 전 다른 database·VM workload에 미치는 영향을 확인한다.

### 8.6 CPU 서버 체크리스트

```text
[ ] DIMM이 모든 채널에 균형 배치됨
[ ] BIOS memory frequency가 예상값인지 확인
[ ] NUMA node·GPU·NIC topology 기록
[ ] thread·membind sweep 수행
[ ] cold/warm mmap 결과 분리
[ ] swap·THP·governor 설정 기록
[ ] 30분 sustained token/s와 전력 측정
```

---

## 9. NVIDIA CUDA

### 9.1 compute capability를 먼저 확인

NVIDIA runtime의 kernel 지원은 제품명보다 compute capability에 의해 결정되는 경우가 많다. 2026-07 기준 공식 CUDA GPU 목록의 대표 계열은 다음과 같다.

| architecture·제품군 예 | compute capability 예 | 운영 의미 |
| --- | ---: | --- |
| Blackwell B200·GB200 | 10.0 | BF16·FP8·FP4 계열 최신 server kernel |
| Blackwell B300·GB300 | 10.3 | 최신 runtime·driver 요구 가능 |
| GB10·DGX Spark 계열 | 12.1 | ARM host·통합 구성 특성 별도 확인 |
| RTX 50·RTX PRO Blackwell | 12.0 | consumer/pro driver와 runtime support matrix 확인 |
| Hopper H100·H200·GH200 | 9.0 | FP8·HBM·NVLink server serving |
| Ada L4·L40·L40S·RTX 40 | 8.9 | FP8 지원 범위와 kernel별 차이 |
| Ampere A100 | 8.0 | BF16·MIG·NVLink |
| Ampere RTX 30·A-series | 8.6 | 소비자·워크스테이션 추론 |
| Turing T4·RTX 20 | 7.5 | vLLM 현재 최소선의 대표 예 |

정확한 값은 [NVIDIA CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus)에서 확인한다.

### 9.2 driver·toolkit·wheel은 서로 다르다

```text
NVIDIA kernel driver
  ≥ wheel/container가 요구하는 최소 driver

PyTorch CUDA runtime
  = torch.version.cuda

system CUDA toolkit
  = nvcc --version

runtime project binary
  = 빌드 당시 CUDA·PyTorch ABI
```

`nvidia-smi`에 표시되는 CUDA 버전과 `torch.version.cuda`가 달라도 정상일 수 있다. 오류가 발생하면 이 네 층을 각각 기록한다.

### 9.3 호환성 정책

NVIDIA는 minor version compatibility와 forward compatibility package를 제공하지만, 모든 기능·PTX·driver 조합이 무조건 호환되는 것은 아니다. production은 공식 compatibility 문서와 runtime의 tested matrix에 맞춘다.

- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html)
- [PyTorch install selector](https://pytorch.org/get-started/locally/)
- 각 runtime의 wheel·container release notes

### 9.4 소비자 GPU와 데이터센터 GPU

| 항목 | 소비자·워크스테이션 | 데이터센터 |
| --- | --- | --- |
| VRAM | 비용 대비 효율적 | 48–288GB 이상 선택지 |
| ECC | 모델별 제한 | 일반적으로 지원 |
| NVLink·NVSwitch | 세대별 제한·부재 | server topology 지원 |
| MIG·partition | 제한 | 일부 모델 지원 |
| driver lifecycle | desktop 중심 | datacenter·enterprise branch |
| 냉각·24시간 | 케이스 설계 의존 | rack·server 설계 |
| 공식 TensorRT-LLM matrix | 일부 미포함 가능 | 명시 지원 장비 중심 |

CUDA가 동작하는 GPU라고 해서 TensorRT-LLM 공식 지원 목록에 자동 포함되는 것은 아니다. TensorRT-LLM의 현재 공식 matrix는 Blackwell·Hopper·일부 Ada·Ampere 데이터센터/프로 장비를 중심으로 확인해야 한다.

### 9.5 CUDA `llama.cpp` 빌드

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j
```

여러 세대용 binary를 만들 때 compute capability를 명시할 수 있다.

```bash
cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='86;89;120'
cmake --build build-cuda --config Release -j
```

실제 CMake가 해당 architecture 번호를 지원하는지 현재 `llama.cpp`와 CUDA toolkit에서 확인한다.

### 9.6 GPU 선택과 offload

```bash
CUDA_VISIBLE_DEVICES=0 \
  ./build-cuda/bin/llama-server \
  -m /models/model.gguf \
  -ngl 99 \
  -c 8192 \
  --host 127.0.0.1
```

부분 offload sweep:

```bash
for ngl in 0 10 20 30 40 60 99; do
  CUDA_VISIBLE_DEVICES=0 ./build-cuda/bin/llama-bench \
    -m /models/model.gguf -ngl "$ngl" -p 512 -n 128
 done
```

### 9.7 managed memory는 마지막 수단

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`은 Linux에서 VRAM 부족 시 system RAM으로 page migration해 crash를 피할 수 있다. 그러나 discrete GPU에서는 PCIe page migration으로 성능이 크게 떨어질 수 있다.

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  ./build-cuda/bin/llama-cli -m /models/too-large.gguf -ngl 99
```

이를 정상 운영 용량으로 간주하지 말고, 더 작은 quant·부분 offload·더 작은 모델과 비교한다.

### 9.8 GPU topology

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi topo -p2p n
```

- `NV*`: NVLink/NVSwitch 경로
- `PIX`·`PXB`: PCIe switch 경로
- `PHB`·`SYS`: host bridge 또는 socket interconnect 경유

TP가 단일 GPU보다 느리면 topology와 P2P를 먼저 본다. consumer multi-GPU에서 PCIe만 사용하는 경우 model이 한 GPU에 들어간다면 replica가 TP보다 유리할 수 있다.

### 9.9 MIG·vGPU

MIG·vGPU는 격리와 자원 분할에 유용하지만, partition별 메모리·compute·encoder resource와 runtime 지원이 다르다.

- 모델 가중치가 각 partition/VM에 중복되는지 확인
- P2P·NCCL·MIG 조합 지원 확인
- memory slice가 KV·workspace까지 포함하는지 계산
- vGPU profile의 framebuffer와 license 조건 확인

### 9.10 NVIDIA 체크리스트

```text
[ ] compute capability 확인
[ ] driver / torch CUDA / toolkit / runtime build 버전 기록
[ ] 단일 GPU에서 baseline
[ ] VRAM peak와 graph reserve 측정
[ ] multi-GPU topology·P2P 확인
[ ] power limit·temperature·clock 로그
[ ] 공식 runtime hardware matrix 확인
[ ] 30–60분 soak test
```


---

## 10. AMD ROCm·HIP

AMD 환경은 **GPU 이름만 보고 ROCm 지원을 추정하지 않는 것**이 가장 중요하다. 동일한 RDNA 세대라도 운영체제, 배포판, ROCm release, PyTorch wheel, 런타임과 모델 kernel에 따라 지원 범위가 달라진다. 설치 전에 [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)에서 다음 교집합을 확인한다.

```text
GPU·APU 제품
  ∩ 운영체제와 kernel
  ∩ ROCm release
  ∩ PyTorch·JAX·ONNX Runtime 계층
  ∩ vLLM·SGLang·llama.cpp 지원
  ∩ 사용할 quant·attention kernel
```

> **검증일 스냅샷:** 2026-07-16에 공개된 ROCm 7.14.0 문서가 최신 계열이지만, 설치 시점의 production release와 장치별 지원표를 다시 확인한다. `rocm/dev-*`나 nightly image가 동작한다는 사실은 production 지원을 의미하지 않는다.

### 10.1 Instinct와 Radeon의 운영 차이

| 구분 | AMD Instinct | Radeon·Ryzen AI |
| --- | --- | --- |
| 주 용도 | 데이터센터 계산·학습·대규모 서빙 | 워크스테이션·데스크톱·개인 추론 |
| 메모리 | HBM, 대용량·고대역폭 | GDDR 또는 공유 RAM |
| 대표 runtime | vLLM·SGLang·PyTorch ROCm·Triton·HIP | `llama.cpp` HIP/Vulkan, Ollama, 지원 범위의 PyTorch·vLLM |
| 다중 GPU | RCCL·Infinity Fabric·PCIe topology 중요 | P2P·메인보드 lane·driver 지원 편차 큼 |
| 배포 기준 | 공식 support matrix와 container 조합 고정 | GPU별 Linux·Windows 지원표를 반드시 확인 |

Instinct에서 잘 동작하는 FP8·FlashAttention·distributed kernel이 Radeon에서 자동으로 지원되는 것은 아니다. 반대로 GGUF의 HIP·Vulkan 경로는 소비자 GPU에서 더 단순한 출발점이 될 수 있다.

### 10.2 장치와 `gfx*` target 확인

```bash
rocminfo | grep -E 'Name:|Marketing Name|gfx' | head -n 30
hipconfig --full
rocm-smi --showproductname --showmeminfo vram --showuse --showtemp
```

AMD SMI를 사용하는 환경:

```bash
amd-smi list
amd-smi static
amd-smi monitor -g -m -u -t -p
```

기록할 값:

```text
ROCm version
GPU marketing name
LLVM target: gfx*
VRAM 용량
PCIe link width·speed
XGMI·P2P topology
firmware와 kernel driver
```

### 10.3 `llama.cpp` HIP 빌드

공식 빌드 문서의 기본 형태:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-hip --config Release -j
```

특정 target을 명시하는 예:

```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build-hip-gfx \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-hip-gfx --config Release -j
```

실제 장치의 `gfx*`와 빌드가 일치해야 한다. 다른 target을 강제로 가장하는 환경변수는 디버깅 실험에는 쓸 수 있으나, 공식 지원·정확성·안정성의 근거로 사용하지 않는다.

### 10.4 GPU 선택과 부분 offload

```bash
HIP_VISIBLE_DEVICES=0 \
  ./build-hip/bin/llama-server \
  -m /models/model.gguf \
  -ngl 99 \
  -c 8192 \
  --host 127.0.0.1
```

부분 offload sweep:

```bash
for ngl in 0 10 20 30 40 60 99; do
  HIP_VISIBLE_DEVICES=0 ./build-hip/bin/llama-bench \
    -m /models/model.gguf -ngl "$ngl" -p 512 -n 128
 done
```

VRAM에 가중치를 거의 다 넣더라도 KV 캐시와 prefill workspace를 위한 여유를 남긴다. 첫 prompt에서 OOM이 나면 `-ngl`, context, batch 순으로 조정한다.

### 10.5 Vulkan fallback

ROCm 공식 지원 밖의 Radeon이거나 Windows에서 GGUF 추론을 할 때 Vulkan이 더 현실적인 경우가 있다.

```bash
cmake -S . -B build-vulkan \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j

GGML_VK_VISIBLE_DEVICES=0 \
  ./build-vulkan/bin/llama-cli \
  -m /models/model.gguf -ngl 99
```

HIP와 Vulkan을 같은 모델·context·batch로 직접 비교한다. Vulkan은 설치 범위가 넓지만, 지원 quant·multi-GPU·attention kernel과 성능은 HIP와 다를 수 있다.

### 10.6 vLLM·SGLang ROCm

AMD 서버에서는 프로젝트가 제공하는 ROCm image나 wheel 조합을 우선한다.

```text
host ROCm driver
  ↔ container ROCm userspace
  ↔ PyTorch ROCm build
  ↔ vLLM/SGLang release
  ↔ model kernel·quantization
```

SGLang container에서 흔히 필요한 device:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  <pinned-sglang-rocm-image>
```

다중 GPU에서는 `/dev/kfd`, render node, shared memory와 RCCL topology가 모두 보여야 한다. image tag와 digest를 manifest에 남긴다.

### 10.7 ONNX Runtime: ROCm EP에서 MIGraphX로

ONNX Runtime의 기존 ROCm Execution Provider는 1.23부터 제거되었다. 신규 AMD 배포는 [MIGraphX Execution Provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) 또는 OpenVINO·Vulkan·vendor runtime을 검토한다.

```text
기존 ROCm EP 고정 배포
  → ORT·ROCm 버전을 동결하고 보안·유지보수 계획 수립

신규 AMD ONNX 배포
  → MIGraphX EP 지원 op·shape·precision 검증
```

EP가 생성되었다고 전체 graph가 GPU에서 실행되는 것은 아니다. fallback node와 host-device copy를 profiling으로 확인한다.

### 10.8 Windows의 AMD 경로

| 목적 | 우선 경로 | 대안 |
| --- | --- | --- |
| GGUF 채팅 | `llama.cpp` Vulkan 또는 지원되는 HIP 빌드 | Ollama Vulkan·ROCm |
| 이미지 생성 | DirectML·지원되는 ROCm·Vulkan backend | ONNX·vendor plugin |
| 서버 추론 | WSL2/Linux ROCm 지원 여부 검증 | 별도 Linux host |
| 범용 ONNX | WinML·DirectML | MIGraphX가 지원되는 Linux |

Windows ROCm은 Linux와 지원 GPU·기능·프레임워크 범위가 같다고 가정하지 않는다.

### 10.9 AMD 성능 점검

```bash
amd-smi monitor -g -m -u -t -p
watch -n 1 'amd-smi metric'
rocprofv3 --help
```

다음 현상을 구분한다.

| 현상 | 가능 원인 |
| --- | --- |
| GPU 사용률은 높지만 token/s가 낮음 | memory-bound, 낮은 clock, 비효율 quant kernel |
| GPU 사용률이 주기적으로 0% | CPU tokenizer·I/O·동기화·queue 병목 |
| VRAM은 남는데 OOM | contiguous workspace·fragmentation·graph reserve |
| 다중 GPU가 더 느림 | PCIe·XGMI topology, RCCL, 작은 batch |
| 첫 실행만 매우 느림 | kernel compile·graph capture·model cache cold start |

### 10.10 AMD 체크리스트

```text
[ ] compatibility matrix에서 GPU·OS·ROCm 교집합 확인
[ ] gfx target과 runtime build 기록
[ ] HIP와 Vulkan을 동일 조건으로 비교
[ ] ROCm·PyTorch·runtime container를 함께 pin
[ ] AMD SMI로 VRAM·clock·power·temperature 기록
[ ] 다중 GPU의 XGMI·PCIe·RCCL topology 확인
[ ] ONNX Runtime 신규 배포는 MIGraphX 검토
[ ] 공식 지원 밖 override를 production 근거로 사용하지 않음
```

---

## 11. Apple Silicon·Metal·MLX

Apple Silicon은 CPU와 GPU가 **통합 메모리**를 공유하므로 대형 모델을 단일 address space에 적재하기 쉽다. 그러나 표기된 통합 메모리 전체가 모델 전용 VRAM은 아니다.

```text
physical unified memory
- macOS wired memory
- WindowServer·display
- 브라우저·IDE·파일 캐시
- runtime·Metal heap
- KV·activation·temporary buffer
= 실제 모델에 쓸 수 있는 여유
```

### 11.1 Apple에서 먼저 볼 지표

Activity Monitor의 Memory 탭에서 다음을 함께 본다.

- **Memory Pressure**: 녹색·노란색·빨간색 상태
- **Memory Used**와 **Cached Files**
- **Wired Memory**
- **Compressed**
- **Swap Used**

명령행:

```bash
sysctl -n machdep.cpu.brand_string
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
memory_pressure
powermetrics --help
```

`free memory`만 보지 않는다. macOS는 파일 캐시를 적극 사용하므로 pressure, compression, swap과 지속 속도를 같이 본다.

### 11.2 통합 메모리별 현실적인 시작점

| 물리 통합 메모리 | 보수적 모델 예산 | 대표 시작점 | 주의점 |
| ---: | ---: | --- | --- |
| 8GB | 4–5GB | 3B Q4, 소형 VLM·ASR | IDE·브라우저를 닫고 2–4K context |
| 16GB | 10–12GB | 8B Q4/Q5, 14B Q3 | swap 진입 전 중지 |
| 24GB | 16–19GB | 14B Q5, 27B Q3/Q4 | VLM·이미지 pipeline 추가 메모리 |
| 32GB | 23–26GB | 27–32B Q4 | 8K+ context는 KV 별도 계산 |
| 48GB | 36–41GB | 32B Q6/Q8, 70B Q3 | 멀티 앱 환경에서 여유 확대 |
| 64GB | 49–55GB | 70B Q4/Q5 | sustained memory bandwidth 측정 |
| 96GB | 74–83GB | 70B Q8, 120B Q4 | 대형 VLM·편집 가능하나 prompt peak 확인 |
| 128GB | 98–110GB | 120B Q5/Q6, 235B Q3 | OS·cache 18–30GB 여유 권장 |
| 192GB | 148–165GB | 235B Q4/Q5 | CPU·GPU 공유 대역폭과 thermals |
| 256GB | 198–220GB | 235B Q6/Q8, 405B Q3 | 장시간 swap 0에 가까운지 확인 |

이 표는 모델별 보장이 아니라 초기 용량 계획이다. 실제 MLX·GGUF artifact 크기와 KV 캐시를 다시 계산한다.

### 11.3 런타임 선택

| 목적 | 첫 선택 | 이유 |
| --- | --- | --- |
| 텍스트 LLM·LoRA | MLX-LM | Apple Silicon용 통합·양자화·학습 경로 |
| VLM | MLX-VLM | vision encoder와 language model의 Apple 경로 |
| 폭넓은 GGUF | `llama.cpp` Metal | 모델·quant 생태계와 CPU+GPU hybrid |
| OpenAI-compatible 서버 | MLX-LM server·`llama.cpp` server | 로컬 API 제공 |
| SGLang API·스케줄러 실험 | SGLang MLX | MLX backend 지원 범위를 확인 |
| 앱 배포 | Core ML·MLC·ONNX CoreML EP | 앱 크기·지원 op·배포 target에 맞춤 |

### 11.4 `llama.cpp` Metal

Apple에서는 Metal backend가 기본 활성화되는 빌드가 일반적이다.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

./build/bin/llama-bench \
  -m /models/model.gguf \
  -ngl 99 -p 512 -n 128
```

CPU와 Metal 비교:

```bash
./build/bin/llama-bench -m /models/model.gguf -ngl 0  -p 512 -n 128
./build/bin/llama-bench -m /models/model.gguf -ngl 99 -p 512 -n 128
```

GPU offload가 항상 전체 시스템 효율을 개선하는 것은 아니다. 작은 모델·짧은 prompt에서는 launch overhead를 포함해 측정한다.

### 11.5 MLX-LM 설치와 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install mlx-lm
```

생성:

```bash
mlx_lm.generate \
  --model mlx-community/<model-id> \
  --prompt '로컬 AI 런타임을 비교해 주세요.' \
  --max-tokens 256
```

대화:

```bash
mlx_lm.chat --model mlx-community/<model-id>
```

양자화 변환:

```bash
mlx_lm.convert \
  --hf-path <org/model> \
  -q \
  --q-bits 4 \
  --upload-repo <optional-repo>
```

실제 CLI 옵션은 설치한 MLX-LM release에서 `--help`로 확인한다.

### 11.6 SGLang MLX와 vLLM-Metal

SGLang의 MLX backend 예:

```bash
SGLANG_USE_MLX=1 \
python -m sglang.launch_server \
  --model-path mlx-community/<model-id> \
  --host 127.0.0.1 \
  --port 30000
```

MLX backend가 지원하는 모델·quant·scheduler 기능은 CUDA backend와 동일하지 않을 수 있다. vLLM의 Apple 경로는 커뮤니티 `vllm-metal`·MLX 통합 상태를 확인하고, production 지원 수준을 별도로 평가한다.

### 11.7 Apple Neural Engine 해석

ANE가 탑재되어 있어도 일반적인 PyTorch·MLX·GGUF LLM이 자동으로 ANE에서 실행되는 것은 아니다. Core ML로 변환 가능한 graph, 지원 op, 정적 shape와 앱 target을 별도로 검증한다.

```text
MLX/Metal GPU 성공
  ≠ ANE 사용

Core ML model load 성공
  ≠ 모든 layer가 ANE에서 실행
```

Core ML Instruments와 performance report로 실제 compute unit 배치를 확인한다.

### 11.8 swap과 SSD

통합 메모리를 초과하면 macOS가 압축과 swap을 사용한다. 모델 load가 성공하더라도 active weights·KV가 지속적으로 swap되면 token/s와 반응성이 급락하고 SSD 쓰기가 증가한다.

```text
Memory Pressure가 노란색·빨간색
+ Swap Used가 계속 증가
+ token/s가 시간에 따라 하락
  → 모델·context·batch·동시성을 낮춤
```

### 11.9 열과 전력

노트북과 소형 데스크톱은 짧은 benchmark보다 30분 이상 지속 부하가 중요하다.

- 전원 연결 여부
- Low Power Mode 여부
- fan·surface temperature
- sustained prompt/decode 속도
- CPU·GPU 동시 사용 시 thermal budget 경쟁
- 외부 디스플레이와 WindowServer 부하

### 11.10 Apple 체크리스트

```text
[ ] physical unified memory와 실제 memory pressure 구분
[ ] MLX와 llama.cpp Metal을 동일 모델·quant로 비교
[ ] swap 0 또는 낮은 상태에서 sustained benchmark
[ ] VLM encoder·VAE·KV 메모리를 합산
[ ] ANE 사용 여부를 추정하지 않고 profiling
[ ] macOS·Xcode·Metal·MLX 버전 기록
[ ] 앱 배포 시 Core ML target과 지원 op 검증
```

---

## 12. Intel CPU·XPU·NPU

Intel 환경은 CPU, Arc·Data Center GPU, Core Ultra NPU를 하나의 vendor 이름으로 묶기보다 **각 compute device와 runtime을 분리**해 선택한다.

### 12.1 권장 경로

| 하드웨어 | 첫 선택 | 대안 |
| --- | --- | --- |
| Xeon·Core CPU | `llama.cpp`·OpenVINO GenAI | PyTorch CPU·SGLang CPU |
| Arc GPU | OpenVINO GPU·native PyTorch XPU | `llama.cpp` SYCL/Vulkan, vLLM XPU |
| Data Center GPU | native PyTorch XPU·vLLM/SGLang XPU | OpenVINO·oneAPI |
| Core Ultra NPU | OpenVINO NPU | 앱별 WinML·vendor 경로 |
| 혼합 CPU+iGPU+NPU | OpenVINO AUTO·HETERO | 개별 device benchmark |

### 12.2 IPEX 상태

Intel Extension for PyTorch 저장소는 2026-03-30에 archive되어 read-only 상태다. 신규 배포는 다음을 우선한다.

```text
native PyTorch XPU
OpenVINO / OpenVINO GenAI
oneAPI·SYCL
runtime 자체 Intel backend
```

기존 IPEX 배포는 PyTorch·oneAPI·driver를 함께 동결하고, 보안 수정과 장기 유지보수 계획을 별도로 세운다.

### 12.3 native PyTorch XPU

현재 PyTorch의 Intel GPU 경로는 `torch.xpu` namespace를 사용한다.

```bash
python3 -m venv .venv-xpu
source .venv-xpu/bin/activate
python -m pip install -U pip
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/xpu
```

확인:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('xpu available:', torch.xpu.is_available())
print('xpu count:', torch.xpu.device_count() if torch.xpu.is_available() else 0)
if torch.xpu.is_available():
    for i in range(torch.xpu.device_count()):
        print(i, torch.xpu.get_device_name(i))
PY
```

모델·dtype·attention kernel 지원은 CUDA와 다를 수 있으므로 BF16·FP16·INT4를 각각 smoke test한다.

### 12.4 Xeon의 AMX·VNNI·AVX

서버 CPU에서 중요한 요소:

- AVX2·AVX-512·VNNI·AMX 지원 여부
- socket당 memory channel 수와 DIMM population
- NUMA locality
- OpenMP·TBB thread affinity
- oneDNN·oneMKL·OpenVINO kernel
- BF16·INT8·INT4 경로

```bash
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|avx|amx|vnni'
numactl --hardware
```

AMX가 있어도 runtime이 해당 kernel을 사용하지 않으면 이점이 없다. `perf`, OpenVINO benchmark와 runtime log로 확인한다.

### 12.5 `llama.cpp` SYCL

oneAPI 환경 예:

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build-sycl \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl --config Release -j
```

장치 확인과 실행:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  ./build-sycl/bin/llama-bench \
  -m /models/model.gguf -ngl 99 -p 512 -n 128
```

환경변수와 selector 문법은 oneAPI·SYCL release에 따라 확인한다.

### 12.6 OpenVINO AUTO·HETERO·MULTI

| mode | 목적 |
| --- | --- |
| `CPU`, `GPU`, `NPU` | 특정 device 지정 |
| `AUTO` | 가용 장치에서 적합한 device 선택 |
| `HETERO` | graph 일부를 다른 device로 분할 |
| `MULTI` | 여러 device로 요청 처리 |
| `AUTO:GPU,CPU` | 우선순위·후보 지정 |

자동 mode가 항상 최적의 지연이나 메모리를 보장하지 않는다. 실제 device 배치와 fallback을 profiling한다.

### 12.7 NPU의 역할

NPU는 저전력·지속적인 소형 모델 처리에 유리할 수 있으나, 다음 제한이 있다.

- 지원되는 op·shape·precision
- 모델 export·compile 시간
- 정적·동적 sequence 처리
- NPU 전용 메모리와 공유 메모리
- LLM 전체가 아닌 일부 encoder·전처리만 배치될 가능성

소형 ASR·vision encoder·embedding·분류부터 검증하고, 대형 autoregressive LLM은 CPU·GPU와 비교한다.

### 12.8 Intel 모니터링

```bash
xpu-smi discovery
xpu-smi dump -d 0 -m 0,1,2,3,4,5
oneapi-cli --help 2>/dev/null || true
```

도구 가용성은 driver 패키지에 따라 다르다. CPU는 `perf`, `pcm-memory`, `numastat`, OpenVINO `benchmark_app`을 함께 사용한다.

### 12.9 Intel 체크리스트

```text
[ ] CPU·GPU·NPU를 별도 device로 인식
[ ] 신규 PyTorch는 native XPU 우선
[ ] OpenVINO device와 fallback 확인
[ ] IPEX 신규 도입을 피하고 기존 배포는 동결 계획
[ ] AMX·VNNI가 실제 kernel에서 사용되는지 검증
[ ] Arc·Data Center GPU별 공식 지원표 확인
[ ] CPU NUMA·memory channel을 기록
```

---

## 13. Vulkan·SYCL·DirectML·WinML·WebGPU

Vendor-native backend가 없거나 범용 배포가 필요할 때 portability backend를 사용한다. 그러나 “폭넓게 실행된다”와 “가장 빠르다”는 같은 뜻이 아니다.

### 13.1 선택표

| backend | 주요 환경 | 장점 | 제약 |
| --- | --- | --- | --- |
| Vulkan | NVIDIA·AMD·Intel·일부 모바일 | 폭넓은 GPU 범위, GGUF fallback | vendor-native보다 kernel·multi-GPU 제약 가능 |
| SYCL·Level Zero | Intel 중심, 일부 이식성 | oneAPI 생태계, CPU·GPU 추상화 | toolchain·device selector 복잡성 |
| DirectML | Windows DirectX 12 GPU | 광범위한 Windows 장치 | sustained engineering, 생성형 모델 성능·op 제약 |
| WinML | Windows 앱 AI runtime | Windows 신규 앱 통합 방향 | model·EP·Windows 버전 확인 |
| WebGPU | 브라우저·웹 앱 | 설치 없는 client inference | 브라우저 quota·shader compile·지원 편차 |
| OpenCL | 일부 모바일·구형 장치 | 넓은 장치 범위 | 최신 LLM kernel 생태계가 상대적으로 제한적 |

### 13.2 `llama.cpp` Vulkan

```bash
cmake -S . -B build-vulkan \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j

./build-vulkan/bin/llama-bench \
  -m /models/model.gguf -ngl 99 -p 512 -n 128
```

장치 선택:

```bash
GGML_VK_VISIBLE_DEVICES=0 \
  ./build-vulkan/bin/llama-server \
  -m /models/model.gguf -ngl 99 -c 8192
```

현재 `llama.cpp` feature matrix에서 parallel multi-GPU 지원은 CUDA·ROCm과 Vulkan·SYCL이 다를 수 있다. sequential split이 동작한다고 TP 수준의 scaling을 기대하지 않는다.

### 13.3 DirectML에서 WinML로의 방향

ONNX Runtime DirectML EP는 계속 사용할 수 있으나 DirectML은 sustained engineering 상태이며, Windows의 신규 개발은 WinML 방향을 검토한다.

DirectML session에서 확인할 항목:

- 지원 ONNX opset과 제외 op
- sequential execution requirement
- memory pattern 설정 제약
- CPU fallback 여부
- static·dynamic shape 성능
- adapter별 driver 차이

```python
import onnxruntime as ort

print(ort.get_available_providers())

sess = ort.InferenceSession(
    "model.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
print(sess.get_providers())
```

provider 목록에 있다고 전체 graph가 DirectML에서 실행된다는 뜻은 아니다.

### 13.4 WebGPU

브라우저 inference는 다음 예산을 별도로 계산한다.

```text
model artifact download
+ IndexedDB·Cache Storage quota
+ WebAssembly heap
+ WebGPU buffer
+ browser tab·renderer memory
+ shader compile cache
```

실무 체크:

- 브라우저·OS·GPU별 `navigator.gpu` 지원
- corporate browser policy
- private mode의 storage quota
- 탭 reload 후 model cache
- cross-origin isolation과 worker
- 입력 데이터가 서버로 전송되지 않는지 명시
- 모델 URL·service worker supply-chain 검증

### 13.5 portability backend 비교법

```text
동일 model revision
동일 quant
동일 context·batch
동일 prompt
동일 출력 token 수
동일 warm-up
```

위 조건으로 CUDA/HIP/Metal과 Vulkan·SYCL·DirectML·WebGPU를 비교한다. output correctness와 peak memory도 기록한다.

---

## 14. Arm·엣지·모바일·기타 가속기

엣지 장치는 peak TOPS보다 **지속 가능한 전력, 메모리 대역폭, runtime 지원과 모델 배포 크기**가 중요하다.

### 14.1 Arm CPU

`llama.cpp`는 Arm·NEON·일부 KleidiAI 최적화를 지원한다. 빌드 시 실제 ISA와 vendor library를 확인한다.

```bash
cmake -S . -B build-arm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm --config Release -j

./build-arm/bin/llama-bench \
  -m /models/small-model.gguf -p 256 -n 64
```

SBC에서는 다음을 제한한다.

- context 2–4K
- batch 1
- 1–3B Q4/Q5
- swap 최소화
- 능동 냉각
- eMMC 대신 충분한 내구성의 SSD

### 14.2 NVIDIA Jetson

Jetson은 CUDA를 사용하지만 데이터센터·데스크톱 GPU와 다음이 다르다.

- JetPack·L4T와 CUDA 결합
- 통합 메모리
- 전력 mode와 fan profile
- ARM64 wheel·container
- TensorRT·TensorRT-LLM 지원 범위

```bash
sudo nvpmodel -q
tegrastats
```

JetPack 버전에 맞는 container와 wheel을 사용하고, desktop CUDA 명령을 그대로 복사하지 않는다.

### 14.3 Android·Adreno·Mali

가능한 경로:

- `llama.cpp` Android/NDK
- Vulkan
- OpenCL·Adreno backend
- MLC LLM Android
- NNAPI·vendor NPU SDK

모바일에서는 모델 파일, 앱 binary, KV, OS 메모리를 합산하고 background kill·thermal throttling·battery drain을 측정한다. 1–3B Q4와 짧은 context부터 시작한다.

### 14.4 Ascend·MUSA·TPU·기타

SGLang·vLLM·vendor runtime은 특정 release에서 다음 플랫폼을 지원할 수 있다.

- Huawei Ascend·CANN
- Moore Threads MUSA
- Google TPU
- AWS Trainium·Inferentia
- Qualcomm·MediaTek NPU

모든 기능이 CUDA backend와 동등하다고 가정하지 않는다. 다음 matrix를 작성한다.

| 확인 항목 | 질문 |
| --- | --- |
| 모델 | 목표 architecture가 지원되는가? |
| quant | INT4·FP8·KV quant kernel이 있는가? |
| attention | Flash·paged·sliding-window·MLA가 있는가? |
| serving | continuous batching·prefix cache가 있는가? |
| distributed | TP·PP·EP와 collective가 있는가? |
| tooling | profiler·monitor·container가 있는가? |
| 유지보수 | stable release·security advisory가 있는가? |

### 14.5 RISC-V·WebAssembly

연구·교육·소형 모델에서는 가능하지만, 일반적으로 최신 x86·Arm·GPU 대비 kernel 생태계와 절대 성능이 제한적이다. “실행 성공”과 “대화형 속도”를 분리해 기록한다.

---

## 15. `llama.cpp`

`llama.cpp`는 CPU부터 CUDA·HIP·Metal·Vulkan·SYCL까지 폭넓게 지원하는 GGUF 중심 런타임이다. 개인 PC, 부분 GPU offload, Apple Silicon, edge에서 기본 비교 기준으로 사용하기 좋다.

### 15.1 적합한 경우

- GGUF Q2–Q8·IQ quant를 직접 선택하려는 경우
- CPU와 GPU layer offload를 혼합하려는 경우
- 단일 binary·OpenAI-compatible server가 필요한 경우
- VLM projector를 함께 실행하려는 경우
- 다양한 운영체제와 GPU vendor를 한 도구로 비교하려는 경우
- `llama-bench`로 prompt·generation 성능을 빠르게 측정하려는 경우

### 15.2 backend 요약

| backend | CMake option | 주 환경 |
| --- | --- | --- |
| CPU | 기본 | x86·Arm·RISC-V |
| BLAS | `GGML_BLAS=ON` | prompt processing 개선 가능 |
| CUDA | `GGML_CUDA=ON` | NVIDIA |
| HIP | `GGML_HIP=ON` | AMD ROCm |
| Metal | 기본 또는 Metal option | Apple Silicon |
| Vulkan | `GGML_VULKAN=ON` | 범용 GPU |
| SYCL | `GGML_SYCL=ON` | Intel oneAPI 중심 |
| OpenCL | 해당 build option | 일부 모바일·범용 장치 |

실제 option 이름은 현재 [공식 빌드 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)를 기준으로 확인한다.

### 15.3 Hugging Face에서 직접 실행

```bash
./build/bin/llama-cli \
  -hf <org>/<gguf-repo>:Q4_K_M \
  -p '메모리 대역폭이 중요한 이유를 설명해 주세요.' \
  -n 256
```

서버:

```bash
./build/bin/llama-server \
  -hf <org>/<gguf-repo>:Q4_K_M \
  -c 8192 \
  -np 1 \
  --host 127.0.0.1 \
  --port 8080
```

repository의 quant tag와 shard 구조를 실제 tree에서 확인한다.

### 15.4 핵심 옵션

| 목적 | 대표 옵션 | 해석 |
| --- | --- | --- |
| GPU offload | `-ngl` | GPU에 올릴 layer 수 |
| context | `-c` | 총 context budget |
| parallel slot | `-np` | server 동시 slot |
| batch | `-b`, `-ub` | logical·physical batch |
| KV dtype | `-ctk`, `-ctv` | K·V cache 정밀도 |
| tensor split | `-ts` | multi-GPU 비율 |
| split mode | `-sm` | layer·row·none 등 |
| CPU thread | `-t`, `-tb` | decode·batch thread |
| mmap | `--mmap` 계열 | model mapping 정책 |
| mlock | `--mlock` | page eviction 방지, RAM 여유 필요 |

옵션은 빠르게 변하므로 `llama-cli --help`, `llama-server --help`를 manifest와 함께 보존한다.

### 15.5 벤치마크

```bash
./build/bin/llama-bench \
  -m /models/model.gguf \
  -p 128,512,2048 \
  -n 64,256 \
  -ngl 0,99 \
  -r 5
```

결과에서 prompt processing와 token generation을 분리한다. 모델·quant·backend가 다르면 같은 표에 섞지 않는다.

### 15.6 server health와 metrics

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/metrics | head
```

metrics endpoint와 이름은 build·server option에 따라 달라질 수 있다. 공개 network에 bind할 때 인증·reverse proxy·rate limit를 적용한다.

### 15.7 공식 container

공식 repository는 CPU·CUDA·ROCm·SYCL·Vulkan·OpenVINO 등 여러 image 변형을 제공한다. 예시는 반드시 release tag와 digest로 고정한다.

```bash
docker pull ghcr.io/ggml-org/llama.cpp:<pinned-tag>
docker image inspect ghcr.io/ggml-org/llama.cpp:<pinned-tag> \
  --format '{{index .RepoDigests 0}}'
```

### 15.8 `llama.cpp` 한계

- 최신 architecture의 model conversion이 늦을 수 있음
- 서버 scheduler가 대규모 데이터센터 runtime과 다름
- backend별 feature parity가 완전하지 않음
- GGUF quant quality가 변환자·imatrix에 따라 달라짐
- 초대형 batch·distributed serving에서는 vLLM·SGLang·TensorRT-LLM이 유리할 수 있음

---

## 16. Ollama

Ollama는 model lifecycle과 로컬 API를 단순화한다. 상세한 kernel·memory 제어보다 **설치와 운영 편의성**이 우선일 때 적합하다.

### 16.1 기본 흐름

```bash
ollama pull <model>
ollama run <model>
ollama ps
```

API 확인:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "<model>",
    "prompt": "RAM과 VRAM의 차이를 설명해 주세요.",
    "stream": false
  }'
```

### 16.2 하드웨어 경로

| 환경 | 일반 경로 |
| --- | --- |
| NVIDIA | CUDA |
| AMD | 지원되는 ROCm 또는 Vulkan 실험 경로 |
| Apple Silicon | Metal |
| CPU | CPU GGML 계열 |
| Windows·Linux 범용 GPU | Vulkan 지원 상태 확인 |

[Ollama hardware support](https://docs.ollama.com/gpu)에서 현재 GPU 목록과 환경변수를 확인한다.

### 16.3 메모리와 동시성

Ollama에서는 다음이 메모리를 크게 바꾼다.

- `num_ctx`
- parallel request 수
- 동시에 load된 model 수
- KV cache dtype
- GPU layer·offload
- multimodal projector

```bash
ollama ps
```

출력의 processor·context·memory 정보를 기록한다. 단순히 model manifest의 파일 크기만으로 운영 메모리를 계산하지 않는다.

### 16.4 환경변수 예시

```bash
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=64
```

정확한 변수와 기본값은 설치한 release의 공식 FAQ·문서를 확인한다. 동시성 증가는 KV 캐시와 context 예산도 증가시킨다.

### 16.5 Vulkan 장치 선택

지원되는 release에서:

```bash
GGML_VK_VISIBLE_DEVICES=0 ollama serve
```

문제가 있으면 Vulkan을 비활성화하거나 vendor-native backend와 비교한다.

### 16.6 Modelfile 재현성

```Dockerfile
FROM <base-model>
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
SYSTEM "근거와 계산 과정을 구분해 답한다."
```

```bash
ollama create ram-local-ai -f Modelfile
ollama show ram-local-ai --modelfile
```

base model digest, Modelfile, Ollama version을 함께 보존한다.

### 16.7 운영 주의

- 기본 API를 공인망에 직접 노출하지 않음
- 모델 blob 저장소 권한 제한
- prompt·response logging에 개인정보가 남는지 확인
- 자동 pull 대신 digest·revision 검증
- GUI·desktop 앱의 background model load 확인
- queue가 무한히 늘지 않도록 upstream backpressure 적용

---

## 17. MLX-LM·MLX-VLM

MLX는 Apple Silicon의 통합 메모리와 Metal을 활용하는 array framework다. MLX-LM·MLX-VLM은 텍스트·비전 모델의 변환, 양자화, 생성, 파인튜닝과 서버 실행을 제공한다.

### 17.1 선택 기준

| 선택 | 적합한 상황 |
| --- | --- |
| MLX-LM | 텍스트 생성·LoRA·QLoRA·Apple 전용 최적화 |
| MLX-VLM | 이미지·문서·VLM |
| `llama.cpp` Metal | GGUF 폭넓은 호환·CPU+GPU hybrid |
| Core ML | 앱 배포·지원 op가 고정된 production pipeline |

### 17.2 설치 격리

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install -U pip
pip install mlx-lm mlx-vlm
python -m pip freeze > requirements-mlx.txt
```

macOS·Python·MLX·MLX-LM·MLX-VLM 버전을 함께 기록한다.

### 17.3 변환 전 dry-run

```bash
hf download <org/model> --dry-run
```

변환에는 원본 weight와 출력 MLX artifact가 동시에 필요할 수 있다.

```text
필요 disk
≈ 원본 weight
+ 변환 중 임시 파일
+ MLX output
+ 20–30% 여유
```

### 17.4 4-bit와 8-bit 비교

```bash
mlx_lm.convert --hf-path <org/model> -q --q-bits 4 --mlx-path ./model-4bit
mlx_lm.convert --hf-path <org/model> -q --q-bits 8 --mlx-path ./model-8bit
```

동일 prompt suite로 다음을 비교한다.

- load memory
- prompt tokens/s
- generation tokens/s
- JSON·code·math 정확도
- 긴 문서 retrieval answer
- memory pressure·swap

### 17.5 MLX server

설치 버전이 server command를 제공하는지 확인:

```bash
mlx_lm.server --help
```

예:

```bash
mlx_lm.server \
  --model mlx-community/<model-id> \
  --host 127.0.0.1 \
  --port 8080
```

외부 노출은 reverse proxy 뒤에서 인증·TLS·rate limit를 적용한다.

### 17.6 분산 MLX

MLX-LM은 Apple Silicon 여러 장치의 분산 inference·training 경로를 제공할 수 있다. 다음 비용을 측정한다.

- Thunderbolt·network interconnect bandwidth
- collective latency
- shard load 시간
- 각 Mac의 메모리 불균형
- process failure와 재시작
- 모델 artifact 중복 저장

단일 128GB Mac과 네트워크로 연결한 2×64GB Mac은 동일한 128GB address space가 아니다.

### 17.7 VLM 주의

VLM은 다음 구성요소를 합산한다.

```text
language model
+ vision encoder
+ projector
+ image preprocessing
+ visual tokens
+ KV cache
```

MLX-VLM 지원 모델 목록과 prompt template을 현재 release에서 확인한다. HF Transformers용 remote code가 MLX에서 자동으로 호환된다고 가정하지 않는다.

---

## 18. vLLM

vLLM은 paged KV cache, continuous batching과 고처리량 API 서버를 제공하는 대표적인 serving runtime이다. 개인 단일 prompt보다 다중 요청·서버 처리량에서 가치가 크다.

### 18.1 현재 하드웨어 범위

공식 설치 문서의 최신 계열은 다음 경로를 다룬다.

| 플랫폼 | 경로 |
| --- | --- |
| NVIDIA | CUDA wheel·container |
| AMD | ROCm wheel·container |
| Intel | XPU wheel·kernel package |
| Apple Silicon | 커뮤니티 vLLM-Metal·MLX 경로 |
| CPU | 전용 CPU install·build |

Windows native는 기본 지원 경로가 아니며 WSL2 또는 커뮤니티 build를 검토한다.

### 18.2 버전 조합

2026-07-21의 개발 문서 스냅샷에서는 기본 NVIDIA binary가 CUDA 12.9 계열이며 다른 CUDA build도 제공된다. 그러나 production은 설치 시점의 **stable release 문서**에 맞춰 다음을 고정한다.

```text
vLLM version
PyTorch version
CUDA 또는 ROCm version
GPU architecture
flash-attention·triton kernel
container digest
```

Blackwell·새 GPU는 최소 CUDA 요구사항을 별도로 확인한다.

### 18.3 독립 환경

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U pip
pip install vllm
```

기존 PyTorch 환경 위에 임의로 덮어쓰지 않는다. vLLM wheel이 요구하는 PyTorch·CUDA 조합과 충돌할 수 있다.

### 18.4 기본 서버

```bash
vllm serve <org/model> \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 8
```

메모리 예산을 더 명시적으로 제어:

```bash
vllm serve <org/model> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192
```

현재 버전에서 `--kv-cache-memory-bytes`, KV dtype, CPU offload 등의 옵션을 확인한다.

### 18.5 GPU 요구사항 해석

NVIDIA에서는 compute capability 하한, AMD에서는 공식 ROCm GPU, Intel에서는 XPU package를 확인한다. wheel 설치 성공만으로 model architecture·quant kernel 지원이 보장되지는 않는다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('cuda:', torch.version.cuda)
print('hip:', torch.version.hip)
print('cuda available:', torch.cuda.is_available())
PY
```

### 18.6 tensor parallel

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve <org/model> \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

TP 전에 다음을 확인한다.

- 한 layer shard의 크기
- GPU memory symmetry
- P2P·NVLink·PCIe topology
- NCCL/RCCL 정상 작동
- single-GPU replica 대비 throughput
- 모델이 TP size로 나누어지는지

### 18.7 quantization

지원 형식은 하드웨어와 release마다 다르다.

- AWQ
- GPTQ·Marlin
- `compressed-tensors`
- FP8
- bitsandbytes
- GGUF
- vendor-specific INT4·FP4

[양자화 가이드](./quantization.md)의 runtime compatibility를 함께 본다. quantized weight가 적재된 뒤 실제 kernel이 dequantize fallback인지 확인한다.

### 18.8 운영 기준

- `/metrics`의 queue·running request·KV usage 관측
- prefix cache hit ratio
- TTFT·TPOT p50/p95/p99
- prefill·decode token/s 분리
- OOM 후 worker recovery
- max model length와 admission control
- model revision·remote code 통제

자세한 계산은 [서빙·동시성 가이드](./serving-concurrency.md)를 참고한다.

---

## 19. SGLang

SGLang은 RadixAttention·prefix caching, structured generation, agent·multimodal serving을 제공하는 고성능 runtime이다. 플랫폼 지원은 넓지만 backend별 기능 차이를 확인해야 한다.

### 19.1 플랫폼

공식 설치 문서는 NVIDIA CUDA, AMD ROCm, Apple Metal·MLX, Intel CPU·XPU, Jetson, Ascend 등 다양한 경로를 제공한다. 모든 backend에서 동일한 quant·attention·distributed 기능이 구현되었다고 가정하지 않는다.

### 19.2 CUDA image 선택

2026-07-21의 최신 문서 스냅샷에서는 기본 image가 CUDA 13 계열을 사용할 수 있으며, CUDA 12·12.9 변형 tag도 안내한다. production에서는 host driver와 호환되는 **고정 release tag·digest**를 사용한다.

```bash
docker pull <sglang-image>:<pinned-tag>
docker image inspect <sglang-image>:<pinned-tag> \
  --format '{{index .RepoDigests 0}}'
```

### 19.3 기본 서버

```bash
python -m sglang.launch_server \
  --model-path <org/model> \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.80 \
  --max-running-requests 8
```

긴 prompt의 prefill OOM 대응:

```bash
python -m sglang.launch_server \
  --model-path <org/model> \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.75
```

### 19.4 AMD ROCm

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  <pinned-sglang-rocm-image>
```

MI300·MI350 등 공식 대상으로 제공되는 kernel과 Radeon 지원을 구분한다.

### 19.5 Apple MLX

```bash
SGLANG_USE_MLX=1 \
python -m sglang.launch_server \
  --model-path mlx-community/<model-id> \
  --host 127.0.0.1 \
  --port 30000
```

MLX 4-bit·8-bit 또는 load-time quant 지원 범위를 현재 문서에서 확인한다. `torch.mps` fallback은 MLX backend보다 기능·성능이 제한될 수 있다.

### 19.6 Intel CPU·XPU

CPU backend는 Intel AMX가 있는 최신 Xeon에서 이점이 있을 수 있다. XPU backend는 native PyTorch XPU·oneAPI 조합과 현재 kernel package를 확인한다.

```text
SGLang version
PyTorch XPU version
oneAPI driver
GPU firmware
model dtype·quant
```

### 19.7 SGLang을 선택할 때

- 반복되는 긴 system prompt·RAG prefix
- structured output·constrained decoding
- agent·tool workflow
- multimodal serving
- disaggregated prefill·decode 연구
- cache-aware routing과 고처리량 API

단일 사용자의 간단한 로컬 채팅은 `llama.cpp`·Ollama·MLX가 더 단순할 수 있다.

### 19.8 장애 예방

- `mem-fraction-static`을 1에 가깝게 설정하지 않음
- max running requests와 context를 동시에 과도하게 늘리지 않음
- prefix cache가 tenant 경계를 넘지 않도록 namespace·salt 검토
- backend별 지원 quant를 실제 log에서 확인
- nightly image를 production에 사용하지 않음
- warm-up과 graph capture 시간을 startup probe에 반영

---

## 20. TensorRT-LLM

TensorRT-LLM은 NVIDIA 지원 GPU에서 model graph와 kernel을 최적화해 높은 처리량·낮은 지연을 목표로 한다. CUDA가 동작하는 모든 GPU를 동일하게 공식 지원하는 runtime으로 해석하면 안 된다.

### 20.1 공식 지원 하드웨어 확인

2026-07-14 문서 스냅샷의 주요 지원 계열에는 다음이 포함된다.

- Blackwell: B200·GB200·B300·GB300·DGX Spark 계열
- Hopper: H100·H200·GH200
- Ada 데이터센터·워크스테이션: L20·L40·L40S
- Ampere: A100

GeForce·기타 CUDA GPU에서 일부 기능이 실행될 수 있어도, 공식 TensorRT-LLM hardware support와 동일하지 않다. [지원 하드웨어 문서](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)를 설치 시점에 확인한다.

### 20.2 적합한 경우

- 모델과 hardware가 장기간 고정됨
- NVIDIA data center GPU에서 최대 처리량이 필요함
- engine build·profiling 비용을 감당할 수 있음
- FP8·FP4·INT4·speculative decoding을 검증함
- Triton Inference Server와 통합함
- production SLO와 회귀 benchmark가 있음

### 20.3 부적합한 경우

- 모델을 자주 바꾸는 개인 연구
- 공식 support matrix 밖의 GPU
- 여러 vendor를 같은 image로 지원해야 함
- unsupported architecture·remote code model
- engine 재빌드와 calibration 관리가 어려움

### 20.4 설치 스냅샷 주의

TensorRT-LLM 공식 설치 문서는 특정 시점의 Ubuntu·CUDA·PyTorch·Python 조합을 제시한다. 예제 버전을 영구 기준으로 복사하지 않고 release branch의 compatibility를 따른다.

```text
NVIDIA driver
CUDA userspace
PyTorch build
TensorRT
TensorRT-LLM
NCCL
Triton Server
```

이 여섯 계층을 container digest와 함께 pin한다.

### 20.5 engine build 흐름

```text
Hugging Face checkpoint
  → checkpoint conversion
  → quantization·calibration
  → TensorRT-LLM engine build
  → runtime·Triton deployment
  → correctness·performance validation
```

engine은 hardware architecture, TP·PP size, max batch, sequence length와 precision에 종속될 수 있다. 원본 모델 revision만 보관해서는 재현되지 않는다.

### 20.6 다중 GPU

- NVLink·NVSwitch topology
- NCCL version
- TP·PP·EP 배치
- GPU memory 균형
- KV cache reuse
- in-flight batching
- host pinned memory
- MIG와 partition 지원

을 함께 검증한다.

### 20.7 최소 manifest

```yaml
runtime:
  name: tensorrt-llm
  version: <version>
  container_digest: sha256:<digest>
model:
  repo: <org/model>
  revision: <commit>
engine:
  precision: fp8
  tensor_parallel: 2
  max_batch_size: 32
  max_input_len: 8192
  max_output_len: 2048
hardware:
  gpu: <exact model>
  driver: <version>
  cuda: <version>
```


---

## 21. OpenVINO GenAI·OVMS

OpenVINO는 Intel CPU·GPU·NPU를 중심으로, 일부 범용 CPU 환경에서 모델 변환·압축·추론을 제공한다. OpenVINO GenAI는 텍스트 생성, VLM, Whisper·ASR, text-to-image 등 생성형 pipeline을 상위 API로 묶는다.

### 21.1 적합한 경우

- Intel Xeon·Core CPU에서 INT4·INT8 추론
- Arc·Data Center GPU
- Core Ultra NPU와 CPU·GPU 혼합
- Windows·Linux에서 동일한 IR 배포
- OpenVINO Model Server로 REST·gRPC serving
- 소형 edge·enterprise 환경에서 device fallback이 필요한 경우

### 21.2 설치

```bash
python3 -m venv .venv-openvino
source .venv-openvino/bin/activate
python -m pip install -U pip
pip install openvino openvino-genai
```

모델 export 도구가 필요한 경우:

```bash
pip install "optimum-intel[openvino]"
```

환경 확인:

```bash
python - <<'PY'
import openvino as ov

core = ov.Core()
print('OpenVINO:', ov.__version__)
print('devices:', core.available_devices)
for dev in core.available_devices:
    try:
        print(dev, core.get_property(dev, 'FULL_DEVICE_NAME'))
    except Exception as exc:
        print(dev, exc)
PY
```

### 21.3 INT4 export

대표 흐름:

```bash
optimum-cli export openvino \
  --model <org/model> \
  --weight-format int4 \
  ./openvino-model-int4
```

다음 옵션은 모델과 설치 버전에 따라 달라진다.

- group size
- ratio·sensitivity
- symmetric·asymmetric
- dataset·calibration
- tokenizer 포함 여부
- stateful model

반드시 `optimum-cli export openvino --help`와 모델 architecture 지원표를 확인한다.

### 21.4 GenAI `LLMPipeline`

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./openvino-model-int4", "CPU")
config = ov_genai.GenerationConfig()
config.max_new_tokens = 256
config.temperature = 0.2

result = pipe.generate(
    "통합 메모리와 전용 VRAM의 차이를 설명해 주세요.",
    config,
)
print(result)
```

GPU 또는 NPU:

```python
pipe = ov_genai.LLMPipeline("./openvino-model-int4", "GPU")
```

모델·device별 지원 precision과 stateful KV 경로를 확인한다.

### 21.5 AUTO·HETERO·MULTI

```python
import openvino as ov

core = ov.Core()
compiled = core.compile_model("model.xml", "AUTO:GPU,CPU")
```

| mode | 사용 목적 | 검증 항목 |
| --- | --- | --- |
| `AUTO` | 적합한 device 자동 선택 | 실제 선택된 device, startup latency |
| `HETERO` | 지원되지 않는 op를 다른 device에 배치 | graph 분할·copy 비용 |
| `MULTI` | 여러 device에서 요청 처리 | load balance·model 복제 메모리 |
| 특정 device | 재현 가능한 단일 target | precision·memory·성능 기준선 |

처음에는 특정 device로 baseline을 만든 뒤 자동 mode를 비교한다.

### 21.6 GGUF 직접 load

OpenVINO GenAI의 일부 최신 경로는 제한된 topology에서 GGUF direct load를 preview로 제공할 수 있다. 다음 조건을 구분한다.

```text
지원되는 GGUF topology·quant
  → direct load 실험

그 외 모델·production
  → OpenVINO IR export를 기본 경로로 사용
```

GGUF가 열렸다는 사실만으로 vision encoder, tool calling, speculative decoding과 모든 generation feature가 지원되는 것은 아니다.

### 21.7 OpenVINO Model Server

OVMS는 OpenVINO 모델을 REST·gRPC로 serving하고, 일부 GenAI pipeline과 continuous batching을 제공한다.

컨테이너 예시의 개념:

```bash
docker run --rm \
  -p 9000:9000 \
  -v "$PWD/models:/models:ro" \
  <pinned-ovms-image> \
  --model_path /models/<model> \
  --model_name local-ai \
  --port 9000
```

실제 GenAI configuration, endpoint와 image tag는 [OVMS 문서](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html)를 따른다.

### 21.8 `benchmark_app`

```bash
benchmark_app \
  -m ./openvino-model/model.xml \
  -d CPU \
  -api async \
  -t 30
```

생성형 LLM의 token latency와 `benchmark_app`의 graph throughput은 동일한 지표가 아니다. pipeline 수준 TTFT·TPOT도 별도로 측정한다.

### 21.9 OpenVINO 체크리스트

```text
[ ] available_devices와 실제 device name 기록
[ ] 원본 모델 revision과 IR export option 기록
[ ] CPU·GPU·NPU 각각 단독 baseline
[ ] AUTO·HETERO의 실제 graph 배치 확인
[ ] INT4 quality regression 평가
[ ] GGUF direct load는 preview 범위 확인
[ ] OVMS image digest·configuration 보존
```

---

## 22. PyTorch·Transformers·Optimum

PyTorch·Transformers는 가장 유연한 연구 기준선이다. 최신 모델의 공식 구현을 빨리 시험하기 좋지만, 자동 device mapping과 custom code 때문에 메모리·성능·보안 결과가 환경마다 달라질 수 있다.

### 22.1 하드웨어 확인 스크립트

```python
from __future__ import annotations

import platform
import torch

print("python platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("hip build:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("accelerator count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, p.total_memory / 1024**3, "GiB")

if hasattr(torch, "xpu"):
    print("xpu available:", torch.xpu.is_available())

if hasattr(torch.backends, "mps"):
    print("mps available:", torch.backends.mps.is_available())
```

AMD PyTorch도 `torch.cuda` API namespace를 사용하는 부분이 있으므로 `torch.version.hip`을 함께 본다.

### 22.2 안전한 기본 load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "<org/model>"
revision = "<commit-sha>"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
model.eval()
```

모델이 custom architecture를 요구하면 remote code를 무조건 활성화하지 않는다. source revision을 검토하고 격리 환경에서 실행한다.

### 22.3 `device_map="auto"` 해석

자동 배치는 편리하지만 다음을 숨길 수 있다.

- 일부 layer의 CPU offload
- GPU 간 비대칭 shard
- PCIe 왕복
- disk offload
- vision encoder와 language model의 다른 장치 배치

```python
print(model.hf_device_map)
```

실제 map과 peak memory를 manifest에 남긴다.

### 22.4 명시적 `max_memory`

```python
max_memory = {
    0: "22GiB",
    "cpu": "48GiB",
}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

표기 VRAM 전체를 지정하지 말고 desktop·runtime·workspace 여유를 남긴다.

### 22.5 attention backend

가능한 경로:

- eager
- PyTorch SDPA
- FlashAttention
- xFormers
- vendor-specific fused attention

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

지원되지 않는 dtype·head dimension·GPU에서는 fallback 또는 error가 발생한다. backend 이름만 기록하지 말고 runtime log와 profiler로 실제 kernel을 확인한다.

### 22.6 `torch.compile`

```python
model = torch.compile(model, mode="reduce-overhead")
```

장점과 비용:

| 항목 | 영향 |
| --- | --- |
| 첫 실행 | compile 시간 증가 |
| shape 변화 | recompile 가능 |
| graph cache | 추가 disk·memory |
| peak memory | capture·workspace 증가 가능 |
| 속도 | 모델·backend·shape에 따라 개선 또는 악화 |

cold start와 steady state를 분리해 측정한다.

### 22.7 Optimum 계열

Optimum은 hardware·runtime별 export와 최적화를 연결한다.

- Optimum Intel: OpenVINO
- Optimum ONNX Runtime
- Optimum NVIDIA 계열 지원 범위
- vendor-specific quantization

동일한 `transformers` model이라도 export 후 output tolerance와 generation config를 비교한다.

### 22.8 메모리 측정

CUDA·ROCm API namespace:

```python
import torch

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    # warm-up and inference
    torch.cuda.synchronize()
    print("allocated GiB:", torch.cuda.memory_allocated() / 1024**3)
    print("reserved GiB:", torch.cuda.memory_reserved() / 1024**3)
    print("peak GiB:", torch.cuda.max_memory_allocated() / 1024**3)
```

process RSS와 system memory도 별도로 측정한다.

### 22.9 연구 기준선 원칙

```text
공식 Transformers output
  → correctness baseline

최적화 runtime output
  → 속도·메모리 비교

허용 오차·task accuracy
  → quant·kernel 회귀 판정
```

최적화 runtime이 더 빠르더라도 output template·stop token·sampling 설정이 다르면 직접 비교하지 않는다.

---

## 23. ONNX Runtime

ONNX Runtime은 여러 Execution Provider를 통해 Windows·Linux·macOS·모바일·edge에서 모델을 배포한다. 다만 최신 autoregressive LLM 전체를 쉽게 export·최적화할 수 있는 범용 해답으로 간주하면 안 된다.

### 23.1 Execution Provider 선택

| EP | 주요 장치 | 비고 |
| --- | --- | --- |
| CUDA EP | NVIDIA GPU | CUDA·cuDNN 조합 확인 |
| TensorRT EP | NVIDIA GPU | graph partition·engine cache |
| MIGraphX EP | AMD ROCm GPU | 신규 AMD ONNX 경로 검토 |
| OpenVINO EP | Intel CPU·GPU·NPU | AUTO·HETERO·device config |
| DirectML EP | Windows DX12 GPU | sustained engineering 상태 |
| CoreML EP | Apple CPU·GPU·ANE | 지원 op·shape·deployment target |
| XNNPACK EP | CPU·모바일 | 소형 모델·edge |
| QNN EP | Qualcomm | 지원 SoC·SDK 확인 |
| NNAPI EP | Android | Android API·op 지원 |
| Web EP | 브라우저 | WebGPU·WASM 경로 |

### 23.2 provider 우선순위

```python
import onnxruntime as ort

print("available:", ort.get_available_providers())

session = ort.InferenceSession(
    "model.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
print("active order:", session.get_providers())
```

fallback provider를 넣으면 실행 성공률은 높아지지만 숨은 host-device copy가 생길 수 있다.

### 23.3 AMD 변경점

기존 ROCm EP는 ONNX Runtime 1.23부터 제거되었다. AMD 신규 배포는 MIGraphX EP의 지원 op·precision·dynamic shape를 검증한다. 기존 ROCm EP deployment를 무심코 최신 ORT로 업그레이드하지 않는다.

### 23.4 DirectML·WinML

DirectML EP는 Windows DX12 장치에서 넓게 동작하지만 다음 제약을 확인한다.

- session의 sequential execution
- memory pattern option
- 지원 opset과 제외 op
- dynamic shape
- CPU fallback
- adapter driver

신규 Windows 앱은 WinML의 현재 개발 방향과 ONNX Runtime 통합 상태를 검토한다.

### 23.5 CoreML EP

Apple 앱 배포에서 확인할 항목:

- 최소 macOS·iOS deployment target
- MLProgram·NeuralNetwork 형식
- CPU·GPU·ANE compute unit
- static shape·dynamic input
- Core ML cache
- unsupported op fallback

LLM token loop 전체보다 encoder·embedding·vision·audio 모델에서 먼저 적용하기 쉽다.

### 23.6 graph partition 확인

ONNX Runtime profiling:

```python
import json
import onnxruntime as ort

opts = ort.SessionOptions()
opts.enable_profiling = True
sess = ort.InferenceSession(
    "model.onnx",
    sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
# run inputs here
profile_path = sess.end_profiling()
print(profile_path)
```

profile에서 node별 provider와 memcpy를 확인한다.

### 23.7 engine·cache 관리

TensorRT·OpenVINO·CoreML 등 EP는 compile·engine cache를 생성할 수 있다.

```text
cache key
≈ model hash
+ EP version
+ driver/runtime
+ device architecture
+ shape·precision options
```

다른 GPU·driver에서 cache를 재사용하지 않고, writable cache directory와 무결성을 관리한다.

### 23.8 ONNX가 적합한 작업

- embedding·reranker
- vision encoder·OCR submodel
- ASR encoder·VAD
- tabular·time-series
- classifier·detector
- 고정된 image/audio pipeline
- 모바일·Windows 앱 배포

매우 동적인 최신 agentic LLM은 Transformers·vLLM·SGLang·OpenVINO GenAI와 비교한다.

---

## 24. MLC LLM·WebLLM

MLC LLM은 모델을 플랫폼별 library·artifact로 compile해 Python·REST·CLI·iOS·Android·WebGPU에서 실행하는 배포 지향 runtime이다. WebLLM은 브라우저에서 WebGPU를 사용한다.

### 24.1 적합한 경우

- 브라우저에서 서버 전송 없는 로컬 채팅
- Electron·웹 앱
- iOS·Android 앱
- Vulkan·Metal·CUDA를 아우르는 클라이언트 배포
- OpenAI-compatible client API
- model artifact를 앱 release와 함께 관리

### 24.2 compile·배포 계층

```text
원본 모델·quant config
  → MLC model conversion
  → target별 compile
  → .so / .dylib / .dll / .wasm / package
  → tokenizer·model weights·config 배포
```

compile 결과는 target triple, GPU backend, thread·tensor parallel 설정에 종속된다.

### 24.3 WebLLM 개념 예제

```javascript
import * as webllm from "@mlc-ai/web-llm";

const engine = await webllm.CreateMLCEngine(
  "<supported-model-id>",
  {
    initProgressCallback: (report) => console.log(report.text),
  },
);

const reply = await engine.chat.completions.create({
  messages: [
    { role: "user", content: "WebGPU 로컬 추론의 장단점은?" },
  ],
});

console.log(reply.choices[0].message.content);
```

지원 model ID와 API는 설치 버전의 [WebLLM 문서](https://github.com/mlc-ai/web-llm)를 따른다.

### 24.4 브라우저 운영

| 항목 | 점검 |
| --- | --- |
| 최초 다운로드 | model 크기·진행률·취소·resume |
| 저장소 | IndexedDB·Cache Storage quota |
| 메모리 | tab·WASM·WebGPU buffer 합산 |
| compile | shader compile와 warm-up UI |
| worker | UI thread 차단 방지 |
| 지원 | browser·OS·GPU matrix |
| 보안 | model origin·service worker·CSP |
| 개인정보 | prompt가 local에 남는지·telemetry 여부 |

### 24.5 브라우저의 한계

- 대형 모델 다운로드 UX
- 모바일 브라우저 background eviction
- vendor driver·WebGPU 구현 편차
- browser memory limit
- long context와 KV growth
- 확장 프로그램·페이지 script의 데이터 접근
- 서버 runtime보다 제한적인 관측성

### 24.6 native 앱

iOS·Android에서는 앱 bundle 또는 first-run download, model update, 코드 서명, disk quota와 thermal budget을 설계한다. 클라우드 fallback이 있으면 사용자의 입력이 언제 외부로 전송되는지 명시한다.

---

## 25. ExLlamaV3와 특화 런타임

모든 작업을 하나의 범용 runtime으로 해결하려 하지 않는다. 특정 hardware·model format·modality에서 특화 런타임이 큰 이점을 줄 수 있다.

### 25.1 ExLlamaV3·EXL3

ExLlamaV3는 NVIDIA CUDA에서 EXL3 저비트 모델의 빠른 추론을 목표로 한다. 적합한 경우:

- NVIDIA GPU
- 지원되는 decoder-only architecture
- 2–8bit 수준의 세밀한 bitrate 선택
- 단일·다중 GPU 저지연 추론
- ExLlama용 KV cache quant
- 모델 변환·calibration을 직접 관리할 수 있음

주의:

- CUDA·PyTorch·extension build 조합
- 지원 architecture
- EXL2와 EXL3 artifact 구분
- 모델당 calibration·conversion 품질
- VLM·MoE·LoRA 지원 상태
- API server의 production maturity

EXL3 파일을 GGUF·GPTQ와 같은 이름만으로 비교하지 않는다.

### 25.2 CTranslate2

CTranslate2는 Transformer translation·ASR 계열의 효율적인 CPU·CUDA 추론에 적합하다.

대표 사용:

- faster-whisper
- 번역 모델
- INT8·FP16 compute type
- batch transcription

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
)
segments, info = model.transcribe("audio.wav", beam_size=5)
for segment in segments:
    print(segment.start, segment.end, segment.text)
```

CPU에서는 `int8`, GPU에서는 `float16`·`int8_float16` 등을 실제 품질·속도로 비교한다.

### 25.3 `whisper.cpp`

GGML 기반 ASR을 CPU·Metal·CUDA·Vulkan 등에서 간단히 실행할 때 적합하다.

```bash
./build/bin/whisper-cli \
  -m /models/ggml-large-v3-turbo.bin \
  -f audio.wav \
  -l ko
```

정확한 binary 이름과 backend option은 현재 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) release를 확인한다.

### 25.4 `stable-diffusion.cpp`

GGML·GGUF 계열 이미지 생성·편집을 단일 binary로 실행하려는 경우 검토한다. 지원 모델·VAE·ControlNet·LoRA와 backend별 기능 차이가 크므로 [이미지 생성 가이드](../modalities/image-generation.md)와 공식 matrix를 함께 본다.

### 25.5 embedding 전용 runtime

대규모 RAG에서는 생성 LLM과 embedding·reranker를 분리한다.

- ONNX Runtime
- OpenVINO
- TensorRT
- Infinity·TEI 계열 serving
- Sentence Transformers

작은 encoder를 대형 LLM server와 같은 GPU에 억지로 적재하면 KV cache fragmentation과 tail latency가 악화될 수 있다.

### 25.6 선택 원칙

```text
범용성·빠른 시작
  → llama.cpp / Ollama / Transformers

고처리량 LLM server
  → vLLM / SGLang / TensorRT-LLM

Apple
  → MLX

Intel
  → OpenVINO

브라우저·앱
  → MLC / ONNX Runtime / Core ML

특정 modality
  → modality 전용 runtime
```

---

## 26. 비전·이미지·오디오 런타임

텍스트 LLM의 runtime 추천을 멀티모달에 그대로 적용하면 메모리를 과소평가하기 쉽다.

### 26.1 작업별 구성

| 작업 | 주요 구성요소 | 대표 runtime |
| --- | --- | --- |
| VLM·OCR | vision encoder, projector, LLM, image preprocessing | Transformers, vLLM, SGLang, `llama.cpp` mtmd, MLX-VLM, OpenVINO |
| 이미지 생성 | text encoder, DiT·UNet, VAE, ControlNet·LoRA | Diffusers, ComfyUI, stable-diffusion.cpp, TensorRT, OpenVINO |
| ASR | audio decoder, feature extractor, encoder-decoder·transducer | whisper.cpp, CTranslate2, NeMo, Transformers, OpenVINO |
| TTS | text encoder, acoustic model, codec·vocoder | Transformers, vendor repo, ONNX, TensorRT, MLX |
| embedding | encoder, tokenizer, batch buffer | Sentence Transformers, TEI, ONNX, OpenVINO, TensorRT |

### 26.2 VLM

```text
peak memory
≈ LLM weights
+ vision encoder
+ projector
+ decoded image pixels
+ visual token activations
+ text KV cache
+ prefill workspace
```

고해상도 이미지나 여러 페이지를 한 요청에 넣으면 visual token이 급증한다. vision encoder가 CPU에 fallback되는지 확인한다.

자세한 모델·메모리 표는 [비전·OCR 가이드](../modalities/vision-ocr.md)를 참고한다.

### 26.3 이미지 생성

```text
peak VRAM
≈ DiT·UNet
+ text encoder
+ VAE
+ latent·attention activations
+ ControlNet·IP-Adapter·LoRA
+ upscale·face restoration
```

- text encoder를 CPU로 offload할 수 있음
- VAE tiling·slicing으로 peak 감소 가능
- batch·resolution이 activation을 크게 증가
- INT4 DiT가 VAE·text encoder까지 INT4라는 뜻은 아님

자세한 내용은 [이미지 생성 가이드](../modalities/image-generation.md)를 참고한다.

### 26.4 오디오

```text
peak memory
≈ ASR/TTS weights
+ feature·waveform buffer
+ streaming state
+ speaker encoder
+ diarization·VAD
+ codec·vocoder
+ LLM·translation·summary
```

실시간에서는 RTF, chunk size, lookahead와 동시 stream 수를 측정한다. 자세한 내용은 [오디오·음성 가이드](../modalities/audio-speech.md)를 참고한다.

### 26.5 pipeline 분리

```text
parser·decoder worker
  → modality model worker
  → LLM worker
  → postprocess worker
```

구성요소별로 다른 장치에 배치할 수 있다.

| 구성 | 예시 |
| --- | --- |
| CPU | PDF decode, image resize, audio resample, tokenizer |
| 소형 GPU·NPU | OCR detector, embedding, VAD |
| 대형 GPU | LLM·DiT·VLM |
| 별도 process | untrusted parser·codec |

단일 process가 모든 library와 driver를 load하지 않게 분리하면 memory isolation과 보안이 개선된다.

---

## 27. Windows·WSL·Linux·macOS

운영체제는 단순 UI 취향이 아니라 driver·runtime·container·filesystem과 유지보수 선택이다.

### 27.1 비교표

| 항목 | Linux | Windows native | WSL2 | macOS |
| --- | --- | --- | --- | --- |
| NVIDIA 서버 runtime | 가장 폭넓음 | 제한·프로젝트별 차이 | 대체로 가능 | 해당 없음 |
| AMD ROCm | 공식 matrix 중심 | 별도 제한된 matrix | 지원 여부 별도 확인 | 해당 없음 |
| Apple Metal·MLX | 해당 없음 | 해당 없음 | 해당 없음 | 최적 경로 |
| Intel OpenVINO | 지원 | 지원 | 가능하나 device 경로 확인 | CPU 중심 일부 지원 |
| Docker GPU | 성숙 | Docker Desktop·WSL 의존 | Linux container | 제한적·Metal passthrough 아님 |
| GUI 앱 | 추가 구성 | 강점 | Windows GUI와 병행 | 강점 |
| production server | 일반적 | 가능하나 runtime 제한 | 개발·소규모 위주 | Apple 전용 workload |

### 27.2 Linux

장점:

- CUDA·ROCm·oneAPI의 주된 서버 지원
- container·Kubernetes
- NUMA·huge page·cgroup 제어
- profiler·driver tooling
- headless 운영

운영 기준:

```bash
uname -a
cat /etc/os-release
lspci -nnk | grep -A3 -E 'VGA|3D|Display'
lsmod | grep -E 'nvidia|amdgpu|i915|xe'
```

kernel·driver 자동 업데이트로 production 조합이 깨지지 않도록 hold·staging test를 적용한다.

### 27.3 Windows native

적합한 경우:

- 데스크톱 GUI·개인 개발
- CUDA·DirectML·WinML·OpenVINO 앱
- Ollama·`llama.cpp`·ComfyUI
- 광범위한 consumer GPU

주의:

- driver branch와 CUDA wheel 조합
- pagefile와 GPU shared memory 표시 해석
- Defender·백신의 model file scan
- 긴 경로·NTFS 권한
- service account와 firewall
- WDDM display reserve·timeout

### 27.4 WSL2

WSL2는 Windows host driver를 통해 Linux CUDA 등 일부 GPU 경로를 제공한다.

```bash
wsl --status
wsl --version
nvidia-smi
```

권장:

- model cache와 repo를 WSL의 ext4 filesystem에 저장
- `/mnt/c`의 작은 파일 I/O 성능을 별도 측정
- WSL memory·swap limit 설정
- Windows·WSL 양쪽에서 같은 GPU를 동시에 과부하하지 않음
- localhost forwarding·firewall 확인
- suspend·resume 후 GPU context 재검증

`.wslconfig` 예시:

```ini
[wsl2]
memory=48GB
processors=16
swap=8GB
```

host 전체 RAM을 할당하지 않는다.

### 27.5 macOS

- Apple Silicon은 MLX·Metal 우선
- Intel Mac은 CPU·일부 Metal 지원 범위 확인
- container가 Linux GPU를 그대로 제공하지 않음
- model cache·APFS snapshot·Time Machine 용량 고려
- Gatekeeper·code signing·quarantine 확인
- launchd service의 환경변수와 권한 고정

### 27.6 운영체제 선택 결론

```text
NVIDIA·AMD 다중 GPU production
  → Linux

Windows desktop AI 앱
  → Windows native 또는 WSL2 병행

Apple Silicon local AI
  → macOS + MLX/Metal

Intel 혼합 CPU·GPU·NPU 앱
  → OpenVINO가 지원하는 Windows·Linux 중 제품 환경에 맞춤
```

---

## 28. 컨테이너·가상화·Kubernetes

컨테이너는 driver를 포함해 모든 것을 격리하지 않는다. GPU kernel driver는 host에 남고, container는 userspace library와 runtime을 가져온다.

### 28.1 계층

```text
host kernel·GPU driver·firmware
  → container runtime·device plugin
  → CUDA·ROCm·oneAPI userspace
  → PyTorch·runtime
  → model·quant
```

host driver가 container의 userspace 요구를 충족해야 한다.

### 28.2 NVIDIA container

```bash
docker run --rm --gpus all \
  nvidia/cuda:<pinned-runtime-tag> \
  nvidia-smi
```

특정 GPU:

```bash
docker run --rm \
  --gpus '"device=0"' \
  nvidia/cuda:<pinned-runtime-tag> \
  nvidia-smi
```

NVIDIA Container Toolkit 설치와 runtime configuration을 공식 문서대로 수행한다.

### 28.3 AMD container

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  <pinned-rocm-image> \
  rocminfo
```

rootless·SELinux·group mapping에 따라 추가 설정이 필요할 수 있다. 필요 이상의 `/dev`와 privileged mode를 주지 않는다.

### 28.4 Intel device

환경에 따라 render node 또는 accelerator device를 전달한다.

```bash
docker run --rm \
  --device=/dev/dri \
  --group-add render \
  <pinned-openvino-image> \
  python -c 'import openvino as ov; print(ov.Core().available_devices)'
```

정확한 device와 group은 host driver를 확인한다.

### 28.5 shared memory와 IPC

PyTorch·NCCL·RCCL·multiprocessing은 `/dev/shm`을 많이 사용할 수 있다.

```bash
docker run --rm \
  --ipc=host \
  --shm-size=16g \
  ...
```

`--ipc=host`는 격리를 약화할 수 있다. 필요한 경우 명시적 `--shm-size`를 우선하고 보안 모델을 검토한다.

### 28.6 image pinning

피해야 할 태그:

```text
latest
nightly
main
dev
```

권장:

```bash
docker pull <image>:<exact-version>
docker image inspect <image>:<exact-version> \
  --format '{{index .RepoDigests 0}}'
```

SBOM, signature, CVE scan과 base image EOL을 관리한다.

### 28.7 Kubernetes GPU

필요 구성:

- vendor GPU Operator 또는 device plugin
- node label·taint·affinity
- RuntimeClass
- huge page·shared memory
- persistent model cache
- topology manager
- health monitoring과 node drain
- pod disruption budget
- model warm-up·readiness probe

예시 resource:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

AMD·Intel resource 이름은 설치한 device plugin을 따른다.

### 28.8 topology-aware scheduling

TP pod는 GPU 수만 맞추지 말고 같은 NVLink island·NUMA node에 배치한다. scheduler가 GPU topology를 모르거나 여러 pod가 interconnect를 공유하면 성능 편차가 커진다.

### 28.9 가상화·passthrough

- PCIe passthrough
- SR-IOV
- NVIDIA vGPU·MIG
- AMD virtualization 기능
- cloud instance GPU partition

에서 확인할 항목:

```text
실제 framebuffer·HBM
P2P·collective 지원
IOMMU group
NUMA locality
license
live migration 제한
performance counter 접근
```

### 28.10 컨테이너 보안

- read-only root filesystem
- non-root user
- seccomp·AppArmor·SELinux
- 최소 device mount
- model cache read-only 가능 여부
- Hugging Face token을 image에 bake하지 않음
- untrusted parser·remote code를 별도 sandbox
- egress allowlist
- container digest와 model hash 기록

---

## 29. 다중 GPU·다중 socket·interconnect

여러 GPU의 VRAM 합계는 자동으로 하나의 큰 메모리가 되지 않는다.

```text
2 × 24GB
  ≠ 단일 48GB address space
```

모델과 KV를 shard하고 통신하는 runtime 기능이 필요하며, 각 GPU에는 layer shard·workspace·communication buffer가 들어가야 한다.

### 29.1 병렬화 선택

| 방식 | 복제·분할 | 주 용도 | 주요 통신 |
| --- | --- | --- | --- |
| Data Parallel·replica | 모델 전체 복제 | 독립 요청 throughput | 요청 routing, 일부 sync |
| Tensor Parallel | tensor를 GPU 간 분할 | 단일 큰 모델·낮은 batch | layer마다 collective |
| Pipeline Parallel | layer stage 분할 | 초대형 모델 | activation 전송·bubble |
| Expert Parallel | MoE expert 분할 | MoE 모델 | all-to-all |
| Context Parallel | 긴 sequence 분할 | 긴 context | attention 관련 collective |
| Sequential layer split | layer 묶음 분할 | GGUF 다중 GPU | layer 경계 activation 전송 |

### 29.2 언제 replica가 유리한가

모델이 GPU 하나에 들어가고 독립 요청이 많다면:

```text
2 × 24GB GPU
  → 24GB model replica 두 개
```

가 TP=2보다 단순하고 throughput이 높을 수 있다. router가 session affinity와 prefix cache locality를 관리한다.

### 29.3 언제 TP가 필요한가

- 모델 weight가 한 GPU에 들어가지 않음
- 단일 요청 latency가 중요함
- 큰 batch로 collective 비용을 상쇄 가능
- 빠른 NVLink·NVSwitch·XGMI가 있음
- runtime의 해당 model TP kernel이 성숙함

### 29.4 topology 확인

NVIDIA:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi topo -p2p n
```

AMD:

```bash
rocm-smi --showtopo
amd-smi topology
```

CPU·NUMA:

```bash
lscpu -e=CPU,NODE,SOCKET,CORE,CACHE
numactl --hardware
lspci -tv
```

도구 옵션은 설치 버전에서 확인한다.

### 29.5 PCIe 예산

부분 offload나 pipeline에서 layer 경계마다 activation이 이동한다. managed memory가 page migration을 반복하면 model weight까지 PCIe를 건널 수 있다.

```text
필요 전송량
≈ 요청당 이동 bytes
× 요청률
× 왕복 횟수
```

PCIe 표기 대역폭이 아니라 실제 bidirectional throughput과 NUMA 경로를 측정한다.

### 29.6 heterogeneous GPU

서로 다른 GPU를 한 TP group에 묶으면 가장 느린 장치와 가장 작은 메모리에 맞춰질 수 있다.

- architecture·kernel 차이
- VRAM 불균형
- clock·power 차이
- PCIe topology
- driver·precision 지원
- shard 비율

가능하면 동일 GPU를 사용하고, heterogeneous 구성은 sequential split·역할 분리와 비교한다.

### 29.7 다중 socket CPU

모델 page가 socket 0 RAM에 있고 thread가 socket 1에서 읽으면 inter-socket traffic이 발생한다.

```bash
numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-bench -m /models/model.gguf
```

두 socket을 모두 쓰는 경우 interleave와 local allocation을 비교한다.

```bash
numactl --interleave=all \
  ./build/bin/llama-bench -m /models/model.gguf
```

### 29.8 collective smoke test

production 전에 NCCL·RCCL test로 bandwidth·latency를 확인한다. driver와 runtime이 모델 실행은 해도 P2P collective가 비정상일 수 있다.

기록:

```text
message size
algorithm
bus bandwidth
error count
GPU pair
PCIe/NVLink/XGMI path
```

### 29.9 capacity 예제

70B Q4 artifact가 약 42–45GiB이고 runtime·KV·workspace에 10GiB가 필요하다고 가정한다.

| 구성 | 가능성 | 설명 |
| --- | --- | --- |
| 1×48GB | 위험 | weight만 들어가고 KV·workspace 부족 가능 |
| 1×64GB | 가능 | context·batch를 보수적으로 시작 |
| 2×24GB | runtime 의존 | 총 48GB라도 각 shard·buffer가 24GB 안에 들어야 함 |
| 2×48GB | 여유 | TP 또는 replica 선택 가능 |
| 128GB unified memory | 용량 가능 | bandwidth·swap·OS 여유를 별도 확인 |

### 29.10 관련 문서

동시 request, KV cache, TP·DP와 disaggregated serving 계산은 [서빙·동시성 가이드](./serving-concurrency.md)를 참고한다.

---

## 30. 스토리지·모델 캐시·I/O

로컬 AI는 메모리뿐 아니라 모델 download·변환·shard·cache 때문에 상당한 disk를 사용한다.

### 30.1 disk 용량 계획

```text
필요 disk
≈ 원본 checkpoint
+ quantized artifact
+ 변환 임시 파일
+ tokenizer·projector·VAE
+ runtime engine·kernel cache
+ benchmark output·log
+ rollback용 이전 version
```

변환 작업에는 최종 artifact의 2–3배가 일시적으로 필요할 수 있다.

### 30.2 Hugging Face dry-run

```bash
hf download <org/model> --dry-run
```

revision 고정:

```bash
hf download <org/model> \
  --revision <commit-sha> \
  --local-dir /models/<name>-<revision>
```

필요한 pattern만 받기:

```bash
hf download <org/repo> \
  --revision <commit-sha> \
  --include '*Q4_K_M*.gguf' '*.json' 'tokenizer*' \
  --local-dir /models/<name>-q4
```

실제 shard와 projector 이름을 먼저 확인한다.

### 30.3 cache 경로

```bash
export HF_HOME=/data/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/hf-cache/hub
```

runtime별 cache:

- Hugging Face hub
- Transformers modules·dynamic code
- Torch extensions
- Triton kernel cache
- TensorRT engine cache
- OpenVINO compiled cache
- Ollama blobs
- MLX converted models
- browser IndexedDB

cache를 같은 directory에 무분별하게 섞지 않는다.

### 30.4 SSD 선택

| 상황 | 중요한 요소 |
| --- | --- |
| model load | sequential read·mmap latency |
| shard download | network와 write speed |
| quantization | read+write·임시 disk·endurance |
| swap·offload | random I/O·latency·endurance |
| 다중 worker cold start | concurrent read·filesystem cache |
| network storage | latency·metadata·cache consistency |

모델 load 후 대부분 RAM·VRAM에 상주하면 초고속 SSD의 token/s 영향은 작을 수 있다. 그러나 cold start, model switching, CPU mmap, offload에서는 중요하다.

### 30.5 mmap와 page cache

GGUF runtime은 mmap을 사용할 수 있다.

장점:

- 초기 load 단순화
- OS page cache 활용
- 여러 process의 read-only page 공유 가능성

주의:

- 첫 access page fault
- network filesystem 지연
- memory pressure에서 eviction
- `mlock` 사용 시 RAM 고정
- 파일 교체 중 inode·mapping 일관성

cold·warm load를 각각 측정한다.

### 30.6 network filesystem

NFS·SMB·object FUSE에서 대형 shard를 직접 load할 때:

- file lock·atomic rename
- stale handle
- metadata storm
- partial download
- cache coherency
- node별 local cache
- checksum

을 확인한다. production에서는 중앙 object storage에서 node-local NVMe로 검증된 artifact를 stage하는 방식이 안정적이다.

### 30.7 model cache 정리

삭제 전에 reference를 확인한다.

```bash
hf cache ls
hf cache prune --dry-run
hf cache rm <cache-id> --dry-run
```

CLI 이름은 huggingface_hub 버전에 따라 달라질 수 있으므로 `hf cache --help`를 확인한다.

운영 정책 예:

```text
현재 production revision
+ 직전 rollback revision
+ 검증 중 candidate
+ 변환 원본
```

만 유지하고 나머지는 checksum·registry가 있으면 정리한다.

### 30.8 checksum

```bash
sha256sum /models/model.gguf > /models/model.gguf.sha256
sha256sum -c /models/model.gguf.sha256
```

sharded model:

```bash
find /models/model -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > /models/model/MANIFEST.sha256
```

### 30.9 engine·compile cache 무효화

다음이 바뀌면 cache를 다시 생성할 수 있다.

- driver·CUDA·ROCm·OpenVINO
- GPU architecture
- runtime version
- model hash
- max sequence·batch
- precision·quant
- attention backend

잘못된 cache 재사용은 crash·성능 저하·잘못된 output을 일으킬 수 있다.

### 30.10 스토리지 체크리스트

```text
[ ] 다운로드 전에 dry-run
[ ] revision·include pattern 고정
[ ] 원본+변환+임시 용량 합산
[ ] node-local cache와 network storage 역할 분리
[ ] model·engine checksum
[ ] cold·warm load time 측정
[ ] cache retention·rollback 정책
[ ] swap·offload의 SSD endurance 검토
```


---

## 31. 전력·열·소음·지속 부하

짧은 benchmark의 최고 속도보다 30–60분 동안 유지되는 처리량이 실제 사용자 경험과 운영비를 더 잘 설명한다.

### 31.1 측정할 값

| 범주 | 지표 |
| --- | --- |
| 전력 | 장치 전력, 시스템 벽전력, idle·load 차이 |
| 열 | GPU hotspot·memory junction·CPU package·SSD 온도 |
| clock | GPU core·memory clock, CPU frequency |
| throttling | thermal·power·voltage·current limit |
| 성능 | prompt·generation tokens/s, TTFT·TPOT |
| 효율 | tokens/J, requests/kWh, images/kWh, audio-hours/kWh |
| 소음 | fan RPM·dBA·사용 환경 허용치 |
| 안정성 | ECC error·GPU reset·driver error·OOM·clock oscillation |

### 31.2 burst와 sustained

```text
cold start
  → warm-up
  → peak burst
  → thermal equilibrium
  → sustained throughput
```

최소 다음 시점을 기록한다.

- 0–2분
- 5분
- 15분
- 30분
- 60분

노트북·미니 PC·소형 워크스테이션은 5분 이후 속도가 크게 달라질 수 있다.

### 31.3 전력 효율 계산

```text
tokens_per_joule
= generated_tokens / consumed_joules
```

```text
energy_per_1m_tokens_kwh
= average_power_watts
× elapsed_seconds
/ 3,600,000
× 1,000,000
/ generated_tokens
```

벽전력 측정이 가능하면 GPU telemetry만 사용한 값보다 전체 시스템 비교에 유리하다. CPU·fan·SSD·PSU 손실이 포함되기 때문이다.

### 31.4 power limit 실험

최대 전력보다 약간 낮춘 limit에서 효율이 개선될 수 있다. 그러나 명령은 장치·권한·보증 조건을 확인하고 적용한다.

NVIDIA 조회:

```bash
nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1
```

AMD 조회:

```bash
amd-smi monitor -p -t -g -m -u
```

power limit 변경은 production baseline과 별도의 실험으로 취급하고, 기본값 복원 절차를 기록한다.

### 31.5 냉각

- GPU 사이 슬롯 간격과 intake 확보
- 고밀도 multi-GPU의 blower·서버 chassis 요구
- RAM DIMM과 VRM airflow
- NVMe heatsink와 변환 작업 온도
- 먼지·필터·ambient temperature
- 노트북 lid·dock·외부 디스플레이 영향

열을 낮추기 위해 팬을 무조건 최대화하기보다, 허용 소음 안에서 sustained throughput을 비교한다.

### 31.6 전원 공급

확인 항목:

- PSU 정격과 transient 대응
- GPU 보조전원 connector 규격
- multi-GPU rail·cable 배치
- UPS 출력과 런타임
- rack PDU·회로 용량
- 노트북 adapter 정격
- 전원 장애 시 model cache·database 일관성

### 31.7 ECC와 신뢰성

ECC가 제공되는 장치에서는 corrected·uncorrected error를 관측한다. ECC가 없다고 사용할 수 없는 것은 아니지만, 장시간 연구·서비스는 다음을 강화한다.

- output checksum·golden test
- periodic restart·health check
- artifact checksum
- GPU reset 감지
- 데이터·checkpoint 백업
- 열·전력 margin

### 31.8 환경별 권장

| 환경 | 권장 운영 |
| --- | --- |
| 노트북 | 전원 연결, 배터리 보호, 15–30분 sustained test |
| 데스크톱 | 케이스 airflow, display 부하와 inference 분리 |
| Apple Silicon | Memory Pressure·swap·fan·CPU/GPU 경쟁 관찰 |
| multi-GPU | GPU 간 온도 편차와 power cap 균형 |
| 서버 | DCGM·AMD SMI·IPMI·PDU telemetry 통합 |
| edge | thermal mode·active cooling·ambient worst case |

---

## 32. 벤치마크 설계

하드웨어 비교는 모델·quant·prompt·context·runtime 설정이 같아야 한다. 다른 조건의 최고 수치를 모아 순위를 만들지 않는다.

### 32.1 최소 실험 단위

```yaml
model:
  repo: <org/model>
  revision: <commit>
  artifact_sha256: <sha256>
  quant: Q4_K_M
runtime:
  name: llama.cpp
  revision: <commit>
hardware:
  accelerator: <exact model>
  memory_gib: 24
workload:
  prompt_tokens: 512
  output_tokens: 256
  context_limit: 8192
  batch: 1
  concurrency: 1
sampling:
  temperature: 0
  seed: 42
```

### 32.2 cold와 warm

| 구간 | 포함하는 비용 |
| --- | --- |
| Cold load | disk read, mmap page fault, model conversion·engine load |
| First request | kernel compile, graph capture, allocator growth |
| Warm steady state | 실제 반복 요청 처리 |
| Soak | 열·전력·fragmentation·cache 안정성 |

각 값을 따로 보고한다.

### 32.3 LLM 지표

- model load time
- prompt processing tokens/s
- generation tokens/s
- TTFT
- TPOT·ITL
- end-to-end latency
- requests/s
- input·output tokens/s
- peak device memory
- peak process RSS
- power·energy
- quality·exact match·task score

### 32.4 `llama-bench`

```bash
./build/bin/llama-bench \
  -m /models/model.gguf \
  -p 128,512,2048 \
  -n 64,256 \
  -ngl 0,99 \
  -r 5
```

CPU thread sweep:

```bash
for t in 4 8 16 24 32; do
  ./build/bin/llama-bench \
    -m /models/model.gguf \
    -t "$t" -p 512 -n 128 -r 3
 done
```

### 32.5 server benchmark

vLLM·SGLang·TensorRT-LLM은 프로젝트의 현재 benchmark CLI를 사용하고 `--help` 결과를 보존한다.

```bash
vllm bench serve --help
python -m sglang.bench_serving --help
```

부하 형태:

- closed-loop: 각 client가 응답 후 다음 요청
- open-loop: 지정 arrival rate
- burst: 짧은 폭증
- steady: 일정 부하
- soak: 장시간

### 32.6 saturation knee

동시성을 1, 2, 4, 8, 16으로 늘려 다음을 그린다.

```text
x축: offered load 또는 concurrency
y축 1: throughput
y축 2: p95 TTFT
```

throughput 증가가 둔화하고 p95가 급증하는 지점 이전을 운영 상한으로 둔다.

### 32.7 quality benchmark

속도와 별도로 평가한다.

| 작업 | 예시 지표 |
| --- | --- |
| 코드 | compile·unit test pass rate |
| 수학 | exact answer·symbolic verification |
| RAG | retrieval recall·citation correctness |
| SQL | execution accuracy·read-only safety |
| OCR | CER·WER·table structure accuracy |
| 이미지 | human preference·text accuracy·artifact rate |
| ASR | WER·CER·RTF |
| TTS | intelligibility·speaker similarity·MOS |

저비트·다른 kernel이 빠르더라도 품질 gate를 통과해야 한다.

### 32.8 VLM·OCR

고정한다.

- 입력 이미지 hash
- 원본 해상도
- resize·crop 정책
- 페이지 수
- visual token budget
- prompt template
- output schema

측정:

- pages/min
- time-to-first-token
- peak VRAM
- table·formula accuracy
- parser+VLM 전체 latency

### 32.9 이미지 생성

고정:

- seed
- prompt·negative prompt
- width·height
- steps
- scheduler·sampler
- CFG
- VAE
- LoRA·ControlNet

측정:

- seconds/image
- images/min
- peak VRAM
- energy/image
- batch scaling
- output quality

### 32.10 오디오

```text
RTF = 처리 시간 / 오디오 길이
```

- offline RTF
- streaming chunk latency
- endpoint delay
- WER·CER
- 동시 stream 수
- peak RAM·VRAM
- audio decode·resample 포함 여부

### 32.11 결과 표준화

```csv
run_id,timestamp,model_revision,quant,runtime_version,device,prompt_tokens,output_tokens,concurrency,pp_tps,tg_tps,ttft_ms,p95_ms,peak_vram_mib,peak_rss_mib,avg_power_w,quality_score
```

원시 로그와 요약 CSV를 함께 저장한다.

### 32.12 잘못된 비교

- 다른 모델 size·quant를 같은 “7B”로 묶음
- chat template가 다름
- output 길이가 다름
- 한쪽만 prefix cache 사용
- 한쪽은 warm, 다른 쪽은 cold
- CPU offload를 숨김
- display GPU와 headless GPU를 비교
- peak가 아닌 평균 memory만 기록
- 샘플 하나만 실행
- quality를 생략

---

## 33. 모니터링·프로파일링

관측성은 “GPU 사용률” 하나로 끝나지 않는다. compute, memory, I/O, queue, power, temperature와 runtime 내부 metric을 함께 본다.

### 33.1 공통 계층

```text
application
  → runtime scheduler·KV cache
  → framework·kernel
  → driver
  → GPU·CPU·memory·storage·network
```

어느 계층의 metric인지 명시한다.

### 33.2 NVIDIA

기본:

```bash
nvidia-smi
nvidia-smi dmon -s pucvmet
nvidia-smi pmon
```

CSV log:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > nvidia-smi.csv
```

심층:

- DCGM·dcgm-exporter
- Nsight Systems
- Nsight Compute
- NCCL tests
- PyTorch profiler

### 33.3 AMD

```bash
amd-smi list
amd-smi static
amd-smi monitor -g -m -u -t -p
```

심층:

- ROCm profiler·`rocprofv3`
- omniperf·Omnitrace 지원 상태
- RCCL tests
- PyTorch profiler

ROCm SMI와 AMD SMI의 세대·명령 차이를 확인한다. 신규 automation은 AMD SMI를 우선 검토한다.

### 33.4 Apple Silicon

- Activity Monitor Memory·Energy·GPU History
- Instruments Metal System Trace
- `memory_pressure`
- `vm_stat`
- `powermetrics`
- MLX memory API·runtime log

`powermetrics`는 권한이 필요하며 metric 이름은 macOS·SoC에 따라 다를 수 있다.

### 33.5 Intel

GPU·XPU:

```bash
xpu-smi discovery
xpu-smi dump --help
```

CPU:

```bash
perf stat -a -- sleep 10
numastat -m
vmstat 1
iostat -xz 1
```

도구:

- Intel VTune Profiler
- Intel PCM
- OpenVINO benchmark·profiling
- oneAPI Level Zero tools

### 33.6 Linux process·memory

```bash
pidstat -r -u -d -p <pid> 1
cat /proc/<pid>/status
cat /proc/<pid>/smaps_rollup
numastat -p <pid>
```

container:

```bash
docker stats
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
```

### 33.7 storage·network

```bash
iostat -xz 1
pidstat -d -p <pid> 1
sar -n DEV 1
ss -s
```

분산 추론에서는 interface별 throughput, retransmit, RDMA·RoCE counter와 congestion을 본다.

### 33.8 runtime metric

수집할 공통 metric:

| 범주 | metric |
| --- | --- |
| queue | waiting requests, queue time |
| execution | running sequences, batch tokens |
| KV | used blocks, free blocks, cache hit |
| latency | TTFT, TPOT, end-to-end p50/p95/p99 |
| throughput | input·output tokens/s, requests/s |
| failure | OOM, timeout, cancellation, worker restart |
| model | load time, active adapter, revision |

### 33.9 Prometheus label 주의

label에 넣지 말 것:

- raw prompt
- user ID·email
- document name·path
- API key
- full model input URL
- arbitrary tenant string

고 cardinality label은 Prometheus 메모리를 폭증시킬 수 있다.

### 33.10 시간 동기화

다중 node에서 trace를 비교하려면 NTP·PTP 상태를 확인한다. clock drift가 있으면 TTFT·network span·queue 분석이 왜곡된다.

### 33.11 알림 예시

```text
GPU memory > 95% for 5m
GPU temperature > vendor limit margin
queue time p95 > SLO
TTFT p95 > SLO
KV cache free blocks < threshold
worker restart > 0
ECC uncorrected error > 0
swap growth > threshold
model checksum mismatch
```

---

## 34. 하드웨어별 배포 레시피

아래 구성은 출발점이다. 모델별 실제 weight·KV·workspace는 각 domain·modality 가이드에서 다시 계산한다.

### 34.1 전용 GPU 없는 32GB RAM PC

```text
runtime: llama.cpp CPU 또는 Ollama
model: 7–8B Q4/Q5, 14B Q3
context: 4–8K
batch/concurrency: 1
```

```bash
./build/bin/llama-server \
  -hf <org/gguf-repo>:Q4_K_M \
  -ngl 0 \
  -c 4096 \
  -np 1 \
  -t 8 \
  --host 127.0.0.1
```

thread를 물리 core 근처에서 sweep하고 RAM channel을 확인한다.

### 34.2 64–128GB RAM 고대역폭 CPU 서버

```text
runtime: llama.cpp·OpenVINO
model: 32B Q4/Q5, 70B Q3/Q4
context: 4–16K
NUMA: node-local 또는 interleave A/B
```

```bash
numactl --interleave=all \
  ./build/bin/llama-bench \
  -m /models/model.gguf \
  -p 512 -n 128
```

동시에 여러 작은 replica가 하나의 큰 process보다 효율적인지 비교한다.

### 34.3 NVIDIA 8GB

```text
model: 3–8B Q4
runtime: llama.cpp CUDA·Ollama
context: 2–4K
여유: display 포함 1–2GB
```

부분 offload를 사용하고 VLM·이미지 생성은 resolution·batch를 낮춘다.

### 34.4 NVIDIA 12GB

```text
model: 8B Q5/Q8 또는 14B Q4
runtime: llama.cpp CUDA, 소형 Transformers
context: 4–8K
```

14B Q4에서 긴 context·VLM projector를 함께 적재하면 OOM 가능성이 있다.

### 34.5 NVIDIA 16GB

```text
model: 14B Q5, 27B Q3
runtime: llama.cpp, 단일-user vLLM·SGLang 실험
image: SDXL·중형 DiT 최적화
```

server runtime은 graph·KV reserve가 커서 GGUF보다 weight 예산을 더 보수적으로 잡는다.

### 34.6 NVIDIA 24GB

```text
model: 27–32B Q4, 14B Q8
runtime: llama.cpp CUDA·vLLM·SGLang·ExLlamaV3
context: 8K 기준 후 확장
```

vLLM 예:

```bash
vllm serve <org/model> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 4
```

20–22GB weight가 적재되면 KV·workspace가 매우 제한될 수 있다.

### 34.7 NVIDIA 48GB

```text
model: 70B Q4, 32B Q8·BF16 일부
runtime: vLLM·SGLang·TensorRT-LLM 지원 범위·llama.cpp
용도: 고품질 단일 모델, 다중 사용자, VLM·이미지 편집
```

model weight와 KV pool을 명시적으로 나누고 10–15% 안전 여유를 둔다.

### 34.8 NVIDIA 80–96GB HBM

```text
model: 70B Q8/BF16 근접, 120B Q4
runtime: vLLM·SGLang·TensorRT-LLM
용도: FP8, 큰 batch, 고동시성
```

MIG·TP·replica를 SLO 기준으로 비교하고 DCGM·NCCL을 사용한다.

### 34.9 AMD Radeon 16GB

```text
runtime: llama.cpp HIP 또는 Vulkan, Ollama
model: 14B Q5, 27B Q3
```

ROCm 공식 지원 여부를 먼저 확인하고 HIP·Vulkan을 동일 조건으로 비교한다.

### 34.10 AMD Radeon 24GB

```text
runtime: llama.cpp HIP/Vulkan, 지원 범위의 vLLM·SGLang
model: 27–32B Q4
```

Radeon에서 data-center 전용 FP8·attention kernel을 기대하지 않는다. ROCm·PyTorch·runtime 조합을 pin한다.

### 34.11 AMD Instinct 64–192GB HBM

```text
runtime: vLLM·SGLang ROCm·PyTorch ROCm
model: BF16·FP8·W4A16 대형 모델
multi-GPU: RCCL·XGMI topology
```

official container, AMD SMI, RCCL test와 장시간 soak를 기본으로 한다.

### 34.12 Apple Silicon 16GB

```text
runtime: MLX-LM·llama.cpp Metal
model: 7–8B Q4/Q5
context: 4K
```

Memory Pressure와 swap을 보며 IDE·브라우저 사용량을 합산한다.

### 34.13 Apple Silicon 32GB

```text
runtime: MLX-LM·MLX-VLM·llama.cpp
model: 27–32B Q4 또는 14B 고정밀
context: 4–8K
```

VLM visual tokens·projector가 있으면 27–32B보다 작은 모델이 더 실용적일 수 있다.

### 34.14 Apple Silicon 64GB

```text
runtime: MLX-LM·llama.cpp Metal
model: 70B Q4/Q5, 32B Q8
```

10–15GB 이상을 OS·KV·앱에 남기고 30분 sustained test를 한다.

### 34.15 Apple Silicon 128GB

```text
runtime: MLX·llama.cpp
model: 120B Q5, 235B Q3, 70B 고정밀
```

큰 모델이 load되더라도 generation bandwidth와 swap을 확인한다. 여러 작은 worker를 동시에 띄울 경우 weight duplication을 측정한다.

### 34.16 Apple Silicon 192–256GB

```text
runtime: MLX·llama.cpp
model: 235B Q4/Q5, 405B 저비트 실험
용도: 대형 단일 사용자 연구·로컬 데이터 처리
```

초대형 모델은 context·multimodal component까지 합산하고 30–50GB 이상의 system 여유를 둘 수 있다.

### 34.17 Intel Arc 16GB

```text
runtime: OpenVINO GPU·native PyTorch XPU·llama.cpp SYCL/Vulkan
model: 14B Q5, 27B Q3
```

OpenVINO IR INT4, GGUF Vulkan/SYCL, Transformers XPU를 동일 task로 비교한다.

### 34.18 Intel Xeon + NPU·iGPU

```text
CPU: 대형 GGUF·OpenVINO
GPU: embedding·소형 LLM·vision
NPU: 지원되는 소형 encoder·저전력 task
```

AUTO mode를 쓰기 전에 device별 단독 baseline을 만든다.

### 34.19 2×24GB consumer GPU

두 선택을 비교한다.

```text
A. 32B Q4 replica × 2
B. 70B Q4 tensor/layer split
```

독립 요청이 많으면 A가, 단일 큰 모델이 필요하면 B가 적합할 수 있다. PCIe·P2P가 B의 핵심이다.

### 34.20 2×80GB·8×80GB 서버

```text
2×80GB
  → 70B BF16·120B 저비트·TP/replica

8×80GB
  → 120B·235B·405B, TP·PP·EP 연구
```

총 VRAM만 계산하지 말고 layer peak, KV, collective, node interconnect와 failure domain을 설계한다.

---

## 35. 보안·공급망·권한

AI runtime은 GPU driver·native extension·container·remote model code를 실행한다. 일반 Python package보다 공격 표면이 작지 않다.

### 35.1 신뢰 경계

```text
model repository
runtime repository
package registry
container registry
GPU driver·firmware
custom CUDA/HIP/SYCL extension
prompt·uploaded file·URL
tool execution
```

각 계층의 owner·revision·signature·update 정책을 기록한다.

### 35.2 모델 다운로드

- repository owner와 license 확인
- commit SHA 고정
- `hf download --dry-run`
- `safetensors`·GGUF 우선
- pickle·custom binary 검사
- SHA-256 manifest 생성
- gated token 최소 권한
- model card의 remote code 요구 확인

### 35.3 `trust_remote_code`

기본값은 `False`다. 필요한 경우:

1. 정확한 revision 고정
2. source 전체 검토
3. network·secret 없는 container
4. read-only model mount
5. non-root user
6. egress 차단
7. output regression test

를 수행한다.

### 35.4 native extension

CUDA·HIP·Triton·C++ extension은 build 시 임의 코드를 실행할 수 있다.

- source package보다 wheel provenance 확인
- compiler·toolchain 고정
- build log·hash 보존
- shared cache 권한 제한
- multi-user host에서 extension cache 분리
- CI에서 SBOM·vulnerability scan

### 35.5 driver·firmware

GPU driver는 높은 권한의 kernel component다.

- vendor security bulletin 구독
- production과 staging 분리
- 자동 최신화 대신 검증된 patch window
- rollback kernel·driver 준비
- Secure Boot·module signing 확인
- GPU reset·Xid·RAS event 모니터링

### 35.6 container

```text
non-root
read-only rootfs
capability drop
no privileged
최소 device mount
seccomp·AppArmor·SELinux
network allowlist
secret file·runtime injection
image digest pin
```

GPU 접근 때문에 무조건 `--privileged`를 사용하지 않는다.

### 35.7 API server

- `127.0.0.1` 기본 bind
- 외부 공개 시 TLS·authentication
- request size·token·image·audio limit
- queue·rate limit
- tenant별 model·adapter ACL
- prompt·response log redaction
- CORS 제한
- SSRF 방지
- tool·shell worker 분리

### 35.8 multi-tenant cache

prefix·KV·compiled graph·adapter cache가 tenant 사이에 공유되면 정보 누출 가능성을 검토한다.

- tenant namespace·salt
- cache key에 model·adapter·prompt policy 포함
- sensitive tenant는 cache 분리 또는 비활성화
- timing side-channel 회귀 test
- GPU memory zeroing·process isolation 정책

### 35.9 parser·멀티모달

PDF·image·audio decoder는 untrusted input을 처리한다.

- file type·magic number 확인
- pixel·page·duration limit
- decompression bomb 방지
- sandbox
- external URL fetch 금지 또는 allowlist
- temporary file TTL
- OCR text를 system instruction으로 취급하지 않음

### 35.10 브라우저·WebGPU

- model CDN integrity·CSP
- service worker update 정책
- IndexedDB에 prompt·model이 남는지 고지
- extension·same-origin script의 접근
- telemetry opt-out
- local-only 주장 검증

### 35.11 라이선스

다음을 별도로 확인한다.

```text
base model license
quantized artifact license
runtime license
custom kernel license
training data·output policy
commercial use
redistribution
acceptable use
```

“오픈 weight”가 OSI 오픈소스 또는 무제한 상업 사용을 뜻하지 않는다.

### 35.12 보안 체크리스트

```text
[ ] model·runtime·container revision 고정
[ ] artifact checksum
[ ] remote code 기본 거부
[ ] native extension build provenance
[ ] GPU device 최소 권한
[ ] API 인증·rate limit
[ ] parser sandbox
[ ] cache tenant 격리
[ ] secret·PII logging 금지
[ ] security bulletin·patch 절차
```

---

## 36. 문제 해결

### 36.1 GPU를 사용하지 않음

| 확인 | 명령·조치 |
| --- | --- |
| 장치 인식 | `nvidia-smi`, `rocminfo`, `xpu-smi`, `system_profiler` |
| framework | `torch.cuda.is_available()`, `torch.xpu.is_available()` |
| runtime build | CUDA·HIP·Metal·Vulkan·SYCL option 확인 |
| device filter | `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, Vulkan selector |
| model log | offloaded layer·active backend 확인 |
| container | device mount·group·runtime 확인 |

### 36.2 모델 load 중 OOM

순서:

1. 다른 process와 display 사용량 확인
2. 더 작은 quant
3. 일부 layer CPU offload
4. model shard·projector 중복 load 확인
5. runtime graph reserve·memory fraction 낮춤
6. context·KV preallocation 낮춤
7. process 재시작으로 fragmentation 제거

### 36.3 load는 되지만 첫 요청 OOM

원인:

- prefill activation
- attention workspace
- KV cache allocation
- CUDA graph capture
- multimodal encoder
- temporary dequantization

조치:

```text
context ↓
batch·ubatch ↓
max batched tokens ↓
parallel slot ↓
GPU layer ↓
KV dtype ↓
resolution·page count ↓
```

### 36.4 더 낮은 quant가 더 느림

가능 원인:

- 해당 bit-width의 최적 kernel 없음
- unpack·dequant 비용
- CPU ISA·GPU tensor core 미사용
- irregular IQ format
- batch가 작아 launch overhead 지배
- 일부 layer fallback

Q3·Q4·Q5를 같은 모델과 조건으로 측정한다.

### 36.5 CPU thread를 늘리면 느려짐

- physical core 초과
- SMT contention
- memory bandwidth 포화
- NUMA remote access
- OpenMP oversubscription
- tokenizer·runtime thread 중복

thread sweep와 socket pinning을 수행한다.

### 36.6 NVIDIA `no kernel image`·illegal instruction

- compute capability와 build architecture 불일치
- CUDA toolkit이 신형 GPU를 지원하지 않음
- wheel에 해당 SASS·PTX가 없음
- driver가 toolkit 요구를 충족하지 않음

GPU compute capability, driver, wheel·container build 정보를 대조한다.

### 36.7 CUDA driver·runtime mismatch

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

`nvidia-smi`에 표시되는 CUDA 값은 driver가 지원하는 최대 계열을 의미할 수 있으며, 설치된 toolkit·PyTorch build와 같은 값이라고 가정하지 않는다.

### 36.8 AMD `invalid device function`·unsupported gfx

- GPU의 `gfx*` target 확인
- ROCm support matrix 확인
- binary target 확인
- official container와 최소 smoke test
- override를 제거한 상태에서 재현
- HIP 대신 Vulkan 비교

공식 지원 밖 GPU를 production에 넣지 않는다.

### 36.9 Apple에서 swap 급증

- 더 작은 quant·모델
- context·batch·parallel 감소
- 브라우저·IDE·Docker 종료
- VLM image 수·해상도 감소
- MLX와 GGUF 중 peak 비교
- memory leak·cache retention 확인

`Memory Pressure`가 안정되는 구성을 선택한다.

### 36.10 Intel XPU가 CPU fallback

- `torch.xpu.is_available()`
- OpenVINO available devices
- model dtype·op 지원
- driver·oneAPI package
- graph partition profile
- native PyTorch XPU wheel 사용 여부

IPEX 의존 예제를 신규 환경에 그대로 적용하지 않는다.

### 36.11 Vulkan은 보이지만 crash

- 최신 GPU driver
- `vulkaninfo`
- device selector
- descriptor·allocation limit
- model quant 지원
- 다른 Vulkan 앱과 충돌
- 작은 모델 smoke test

vendor-native backend와 비교해 driver 문제인지 runtime 문제인지 분리한다.

### 36.12 다중 GPU가 단일 GPU보다 느림

- model이 한 GPU에 이미 들어감
- batch·concurrency가 너무 작음
- PCIe·NUMA 경로
- P2P 비활성화
- collective algorithm
- heterogeneous GPU
- shard 불균형
- CPU feeder·network 병목

replica와 TP를 직접 비교한다.

### 36.13 첫 실행만 매우 느림

정상 가능 원인:

- model download
- mmap page fault
- shader·Triton compile
- TensorRT engine build
- OpenVINO compile cache
- CUDA graph capture

cold start SLO가 필요하면 cache를 미리 생성하고 readiness 전에 warm-up한다.

### 36.14 시간이 지나며 느려짐

- thermal throttling
- power cap
- swap·memory pressure
- KV·prefix cache 포화
- allocator fragmentation
- queue 증가
- SSD·network congestion
- log volume

시간축으로 power·temperature·clock·queue·memory를 같이 그린다.

### 36.15 출력이 깨짐·무한 반복

- 잘못된 chat template
- tokenizer revision 불일치
- EOS·stop token
- quant 품질
- KV dtype 오류
- unsupported model architecture
- speculative decoder 불일치
- corrupted artifact

공식 Transformers FP16/BF16 output을 correctness baseline으로 비교한다.

### 36.16 JSON·tool call만 실패

- Q2·Q3 quant 민감도
- constrained decoding 지원
- chat template·tool schema 형식
- tokenizer special token
- temperature·sampling
- context truncation

Q4·Q5 또는 더 작은 고정밀 모델과 비교한다.

### 36.17 컨테이너에서 GPU가 안 보임

NVIDIA:

```bash
docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi
```

AMD:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  <rocm-image> rocminfo
```

host에서 먼저 장치가 정상인지 확인한다.

### 36.18 model load가 network storage에서 불안정

- node-local NVMe stage
- checksum 후 atomic rename
- read-only mount
- partial file 제거
- concurrent download lock
- cache warm-up
- NFS timeout·retransmit 확인

### 36.19 service가 응답하지 않음

- model load·warm-up 중인지
- health와 readiness 분리
- port bind·firewall
- reverse proxy timeout
- queue saturation
- dead worker
- GPU reset
- disk full

startup log와 GPU telemetry를 함께 본다.

### 36.20 재현되지 않는 성능 차이

다음을 모두 고정한다.

```text
model commit
artifact hash
runtime commit
container digest
driver
OS kernel
power mode
prompt·output length
context·batch·concurrency
warm-up
sampling seed
ambient·thermal state
```

---

## 37. 재현성 manifest와 기여 형식

이 레포지토리에 benchmark·hardware 결과를 추가할 때는 마케팅 명칭만 쓰지 않고 재현 가능한 manifest를 함께 제출한다.

### 37.1 권장 manifest

```yaml
schema_version: 1
run:
  id: 2026-07-21-host01-qwen-q4-001
  timestamp_utc: "2026-07-21T12:00:00Z"
  operator: anonymous

host:
  os: Ubuntu 24.04
  kernel: <kernel>
  cpu: <exact model>
  sockets: 1
  physical_cores: 16
  ram_gib: 128
  memory_channels: 4
  numa_nodes: 1
  storage: <model and filesystem>

accelerators:
  - vendor: NVIDIA
    model: <exact model>
    architecture: <architecture>
    memory_gib: 24
    driver: <version>
    compute_capability: <major.minor>
    power_limit_w: <value>

software:
  runtime: llama.cpp
  runtime_revision: <git-sha>
  build_backend: cuda
  compiler: <version>
  cuda: <version>
  container_digest: null

model:
  repo: <org/repo>
  revision: <commit-sha>
  file: <filename>
  sha256: <sha256>
  architecture: <architecture>
  parameters: <count>
  quantization: Q4_K_M

settings:
  context: 8192
  kv_dtype: q8_0
  gpu_layers: 99
  batch: 512
  ubatch: 128
  concurrency: 1
  cpu_threads: 8

workload:
  prompt_dataset: ./bench/prompts-v1.jsonl
  prompt_tokens: 512
  output_tokens: 256
  repetitions: 5
  warmup_runs: 2
  seed: 42

results:
  model_load_seconds: <value>
  prompt_tokens_per_second: <value>
  generation_tokens_per_second: <value>
  ttft_ms_p50: <value>
  ttft_ms_p95: <value>
  peak_device_memory_mib: <value>
  peak_rss_mib: <value>
  average_power_w: <value>
  maximum_temperature_c: <value>
  quality_score: <value>

artifacts:
  raw_log: ./results/<id>.log
  metrics_csv: ./results/<id>.csv
  environment: ./results/<id>-env.txt
```

### 37.2 환경 수집

Linux 예:

```bash
{
  date -u --iso-8601=seconds
  uname -a
  cat /etc/os-release
  lscpu
  numactl --hardware 2>/dev/null || true
  nvidia-smi -q 2>/dev/null || true
  amd-smi static 2>/dev/null || true
  rocminfo 2>/dev/null | head -n 200 || true
  python -m pip freeze 2>/dev/null || true
} > environment.txt
```

secret·hostname·serial·user path가 포함되는지 검토한 뒤 공개한다.

### 37.3 benchmark PR 필수 정보

```text
[ ] 정확한 하드웨어 모델
[ ] 물리·실사용 메모리
[ ] driver·runtime·OS
[ ] model repo·revision·hash
[ ] quant·KV dtype
[ ] context·batch·concurrency
[ ] cold·warm 구분
[ ] prompt·output token 수
[ ] peak memory·power·temperature
[ ] raw log
[ ] 품질 또는 correctness 검증
```

### 37.4 결과 상태

| 상태 | 의미 |
| --- | --- |
| `verified` | maintainer 또는 CI가 동일 조건을 재현 |
| `reported` | 기여자가 manifest·log와 함께 제출 |
| `estimated` | 계산식 기반, 실측 아님 |
| `experimental` | nightly·비공식 GPU·override 사용 |
| `deprecated` | runtime·driver·model이 더 이상 권장되지 않음 |

표에서 상태를 숨기지 않는다.

### 37.5 파일 구조 제안

```text
benchmarks/
├── manifests/
│   └── <run-id>.yaml
├── results/
│   ├── <run-id>.csv
│   └── <run-id>.log
├── prompts/
│   └── prompts-v1.jsonl
└── schemas/
    └── benchmark-manifest.schema.json
```

### 37.6 자동 검증

CI에서 확인할 수 있는 항목:

- YAML schema
- model·runtime URL 형식
- SHA-256 길이
- 필수 필드
- 단위 일관성
- impossible memory 수치
- duplicate run ID
- raw log 존재
- Markdown 표 자동 생성

hardware 성능 자체는 CI runner가 재현하기 어렵기 때문에 제출 상태를 구분한다.

### 37.7 문서 갱신 규칙

런타임·하드웨어 정보 변경 시 다음을 함께 갱신한다.

```text
검증일
공식 source URL
지원 시작·종료 version
OS·driver 조건
기존 권장안의 deprecated 표시
migration 경로
```

기존 결과를 삭제하기보다 archive하고 현재 권장안을 명확히 표시한다.

---

## 38. 주요 출처·최종 권장안

### 38.1 주요 런타임·프로젝트 문서

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`llama.cpp` 빌드 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [`llama.cpp` backend feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
- [Ollama 하드웨어 지원](https://docs.ollama.com/gpu)
- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [vLLM 설치 문서](https://docs.vllm.ai/en/latest/getting_started/installation/)
- [SGLang 설치 문서](https://docs.sglang.io/docs/get-started/install)
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
- [OpenVINO GenAI](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html)
- [OpenVINO Model Server](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [MLC LLM](https://llm.mlc.ai/)
- [WebLLM](https://github.com/mlc-ai/web-llm)
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [CTranslate2](https://opennmt.net/CTranslate2/)
- [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp)

### 38.2 하드웨어·플랫폼 문서

- [NVIDIA CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [PyTorch Intel GPU support](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel Extension for PyTorch archive](https://github.com/intel/intel-extension-for-pytorch)
- [OpenVINO supported devices](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-devices.html)
- [Apple Activity Monitor 메모리 해석](https://support.apple.com/guide/activity-monitor/view-memory-usage-actmntr1004/mac)
- [Apple MLX 문서](https://ml-explore.github.io/mlx/build/html/index.html)
- [Linux NUMA 문서](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)

### 38.3 ONNX·Windows 전환 주의

- [ONNX Runtime MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- [ONNX Runtime ROCm EP](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- [ONNX Runtime DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [ONNX Runtime OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

현재 신규 AMD ONNX deployment는 제거된 ROCm EP 대신 MIGraphX를 검토하고, Windows 신규 앱은 DirectML의 sustained-engineering 상태와 WinML 방향을 함께 확인한다.

### 38.4 관련 레포지토리 가이드

- [양자화](./quantization.md)
- [파인튜닝 메모리](./fine-tuning-memory.md)
- [서빙·동시성](./serving-concurrency.md)
- [생산성·문서·RAG](../domains/productivity-rag.md)
- [데이터 분석](../domains/data-analysis.md)
- [비전·OCR](../modalities/vision-ocr.md)
- [이미지 생성](../modalities/image-generation.md)
- [오디오·음성](../modalities/audio-speech.md)

### 38.5 최종 선택 규칙

```text
1. 공식 지원 matrix에서 hardware·OS·driver 교집합 확인
2. 모델 architecture·quant를 지원하는 runtime 선택
3. physical memory가 아니라 실사용 가능 memory 계산
4. weights + KV + workspace + modality component 합산
5. vendor-native backend를 첫 기준선으로 측정
6. portability backend를 동일 조건으로 비교
7. cold·warm·soak benchmark와 품질 gate 수행
8. topology·전력·열·swap을 함께 관측
9. model·runtime·container·driver revision 고정
10. manifest·raw log·checksum과 함께 결과 공개
```

### 38.6 장비 구매 전 질문

```text
[ ] 반드시 실행해야 하는 모델 규모와 quant는?
[ ] 목표 context와 동시 요청 수는?
[ ] 텍스트만인가, VLM·이미지·오디오도 필요한가?
[ ] 단일 큰 메모리인가, 여러 GPU로 shard할 것인가?
[ ] vendor-native runtime이 공식 지원하는가?
[ ] 메모리 대역폭과 interconnect는 충분한가?
[ ] Linux·Windows·macOS 중 유지보수 가능한 환경은?
[ ] 24시간 전력·열·소음·냉각 비용은?
[ ] 상업 이용·모델 라이선스 조건은?
[ ] driver·runtime 업데이트와 rollback을 누가 관리하는가?
```

### 38.7 실무 기본값

| 상황 | 기본값 |
| --- | --- |
| 처음 실행 | `llama.cpp` GGUF Q4_K_M 또는 vendor-native 간단 runtime |
| Apple Silicon | MLX 4-bit와 `llama.cpp` Q4를 A/B |
| NVIDIA 서버 | vLLM·SGLang, 지원 장비는 TensorRT-LLM 비교 |
| AMD 서버 | 공식 ROCm 조합의 vLLM·SGLang, GGUF는 HIP 비교 |
| Intel | OpenVINO와 native PyTorch XPU, CPU GGUF 기준선 |
| 범용 Windows GPU | CUDA·Vulkan·WinML/DirectML 중 공식 지원 경로 |
| 브라우저 | WebLLM·WebGPU, 작은 모델과 명확한 다운로드 UX |
| 긴 context | 모델보다 KV·동시성 예산부터 줄임 |
| 메모리 부족 | 작은 모델 Q4를 극저비트 큰 모델과 직접 비교 |
| production | stable release·digest·revision·quality gate |

### 38.8 문서 갱신 주의

하드웨어 지원과 기본 wheel·container의 CUDA·ROCm 버전은 수개월 안에도 바뀔 수 있다. 특히 다음은 설치 직전에 다시 확인한다.

- 신규 GPU architecture의 최소 driver·toolkit
- vLLM·SGLang stable build matrix
- ROCm GPU·OS support
- PyTorch XPU·MPS·ROCm 지원
- TensorRT-LLM official hardware matrix
- OpenVINO device·model support
- ONNX Runtime EP deprecation
- `llama.cpp` backend·quant feature matrix
- Ollama experimental Vulkan·GPU support
- 모델별 license와 conversion artifact

> **결론:** 가장 좋은 로컬 AI 하드웨어는 가장 큰 VRAM을 가진 장치가 아니라, 목표 모델·정밀도·context·동시성을 공식 지원하는 런타임에서 충분한 메모리 대역폭과 안정적인 운영 여유를 제공하는 장치다. 구매 전에는 예상 용량표를 사용하고, 배포 전에는 반드시 실제 workload로 cold·warm·soak benchmark와 품질 검증을 수행한다.

**문서 종료**
