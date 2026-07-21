# RAM-for-Local-AI

> 보유한 **시스템 RAM, GPU VRAM, Apple Silicon 통합 메모리**를 기준으로 로컬 AI 모델·양자화·런타임·운영 구성을 선택하기 위한 한국어 가이드와 정적 웹 계산기

`RAM-for-Local-AI`는 단순한 모델 순위표가 아니다. 이 레포지토리는 다음 질문에 답하는 것을 목표로 한다.

- 내 장비에서 어떤 모델과 양자화를 실제로 실행할 수 있는가?
- 모델 파일이 메모리에 들어간 뒤 KV 캐시와 작업 버퍼를 얼마나 남길 수 있는가?
- RAG, 데이터 분석, 비전·OCR, 이미지 생성, 음성 파이프라인에서는 어떤 추가 메모리가 필요한가?
- 여러 사용자를 동시에 처리하거나 LoRA·QLoRA 파인튜닝을 수행하려면 용량을 어떻게 계산해야 하는가?
- NVIDIA·AMD·Apple·Intel·CPU 환경에서 어떤 런타임과 형식을 우선 검토해야 하는가?

현재 레포지토리는 **11개 가이드**와 브라우저에서 정적으로 실행되는 **로컬 AI 메모리 계산기**로 구성되어 있다. 각 가이드는 RAM·VRAM·통합 메모리 구간별 추천, Q2·Q3·Q4 및 다른 저정밀 형식, Hugging Face 링크, 실행 예제, 평가·보안·재현성 항목을 함께 다룬다.

> **문서 스냅샷:** 2026-07-20~2026-07-21 KST에 검증된 모델·런타임 정보를 기준으로 한다. 모델 파일, 라이선스, 런타임 API와 하드웨어 지원표는 변경될 수 있으므로 실제 다운로드와 배포 직전에 공식 저장소를 다시 확인해야 한다.

[바로 시작](#바로-시작하기) · [메모리 계산기](#메모리-계산기) · [가이드 목록](#가이드-목록) · [계산 원칙](#이-레포지토리의-계산-원칙) · [GitHub Pages](#github-pages-배포)

---

## 바로 시작하기

### 1. 웹 계산기로 먼저 범위를 좁히기

정적 계산기 소스는 [`tools/memory-calculator/`](./tools/memory-calculator/)에 있다.

```text
https://jtech-co.github.io/RAM-for-Local-AI/tools/memory-calculator/
```

로컬에서는 레포지토리 루트에서 간단한 정적 서버를 실행할 수 있다.

```bash
python -m http.server 8000
```

그다음 주소를 연다.

```text
http://localhost:8000/tools/memory-calculator/
```

계산기는 다음 네 가지 운용 형태를 지원한다.

| 운용 형태 | 계산에 포함하는 주요 항목 |
|---|---|
| 추론 | 모델 가중치, KV 캐시, 런타임 workspace, 멀티모달 버퍼, 운영 여유 |
| RAG | 생성 모델, 임베딩, reranker, vector index, 문서 캐시, parser·OCR worker |
| 동시 사용자 서빙 | replica, active sequence, KV pool, prefix cache, prefill·decode workspace |
| 파인튜닝 | 가중치, gradient, optimizer state, activation, adapter, checkpoint staging |

현재 계산기 카탈로그에는 다음 구성이 포함되어 있다.

- 7개 분야
- 24개 세부 작업
- 66개 모델 프리셋
- Q2·Q3·Q4·Q5·Q6·Q8, FP16·BF16·FP8, NF4·AWQ 등 다중 정밀도
- 전용 GPU, Apple·APU 통합 메모리, CPU 전용, 다중 GPU 배치
- RAG, 데이터 분석, 비전·OCR, 이미지 생성, 오디오 파이프라인별 추가 입력

### 2. 장비와 작업에 맞는 상세 가이드 열기

계산기 결과는 초기 용량 계획용 근사치다. 최종 모델 선택, 정확한 파일명, 런타임 옵션, 품질 검증 방법은 아래 분야·모달리티·운영 가이드에서 확인한다.

### 3. 다운로드 직전에 실제 파일 크기 확인하기

Hugging Face CLI를 사용할 수 있다면 전체 shard와 예상 다운로드 크기를 먼저 확인한다.

```bash
hf download <organization>/<repository> --dry-run
```

GGUF, MLX, AWQ, GPTQ, safetensors는 같은 비트 수로 표시되더라도 실제 크기와 실행 특성이 다를 수 있다. 파일명이나 양자화 태그를 추정하지 말고 모델 카드와 저장소 파일 목록을 확인한다.

---

## 메모리 계산기

### 주요 기능

- 시스템 RAM, GPU VRAM, Apple 통합 메모리 수동 입력
- 브라우저가 제공하는 범위에서 RAM·GPU 정보 자동 탐지
- 모델·양자화별 resident weight 추정 또는 실제 파일 크기 직접 입력
- context length, output tokens, KV cache dtype와 동시 요청 반영
- RAG index, embedding, reranker, 문서 cache와 parser worker 반영
- 데이터 working set, DataFrame expansion과 notebook kernel 반영
- 이미지 해상도, batch, ControlNet, LoRA, upscaler 반영
- 오디오 길이, 동시 stream, diarization, ASR→LLM→TTS cascade 반영
- LoRA·QLoRA·full fine-tuning 방식별 학습 메모리 추정
- Hugging Face와 관련 가이드 링크 제공
- 계산 결과 복사와 JSON 저장
- 장식적 글로우·깜빡임 없이 가독성을 우선한 반응형 CLI 스타일 UI

### 자동 탐지의 한계

일반 웹 브라우저는 정확한 총 RAM과 VRAM을 항상 제공하지 않는다.

- `navigator.deviceMemory`는 지원되는 브라우저에서도 반올림된 근삿값일 수 있다.
- 표준 WebGPU·WebGL API만으로는 전용 VRAM 총량을 신뢰성 있게 읽기 어렵다.
- 계산기는 가능한 경우 GPU 이름과 알려진 제품 사양을 대조하지만, 같은 제품명에 8GB·16GB처럼 여러 변형이 있을 수 있다.
- Apple Silicon은 CPU와 GPU가 같은 통합 메모리를 사용하므로 전용 VRAM처럼 해석하면 안 된다.
- 브라우저 개인정보 보호 설정에 따라 GPU 이름 자체가 숨겨질 수 있다.

따라서 **자동 탐지 결과는 후보값**이며, 운영체제 도구에서 확인한 수동 입력값을 최종 기준으로 사용한다.

### 개인정보와 네트워크

계산기는 프레임워크, 서버 API, CDN, 외부 폰트와 텔레메트리를 사용하지 않는 정적 HTML·CSS·JavaScript 구성이다.

- 계산과 장치 탐지는 브라우저 안에서 수행한다.
- 입력값은 외부 서버로 전송하지 않는다.
- 사용자 설정은 브라우저의 로컬 저장소에만 저장될 수 있다.
- Hugging Face와 문서 링크는 사용자가 직접 클릭할 때만 열린다.

정적 실행에 필요한 주요 파일은 다음과 같다.

```text
tools/memory-calculator/
├── index.html
├── styles.css
├── catalog.js
└── app.js
```

---

## 가이드 목록

### 분야별 가이드

| 분야 | 주요 범위 | 가이드 | 검증일 |
|---|---|---|---:|
| 버그바운티·사이버보안 | 승인된 버그바운티, CTF·교육, CVE·위협 인텔리전스, 코드 감사, 보안 자동화 | [cybersecurity.md](./guides/domains/cybersecurity.md) | 2026-07-20 |
| 프로그래밍·수학·과학 | 범용 코딩, 저장소 에이전트, 수학·과학 추론, 논문 분석, Lean 4 형식증명 | [programming-stem.md](./guides/domains/programming-stem.md) | 2026-07-20 |
| 생산성·문서·RAG | 문서 요약·질의응답, 번역, 임베딩, reranker, 개인·조직 지식베이스 | [productivity-rag.md](./guides/domains/productivity-rag.md) | 2026-07-21 |
| 데이터 분석·BI·SQL | Text-to-SQL, EDA, Python·R, 표형 모델, 시계열, 통계·ML 파이프라인 | [data-analysis.md](./guides/domains/data-analysis.md) | 2026-07-21 |

### 모달리티별 가이드

| 모달리티 | 주요 범위 | 가이드 | 검증일 |
|---|---|---|---:|
| 비전·OCR·문서 이해 | PDF·문서 OCR, 표·수식·차트, screenshot·UI 분석, VLM projector | [vision-ocr.md](./guides/modalities/vision-ocr.md) | 2026-07-21 |
| 이미지 생성·편집 | Diffusion·DiT, SD·FLUX 계열, ControlNet, LoRA, VAE, upscaler, ComfyUI | [image-generation.md](./guides/modalities/image-generation.md) | 2026-07-21 |
| 오디오·음성 | ASR, streaming transcription, diarization, TTS, voice cloning, 오디오 이해 | [audio-speech.md](./guides/modalities/audio-speech.md) | 2026-07-21 |

### 공통 운영 가이드

| 운영 주제 | 주요 범위 | 가이드 | 검증일 |
|---|---|---|---:|
| 양자화 | GGUF Q2~Q8, IQ, AWQ, GPTQ, NF4, FP8·FP4, MLX, KV cache quantization | [quantization.md](./guides/operations/quantization.md) | 2026-07-21 |
| 파인튜닝 메모리 | Full FT, LoRA, QLoRA, DoRA, SFT·DPO·GRPO, FSDP2·ZeRO | [fine-tuning-memory.md](./guides/operations/fine-tuning-memory.md) | 2026-07-21 |
| 서빙·동시성 | KV pool, continuous batching, queue, SLO, prefix cache, 다중 GPU·tenant | [serving-concurrency.md](./guides/operations/serving-concurrency.md) | 2026-07-21 |
| 런타임·하드웨어 | CPU, NVIDIA CUDA, AMD ROCm, Apple MLX·Metal, Intel, Vulkan·WebGPU | [runtime-hardware.md](./guides/operations/runtime-hardware.md) | 2026-07-21 |

루트의 날짜 포함 두 파일은 초기 공개 스냅샷과 기존 링크 호환을 위해 남겨 둔 복제본이다. 새 링크에서는 `guides/domains/cybersecurity.md`와 `guides/domains/programming-stem.md`를 기준 경로로 사용한다.

---

## 이 레포지토리의 계산 원칙

### 추론

```text
추론 peak device memory
≈ resident model weights
+ KV cache
+ activation·attention·quantization workspace
+ graph·allocator reserve
+ vision·audio·VAE·projector 등 추가 구성요소
+ 운영 안전 여유
```

### RAG

```text
RAG 전체 메모리
≈ 생성 모델과 KV cache
+ embedding model
+ reranker
+ vector index
+ 원문·metadata cache
+ parser·OCR worker
+ query engine와 운영 여유
```

### 온라인 서빙

```text
온라인 서빙 메모리
≈ model weights × replica 수
+ active sequence별 KV cache
+ prefix cache
+ prefill·decode workspace
+ queue·tokenizer·multimodal buffer
+ 통신·관측성·운영 여유
```

### 파인튜닝

```text
학습 peak accelerator memory
≈ model weights
+ trainable parameter copy
+ gradients
+ optimizer states
+ saved activations
+ logits·temporary workspace
+ communication buffer
+ checkpoint·allocator peak
```

모델 파일 크기는 위 식의 한 항목일 뿐이다. “다운로드 파일이 10GB이므로 12GB VRAM에서 안전하다”는 식으로 판단하면 첫 긴 prompt, 멀티모달 입력, 동시 요청 또는 checkpoint 저장 시 OOM이 발생할 수 있다.

---

## 메모리 종류를 읽는 방법

| 환경 | 해석 기준 |
|---|---|
| NVIDIA·AMD 전용 GPU | VRAM에 가중치, KV cache와 주요 workspace가 모두 들어갈 때 가장 빠르다. 시스템 RAM offload는 용량을 늘리지만 PCIe 전송 비용이 생긴다. |
| Apple Silicon | CPU·GPU·OS가 통합 메모리를 공유한다. 장착 용량 전체를 모델에 배정하지 말고 memory pressure와 swap을 함께 본다. |
| CPU 전용 | RAM 용량 외에 메모리 채널, 대역폭, NUMA, ISA와 quantized kernel 지원이 처리량을 좌우한다. |
| 다중 GPU | VRAM 단순 합산만으로 판단하지 않는다. tensor·pipeline·expert parallel 지원과 interconnect를 확인한다. |
| APU·공유 메모리 | BIOS에서 확보한 GPU 메모리 표시보다 실제 공유 메모리 대역폭과 운영체제 정책이 중요할 수 있다. |

### 장착 메모리와 실사용 가능 메모리

가이드의 메모리 구간은 장착 용량을 빠르게 분류하기 위한 출발점이다. 실제 계획에서는 다음을 먼저 제외한다.

- 운영체제와 백그라운드 서비스
- 디스플레이와 브라우저·IDE
- 드라이버·런타임 context
- 파일 cache와 memory-mapped page
- allocator fragmentation
- 장시간 운용을 위한 안전 여유

개인 워크스테이션에서는 처음부터 장착 메모리의 100%를 모델에 할당하지 않는다. 계산기 기본 예약률을 출발점으로 사용하고 실제 peak를 측정해 조정한다.

---

## 양자화 빠른 기준

같은 “4비트”라도 `Q4_K_M`, `IQ4_XS`, AWQ W4A16, GPTQ 4-bit, NF4, MLX 4-bit, NVFP4는 같은 형식이 아니다. 런타임과 하드웨어가 실제로 최적화한 형식을 선택해야 한다.

| 정밀도·형식 | 일반적인 위치 | 우선 검토하는 상황 |
|---|---|---|
| Q2·IQ2 | 극단적인 용량 절감 | 기능 확인, 초대형 모델 실험, 결과를 별도 검증하는 제한된 작업 |
| Q3·IQ3 | Q4가 들어가지 않는 경우 | 한 단계 큰 모델을 시험하되 코드·수학·JSON·도구 호출 품질을 직접 평가할 때 |
| Q4 | 범용 기본값 | CPU·GPU·Apple 환경의 일상 추론과 단일 사용자 작업 |
| Q5·Q6 | 품질 우선 | 수학·과학, 정밀 코드 수정, 장문 비교, 메모리 여유가 있는 환경 |
| Q8·INT8 | 고품질 양자화 기준선 | Q4와의 품질 차이를 측정하거나 충분한 메모리가 있을 때 |
| FP16·BF16 | 원본에 가까운 기준선 | 품질 평가, 학습, 고정밀 구성, 서버급 메모리 |
| FP8·FP4 | 지원 하드웨어 특화 | 해당 GPU와 런타임에 native·optimized kernel이 확인된 서버 배포 |
| NF4 | QLoRA | frozen 4-bit base에 adapter를 학습하는 파인튜닝 |
| AWQ·GPTQ·compressed-tensors | CUDA 중심 W4A16·W8A8 | vLLM·SGLang·TensorRT 계열의 지원 matrix가 맞는 서버 추론 |

특별한 이유가 없다면 다음 순서로 시작한다.

```text
Q4 또는 하드웨어가 잘 지원하는 4-bit 형식
→ context 4K~8K, 동시 요청 1개
→ 실제 peak RAM·VRAM과 품질 측정
→ Q5/Q6 또는 더 큰 모델 검토
→ 긴 context와 동시성 단계적 증가
```

메모리에 간신히 들어가는 대형 Q2 모델보다 운영 여유를 남긴 한 단계 작은 Q4·Q5 모델이 실제 작업에서 더 안정적인 경우가 많다.

---

## 작업별 시작 경로

| 목표 | 먼저 열 문서 | 다음에 확인할 문서 |
|---|---|---|
| 개인 PC에서 코딩 모델 실행 | [프로그래밍·STEM](./guides/domains/programming-stem.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 로컬 문서 검색·질의응답 | [생산성·RAG](./guides/domains/productivity-rag.md) | [비전·OCR](./guides/modalities/vision-ocr.md), [서빙·동시성](./guides/operations/serving-concurrency.md) |
| SQL·CSV·Parquet 분석 | [데이터 분석](./guides/domains/data-analysis.md) | [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| PDF·표·수식 OCR | [비전·OCR](./guides/modalities/vision-ocr.md) | [생산성·RAG](./guides/domains/productivity-rag.md) |
| 로컬 이미지 생성 | [이미지 생성](./guides/modalities/image-generation.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 회의 녹취·TTS·음성 에이전트 | [오디오·음성](./guides/modalities/audio-speech.md) | [서빙·동시성](./guides/operations/serving-concurrency.md) |
| LoRA·QLoRA 학습 | [파인튜닝 메모리](./guides/operations/fine-tuning-memory.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 여러 사용자에게 API 제공 | [서빙·동시성](./guides/operations/serving-concurrency.md) | [런타임·하드웨어](./guides/operations/runtime-hardware.md), [양자화](./guides/operations/quantization.md) |
| 승인된 보안 연구·코드 감사 | [사이버보안](./guides/domains/cybersecurity.md) | [서빙·동시성](./guides/operations/serving-concurrency.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |

---

## 레포지토리 구조

핵심 공개 구조는 다음과 같다.

```text
RAM-for-Local-AI/
├── README.md
├── guides/
│   ├── domains/
│   │   ├── cybersecurity.md
│   │   ├── programming-stem.md
│   │   ├── productivity-rag.md
│   │   └── data-analysis.md
│   ├── modalities/
│   │   ├── vision-ocr.md
│   │   ├── image-generation.md
│   │   └── audio-speech.md
│   └── operations/
│       ├── quantization.md
│       ├── fine-tuning-memory.md
│       ├── serving-concurrency.md
│       └── runtime-hardware.md
└── tools/
    └── memory-calculator/
        ├── index.html
        ├── styles.css
        ├── catalog.js
        ├── app.js
        └── ...  # 정적 데이터·계산 보조 모듈
```

`guides/` 아래의 날짜 없는 파일을 최신 기준 경로로 사용한다. 루트의 날짜 포함 파일은 초기 스냅샷 보존과 기존 외부 링크 호환을 위한 것이다.

---

## GitHub Pages 배포

계산기는 별도 빌드 없이 GitHub Pages에서 실행할 수 있다.

1. GitHub 레포지토리의 **Settings → Pages**를 연다.
2. **Build and deployment**에서 `Deploy from a branch`를 선택한다.
3. branch를 `main`, folder를 `/ (root)`로 지정한다.
4. 배포가 끝나면 다음 경로를 연다.

```text
https://<GitHub-ID>.github.io/RAM-for-Local-AI/tools/memory-calculator/
```

자동 장치 탐지는 secure context에서 더 많은 정보를 제공할 수 있으므로 `file://` 직접 실행보다 GitHub Pages 또는 로컬 HTTP 서버를 권장한다.

---

## 업데이트와 기여

모델 또는 하드웨어 결과를 추가할 때 가능한 한 다음 정보를 포함한다.

1. 모델명, 개발 조직과 공식 모델 카드
2. 원본 가중치와 실제 배포 저장소 링크
3. 정확한 파일명, 양자화 태그, shard 수와 총 다운로드 크기
4. dense·MoE 구조, 전체·활성 파라미터, context 한도
5. 하드웨어 모델, 장착 RAM·VRAM·통합 메모리
6. 운영체제, driver, runtime와 버전
7. GPU offload, KV dtype, context, batch, 동시 요청 수
8. 실제 peak RAM·VRAM과 처리 속도
9. 대표 작업셋의 품질 결과와 실패 사례
10. 모델·양자화 라이선스와 검증일

“실행됨”만 기록하지 말고, 어떤 설정에서 어느 정도의 여유와 속도로 실행됐는지 재현 가능하게 남긴다.

### 권장 benchmark 항목

```text
model repository + revision
quantization + exact filename
runtime + commit/version
hardware + driver + OS
context + input/output length
KV cache dtype
batch + concurrency
prompt processing tokens/s
output tokens/s
TTFT + TPOT
peak RAM + VRAM
power + temperature
quality evaluation result
```

### 카탈로그 유지보수

계산기 모델 데이터는 [`tools/memory-calculator/catalog.js`](./tools/memory-calculator/catalog.js)에 있다. 가이드에서 모델 링크, 파일 크기, 최소 메모리 또는 권장 런타임을 수정할 때 계산기 카탈로그도 함께 검토해야 한다.

향후 구조화 데이터와 문서 생성 스크립트를 추가할 경우 다음 흐름을 권장한다.

```text
data/models.yaml 또는 data/models.json
        ↓
검증·생성 스크립트
        ↓
guides/*.md + tools/memory-calculator/catalog.js
        ↓
GitHub Actions 링크·schema·중복·검증일 검사
```

---

## 안전·정확성·라이선스

- 사이버보안 가이드와 모델은 소유하거나 명시적으로 허가받은 시스템, 승인된 버그바운티 범위, CTF, 격리된 연구 환경에서만 사용한다.
- 로컬 실행은 외부 전송을 줄일 수 있지만 prompt, log, vector DB, cache, swap, 임시 이미지·오디오와 생성 파일에 민감정보가 남을 수 있다.
- 모델이 생성한 코드, SQL, 수식, 과학적 주장, 인용, 보안 판단과 고위험 도메인 결과는 독립적인 도구와 신뢰할 수 있는 출처로 검증한다.
- `trust_remote_code`, custom node, adapter, pickle, 임의 container와 비공식 quantization artifact는 공급망 위험을 검토한다.
- 이 레포지토리는 모델 가중치를 재배포하지 않는다. 각 모델, quantization, dataset와 runtime의 라이선스·사용 제한은 해당 공식 저장소를 따른다.
- 파일 크기와 메모리 표는 선택 보조 자료이며 특정 하드웨어의 속도, 품질, 안정성 또는 상업적 이용 가능성을 보장하지 않는다.

---

## 문서 상태

| 구분 | 수량 | 현재 상태 |
|---|---:|---|
| 분야별 가이드 | 4 | 작성 완료 |
| 모달리티별 가이드 | 3 | 작성 완료 |
| 공통 운영 가이드 | 4 | 작성 완료 |
| 정적 웹 계산기 | 1 | 작성 완료 |
| 계산기 모델 프리셋 | 66 | 2026-07-21 카탈로그 기준 |

모델 생태계와 런타임은 빠르게 변한다. 표의 수치는 영구적인 사양이 아니라 **검증일 당시 공개된 모델·파일·문서를 기준으로 한 용량 계획용 스냅샷**이다.
