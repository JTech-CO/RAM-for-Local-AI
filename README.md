# RAM-for-Local-AI - README

> 보유한 **시스템 RAM, GPU VRAM, Apple Silicon 통합 메모리**를 기준으로 로컬 AI 모델·양자화·런타임을 선택하고, 추론·RAG·서빙·파인튜닝에 필요한 메모리를 계획하기 위한 한국어 가이드

**[웹 메모리 계산기 열기](https://jtech-co.github.io/RAM-for-Local-AI/tools/memory-calculator/)** · [가이드 목록](#가이드-목록) · [작업별 시작 경로](#작업별-시작-경로) · [계산 원칙](#메모리-계산-원칙)

---

## 개요

`RAM-for-Local-AI`는 모델 순위만 나열하는 레포지토리가 아니다. 다음과 같은 실제 선택 문제를 다룬다.

- 내 장비에서 어떤 모델과 양자화를 **안정적으로 실행할 수 있는가?**
- 모델 가중치를 적재한 뒤 KV 캐시, 작업 버퍼와 운영 여유를 얼마나 남길 수 있는가?
- RAG, 데이터 분석, 비전·OCR, 이미지 생성과 음성 처리에서는 어떤 추가 메모리가 필요한가?
- 여러 사용자를 처리하거나 LoRA·QLoRA 파인튜닝을 수행할 때 메모리 요구량이 어떻게 달라지는가?
- NVIDIA·AMD·Apple·Intel·CPU 환경에서 어떤 런타임과 배포 형식을 우선 검토해야 하는가?

현재 레포지토리는 **11개 가이드**와 브라우저에서 정적으로 실행되는 **로컬 AI 메모리 계산기**로 구성되어 있다.

| 구성 | 현재 범위 |
|---|---|
| 분야별 가이드 | 사이버보안, 프로그래밍·수학·과학, 생산성·RAG, 데이터 분석 |
| 모달리티별 가이드 | 비전·OCR, 이미지 생성·편집, 오디오·음성 |
| 운영 가이드 | 양자화, 파인튜닝 메모리, 서빙·동시성, 런타임·하드웨어 |
| 계산기 | 7개 분야, 24개 세부 작업, 66개 모델 프리셋 |
| 운용 형태 | 단일 사용자 추론, RAG, API 서빙, 파인튜닝 |
| 실행 방식 | 서버 API·빌드 과정·텔레메트리가 없는 정적 HTML·CSS·JavaScript |

> **콘텐츠 기준일:** 각 문서는 2026-07-20~2026-07-21 KST에 공개된 모델·런타임·하드웨어 정보를 기준으로 검증했다. 모델 파일, 양자화 저장소, 라이선스, 런타임 API와 하드웨어 지원 상태는 변경될 수 있으므로 실제 다운로드와 배포 직전에 공식 저장소를 다시 확인해야 한다.

### 이 레포지토리가 제공하는 것과 제공하지 않는 것

| 제공 | 제공하지 않음 |
|---|---|
| RAM·VRAM 구간별 후보 모델과 양자화 | 특정 장비에서의 속도·품질·안정성 보장 |
| Hugging Face 저장소와 실행 예제 | 모델 가중치 재배포 또는 자동 설치 |
| KV 캐시·작업 버퍼·RAG·동시성·학습 상태를 포함한 용량 계획 | 모든 런타임·드라이버 조합의 호환성 보장 |
| 작업별 평가·보안·재현성 체크리스트 | 생성 결과의 정확성·합법성·상업 이용 가능성 보증 |

---

## 빠른 시작

### 1. 장착 용량이 아니라 사용 가능한 메모리를 확인한다

| 환경 | 확인할 값 | 참고 |
|---|---|---|
| NVIDIA·AMD 전용 GPU | 총 VRAM과 현재 사용량 | 디스플레이·드라이버·다른 프로세스가 사용하는 VRAM을 제외한다. |
| Apple Silicon | 통합 메모리와 현재 memory pressure | CPU·GPU·OS가 같은 메모리를 공유한다. |
| CPU 전용 | 사용 가능한 시스템 RAM | 운영체제, 파일 캐시, IDE와 데이터 처리 메모리를 함께 남긴다. |
| 다중 GPU | GPU별 VRAM과 연결 구조 | 런타임이 tensor·pipeline·expert parallel을 지원하는지 먼저 확인한다. |

운영체제 도구에서 확인한 값이 브라우저 자동 탐지보다 우선한다.

### 2. 웹 계산기로 실행 가능한 범위를 좁힌다

**[RAM-for-Local-AI 메모리 계산기](https://jtech-co.github.io/RAM-for-Local-AI/tools/memory-calculator/)**

계산기에서 다음 항목을 선택하거나 입력한다.

1. RAM·VRAM 또는 통합 메모리
2. 분야와 세부 작업
3. 모델과 양자화
4. 컨텍스트 길이, 출력 토큰과 동시 요청 수
5. RAG·데이터·비전·이미지·오디오별 추가 구성
6. 추론·서빙·파인튜닝 운용 방식

계산 결과는 후보를 좁히기 위한 **용량 계획값**이다. 최종 판단은 실제 런타임에서 peak RAM·VRAM과 품질을 측정해 확정한다.

### 3. 관련 가이드에서 모델·파일·런타임을 확인한다

계산 결과의 가이드 링크 또는 아래 [가이드 목록](#가이드-목록)에서 정확한 모델 저장소, 양자화 파일명, 실행 옵션과 평가 방법을 확인한다.

### 4. 다운로드 전에 실제 파일 크기를 검증한다

Hugging Face CLI를 사용할 수 있다면 전체 shard와 예상 다운로드 크기를 먼저 확인한다.

```bash
hf download <organization>/<repository> --dry-run
```

GGUF, MLX, AWQ, GPTQ, `compressed-tensors`, safetensors는 같은 비트 수로 표시되더라도 실제 파일 크기와 실행 특성이 다를 수 있다. 파일명이나 양자화 태그를 추정하지 말고 모델 카드와 저장소 파일 목록을 확인한다.

---

## 메모리 계산기

### 지원하는 운용 형태

| 운용 형태 | 계산에 포함하는 주요 항목 |
|---|---|
| 단일 사용자 추론 | 모델 가중치, KV 캐시, 런타임 작업 버퍼, 멀티모달 버퍼, 운영 여유 |
| RAG | 생성 모델, 임베딩, reranker, 벡터 인덱스, 문서 캐시, parser·OCR worker |
| API 서빙·동시 사용자 | replica, active sequence, KV pool, prefix cache, prefill·decode 작업 버퍼 |
| 파인튜닝 | 가중치, trainable parameter, gradient, optimizer state, activation, checkpoint staging |

분야에 따라 다음 입력이 추가된다.

- **RAG:** 청크 수, 임베딩 차원, 벡터 정밀도, 인덱스 오버헤드, 문서 캐시
- **데이터 분석:** 원본 데이터 크기, DataFrame 확장률, working set, notebook kernel
- **비전·OCR:** 페이지·이미지 수, 해상도, visual token, parser worker
- **이미지 생성:** 해상도, batch, ControlNet, LoRA, 참조 이미지, upscaler
- **오디오·음성:** 오디오 길이, 동시 stream, diarization, ASR→LLM→TTS cascade

### 적합도 표시를 읽는 법

| 표시 | 의미 |
|---|---|
| 여유 | 현재 설정에서 운영 안전 여유를 확보한 시작점 |
| 적합 | 일반적인 권장 범위 안에서 실행 가능 |
| 경계 | 적재 가능성이 있으나 첫 요청·긴 입력·동시성에서 peak 측정 필요 |
| 조정 필요 | offload, context, batch, 동시성 또는 양자화 조정 필요 |
| 부족 | 현재 설정의 예상 요구량이 입력한 메모리 예산을 초과 |

`여유`나 `적합`은 처리 속도와 답변 품질까지 보장하는 등급이 아니다. 같은 용량에서도 메모리 대역폭, 커널 지원, 드라이버, 컨텍스트와 데이터 형태에 따라 결과가 크게 달라질 수 있다.

### 자동 장치 탐지의 한계

일반 웹 브라우저는 정확한 총 RAM과 VRAM을 항상 제공하지 않는다.

- `navigator.deviceMemory`는 지원되는 브라우저에서도 반올림된 근삿값일 수 있다.
- 표준 WebGPU·WebGL API만으로는 전용 VRAM 총량을 일관되게 읽기 어렵다.
- GPU 이름을 확인하더라도 같은 제품에 8GB·16GB처럼 여러 메모리 변형이 있을 수 있다.
- Apple Silicon은 전용 VRAM이 아니라 CPU·GPU·OS가 공유하는 통합 메모리를 사용한다.
- 브라우저 개인정보 보호 설정에 따라 GPU 이름이나 adapter 정보가 숨겨질 수 있다.

따라서 자동 탐지 결과는 **후보값**이며, 운영체제 도구에서 확인한 수동 입력값을 최종 기준으로 사용한다.

### 개인정보와 네트워크

계산기는 프레임워크, 서버 API, CDN, 외부 폰트와 텔레메트리를 사용하지 않는다.

- 계산과 장치 탐지는 브라우저 안에서 수행한다.
- 입력값과 계산 결과를 외부 서버로 전송하지 않는다.
- 사용자 설정은 브라우저의 `localStorage`에만 저장될 수 있다.
- Hugging Face와 가이드 링크는 사용자가 직접 클릭할 때만 열린다.

---

## 가이드 목록

### 분야별 가이드

| 분야 | 주요 범위 | 문서 | 검증일 |
|---|---|---|---:|
| 버그바운티·사이버보안 | 승인된 버그바운티, CTF·교육, CVE·위협 인텔리전스, 코드 감사, 보안 자동화 | [cybersecurity.md](./guides/domains/cybersecurity.md) | 2026-07-20 |
| 프로그래밍·수학·과학 | 범용 코딩, 저장소 에이전트, 수학·과학 추론, 논문 분석, Lean 4 형식증명 | [programming-stem.md](./guides/domains/programming-stem.md) | 2026-07-20 |
| 생산성·문서·RAG | 문서 요약·질의응답, 번역, 임베딩, reranker, 개인·조직 지식베이스 | [productivity-rag.md](./guides/domains/productivity-rag.md) | 2026-07-21 |
| 데이터 분석·BI·SQL | Text-to-SQL, EDA, Python·R, 표형 모델, 시계열, 통계·ML 파이프라인 | [data-analysis.md](./guides/domains/data-analysis.md) | 2026-07-21 |

### 모달리티별 가이드

| 모달리티 | 주요 범위 | 문서 | 검증일 |
|---|---|---|---:|
| 비전·OCR·문서 이해 | PDF·문서 OCR, 표·수식·차트, screenshot·UI 분석, VLM projector | [vision-ocr.md](./guides/modalities/vision-ocr.md) | 2026-07-21 |
| 이미지 생성·편집 | Diffusion·DiT, SD·FLUX 계열, ControlNet, LoRA, VAE, upscaler, ComfyUI | [image-generation.md](./guides/modalities/image-generation.md) | 2026-07-21 |
| 오디오·음성 | ASR, streaming transcription, diarization, TTS, voice cloning, 오디오 이해 | [audio-speech.md](./guides/modalities/audio-speech.md) | 2026-07-21 |

### 공통 운영 가이드

| 운영 주제 | 주요 범위 | 문서 | 검증일 |
|---|---|---|---:|
| 양자화 | GGUF Q2~Q8, IQ, AWQ, GPTQ, NF4, FP8·FP4, MLX, KV cache quantization | [quantization.md](./guides/operations/quantization.md) | 2026-07-21 |
| 파인튜닝 메모리 | Full FT, LoRA, QLoRA, DoRA, SFT·DPO·GRPO, FSDP2·ZeRO | [fine-tuning-memory.md](./guides/operations/fine-tuning-memory.md) | 2026-07-21 |
| 서빙·동시성 | KV pool, continuous batching, queue, SLO, prefix cache, 다중 GPU·tenant | [serving-concurrency.md](./guides/operations/serving-concurrency.md) | 2026-07-21 |
| 런타임·하드웨어 | CPU, NVIDIA CUDA, AMD ROCm, Apple MLX·Metal, Intel, Vulkan·WebGPU | [runtime-hardware.md](./guides/operations/runtime-hardware.md) | 2026-07-21 |

`guides/` 아래의 날짜 없는 문서를 최신 기준 경로로 사용한다. 루트의 날짜 포함 두 문서는 초기 공개 스냅샷과 기존 외부 링크 호환을 위해 보존한다.

---

## 메모리 계산 원칙

모든 운용 형태의 기본 구조는 다음과 같다.

```text
필요 메모리
≈ resident model weights
+ 작업에 따라 증가하는 동적 메모리
+ 런타임·통신·allocator 작업 버퍼
+ 운영체제·도구·데이터 처리 메모리
+ 운영 안전 여유
```

모델 파일 크기는 위 식의 한 항목일 뿐이다. “다운로드 파일이 10GB이므로 12GB VRAM에서 안전하다”는 식으로 판단하면 긴 prompt, 멀티모달 입력, 동시 요청 또는 checkpoint 저장 시 OOM이 발생할 수 있다.

| 운용 형태 | 대표적인 동적 메모리 |
|---|---|
| 추론 | KV 캐시, attention·activation workspace, graph reserve, vision·audio·VAE·projector |
| RAG | 임베딩·reranker, 벡터 인덱스, 원문·metadata cache, parser·OCR worker |
| 온라인 서빙 | active sequence별 KV, prefix cache, prefill·decode buffer, queue와 replica |
| 파인튜닝 | gradient, optimizer state, saved activation, communication buffer, checkpoint peak |

### 메모리 종류를 읽는 방법

| 환경 | 해석 기준 |
|---|---|
| NVIDIA·AMD 전용 GPU | VRAM에 가중치, KV 캐시와 주요 작업 버퍼가 모두 들어갈 때 가장 빠르다. 시스템 RAM offload는 용량을 늘리지만 PCIe 전송 비용이 생긴다. |
| Apple Silicon | CPU·GPU·OS가 통합 메모리를 공유한다. 장착 용량 전체를 모델에 배정하지 말고 memory pressure와 swap을 함께 본다. |
| CPU 전용 | RAM 용량 외에 메모리 채널, 대역폭, NUMA, ISA와 quantized kernel 지원이 처리량을 좌우한다. |
| 다중 GPU | VRAM을 단순 합산하지 않는다. tensor·pipeline·expert parallel 지원과 interconnect를 확인한다. |
| APU·공유 메모리 | BIOS의 고정 GPU 메모리 표시보다 실제 공유 메모리 대역폭과 운영체제 정책이 중요할 수 있다. |

### 장착 메모리와 실사용 가능 메모리

실제 계획에서는 다음 항목을 먼저 제외한다.

- 운영체제와 백그라운드 서비스
- 디스플레이, 브라우저, IDE와 notebook
- 드라이버·런타임 context
- 파일 캐시와 memory-mapped page
- allocator fragmentation과 임시 작업 버퍼
- 장시간 운용을 위한 안전 여유

개인 워크스테이션에서는 장착 메모리의 100%를 모델에 할당하지 않는다. 계산기의 기본 예약률을 출발점으로 사용하고 실제 peak를 측정해 조정한다.

---

## 양자화 빠른 기준

같은 “4비트”라도 `Q4_K_M`, `IQ4_XS`, AWQ W4A16, GPTQ 4-bit, NF4, MLX 4-bit와 NVFP4는 같은 형식이 아니다. 런타임과 하드웨어가 실제로 최적화한 형식을 선택해야 한다.

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
| AWQ·GPTQ·`compressed-tensors` | CUDA 중심 W4A16·W8A8 | vLLM·SGLang·TensorRT 계열의 지원 matrix가 맞는 서버 추론 |

특별한 이유가 없다면 다음 순서로 시작한다.

```text
Q4 또는 하드웨어가 잘 지원하는 4-bit 형식
→ context 4K~8K, 동시 요청 1개
→ 실제 peak RAM·VRAM과 품질 측정
→ Q5·Q6 또는 더 큰 모델 검토
→ 긴 context와 동시성 단계적 증가
```

메모리에 간신히 들어가는 대형 Q2 모델보다 운영 여유를 남긴 한 단계 작은 Q4·Q5 모델이 실제 작업에서 더 안정적인 경우가 많다.

---

## 작업별 시작 경로

| 목표 | 먼저 확인할 문서 | 함께 확인할 문서 |
|---|---|---|
| 개인 PC에서 코딩 모델 실행 | [프로그래밍·STEM](./guides/domains/programming-stem.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 로컬 문서 검색·질의응답 | [생산성·RAG](./guides/domains/productivity-rag.md) | [비전·OCR](./guides/modalities/vision-ocr.md), [서빙·동시성](./guides/operations/serving-concurrency.md) |
| SQL·CSV·Parquet 분석 | [데이터 분석](./guides/domains/data-analysis.md) | [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| PDF·표·수식 OCR | [비전·OCR](./guides/modalities/vision-ocr.md) | [생산성·RAG](./guides/domains/productivity-rag.md) |
| 로컬 이미지 생성·편집 | [이미지 생성](./guides/modalities/image-generation.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 회의 녹취·TTS·음성 에이전트 | [오디오·음성](./guides/modalities/audio-speech.md) | [서빙·동시성](./guides/operations/serving-concurrency.md) |
| LoRA·QLoRA 학습 | [파인튜닝 메모리](./guides/operations/fine-tuning-memory.md) | [양자화](./guides/operations/quantization.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |
| 여러 사용자에게 API 제공 | [서빙·동시성](./guides/operations/serving-concurrency.md) | [런타임·하드웨어](./guides/operations/runtime-hardware.md), [양자화](./guides/operations/quantization.md) |
| 승인된 보안 연구·코드 감사 | [사이버보안](./guides/domains/cybersecurity.md) | [서빙·동시성](./guides/operations/serving-concurrency.md), [런타임·하드웨어](./guides/operations/runtime-hardware.md) |

---

## 레포지토리 구조 (이미지는 제외)

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
        └── app.js
```

`guides/` 아래의 날짜 없는 파일이 최신 기준 문서다. 계산기에서 실제로 로드하는 핵심 파일은 `index.html`, `styles.css`, `catalog.js`, `app.js` 네 개다.

---

## 업데이트와 기여

모델·양자화·하드웨어 결과를 추가할 때 가능한 한 다음 정보를 함께 기록한다.

| 구분 | 필요한 정보 |
|---|---|
| 모델 | 공식 모델 카드, revision, dense·MoE 구조, 전체·활성 파라미터, context 한도 |
| 배포 파일 | 저장소, 정확한 파일명·quant tag, shard 수, 총 다운로드 크기, checksum |
| 환경 | 하드웨어, RAM·VRAM, 운영체제, driver, runtime와 버전 |
| 실행 설정 | GPU offload, KV dtype, context, batch, 동시 요청 수 |
| 결과 | peak RAM·VRAM, prompt·generation 처리량, TTFT·TPOT, 실패 사례 |
| 평가 | 대표 작업셋의 품질 결과, 기준 모델과 비교, 라이선스·검증일 |

“실행됨”만 기록하지 말고, 어떤 설정에서 어느 정도의 여유와 속도로 실행됐는지 재현 가능하게 남긴다.

계산기 모델 데이터는 [`tools/memory-calculator/catalog.js`](./tools/memory-calculator/catalog.js)에 있다. 가이드에서 모델 링크, 파일 크기, 최소 메모리 또는 권장 런타임을 수정할 때 계산기 카탈로그도 함께 검토해야 한다.

장기적으로는 다음과 같은 단일 데이터 원본 구조가 적합하다.

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

## 개발과 배포

<details>
<summary>로컬 실행 및 GitHub Pages 설정 보기</summary>

### 로컬 실행

레포지토리 루트에서 정적 서버를 실행한다.

```bash
python -m http.server 8000
```

다음 주소를 연다.

```text
http://localhost:8000/tools/memory-calculator/
```

`file://` 직접 실행보다 HTTP 또는 HTTPS 환경을 권장한다. 일부 브라우저의 장치 정보 API와 모듈 로딩 동작은 secure context 또는 HTTP 환경에서 더 일관적이다.

### GitHub Pages

1. 레포지토리의 **Settings → Pages**를 연다.
2. **Build and deployment**에서 `Deploy from a branch`를 선택한다.
3. branch를 `main`, folder를 `/ (root)`로 지정한다.
4. 배포가 끝나면 다음 경로를 연다.

```text
https://<GitHub-ID>.github.io/RAM-for-Local-AI/tools/memory-calculator/
```

현재 공개 계산기:

- <https://jtech-co.github.io/RAM-for-Local-AI/tools/memory-calculator/>

</details>

---

## 안전·정확성·라이선스

- 사이버보안 가이드와 모델은 소유하거나 명시적으로 허가받은 시스템, 승인된 버그바운티 범위, CTF와 격리된 연구 환경에서만 사용한다.
- 로컬 실행은 외부 전송을 줄일 수 있지만 prompt, log, vector DB, cache, swap, 임시 이미지·오디오와 생성 파일에 민감정보가 남을 수 있다.
- 모델이 생성한 코드, SQL, 수식, 과학적 주장, 인용, 보안 판단과 고위험 도메인 결과는 독립적인 도구와 신뢰할 수 있는 출처로 검증한다.
- `trust_remote_code`, custom node, adapter, pickle, 임의 container와 비공식 quantization artifact는 공급망 위험을 검토한다.
- 이 레포지토리는 모델 가중치를 재배포하지 않는다. 각 모델, quantization, dataset와 runtime의 라이선스·사용 제한은 해당 공식 저장소를 따른다.
- 파일 크기와 메모리 표는 선택 보조 자료이며 특정 하드웨어의 속도, 품질, 안정성 또는 상업적 이용 가능성을 보장하지 않는다.

---

## 문서 상태

| 구분 | 수량 | 상태 |
|---|---:|---|
| 분야별 가이드 | 4 | 작성 완료 |
| 모달리티별 가이드 | 3 | 작성 완료 |
| 공통 운영 가이드 | 4 | 작성 완료 |
| 정적 웹 계산기 | 1 | 운영 중 |
| 계산기 모델 프리셋 | 66 | 2026-07-21 카탈로그 기준 |

모델 생태계와 런타임은 빠르게 변한다. 이 레포지토리의 표와 계산 결과는 영구적인 사양이 아니라 **검증일 당시 공개된 모델·파일·문서를 기준으로 한 용량 계획용 스냅샷**이다.
