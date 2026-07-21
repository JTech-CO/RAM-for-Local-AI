# 로컬 AI 모델 선택 가이드
## 범용 프로그래밍·수학·과학·연구용 — RAM/VRAM/Apple 통합 메모리별

> **최종 검증일:** 2026-07-20 (KST)  
> **주요 실행 형식:** GGUF + `llama.cpp`; Apple Silicon에서는 MLX도 병행 가능  
> **범위:** 범용 프로그래밍, 저장소 수준 코딩 에이전트, 수학·과학 추론, 논문·기술문서 분석, RAG, 데이터 분석, Lean 4 형식증명 및 연구 자동화

이 문서는 보유한 **시스템 RAM**, **GPU VRAM**, 또는 **Apple Silicon 통합 메모리**만 알아도 적절한 로컬 모델과 양자화를 고르고, Hugging Face에서 바로 내려받아 실행할 수 있도록 구성한 실전 가이드다.

모델 파일과 양자화 저장소는 계속 수정된다. 아래 크기는 2026-07-20에 확인한 대표값이며, 다운로드 직전 반드시 모델 카드의 **파일명, 전체 shard 수, 총 크기, 라이선스, 수정일, 지원 백엔드**를 다시 확인해야 한다.

> **핵심 원칙:** 코딩·수학·과학에서는 메모리에 겨우 들어가는 큰 Q2 모델보다, 충분한 여유를 남긴 한 단계 작은 **Q4/Q5 모델**이 더 안정적인 경우가 많다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [메모리 용량을 올바르게 읽는 방법](#2-메모리-용량을-올바르게-읽는-방법)
3. [Q2·Q3·Q4 양자화 선택법](#3-q2q3q4-양자화-선택법)
4. [모델별 상세 표](#4-모델별-상세-표)
5. [작업별 추천](#5-작업별-추천)
6. [Hugging Face 다운로드와 실행](#6-hugging-face-다운로드와-실행)
7. [컨텍스트·KV 캐시·멀티모달 메모리](#7-컨텍스트kv-캐시멀티모달-메모리)
8. [연구용 도구 체인](#8-연구용-도구-체인)
9. [RAG·논문 검색 구성](#9-rag논문-검색-구성)
10. [평가·재현성·검증](#10-평가재현성검증)
11. [안전한 로컬 에이전트 운영](#11-안전한-로컬-에이전트-운영)
12. [초대형·실험 모델](#12-초대형실험-모델)
13. [주요 출처와 저장소](#13-주요-출처와-저장소)

---

# 1. 30초 선택표

아래의 메모리는 **장착된 총 RAM/VRAM/통합 메모리** 기준이다. 운영체제, IDE, 브라우저, 디스플레이 공유 메모리와 KV 캐시가 추가로 필요하므로 파일 크기와 총 메모리를 동일하게 보면 안 된다.

| 장착 메모리 | 가장 무난한 기본 모델 | 권장 양자화 | 대표 가중치 크기 | 시작 컨텍스트 | 판단 |
| --- | --- | --- | --- | --- | --- |
| **4 GB** | [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | Q4 계열 | 약 0.54 GB | 4K | 초경량 코드 설명·문서 요약. 2B Q2/Q3도 가능하지만 OS 여유가 매우 작다. |
| **6 GB** | [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Q3 계열 | 약 2.11 GB | 4K–8K | 간단한 Python/JS, 테스트 초안, 기초 수학. Q4는 경량 OS에서만. |
| **8 GB** | [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Q4 계열 | 약 2.78 GB | 8K | 저사양 노트북의 범용 기본값. 형식증명은 7B prover Q3/Q4 후보. |
| **12 GB** | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Q4 계열 | 약 5.84 GB | 8K | 코딩·수학·과학 질의·도구 호출의 균형형. |
| **16 GB** | [Ministral 3 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF) | Q4_K_M | 약 8.24 GB | 8K–16K | STEM 추론과 코딩에 유리. gpt-oss-20b는 16 GB에서 매우 빠듯하다. |
| **24 GB** | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | Q3_K_M | 약 13.6 GB | 8K–16K | 범용 연구 조수. 저장소 작업은 Devstral Small 2 Q4가 강한 대안. |
| **32 GB** | [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | UD-Q4_K_M | 약 22.1 GB | 8K부터 | 코딩·수학·도구 사용·멀티모달 연구의 강력한 단일 모델. 긴 컨텍스트는 실측 후 확대. |
| **48 GB** | [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | UD-Q3_K_S | 약 33.3 GB | 8K–16K | 대형 코드베이스와 장기 코딩 에이전트. 일반 연구에는 Qwen3.6 고정밀도도 합리적. |
| **64 GB** | [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | Q4_K_M | 약 48.5 GB | 8K–16K | 프로그래밍 우선. 범용·비전은 Mistral Small 4의 IQ2/저 Q3 대안. |
| **80 GB** | [Mistral Small 4 119B-A6.5B](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) | UD-Q3_K_M | 약 54.4 GB | 8K–16K | 대형 MoE 범용·코딩·비전. gpt-oss-120b는 이 구간의 최소선. |
| **96 GB** | [Mistral Medium 3.5 128B](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF) | Q4_K_S / Q4_K_M | 약 73.0 / 78.4 GB | 8K부터 | 고급 범용·코딩·수학·멀티모달. Lean 4는 Leanstral 1.5 Q4가 전용 선택. |
| **128 GB** | [Mistral Medium 3.5 128B](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF) | Q5_K_M / Q6_K | 약 91.1 / 107.8 GB | 16K–32K | 정확도 우선 단일 모델 또는 병렬 연구 서비스. |
| **192 GB** | [Mistral Medium 3.5 128B](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF) | Q8_0 | 약 132.9 GB | 32K부터 실측 | 고정밀 기준선·다중 사용자·별도 임베딩/재순위 서비스 동시 운용. |
| **256 GB+** | [MiniMax-M2.5 GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) | 저비트부터 검증 | 배포별 상이 | 4K–8K부터 | 초대형 실험 구간. 전체 shard, 백엔드 지원, NUMA·대역폭을 먼저 확인한다. |

## 용도별 한 줄 결론

- **8 GB 이하:** Qwen3.5-4B Q4가 저사양 범용 기준선이다.
- **12–16 GB:** Qwen3.5-9B Q4 또는 Ministral 3 14B Reasoning Q4가 실용적이다.
- **24–32 GB 범용 연구:** Qwen3.6-27B/35B-A3B Q3/Q4가 중심 선택이다.
- **24–32 GB 저장소 에이전트:** Devstral Small 2 24B Q4가 명확한 전용 후보다.
- **48–64 GB 대형 프로그래밍:** Qwen3-Coder-Next Q3/Q4가 유력하다.
- **24–32 GB 자연어 수학·과학:** SU-01 Q3/Q4와 Python·SymPy 검산을 함께 쓴다.
- **Lean 4 형식증명:** 8–12 GB는 DeepSeek/Goedel 7–8B, 32 GB는 Goedel 32B, 96 GB는 Leanstral 1.5 Q4다.
- **96 GB급 단일 범용 모델:** Mistral Medium 3.5 Q4 또는 Mistral Small 4 Q4가 현실적인 상한선이다.
- **논문·최신 지식:** 모델 크기보다 RAG, 출처 메타데이터, 재순위기와 인용 검증이 중요하다.

## 빠른 의사결정 규칙

1. 총 메모리의 **65–75% 이하**에 가중치가 들어가는 Q4를 먼저 찾는다.
2. 8K 컨텍스트로 부팅한 뒤 실제 peak memory를 측정한다.
3. Q4가 안 들어가면 컨텍스트를 줄이고, 그래도 부족할 때 Q3로 내린다.
4. Q2는 기능 확인과 실험에 우선 사용하고, 정밀 코드·수학·과학 결과는 재검증한다.
5. 같은 모델의 Q3와 한 단계 작은 모델의 Q5/Q6를 실제 작업셋으로 비교한다.

---

# 2. 메모리 용량을 올바르게 읽는 방법

## 2.1 GGUF 파일 크기와 실제 실행 메모리는 다르다

```text
실제 필요 메모리
≈ GGUF 가중치 파일
+ KV 캐시
+ 계산·그래프·토크나이저·백엔드 버퍼
+ 비전 projector 및 이미지/영상 입력 버퍼
+ 운영체제·디스플레이·IDE·브라우저 점유량
+ 동시 요청과 병렬 슬롯의 추가 캐시
```

실전에서는 다음 여유를 권장한다.

| 장착 메모리 | OS·앱에 남길 최소 여유 | 모델 가중치 목표 | 권장 시작점 |
|---:|---:|---:|---|
| 4–8 GB | 2–4 GB | 총 메모리의 40–60% | 4K–8K, 단일 요청 |
| 12–16 GB | 3–6 GB | 총 메모리의 50–65% | 8K, 작은 배치 |
| 24–32 GB | 5–8 GB | 총 메모리의 55–70% | 8K–16K |
| 48–64 GB | 8–12 GB | 총 메모리의 60–75% | 8K–16K, peak 측정 |
| 80–128 GB | 12–20 GB | 총 메모리의 65–78% | 8K부터 확대 |
| 192 GB 이상 | 20 GB 이상 | 서비스 구조별 산정 | NUMA·병렬 슬롯 포함 실측 |

이 비율은 절대식이 아니라 보수적 운용 휴리스틱이다. 모델 구조, KV 캐시 타입, 백엔드와 동시성에 따라 달라진다.

## 2.2 NVIDIA·AMD 전용 GPU

- 완전 GPU 오프로딩은 **가중치 + KV 캐시 + GPU 버퍼**가 VRAM에 들어갈 때 가장 빠르다.
- 일부 레이어를 시스템 RAM으로 내리는 하이브리드 오프로딩은 가능하지만 PCIe 왕복 때문에 느려질 수 있다.
- `VRAM 24 GB + 시스템 RAM 64 GB`는 용량 측면에서 큰 모델을 실행할 수 있게 해도 `VRAM 88 GB`와 같은 처리량을 제공하지 않는다.
- 디스플레이가 같은 GPU를 사용하면 최소 1–2 GB 이상의 VRAM을 비워 두는 편이 안정적이다.
- 멀티 GPU는 단순 합산이 아니다. 텐서 분할, GPU 간 링크, PCIe 토폴로지와 백엔드 지원을 확인한다.

## 2.3 Apple Silicon 통합 메모리

- CPU와 GPU가 동일 메모리 풀을 사용하므로 GGUF·MLX 대형 모델 배치가 단순하다.
- 표시된 32/64/96/128 GB 전체를 모델이 사용할 수 있는 것은 아니다. macOS, WindowServer, IDE, 브라우저와 파일 캐시가 함께 점유한다.
- 메모리 압축과 swap이 시작되면 실행은 되더라도 token/s가 급격히 떨어질 수 있다.
- 64 GB Mac에서 48.5 GB 가중치를 쓰는 경우 긴 컨텍스트와 다중 요청은 보수적으로 제한한다.
- MLX 4bit와 GGUF Q4는 양자화 방식·커널·메모리 사용이 동일하지 않다. 같은 “4bit”라는 이름만으로 직접 비교하지 않는다.

## 2.4 CPU 전용 서버

- 모델이 RAM에 들어가는 것만으로 충분하지 않다. 메모리 채널 수, DDR 세대, NUMA, CPU 벡터 명령과 대역폭이 생성 속도를 좌우한다.
- 100B 이상 모델은 용량보다 **메모리 대역폭과 NUMA 교차 접근**이 병목인 경우가 많다.
- 모델 shard와 mmap 파일을 빠르게 읽을 수 있도록 NVMe 여유 공간을 확보한다.
- 다중 소켓에서는 프로세스와 메모리의 NUMA 바인딩을 실측해 최적화한다.

## 2.5 MoE의 “활성 파라미터”를 오해하지 않기

`35B-A3B`, `80B-A3B`, `119B-A6.5B`는 토큰당 계산에 주로 참여하는 활성 파라미터가 작다는 뜻이다. 일반적인 로컬 추론에서는 **전체 전문가 가중치가 메모리 또는 mmap 가능한 저장장치에 준비되어야 한다.**

따라서 80B-A3B를 3B 모델처럼 4 GB 장비에 올릴 수 있다고 판단하면 안 된다. 활성 파라미터는 주로 연산량과 속도에 영향을 주며, GGUF 파일 크기는 전체 저장 파라미터에 더 가깝다.

## 2.6 “사용 가능한 메모리”를 먼저 확인하기

### Linux

```bash
free -h
nvidia-smi
rocm-smi  # AMD 환경
```

### macOS

```bash
system_profiler SPHardwareDataType | grep -E "Chip|Memory"
vm_stat
```

### Windows PowerShell

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
```

운영체제 표시값과 실제 GPU API가 보고하는 가용 VRAM이 다를 수 있으므로 실행 직전 백엔드 로그도 확인한다.

---

# 3. Q2·Q3·Q4 양자화 선택법

## 3.1 실전 우선순위

| 양자화 | 권장도 | 특성 | 적합한 작업 |
|---|---:|---|---|
| **Q4_K_M / UD-Q4_K_M** | 기본값 | 품질과 용량의 균형 | 일반 코딩, 코드 리뷰, 수학·과학 질의, 연구 에이전트 |
| **Q3_K_M / UD-Q3_K_M** | 메모리 부족 시 | Q4보다 누락·오류 가능성이 증가 | 큰 모델 기능 활용, 초안·분류·요약, 제한적 코딩 |
| **Q2_K / IQ2 / UD-Q2** | 최후 수단 | 압축률은 높지만 정밀 추론 손실 가능 | 대형 모델 기능 확인, 실험, 낮은 위험의 초안 |
| **Q5_K_M / Q6_K** | 메모리 여유 시 | 코드·수학 정확도와 지시 안정성 개선 가능 | 정밀 리팩터링, 수치·논리 문제, 장기 에이전트 |
| **Q8_0 / BF16** | 서버급 | 원본에 가까운 기준선, 대역폭 요구 큼 | 평가 기준선, 양자화 영향 측정, 품질 우선 서비스 |

## 3.2 코딩·수학·과학에서 Q2가 특히 위험한 이유

자연어 문체 품질은 그럴듯해 보여도 다음과 같은 작은 오류가 늘 수 있다.

- API 이름·인수·타입의 미세한 오류
- 긴 의존관계에서 변수나 조건 누락
- 부호, 지수, 단위, 인덱스 범위 오류
- 증명에서 전제가 누락되거나 결론이 비약되는 문제
- 논문 제목·저자·DOI·인용을 그럴듯하게 조작하는 문제
- 도구 호출 JSON·스키마의 작은 형식 오류

따라서 Q2 출력은 **컴파일러, 테스트, Python, CAS, Lean 4, 검색 인덱스 또는 원문**으로 반드시 검증한다.

## 3.3 접미사의 의미

- `K_S`, `K_M`, `K_L`: 같은 비트 계열에서도 텐서별 정밀도 배분과 크기가 다르다. 보통 `K_M`이 무난하다.
- `IQ2_XXS`, `IQ2_XS`, `IQ4_XS`: 중요도 기반 저비트 계열이다. 작지만 백엔드·품질을 실제 작업으로 확인한다.
- `UD-Qx_K_*`, `Qx_K_XL`: 동적 혼합 양자화다. 숫자만 같아도 파일 크기와 품질이 다를 수 있다.
- `imatrix`: 중요도 행렬을 이용한 양자화다. 동일 크기라면 신뢰할 수 있는 calibration/imatrix 빌드를 비교할 가치가 있다.
- `MXFP4`: 일반적인 GGUF Q4와 다른 네이티브 저정밀 형식이다.

## 3.4 `gpt-oss`의 예외

`gpt-oss-20b`와 `gpt-oss-120b`는 전문가 가중치가 네이티브 MXFP4 형태이므로 일부 GGUF 배포에서 Q2·Q3·Q4 파일 크기가 거의 같다.

| 모델 | Q2 계열 | Q3 계열 | Q4/MXFP4 계열 |
|---|---:|---:|---:|
| gpt-oss-20b | 약 11.5 GB | 약 11.5 GB | 약 11.6 GB |
| gpt-oss-120b | 약 62.6 GB | 약 62.6 GB | 약 62.8 GB |

Q2로 내려도 일반 모델처럼 큰 절감이 없으므로, 대화 템플릿과 백엔드 호환성을 확인한 뒤 Q4/MXFP4 계열을 우선한다. `gpt-oss`는 Harmony 형식 지원도 중요하다.

## 3.5 기본 선택 알고리즘

```text
1. 원하는 모델의 Q4_K_M 총 shard 크기를 확인한다.
2. Q4 가중치가 총 메모리의 65–75% 이하인지 본다.
3. 8K 컨텍스트 + 단일 요청으로 부팅한다.
4. peak RAM/VRAM, 속도, swap 여부를 측정한다.
5. 여유가 있으면 16K/32K 또는 Q5/Q6로 올린다.
6. 부족하면 컨텍스트 → KV 정밀도 → GPU 오프로딩 → Q3 순으로 조정한다.
7. Q2는 별도 평가를 통과한 작업에만 사용한다.
```

---

# 4. 모델별 상세 표

## 표 읽기

- 크기는 Hugging Face 모델 카드나 파일 목록에서 확인한 **십진 GB 기준 대표 가중치 크기**다.
- 범위는 같은 비트 계열의 `IQ`, `UD`, `K_S/K_M/K_L/XL` 차이를 반영한다.
- 멀티모달 projector, KV 캐시, 런타임 버퍼, 운영체제 메모리는 포함하지 않았다.
- “권장 총 메모리”는 8K–16K 시작 컨텍스트와 기본 OS 여유를 고려한 보수적 구간이다.
- 저장소 업데이트로 파일명이 바뀔 수 있으므로 실제 다운로드 전에 `hf download ... --dry-run`을 실행한다.

## 4.1 소형 범용·STEM·입문형 증명 모델

| 모델 | 주용도 | Q2 계열 GB | Q3 계열 GB | Q4 계열 GB | 권장 총 메모리 | Hugging Face |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3.5-0.8B** | 초경량 범용·코딩·다국어·비전 | 약 0.42 | 약 0.47 | **약 0.54** | 3–4 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) |
| **Qwen3.5-2B** | 저사양 코딩·도구 보조·요약 | 약 0.97 | 약 1.11 | **약 1.29** | 4–6 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) |
| **Ministral 3 3B Reasoning** | 소형 STEM 추론·코딩·비전 | 공식 GGUF 미제공 | 공식 GGUF 미제공 | **약 2.15** | 6–8 GB | [공식 GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF) |
| **Qwen3.5-4B** | 저사양 범용 코딩·수학·멀티모달 | 약 1.94 | 약 2.11 | **약 2.78** | 6–8 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| **DeepSeek-Prover-V2-7B** | Lean 4 형식증명 전용 | 약 2.2–2.9 | 약 3.46 | **약 4.22** | 8–12 GB | [GGUF](https://huggingface.co/unsloth/DeepSeek-Prover-V2-7B-GGUF) |
| **Goedel-Prover-V2-8B** | Lean 4 자동 정리 증명 | 3.28 | 4.12 | **5.03** | 8–12 GB | [GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-8B-GGUF) |
| **Ministral 3 8B Reasoning** | STEM 추론·비전·엣지 배포 | 공식 GGUF 미제공 | 공식 GGUF 미제공 | **약 5.20** | 12–16 GB | [공식 GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512-GGUF) |
| **Qwen3.5-9B** | 범용 코딩·수학·과학 문서·도구 사용 | 약 4.12 | 약 4.32 | **약 5.84** | 12–16 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| **Ministral 3 14B Reasoning** | 중형 수학·과학 추론·코딩·비전 | 공식 GGUF 미제공 | 공식 GGUF 미제공 | **8.24** | 16–24 GB | [공식 GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF) |

### 이 구간의 판단

- **4–8 GB:** Qwen3.5-0.8B/2B/4B는 로컬 툴 체인과 프롬프트를 검증하기 좋다.
- **12 GB:** Qwen3.5-9B Q4가 범용성이 높다.
- **16 GB:** Ministral 3 14B Reasoning Q4는 STEM 추론용으로 강한 선택이다.
- **Lean 4 입문:** DeepSeek-Prover-V2-7B 또는 Goedel-Prover-V2-8B를 Lean compiler feedback loop와 함께 사용한다.
- 작은 모델은 복잡한 연구 결론보다 **검색 질의 생성, 코드 뼈대, 테스트 초안, 데이터 정리**에 배치하는 편이 안전하다.

## 4.2 중형 범용·코딩·수학·과학 모델

| 모델 | 구조·주용도 | Q2 계열 GB | Q3 계열 GB | Q4 계열 GB | 권장 총 메모리 | Hugging Face |
| --- | --- | --- | --- | --- | --- | --- |
| **gpt-oss-20b** | 약 21B 총/3.6B 활성 MoE; 추론·에이전트 | 약 11.5 | 약 11.5 | **약 11.6** | 16 GB 최소 / 24 GB 권장 | [GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) · [공식](https://huggingface.co/openai/gpt-oss-20b) |
| **Devstral Small 2 24B** | 저장소 수준 소프트웨어 엔지니어링·비전 | 8.89–9.29 | 약 11.5 | **14.3** | 24–32 GB | [GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |
| **Qwen3.6-27B** | 27B dense; 범용·코딩·수학·비전·도구 사용 | 11.8 | 13.6 | **16.8** | Q3 24 GB / Q4 32 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) · [공식](https://huggingface.co/Qwen/Qwen3.6-27B) |
| **Qwen3.6-35B-A3B** | 35B 총/3B 활성 MoE; 범용 에이전트·코딩·비전 | 12.3 | 16.6 | **22.1** | Q3 24–32 GB / Q4 32 GB | [GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · [공식](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| **SU-01** | 약 31B-A3B; 자연어 수학·과학 올림피아드형 장기 추론 | 10.3–11.8 | 12.9–14.7 | **16.4–18.6** | Q3 24 GB / Q4 32 GB | [GGUF](https://huggingface.co/axi0mX/SU-01-GGUF) |
| **Goedel-Prover-V2-32B** | Lean 4 형식증명 전용 | 12.4 | 16.1 | **19.9** | Q3 24 GB / Q4 32 GB | [GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF) |

### 이 구간의 판단

- **Devstral Small 2:** 저장소 탐색, 여러 파일 수정, 테스트 실행과 패치 작성에 초점을 둔다.
- **Qwen3.6-27B:** 24–32 GB에서 범용성과 정밀도의 균형이 좋다.
- **Qwen3.6-35B-A3B:** 코딩·비전·도구 사용을 하나의 모델로 처리하려는 32 GB급 장비의 중심 후보다.
- **SU-01:** 자연어 수학·과학 올림피아드형 추론에 특화되어 있으며, 일반 소프트웨어 에이전트 또는 Lean 증명기와 동일한 모델로 보지 않는다.
- **Goedel 32B:** 자연어 풀이보다 컴파일 가능한 Lean 4 증명 생성에 배치한다.

## 4.3 대형 범용·코딩·연구·형식증명 모델

| 모델 | 구조·주용도 | Q2 계열 GB | Q3 계열 GB | Q4 계열 GB | 권장 총 메모리 | Hugging Face |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-Coder-Next** | 80B 총/3B 활성 MoE; 256K, 코딩 에이전트 | 23.3–29.3 | 33.3–38.3 | **38.4–49.6; Q4_K_M 48.5** | Q3 48 GB / Q4 64 GB | [GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) |
| **Mistral Small 4 119B-A6.5B** | 범용·코딩·추론·비전 MoE, 256K | 34.9–40.2 | 약 54.4 | **약 73.8** | Q2 64 GB / Q3 80 GB / Q4 96 GB | [GGUF](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) |
| **gpt-oss-120b** | 약 117B 총/5.1B 활성 MoE; 추론·에이전트 | 약 62.6 | 약 62.6 | **약 62.8** | 80 GB 최소 / 96 GB 권장 | [GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF) · [공식](https://huggingface.co/openai/gpt-oss-120b) |
| **Devstral 2 123B** | 대형 저장소·소프트웨어 엔지니어링 에이전트 | 47.9 | 54.4 | **75.5–78.5** | Q3 80 GB / Q4 96 GB | [GGUF](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF) |
| **Mistral Medium 3.5 128B** | 128B dense; 범용·코딩·수학·비전, 256K | 49.86 | 63.28 | **78.41** | Q3 80–96 GB / Q4 96 GB | [GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF) · [공식](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) |
| **Leanstral 1.5 119B-A6.5B** | Lean 4 증명공학·자동형식화·코드 검증 | 신뢰 가능한 현재 배포 미확인 | 신뢰 가능한 현재 배포 미확인 | **72.2** | 96 GB | [공식](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B) · [Q4 GGUF](https://huggingface.co/Abiray/Leanstral-1.5-119B-A6B-Q4KM-GGUF) |

### 이 구간의 판단

- **Qwen3-Coder-Next:** 활성 파라미터는 3B급이지만 전체 80B 가중치가 필요하다. 저장소 수준 코딩과 도구 사용이 주목적이다.
- **Mistral Small 4:** 대형 MoE와 비전이 필요하지만 dense 128B보다 낮은 파일 크기를 원하는 경우 적합하다.
- **Mistral Medium 3.5:** 코딩·수학·추론·멀티모달을 한 모델로 묶는 96–128 GB급 범용 상위 선택이다.
- **gpt-oss-120b:** Q2/Q3/Q4 크기가 거의 같다는 예외와 Harmony 호환성을 먼저 확인한다.
- **Leanstral 1.5:** 일반 챗봇이 아니라 Lean 4 compiler/LSP와 반복 상호작용하는 증명공학 에이전트로 운영해야 한다.

## 4.4 Lean 4 형식증명 전용 모델

자연어 수학 모델은 그럴듯한 설명을 만들 수 있지만, 형식증명 모델은 Lean 4가 수용하는 항·전술·보조정리를 생성해 **컴파일러로 검증 가능**하다는 점이 다르다.

| 모델 | 규모 | 주용도 | 대표 양자화 크기 | 권장 총 메모리 | Hugging Face |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek-Prover-V2-7B** | 7B | Lean 4 정리 증명·증명 완성 | Q2 약 2.2–2.9 / Q3_K_M 약 3.46 / Q4_K_M 약 4.22 GB | 8–12 GB | [GGUF](https://huggingface.co/unsloth/DeepSeek-Prover-V2-7B-GGUF) |
| **Goedel-Prover-V2-8B** | 8B | Lean 4 자동 증명·작은 프로젝트 | Q2_K 3.28 / Q3_K_M 4.12 / Q4_K_M 5.03 GB | 8–12 GB | [GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-8B-GGUF) |
| **Goedel-Prover-V2-32B** | 32B | 고난도 Lean 4 증명·긴 증명 탐색 | Q2_K 12.4 / Q3_K_M 16.1 / Q4_K_M 19.9 GB | 24–32 GB | [GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF) |
| **OProver-8B/32B post-trained** | 8B / 32B | 검색·컴파일러 피드백을 쓰는 agentic Lean prover | 현재 신뢰할 수 있는 통합 GGUF 표는 저장소별 확인 | 모델별 상이 | [컬렉션](https://huggingface.co/collections/m-a-p/oprover) · [8B Round2](https://huggingface.co/m-a-p/OProver-8B-Round2) · [32B Round1](https://huggingface.co/m-a-p/OProver-32B-Round1) |
| **Leanstral 1.5** | 119B 총 / 6.5B 활성 | 장기 Lean 4 증명공학·자동형식화·코드 검증 | 커뮤니티 Q4_K_M 약 72.2 GB; Q2/Q3는 배포·품질을 개별 검증 | 96 GB | [공식](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B) · [Q4 GGUF](https://huggingface.co/Abiray/Leanstral-1.5-119B-A6B-Q4KM-GGUF) · [MLX 4bit](https://huggingface.co/mvid/Leanstral-1.5-119B-A6B-MLX-4bit) |

### 형식증명 모델 운용 원칙

1. 모델 출력을 최종 증명으로 간주하지 말고 `lake build`, Lean LSP 또는 compiler feedback으로 검증한다.
2. 한 번의 긴 답보다 “현재 goal → 후보 tactic → 오류 → 수정” 반복 루프가 효과적이다.
3. 프로젝트의 Lean 버전과 `mathlib` revision을 고정한다.
4. 검색 가능한 local theorem index와 관련 lemma retrieval을 붙인다.
5. `sorry`, `admit`, 불신 가능한 axiom이 섞이지 않았는지 CI에서 검사한다.
6. 자연어 문제를 Lean 명제로 옮기는 autoformalization 단계와 실제 증명 단계를 분리 평가한다.

## 4.5 RAG용 임베딩·재순위 모델

생성 모델과 검색 모델은 역할이 다르다. 최신 논문·사내 문서·코드베이스를 다룰 때는 별도 임베딩 모델과 reranker를 붙이는 편이 정확하다.

| 모델 | 권장 역할 | Hugging Face |
| --- | --- | --- |
| **Qwen3-Embedding-0.6B** | 저메모리 논문·코드 검색, 한국어 포함 다국어 | [모델](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| **Qwen3-Embedding-4B** | 중형 연구 코퍼스·코드베이스 검색 | [모델](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| **Qwen3-Embedding-8B** | 정확도 우선 대형 검색 서비스 | [모델](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| **Qwen3-Reranker-0.6B / 4B / 8B** | 초기 검색 결과를 재순위해 근거 정확도를 높임 | [0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) · [8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) |
| **BGE-M3** | 다국어 dense·sparse·multi-vector 검색의 안정적 대안 | [모델](https://huggingface.co/BAAI/bge-m3) |

임베딩·reranker는 GGUF Q2/Q3/Q4보다 FP16/BF16/INT8, ONNX, TEI, vLLM 등 다른 배포 형식이 흔하다. 생성 모델 표의 메모리 규칙을 그대로 적용하지 말고 별도 서비스로 측정한다.

---

# 5. 작업별 추천

## 5.1 일상 프로그래밍·QA·코드 리뷰

| 메모리 | 추천 | 배치 방식 |
|---:|---|---|
| 8 GB | Qwen3.5-4B Q4 | 짧은 함수, 테스트 케이스, 타입 오류 설명 |
| 12–16 GB | Qwen3.5-9B Q4/Q5 또는 Ministral 14B Q4 | 단일 파일·작은 모듈, 테스트 초안과 리뷰 |
| 24 GB | Devstral Small 2 Q4 또는 Qwen3.6-27B Q3 | 중형 저장소, 도구 호출과 테스트 실행 |
| 32 GB | Qwen3.6-27B/35B Q4 | 복수 파일 수정, 프런트엔드·백엔드·데이터 작업 |
| 48–64 GB | Qwen3-Coder-Next Q3/Q4 | 대형 코드베이스, 장기 에이전트, 복잡한 리팩터링 |
| 96 GB+ | Mistral Medium 3.5 Q4+ 또는 Devstral 2 | 대형 저장소와 광범위한 설계·리뷰 |

코드 에이전트에는 다음을 함께 제공한다.

- 저장소 트리와 명확한 작업 범위
- formatter, linter, type checker, compiler
- 단위·통합·회귀 테스트
- 변경 가능한 경로 allowlist
- 최대 반복 횟수와 명령 실행 timeout
- patch/diff 검토 단계

## 5.2 수학 문제·과학 추론

| 작업 | 추천 모델 | 필수 검증 도구 |
|---|---|---|
| 기초·중급 계산과 설명 | Qwen3.5-9B Q4/Q5, Ministral 14B Q4 | Python, 단위 테스트, 수치 재계산 |
| 올림피아드형 자연어 추론 | SU-01 Q3/Q4 | SymPy, 수치 반례 탐색, 사람 검토 |
| 과학 문헌 기반 질의 | Qwen3.6-27B/35B + RAG | 원문 chunk, DOI/서지 메타데이터 |
| 통계·데이터 분석 | Qwen3.6 또는 Mistral Medium + Python/R | 재현 가능한 notebook, 고정 seed |
| 형식 수학 | DeepSeek/Goedel/Leanstral | Lean 4 compiler, mathlib revision |
| 수치해석·시뮬레이션 | 모델은 계획·코드 담당 | NumPy/SciPy/JAX, 오차·수렴성 검사 |

수학·과학 응답에는 다음 질문을 자동으로 붙이는 것이 좋다.

```text
- 가정과 적용 범위는 무엇인가?
- 단위와 차원이 일치하는가?
- 경계조건과 초기조건은 명시됐는가?
- 독립적인 수치 계산으로 확인했는가?
- 반례 또는 실패 조건은 무엇인가?
- 결과를 뒷받침하는 실제 출처는 무엇인가?
```

## 5.3 논문·기술문서 분석

- **작은 문서·저메모리:** Qwen3.5-9B Q4 + Qwen3-Embedding-0.6B
- **중형 연구 워크스테이션:** Qwen3.6-27B/35B Q4 + Qwen3-Embedding-4B + reranker
- **이미지·도표·수식이 많은 PDF:** Qwen3.6, Mistral Small 4 또는 Medium 3.5의 비전 입력을 사용하되 projector/vision memory를 별도 산정한다.
- **대규모 코퍼스:** 전체 문서를 프롬프트에 넣지 말고 chunk retrieval, metadata filter, reranking과 citation trace를 사용한다.
- **최신 연구:** 정적 모델 지식만으로 답하게 하지 말고 논문 원문·초록·데이터셋 카드와 코드 저장소를 검색한다.

## 5.4 데이터 분석·연구 프로그래밍

권장 역할 분리:

```text
LLM
 ├─ 분석 계획·가설·코드 초안
 ├─ 도구 선택·오류 해석
 └─ 결과 설명 초안

실행 도구
 ├─ Python / R / Julia
 ├─ NumPy / pandas / SciPy / statsmodels
 ├─ SymPy / SageMath
 ├─ JAX / PyTorch
 ├─ SQL 엔진
 └─ 시각화·리포트 생성

검증 계층
 ├─ schema·type 검사
 ├─ 단위·범위·결측치 검사
 ├─ seed·환경 고정
 ├─ 독립 계산
 └─ 원자료·출처 추적
```

LLM이 표나 수치를 직접 “계산했다고 주장”하게 두지 말고 실행 가능한 코드와 실제 출력 로그를 남긴다.

## 5.5 연구 에이전트

연구 에이전트는 단일 모델보다 다음 조합이 중요하다.

1. 생성/추론 모델
2. 검색 엔진 또는 로컬 RAG
3. 임베딩·reranker
4. Python/셸/브라우저/논문 API 도구
5. 출처 저장소와 인용 formatter
6. 작업 예산·timeout·allowlist
7. 결과 검증기
8. 실행 기록과 재현성 manifest

32 GB에서는 Qwen3.6-35B-A3B Q4 + 경량 embedding을 순차 실행하고, 64–128 GB에서는 생성 모델과 검색 서비스를 분리해 동시 운용하는 구성이 효율적이다.

## 5.6 멀티모달 연구

이미지·도표·UI·현미경/실험 이미지를 처리할 때:

- 모델이 텍스트 전용 GGUF인지 vision 포함 배포인지 확인한다.
- 같은 저장소의 `mmproj*.gguf` 또는 백엔드가 요구하는 projector를 내려받는다.
- 이미지 수, 해상도, 영상 frame 수에 따라 메모리와 prefill 시간이 크게 증가한다.
- 도표에서 읽은 숫자는 원 데이터나 표와 대조한다.
- 의료·생명과학 이미지는 모델 출력만으로 진단·결론을 확정하지 않는다.

---

# 6. Hugging Face 다운로드와 실행

## 6.1 Hugging Face CLI 설치

```bash
python -m pip install -U "huggingface_hub[cli]"
```

토큰이 필요한 gated 모델은 먼저 로그인한다.

```bash
hf auth login
```

## 6.2 다운로드 전에 전체 파일과 크기 확인

```bash
hf download unsloth/Qwen3.6-27B-GGUF --dry-run
```

대형 모델은 여러 shard로 나뉠 수 있다. 한 shard만 내려받으면 실행되지 않는다.

## 6.3 Q4 파일 다운로드 예시

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir ./models/qwen3.6-27b
```

Qwen3.6-35B-A3B의 동적 Q4 예시:

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "*UD-Q4_K_M*.gguf" \
  --local-dir ./models/qwen3.6-35b-a3b
```

Mistral Medium 3.5의 split Q4 예시:

```bash
hf download bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF \
  --include "*Q4_K_M*" \
  --local-dir ./models/mistral-medium-3.5-q4
```

> `--include` 패턴이 README나 projector만 잡는 경우가 있으므로 `--dry-run` 결과에서 모든 `.gguf` shard와 총합을 확인한다.

## 6.4 `llama.cpp`에서 Hugging Face 직접 실행

최근 `llama.cpp` 빌드는 `-hf <user>/<repo>:<quant>` 형태로 모델을 자동 선택·다운로드할 수 있다.

### 12–16 GB 범용

```bash
llama-cli \
  -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 999
```

### 32 GB 범용·코딩

```bash
llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  -c 8192 \
  -ngl 999
```

### 48–64 GB 코딩 에이전트

```bash
llama-server \
  -hf unsloth/Qwen3-Coder-Next-GGUF:UD-Q3_K_S \
  -c 16384 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080
```

### 96 GB급 범용 모델

```bash
llama-server \
  -hf bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080
```

설치 방식에 따라 실행 파일 이름이 `llama-cli`/`llama-server` 또는 `llama cli`/`llama serve`일 수 있다. `llama --help`와 해당 release 문서를 확인한다.

## 6.5 CPU 전용·부분 GPU 오프로딩

```bash
# CPU 전용
llama-cli -m ./model.gguf -c 8192 -ngl 0

# 일부 레이어만 GPU에 올리기
llama-cli -m ./model.gguf -c 8192 -ngl 20
```

최적 `-ngl` 값은 모델과 VRAM에 따라 달라진다. OOM 직전까지 올리는 것보다 KV 캐시와 디스플레이 여유를 남기는 편이 안정적이다.

## 6.6 OpenAI 호환 로컬 API

```bash
llama-server \
  -m ./models/model.gguf \
  -c 16384 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080
```

테스트:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "Answer with verifiable code and cite provided sources."},
      {"role": "user", "content": "Write a typed Python function and tests."}
    ],
    "temperature": 0.2
  }'
```

## 6.7 Ollama에서 Hugging Face GGUF 직접 실행

```bash
ollama run hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
```

```bash
ollama run hf.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF:Q4_K_M
```

Ollama가 해당 아키텍처·chat template·vision projector를 정확히 지원하는지 버전별로 확인한다.

## 6.8 멀티모달 projector

Qwen3.6-35B-A3B용 F16 projector 다운로드 예시:

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "mmproj-F16.gguf" \
  --local-dir ./models/qwen3.6-35b-a3b
```

같은 저장소의 Q4 모델과 실행하는 예시:

```bash
llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --mmproj ./models/qwen3.6-35b-a3b/mmproj-F16.gguf \
  -c 8192 \
  -ngl 999
```

모델마다 projector 이름과 요구 형식이 다르므로 반드시 같은 저장소·revision 조합을 사용한다. 이 저장소에는 `mmproj-F16.gguf` 외에도 BF16/F32 변형이 있다.

## 6.9 체크섬·revision 고정

재현성이 중요한 연구에서는 branch 이름 `main`만 기록하지 말고 commit SHA와 파일 SHA-256을 남긴다.

```bash
sha256sum ./models/model/*.gguf > model-sha256.txt
```

다운로드 manifest 예시:

```yaml
model_repo: unsloth/Qwen3.6-35B-A3B-GGUF
revision: <hugging-face-commit-sha>
quant: UD-Q4_K_M
files:
  - Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
runtime: llama.cpp
runtime_revision: <git-sha-or-release>
context: 8192
kv_cache: default
seed: 42
prompt_template: <name-and-version>
```

---

# 7. 컨텍스트·KV 캐시·멀티모달 메모리

## 7.1 광고된 최대 컨텍스트는 실용 기본값이 아니다

256K 또는 1M 컨텍스트를 지원한다고 해서 로컬 장비에서 그 길이를 기본으로 잡아야 하는 것은 아니다. 컨텍스트가 커질수록:

- KV 캐시 메모리가 대체로 선형 증가한다.
- prefill 시간이 길어진다.
- 검색되지 않은 잡음이 늘어 답변 품질이 오히려 떨어질 수 있다.
- 다중 요청의 총 캐시가 빠르게 증가한다.
- 특정 RoPE 확장이나 장문 설정이 필요할 수 있다.

권장 절차:

```text
8K → peak memory/품질 측정
16K → 동일 평가 반복
32K → 실제 장문 작업에서만
64K+ → RAG와 요약 계층으로 대체 가능한지 먼저 검토
```

## 7.2 KV 캐시 정밀도

지원되는 `llama.cpp` 빌드에서는 다음과 같은 캐시 양자화를 시험할 수 있다.

```bash
llama-server \
  -m ./model.gguf \
  -c 32768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

더 낮은 KV 정밀도는 메모리를 줄일 수 있지만 긴 문맥 회수와 정밀 추론에 영향을 줄 수 있으므로 실제 평가가 필요하다.

## 7.3 병렬 요청

동시 sequence·parallel slot 수가 늘면 KV 캐시와 임시 버퍼가 증가한다. 단일 사용자 테스트가 성공했다고 동일 설정으로 여러 사용자를 바로 받지 않는다.

측정 항목:

- idle RSS/VRAM
- 첫 요청 peak
- 긴 입력 prefill peak
- 동시 2/4/8 요청 peak
- swap·OOM·GPU reset 여부
- prompt tokens/s와 generation tokens/s
- p50/p95 latency

## 7.4 RAG가 컨텍스트 확장보다 유리한 경우

- 수천 개 논문에서 특정 근거를 찾을 때
- 수백만 줄 코드 중 관련 파일만 필요할 때
- 최신 문헌과 변경 이력이 중요할 때
- 인용 가능한 chunk와 문서 위치가 필요할 때
- 동일 문서를 반복 질의할 때

전체 코퍼스를 256K에 밀어 넣기보다 검색 → rerank → 제한된 근거 → 답변의 구조가 일반적으로 비용과 검증성 측면에서 낫다.

---

# 8. 연구용 도구 체인

## 8.1 프로그래밍

- Git worktree 또는 임시 branch
- `ripgrep`, language server, AST/semantic search
- compiler, formatter, linter, type checker
- 단위·통합·E2E 테스트
- dependency vulnerability/lockfile 검사
- patch size 제한과 diff review

모델의 성공 기준을 “설명 생성”이 아니라 다음과 같이 정의한다.

```text
컴파일 성공
+ 모든 지정 테스트 통과
+ 새 회귀 테스트 포함
+ lint/type check 통과
+ 변경 범위 준수
+ 성능·보안 회귀 없음
```

## 8.2 수학·기호 계산

- SymPy, SageMath, Mathematica 또는 Maple
- rational/exact arithmetic
- 수치 반례 탐색
- interval arithmetic
- dimensional analysis
- Lean 4/Coq/Isabelle 같은 proof assistant

LLM은 식 변형과 전략을 제안하고, CAS와 proof assistant가 검증을 담당하게 한다.

## 8.3 수치·과학 계산

- NumPy, SciPy, JAX, PyTorch
- pandas, Polars, DuckDB
- statsmodels, scikit-learn
- R, Julia
- 실험 파라미터와 seed 기록
- float precision, tolerance, convergence criterion 기록

“코드가 실행됐다”와 “과학적으로 타당하다”는 별개다. 모델 선택보다 실험 설계와 검증 절차가 중요하다.

## 8.4 문헌·서지

- DOI, PMID, arXiv ID, 공식 데이터셋 ID를 canonical key로 저장
- 제목·저자·연도·학회·버전을 원문 메타데이터에서 읽음
- 모델이 생성한 인용을 검색 결과와 대조
- preprint와 peer-reviewed version 구분
- retract/erratum/correction 확인
- 근거 chunk의 문서·페이지·섹션 위치 저장

## 8.5 에이전트 역할 분리

```text
Planner: 작업 분해와 검증 계획
Retriever: 코드·논문·데이터 검색
Executor: Python/셸/Lean/테스트 실행
Reviewer: 결과·diff·수치·인용 검토
Reporter: 근거와 불확실성을 포함한 최종 문서화
```

메모리가 작으면 같은 모델을 역할별 system prompt로 순차 호출하고, 메모리가 크면 생성 모델과 검색/검증 서비스를 분리한다.

---

# 9. RAG·논문 검색 구성

## 9.1 권장 파이프라인

```text
문서 수집
→ 텍스트·표·수식·코드 추출
→ 문서/섹션/페이지 메타데이터 보존
→ 의미 단위 chunking
→ embedding
→ metadata filter + hybrid retrieval
→ reranking
→ 근거 제한
→ 생성 모델 답변
→ 인용·수치·주장 검증
```

## 9.2 chunking

고정 1,000자 절단보다 구조를 보존한다.

- 논문: 제목, 초록, 방법, 결과, 표/그림 caption, 부록
- 코드: 파일, 클래스, 함수, symbol, call graph
- 수학: 정의, 정리, 증명, 식 번호
- API 문서: endpoint, schema, example, version

overlap을 과도하게 키우면 중복 결과가 많아지고 reranker 비용이 증가한다.

## 9.3 인용 가능한 답변 형식

```json
{
  "claim": "주장",
  "evidence": [
    {
      "document_id": "doi-or-repo-revision",
      "location": "page/section/line",
      "quote_or_paraphrase": "근거 요약"
    }
  ],
  "calculation": "재현 가능한 코드 또는 식",
  "uncertainty": "미확인 전제와 한계"
}
```

생성 모델에게 존재하지 않는 DOI나 논문을 만들지 말라고 지시하는 것만으로 충분하지 않다. 검색 결과에 없는 citation ID는 후처리에서 거부한다.

## 9.4 최신성

로컬 모델의 학습 지식은 특정 시점에 고정된다. 다음은 반드시 외부 또는 로컬 최신 인덱스에서 가져온다.

- 최신 논문·프리프린트
- 라이브러리 API와 breaking change
- 데이터셋 revision
- 규격·표준·법규
- 모델·백엔드 릴리스와 보안 공지
- 재현 연구와 정정·철회 정보

---

# 10. 평가·재현성·검증

## 10.1 벤치마크보다 자신의 작업셋

공개 벤치마크 점수만으로 양자화를 고르지 않는다. 실제 용도에서 30–100개 정도의 고정 평가셋을 만든다.

### 프로그래밍 평가

- compile/test pass rate
- 저장소 탐색 성공률
- 올바른 파일 수정 비율
- 회귀 테스트 생성 품질
- tool-call schema 성공률
- 불필요한 diff 크기
- 평균 반복 횟수와 총 토큰

### 수학·과학 평가

- exact answer 또는 tolerance 내 수치 정답률
- 가정·단위·경계조건 누락률
- CAS/수치 검증 통과율
- 반례 탐지율
- 출처 정확도와 citation precision
- “모름/불확실”을 올바르게 표시한 비율

### 형식증명 평가

- Lean compile success
- `sorry`/`admit` 부재
- 프로젝트 revision 재현성
- pass@1/pass@k
- 평균 compiler feedback round
- 증명 시간과 token budget

## 10.2 양자화 A/B 테스트

같은 모델의 Q3/Q4/Q5를 비교할 때 다음을 고정한다.

- 모델 revision
- chat template
- system prompt
- temperature, top-p, seed
- 컨텍스트와 KV 설정
- runtime revision
- tool schema와 timeout
- 평가 입력 순서

품질 외에도 다음을 함께 기록한다.

```csv
model,quant,context,peak_ram_gb,peak_vram_gb,prompt_tps,gen_tps,pass_rate,citation_precision,tool_success
```

## 10.3 연구 재현성 manifest

```yaml
experiment_id: local-llm-research-001
date_utc: 2026-07-20T00:00:00Z
hardware:
  cpu: <model>
  gpu: <model-and-count>
  ram_gb: <value>
  vram_gb: <value>
model:
  repo: <hf-repo>
  revision: <commit-sha>
  quant: <quant-name>
  sha256: <file-sha256>
runtime:
  name: llama.cpp
  revision: <git-sha>
  flags: "-c 8192 -ngl 999"
generation:
  temperature: 0.2
  top_p: 0.95
  seed: 42
tools:
  python: <version>
  lean: <version>
  mathlib: <revision>
data:
  dataset_id: <id>
  revision: <revision>
results:
  artifact_dir: <path>
```

## 10.4 자동 검증 게이트

최종 결과가 다음 조건을 통과하지 않으면 사람에게 넘기지 않는 구조가 유용하다.

```text
코드: compile && test && lint && typecheck
수학: symbolic_check || numerical_cross_check
과학: units_check && source_check && reproducibility_check
인용: every_citation_resolves && every_claim_has_evidence
Lean: lake_build && no_sorry && no_untrusted_axiom
```

## 10.5 모델의 불확실성을 보존하기

모델이 불확실한 부분을 자신 있게 덮어쓰지 않도록 출력 스키마에 다음 필드를 둔다.

- 확인된 사실
- 계산 또는 도구 출력
- 추론·가설
- 미확인 전제
- 필요한 추가 실험
- 반대 근거
- 신뢰 수준

---

# 11. 안전한 로컬 에이전트 운영

로컬 실행은 데이터가 자동으로 안전하다는 뜻이 아니다. 모델이 셸·브라우저·파일·Git·클라우드 자격증명에 접근하면 별도의 보안 경계가 필요하다.

## 11.1 최소 권한

- 읽기 전용 저장소 clone 또는 임시 worktree 사용
- 쓰기 가능한 디렉터리 allowlist
- 홈 디렉터리·SSH key·클라우드 credential 차단
- 컨테이너·VM·샌드박스에서 실행
- root 및 Docker socket 접근 금지
- 네트워크 egress allowlist
- 명령 timeout·CPU·RAM·디스크 quota

## 11.2 로컬 서버 노출

기본적으로 다음처럼 loopback에만 bind한다.

```bash
llama-server \
  -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080
```

외부 접근이 필요하면 reverse proxy, TLS, 인증, rate limit, 요청·도구 audit log를 추가한다. `0.0.0.0`으로 무인증 노출하지 않는다.

## 11.3 프롬프트 인젝션

논문·웹페이지·코드 주석도 공격 입력일 수 있다.

- 검색된 문서의 지시를 system instruction으로 승격하지 않는다.
- 데이터와 명령을 명시적으로 구분한다.
- 도구 호출을 별도 정책 엔진이 승인한다.
- 비밀정보 요청·외부 전송·권한 상승을 차단한다.
- destructive command는 사람 승인 또는 금지 정책을 둔다.

## 11.4 라이선스와 데이터 정책

- 모델 라이선스와 quant 저장소 라이선스를 각각 확인한다.
- 상업적 사용·재배포·파생 모델 조건을 검토한다.
- 연구 데이터의 개인정보·기밀·저작권 정책을 확인한다.
- 모델 출력에 학습 데이터나 사내 비밀이 포함될 가능성을 고려한다.
- 모델 카드의 acceptable use와 제한 사항을 배포 문서에 기록한다.

---

# 12. 초대형·실험 모델

아래 모델은 최신성과 잠재력은 높지만 일반 워크스테이션보다 서버·분산 추론에 가깝다. GGUF가 존재해도 `llama.cpp`의 해당 아키텍처 지원, 채팅 템플릿, tool calling, 멀티모달 projector와 실제 token/s를 먼저 확인한다.

| 모델 | 개요 | 로컬 판단 | 메모리 구간 | Hugging Face |
| --- | --- | --- | --- | --- |
| **MiniMax-M2.5** | 약 229B, 코딩·도구 사용·검색 에이전트 | Q2/Q3/Q4 community GGUF 존재. shard 수와 총합은 배포별로 다르므로 `hf download --dry-run`으로 확인 | 192–256 GB+ 실험 권장 | [공식](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) · [Unsloth GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) · [LM Studio GGUF](https://huggingface.co/lmstudio-community/MiniMax-M2.5-GGUF) |
| **MiniMax-M3** | 약 428B 총/23B 활성, 네이티브 멀티모달, 최대 1M 컨텍스트 | 최신 백엔드의 MSA 지원과 quant 호환성을 우선 확인. 광고 컨텍스트를 로컬 기본값으로 사용하지 말 것 | 256 GB 이상 서버급부터 검토 | [공식](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| **Qwen3.5 122B/397B 계열** | 대형 네이티브 멀티모달·에이전트 | 공식 가중치와 현재 GGUF/MLX 배포의 파일 목록을 직접 확인 | 저비트도 서버급 | [Qwen 조직](https://huggingface.co/Qwen) · [Qwen3.5 소개](https://qwen.ai/blog?id=qwen3.5) |

## 초대형 모델 체크리스트

1. 모든 shard의 총 크기를 합산했는가?
2. 현재 runtime release가 정확한 아키텍처와 tokenizer를 지원하는가?
3. KV 캐시와 병렬 슬롯을 포함한 peak memory를 계산했는가?
4. CPU RAM offload가 발생할 때 허용 가능한 속도인가?
5. NUMA와 multi-GPU tensor split을 실측했는가?
6. SSD에 모델 2–3배의 여유 공간이 있는가?
7. Q2 모델이 실제 작업에서 작은 Q4/Q5보다 나은가?
8. 라이선스와 상업 이용 조건을 검토했는가?
9. 256K/1M 컨텍스트가 실제로 필요한가, RAG로 대체 가능한가?
10. 최신 backend 버그·known issue를 확인했는가?

---

# 13. 주요 출처와 저장소

## 최신 모델 계열

- [Qwen3.6 공식 GitHub](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-27B 공식 모델](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6-35B-A3B 공식 모델](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 소개](https://qwen.ai/blog?id=qwen3.5)
- [Qwen Hugging Face 조직](https://huggingface.co/Qwen)
- [Mistral Medium 3.5 공식 모델](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B)
- [Mistral Medium 3.5 Bartowski GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF)
- [Mistral Small 4 GGUF](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF)
- [Ministral 3 Reasoning 3B](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF)
- [Ministral 3 Reasoning 8B](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512-GGUF)
- [Ministral 3 Reasoning 14B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF)
- [Devstral Small 2 GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF)
- [Qwen3-Coder-Next GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- [gpt-oss-20b 공식 모델](https://huggingface.co/openai/gpt-oss-20b)
- [gpt-oss-120b 공식 모델](https://huggingface.co/openai/gpt-oss-120b)
- [SU-01 GGUF](https://huggingface.co/axi0mX/SU-01-GGUF)

## 형식증명

- [Leanstral 1.5 공식 모델](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B)
- [Leanstral 1.5 발표](https://mistral.ai/news/leanstral-1-5/)
- [DeepSeek-Prover-V2-7B GGUF](https://huggingface.co/unsloth/DeepSeek-Prover-V2-7B-GGUF)
- [Goedel-Prover-V2-8B GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-8B-GGUF)
- [Goedel-Prover-V2-32B GGUF](https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF)
- [OProver 컬렉션](https://huggingface.co/collections/m-a-p/oprover)

## 검색·실행 도구

- [`llama.cpp` 공식 저장소](https://github.com/ggml-org/llama.cpp)
- [Hugging Face CLI 가이드](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Qwen3 Embedding·Reranker](https://huggingface.co/collections/Qwen/qwen3-embedding)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Lean 4](https://github.com/leanprover/lean4)
- [mathlib](https://github.com/leanprover-community/mathlib4)

---

# 최종 선택 체크리스트

- [ ] 장착 메모리와 실제 가용 메모리를 구분했다.
- [ ] 가중치가 총 메모리의 65–75% 이하인 Q4를 우선 검토했다.
- [ ] `hf download --dry-run`으로 모든 shard와 총 크기를 확인했다.
- [ ] 8K 컨텍스트에서 peak RAM/VRAM과 swap을 측정했다.
- [ ] Q2/Q3/Q4를 실제 코드·수학·과학 평가셋으로 비교했다.
- [ ] chat template, reasoning mode, tool-call parser가 runtime과 일치한다.
- [ ] 비전 모델은 projector와 이미지 메모리를 포함했다.
- [ ] 최신 논문·API·표준은 RAG 또는 검색으로 보강한다.
- [ ] 수치 결과는 Python/CAS로, 증명은 Lean으로, 코드는 테스트로 검증한다.
- [ ] 모델·runtime·데이터 revision과 SHA-256을 기록했다.
- [ ] 로컬 서버는 기본적으로 `127.0.0.1`에만 bind했다.
- [ ] 에이전트 파일·셸·네트워크 권한을 최소화했다.
- [ ] 모델과 quant 저장소의 라이선스를 각각 확인했다.

> **결론:** 메모리 용량이 허용하는 가장 큰 모델을 고르는 것이 목표가 아니다. **Q4 이상에서 안정적으로 실행되고, 실제 작업셋에서 검증되며, 검색·계산·컴파일러·테스트와 연결된 모델**이 연구와 프로그래밍에 가장 유용하다.
