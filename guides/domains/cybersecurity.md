# 로컬 AI 모델 선택 가이드
## 버그바운티·사이버보안 연구·학습·프로그래밍용 — RAM/VRAM/Apple 통합 메모리별

> **최종 검증일:** 2026-07-20 (KST)  
> **주요 실행 형식:** GGUF + `llama.cpp`  
> **범위:** 승인된 버그바운티, 사내 보안 점검, CTF/교육, 악성코드 분석 샌드박스, 코드 감사, 보안 자동화 및 일반 프로그래밍

이 문서는 보유한 **시스템 RAM**, **GPU VRAM**, 또는 **Apple Silicon 통합 메모리**만 알아도 적절한 로컬 모델과 양자화를 빠르게 고르고, Hugging Face에서 바로 내려받아 실행할 수 있도록 구성한 실전 가이드다.

모델·GGUF 파일은 계속 갱신된다. 링크된 모델 카드에서 **라이선스, 파일명, SHA/수정일, 지원 백엔드**를 실행 직전에 다시 확인해야 한다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [메모리 용량을 읽는 방법](#2-메모리-용량을-읽는-방법)
3. [Q2·Q3·Q4 양자화 선택법](#3-q2q3q4-양자화-선택법)
4. [모델별 상세 표](#4-모델별-상세-표)
5. [보안·프로그래밍 작업별 추천](#5-보안프로그래밍-작업별-추천)
6. [다운로드 및 실행 명령](#6-다운로드-및-실행-명령)
7. [컨텍스트와 KV 캐시 관리](#7-컨텍스트와-kv-캐시-관리)
8. [안전한 로컬 보안 연구 환경](#8-안전한-로컬-보안-연구-환경)
9. [초대형·실험적 모델](#9-초대형실험적-모델)
10. [제약·검증 체크리스트](#10-제약검증-체크리스트)
11. [주요 출처](#11-주요-출처)

---

# 1. 30초 선택표

아래의 메모리는 **장착된 총 RAM/VRAM/통합 메모리** 기준이다. 운영체제와 다른 프로그램이 사용하는 용량을 빼면 실제 모델에 쓸 수 있는 메모리는 더 적다.

| 장착 메모리 | 우선 추천 모델 | 권장 양자화 | 가중치 파일 크기 | 시작 컨텍스트 | 실전 판단 |
|---:|---|---:|---:|---:|---|
| **4 GB** | [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | Q4_K_M | 0.535 GB | 4K | 학습·간단한 셸/Python 보조. 4 GB 시스템은 프로토타입 수준 |
| **6 GB** | [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | Q4_K_M | 1.29 GB | 4K–8K | 짧은 코드 설명·정규식·로그 요약에 적합 |
| **8 GB** | [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Q4_K_M | 2.78 GB | 8K | 저사양 노트북의 기본 선택. 비전이 필요하면 Gemma 4 E2B도 후보 |
| **12 GB** | [Foundation-Sec-8B](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF) | Q4_K_M | 4.92 GB | 8K | 보안 용어·CVE·위협 인텔리전스·트리아지에 특화 |
| **12 GB** | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Q4_K_M | 5.84 GB | 8K | 일반 코딩·도구 호출·문서 분석 균형형 |
| **16 GB** | [Ministral 3 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF) | Q4_K_M | 8.24 GB | 8K–16K | 추론·코드·문서 질의의 안정적 상한선 |
| **16 GB 최소 / 24 GB 권장** | [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) | Q4_K_M/MXFP4 | 11.6 GB | 8K | 에이전트·추론에 강함. 16 GB는 OS 여유가 매우 작음 |
| **24 GB** | [Devstral Small 2 24B](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) | Q4_K_M | 14.9 GB | 8K–16K | 저장소 수준 코드 수정·소프트웨어 엔지니어링 에이전트 |
| **24 GB** | [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | UD-Q3_K_M | 15.4 GB | 8K | MoE 코딩·프런트엔드·도구 사용. Q4는 32 GB 권장 |
| **32 GB** | [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | UD-Q4_K_M | 18.0 GB | 16K | 현재 중형 로컬 코딩/에이전트의 강력한 기본 선택 |
| **32 GB** | [Gemma 4 31B](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | Q4_K_M | 19.1 GB | 8K–16K | 범용 추론·멀티모달 분석. 영상/이미지는 추가 메모리 필요 |
| **48 GB** | [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | Q3_K_M | 33.3 GB | 8K–16K | 대형 코드베이스·에이전트. Q4 39.2 GB는 48 GB에서 빠듯함 |
| **48 GB** | [Mistral Small 4 119B](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) | IQ2/UD-Q2 | 34.9–40.2 GB | 8K | 품질 저하를 감수한 대형 모델 실험용 |
| **64 GB** | [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | Q4_K_M | 39.2 GB* | 16K | 코드 에이전트의 실용 구간. 저장소별 Q4 크기 차이 확인 필요 |
| **64 GB** | [Mistral Small 4 119B](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) | UD-Q3_K_M | 49.6 GB | 8K | 낮은 컨텍스트에서 가능. 통합 메모리 64 GB는 여유가 작음 |
| **96 GB** | [gpt-oss-120b](https://huggingface.co/unsloth/gpt-oss-120b-GGUF) | Q4_K_M/MXFP4 | 약 62.8 GB | 8K–16K | 네이티브 저비트 MoE. 64 GB 장비보다 80/96 GB급이 안전 |
| **96 GB** | [Mistral Small 4 119B](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) | UD-Q4_K_M | 59.2 GB | 16K | 대형 범용/에이전트 모델의 현실적인 Q4 구간 |
| **96 GB** | [Devstral 2 123B](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF) | Q4 계열 | 75.5–78.5 GB | 8K–16K | 고급 저장소 에이전트. 96 GB에서도 긴 컨텍스트는 주의 |
| **128 GB** | Devstral 2 / Mistral Small 4 | Q4–Q5 | 60–90+ GB | 16K–32K | 더 긴 저장소 컨텍스트와 병렬 슬롯 운용에 적합 |
| **192 GB** | [DeepSeek-V4-Flash](https://huggingface.co/Preyazz/DeepSeek-V4-Flash-GGUF) | Q3/Q4 | 약 125/161 GB | 8K–16K | 최신 아키텍처 지원 여부를 먼저 확인하는 실험 구간 |
| **256 GB 이상** | V4 Flash 고품질 양자화·다중 모델 | Q4–Q6 | 모델별 상이 | 측정 후 확대 | 서버급. NUMA·메모리 대역폭·스토리지 속도가 중요 |
| **768 GB–1 TB+** | DeepSeek V4 Pro·초대형 MoE | 저비트부터 | 수백 GB | 4K부터 | 연구용. 전용 포크/백엔드 제약과 매우 긴 로딩 시간을 감수 |

\* Qwen3-Coder-Next Q4는 저장소와 동적 양자화 방식에 따라 약 **39.2 GB에서 48.4 GB**까지 차이가 난다. 파일명을 보고 판단해야 한다.

## 용도별 한 줄 결론

- **보안 도메인 지식·CVE 트리아지:** Foundation-Sec-8B Q4
- **저사양 범용 코딩:** Qwen3.5-4B 또는 9B Q4
- **24–32 GB 저장소 에이전트:** Devstral Small 2 24B Q4 또는 Qwen3.6-35B-A3B Q4
- **48–64 GB 대형 코드 에이전트:** Qwen3-Coder-Next Q3/Q4
- **96 GB급 고급 추론·에이전트:** gpt-oss-120b 또는 Mistral Small 4 119B Q4
- **한 모델만 고를 때:** 메모리에 들어가는 가장 큰 모델보다 **Q4에서 충분한 여유가 남는 모델**이 일반적으로 더 안정적이다.

---

# 2. 메모리 용량을 읽는 방법

## 2.1 표의 파일 크기와 실제 필요 메모리는 다르다

표에 적힌 Q2/Q3/Q4 크기는 대부분 **GGUF 가중치 파일 자체의 크기**다. 실제 실행에는 다음이 더해진다.

```text
실제 필요 메모리
≈ GGUF 가중치
+ KV 캐시
+ 계산/그래프/토크나이저/백엔드 버퍼
+ 비전·오디오 projector 및 입력 버퍼
+ 운영체제와 다른 프로그램의 점유량
```

실전 여유분의 보수적 기준:

| 장착 메모리 | OS·앱에 남길 최소 여유 | 모델 선택 원칙 |
|---:|---:|---|
| 4–8 GB | 2–4 GB | 0.8B–4B, 4K–8K 컨텍스트 |
| 12–16 GB | 3–6 GB | 8B–14B Q4 또는 20B급 특수 저비트 |
| 24–32 GB | 5–8 GB | 24B–35B Q3/Q4 |
| 48–64 GB | 8–12 GB | 80B MoE Q3/Q4 또는 119B Q2/Q3 |
| 96–128 GB | 12–20 GB | 119B–125B Q4, 긴 컨텍스트는 실측 |
| 192 GB 이상 | 20 GB 이상 | 모델 버퍼·NUMA·병렬 슬롯까지 별도 산정 |

이 값은 운영체제, 디스플레이 공유 메모리, 백엔드, 컨텍스트, 배치 크기와 동시 요청 수에 따라 달라지는 **실무용 휴리스틱**이다.

## 2.2 NVIDIA/AMD 전용 GPU

- 완전 GPU 오프로딩은 **가중치 + KV 캐시 + GPU 버퍼가 VRAM에 모두 들어갈 때** 가장 빠르다.
- 일부 레이어를 시스템 RAM에 두는 하이브리드 오프로딩은 가능하지만 PCIe 전송 때문에 크게 느려질 수 있다.
- `VRAM 16 GB + RAM 32 GB = VRAM 48 GB 성능`이 아니다. 용량 측면의 우회일 뿐이다.
- 여러 GPU를 사용할 경우 백엔드와 모델 구조에 따라 텐서 분할 효율이 다르다.

## 2.3 Apple Silicon 통합 메모리

- CPU와 GPU가 같은 메모리 풀을 공유하므로 모델 배치가 단순하고 대형 GGUF 운용에 유리하다.
- 표시된 32/64/96/128 GB 전체를 모델이 쓸 수 있는 것은 아니다. macOS, WindowServer, 브라우저, IDE가 함께 사용한다.
- 메모리 압축과 스왑이 발생하면 “실행은 되지만 토큰 생성 속도가 급감”할 수 있다.
- 대형 모델은 SSD 여유 공간과 쓰기 수명도 고려해야 한다.

## 2.4 CPU 전용 서버

- 모델이 RAM에 들어가도 메모리 대역폭이 낮으면 생성 속도가 제한된다.
- 채널 수, DDR 세대, NUMA 배치, CPU 벡터 명령 지원이 중요하다.
- 100B+ 모델은 용량보다 **메모리 대역폭과 NUMA 교차 접근**이 병목인 경우가 많다.

## 2.5 MoE의 “활성 파라미터”를 오해하지 않기

`35B-A3B`, `80B-A3B`, `117B-A5B` 같은 표기는 토큰당 계산에 주로 참여하는 활성 파라미터가 작다는 의미다. 그러나 일반적인 로컬 추론에서는 **전체 전문가 가중치가 메모리 또는 저장장치에서 준비되어야 한다.**

따라서 80B-A3B를 3B 모델처럼 4 GB 장비에서 실행할 수 있다고 판단하면 안 된다. 활성 파라미터는 주로 계산량과 속도에 영향을 주고, GGUF 파일 크기는 전체 저장 파라미터에 가깝다.

---

# 3. Q2·Q3·Q4 양자화 선택법

## 3.1 실전 우선순위

| 양자화 | 권장도 | 품질/용량 특성 | 적합한 작업 |
|---|---:|---|---|
| **Q4_K_M / UD-Q4_K_M** | 기본값 | 품질과 용량의 균형이 가장 좋음 | 코드 생성, 코드 리뷰, 보안 분석, 에이전트 |
| **Q3_K_M / UD-Q3_K_M** | 메모리 부족 시 | Q4보다 오류·누락 가능성이 증가 | 트리아지, 요약, 제한된 코드 작업 |
| **Q2_K / IQ2 / UD-Q2** | 최후 수단 | 큰 품질 저하 가능. 모델 규모 이점이 일부 상쇄 | 대형 모델 기능 확인, 분류·초안·실험 |
| **Q5_K_M / Q6_K** | 메모리 여유 시 | 코드 정확도와 지시 이행 안정성이 개선되는 경우가 많음 | 정밀 코드 감사, 장기 에이전트, 중요 분석 |
| **Q8_0 / BF16** | 서버급 | 원본에 가까우나 메모리와 대역폭 요구가 큼 | 평가 기준선, 재현성·품질 우선 |

## 3.2 이름에 붙는 접미사

- `K_S`, `K_M`, `K_L`: 같은 비트 계열에서도 블록·텐서별 배분이 다르다. 일반적으로 `K_M`이 무난하다.
- `IQ2_XXS`, `IQ2_XS`: 극저비트 중요도 기반 양자화. 파일은 작지만 지원 백엔드와 품질을 반드시 검증한다.
- `UD-Qx_K_*`, `Qx_K_XL`: 텐서별 동적/혼합 양자화다. 이름의 숫자만으로 파일 크기를 추정하면 안 된다.
- `imatrix`: 중요도 행렬을 사용해 양자화 오차를 줄인 배포다. 같은 크기라면 신뢰할 수 있는 imatrix 빌드를 선호할 수 있다.

## 3.3 기본 선택 규칙

1. 먼저 **Q4_K_M**이 가중치와 8K–16K KV 캐시를 포함해 들어가는지 본다.
2. Q4가 간당간당하면 컨텍스트를 먼저 줄이고, 그래도 안 되면 Q3_K_M으로 내린다.
3. Q2를 선택하기 전에 한 단계 작은 모델의 Q4와 비교한다. 코딩·보안 분석은 작은 Q4가 큰 Q2보다 안정적인 경우가 많다.
4. 메모리가 충분하면 Q5/Q6로 올리되, 실제 평가셋과 저장소 작업으로 개선 폭을 측정한다.
5. 같은 양자화명이라도 저장소별 파일 크기가 다를 수 있으므로 **표의 숫자보다 실제 파일 목록을 최종 기준**으로 삼는다.

## 3.4 `gpt-oss` 예외

`gpt-oss-20b`와 `gpt-oss-120b`의 전문가 가중치는 네이티브 MXFP4 형태를 사용한다. 일부 GGUF 배포에서 Q2, Q3, Q4 파일 크기가 거의 동일하다.

- gpt-oss-20b: 약 11.5 / 11.5 / 11.6 GB
- gpt-oss-120b: 약 62.6 / 62.6 / 62.8 GB

따라서 `gpt-oss`는 Q2로 내려도 일반 모델처럼 큰 메모리 절감이 발생하지 않는다. 품질과 백엔드 호환성을 보고 Q4/MXFP4 계열을 선택하는 편이 합리적이다.

---

# 4. 모델별 상세 표

## 표 읽기

- 크기는 Hugging Face 모델 카드 또는 파일 목록에서 확인한 **십진 GB 기준의 대략적인 가중치 크기**다.
- `공식 미제공`은 해당 공식 GGUF 저장소에서 그 양자화를 제공하지 않는다는 뜻이다. 다른 커뮤니티 저장소에는 존재할 수 있다.
- `UD`, `XL`, `IQ` 등 변형에 따라 파일 크기가 달라진다. 표에는 비교에 유용한 대표값을 넣었다.
- 멀티모달 모델의 `mmproj`/projector 파일과 이미지·오디오 처리 버퍼는 포함하지 않았다.
- “권장 총 메모리”는 모델 가중치만의 크기가 아니라, OS·KV 캐시·런타임 여유를 고려한 보수적 구간이다.

## 4.1 소형·중형: 0.8B–14B

| 모델 | 구조·주용도 | Q2 대표 크기 | Q3 대표 크기 | Q4 대표 크기 | 권장 총 메모리 | GGUF / 모델 카드 |
|---|---|---:|---:|---:|---:|---|
| **Qwen3.5-0.8B** | 초경량 범용·코딩·다국어·비전 | 0.418 GB | 0.470 GB | **0.535 GB** | 4 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) |
| **Qwen3.5-2B** | 저사양 코딩·로그 요약·도구 보조 | 0.967 GB | 1.11 GB | **1.29 GB** | 4–6 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) |
| **Ministral 3 3B Reasoning** | 소형 추론·비전·온디바이스 | 공식 미제공 | 공식 미제공 | **2.15 GB** | 6–8 GB | [Mistral 공식 GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF) |
| **Qwen3.5-4B** | 저사양 범용 코딩·비전·에이전트 입문 | 1.94 GB | 2.11 GB | **2.78 GB** | 8 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| **Gemma 4 E2B** | 약 5B 저장 파라미터, 멀티모달·범용 | 2.40 GB | 2.45 GB | **3.15 GB** | 8–12 GB | [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| **Foundation-Sec-8B Reasoning** | 사이버보안 특화: CVE·위협 인텔·익스플로잇 문서·컴플라이언스 | 3.18 GB | 4.02 GB | **4.92 GB** | 12–16 GB | [공식 Q4](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF) · [다중 양자화](https://huggingface.co/mradermacher/Foundation-Sec-8B-Reasoning-GGUF) · [기본 모델](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning) |
| **Gemma 4 E4B** | 약 8B 저장 파라미터, 멀티모달·범용 | 3.76 GB | 3.86 GB | **5.07 GB** | 12–16 GB | [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| **Ministral 3 8B Reasoning** | 추론·비전·엣지 배포 | 공식 미제공 | 공식 미제공 | **5.20 GB** | 12–16 GB | [Mistral 공식 GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512-GGUF) |
| **Qwen3.5-9B** | 범용 코딩·도구 사용·문서 분석 | 4.12 GB | 4.32 GB | **5.84 GB** | 12–16 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| **Gemma 4 12B** | 멀티모달·추론·문서 분석 | 4.66 GB | 5.14 GB | **7.40 GB** | 16–24 GB | [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |
| **Ministral 3 14B Reasoning** | 중형 추론·코드·비전 | 공식 미제공 | 공식 미제공 | **8.24 GB** | 16–24 GB | [Mistral 공식 GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF) |

### 이 구간의 추천

- **8 GB 이하:** Qwen3.5-4B Q4를 기준선으로 사용한다.
- **12–16 GB 보안 연구:** Foundation-Sec-8B Q4로 도메인 질의·트리아지를 하고, Qwen3.5-9B Q4로 일반 코딩을 보완한다.
- **16 GB 단일 모델:** Ministral 3 14B Q4가 균형적이다. 멀티모달 기능을 쓰면 컨텍스트를 줄인다.

## 4.2 중대형: 20B–35B

| 모델 | 구조·주용도 | Q2 대표 크기 | Q3 대표 크기 | Q4 대표 크기 | 권장 총 메모리 | GGUF / 모델 카드 |
|---|---|---:|---:|---:|---:|---|
| **gpt-oss-20b** | 약 21B 총/3.6B 활성 MoE, 추론·에이전트 | 11.5 GB | 11.5 GB | **11.6 GB** | 16 GB 최소, 24 GB 권장 | [Unsloth GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) · [OpenAI 기본 모델](https://huggingface.co/openai/gpt-oss-20b) |
| **Devstral Small 2 24B** | 저장소 수준 소프트웨어 엔지니어링·코드 에이전트·비전 | 8.89 GB | 10.4 GB | **14.9 GB** | Q3 16–24 GB, Q4 24–32 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |
| **Gemma 4 26B-A4B** | 약 25B 저장/4B 활성 MoE, 멀티모달 | 10.5 GB | 11.3 GB | **13.6 GB** | 24–32 GB | [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| **Gemma 4 31B** | 대형 멀티모달·추론·문서 분석 | 11.8 GB | 13.2 GB | **19.1 GB** | Q3 24 GB, Q4 32 GB | [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| **Qwen3.6-27B** | 범용·코딩·에이전트 | 공식 저장소에서 확인 필요 | 공식 저장소에서 확인 필요 | **19.1 GB** | 32 GB | [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF) |
| **Qwen3.6-35B-A3B** | 35B 총/3B 활성 MoE, 프런트엔드·코딩·도구 사용·저장소 추론 | 12.3 GB | 15.4 GB | **18.0 GB** | Q3 24 GB, Q4 32 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |

### 이 구간의 추천

- **저장소 자동 수정:** Devstral Small 2 24B Q4
- **코딩 + 일반 추론 + 도구 사용:** Qwen3.6-35B-A3B Q4
- **추론/에이전트와 16 GB 최소 장비:** gpt-oss-20b. 단, 16 GB 통합 메모리에서는 브라우저·IDE를 정리하고 8K 컨텍스트부터 시작한다.
- **스크린샷·UI·문서 이미지 분석:** Gemma 4 또는 Devstral 비전 기능. projector와 이미지 버퍼 용량을 별도로 남긴다.

## 4.3 대형: 80B–125B

| 모델 | 구조·주용도 | Q2 대표 크기 | Q3 대표 크기 | Q4 대표 크기 | 권장 총 메모리 | GGUF / 모델 카드 |
|---|---|---:|---:|---:|---:|---|
| **Qwen3-Coder-Next** | 80B 총/3B 활성 MoE, 256K, 코딩 에이전트·로컬 개발 | 23.3 GB Q2_K; XL은 약 26.8–29.3 GB | **33.3 GB** | **39.2 GB**, 다른 배포는 약 48.4 GB | Q3 48 GB, Q4 64 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) · [Qwen 공식 GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) |
| **gpt-oss-120b** | 약 117B 총/5.1B 활성 MoE, 추론·에이전트 | 62.6 GB | 62.6 GB | **약 62.8 GB** | 80–96 GB 이상 | [Unsloth GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF) · [OpenAI 기본 모델](https://huggingface.co/openai/gpt-oss-120b) |
| **Mistral Small 4 119B** | 대형 범용·추론·에이전트·멀티모달 | 34.9 GB IQ2 또는 40.2 GB Q2_K_XL | **49.6 GB** | **59.2 GB** | Q3 64 GB, Q4 96 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) |
| **Devstral 2 123B** | 모델 카드 표기 약 123/125B급, 대형 코드·저장소 에이전트 | 34.4 GB Q2_K; XL 47.9 GB | **54.4 GB** | **75.5–78.5 GB** | Q3 64–96 GB, Q4 96–128 GB | [Unsloth GGUF](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF) |

### 이 구간의 추천

- **48 GB:** Qwen3-Coder-Next Q3가 가장 실용적이다. 119B Q2는 모델 규모는 크지만 코딩 정확도 저하와 느린 속도를 감수해야 한다.
- **64 GB:** Qwen3-Coder-Next Q4 또는 Mistral Small 4 Q3. Apple 64 GB에서 60 GB급 가중치는 OS 여유가 없어 비추천이다.
- **96 GB:** gpt-oss-120b, Mistral Small 4 Q4, Devstral 2 Q4가 현실적인 상위 선택이다.
- **128 GB:** 긴 컨텍스트, 여러 동시 슬롯, Q5급 양자화에 유리하다.

---

# 5. 보안·프로그래밍 작업별 추천

## 5.1 버그바운티 프로그램 정책·스코프 분석

**추천:** Foundation-Sec-8B Q4 + 로컬 RAG

입력 자료:

- 프로그램 정책과 인스코프 자산
- 허용/금지 테스트 방식
- 속도 제한, 자동화 제한, 데이터 취급 조건
- 과거 공개 보고서와 중복 규칙

모델이 정책 문구를 임의로 확대 해석하지 않도록, 모든 결론에 **근거 문단과 원문 위치**를 함께 출력하게 한다.

## 5.2 CVE·어드바이저리·위협 인텔리전스 트리아지

**추천:** Foundation-Sec-8B Q4 또는 Qwen3.5-9B Q4

정적 모델 가중치는 최신 CVE를 알지 못하거나 내용을 혼동할 수 있다. NVD, GitHub Security Advisories, OSV, 벤더 권고문, 사내 SBOM을 로컬 인덱스로 연결하고 다음을 구조화한다.

```text
CVE/권고 ID
영향 버전
공격 전제조건
공격 벡터
실제 사용 중인 컴포넌트와 도달 가능성
완화책과 패치
검증 상태 및 출처
```

## 5.3 코드 감사·취약점 후보 탐색

**추천 순서:**

1. 24–32 GB: Devstral Small 2 24B Q4 또는 Qwen3.6-35B-A3B Q4
2. 48–64 GB: Qwen3-Coder-Next Q3/Q4
3. 96 GB 이상: Devstral 2 또는 Mistral Small 4 Q4

권장 워크플로:

1. 저장소 구조와 신뢰 경계를 먼저 요약한다.
2. 입력 지점 → 변환 → 권한 검사 → 위험 sink의 데이터 흐름을 추적한다.
3. 각 후보에 파일·함수·라인·전제조건·반증 조건을 기록한다.
4. 정적 분석기, 테스트, 재현 코드로 검증한다.
5. 검증되지 않은 결과는 “취약점”이 아니라 “가설”로 표시한다.

## 5.4 PoC·회귀 테스트·패치 작성

**추천:** Devstral Small 2 / Qwen3.6-35B-A3B / Qwen3-Coder-Next

모델에게 한 번에 공격 코드 전체를 맡기기보다 다음 단위로 분리하면 결과가 안정적이다.

- 최소 재현 테스트
- 실패 전제조건과 안전한 입력
- 패치 후보
- 패치 전후 테스트
- 부작용 및 호환성 검토

실제 실행은 격리된 환경에서 사람이 승인한 명령만 수행한다.

## 5.5 악성코드·의심 파일 분석

**추천:** 코드 모델 + 격리 VM + 정적 분석 도구

- 샘플을 모델 서버나 일반 개발 PC에서 직접 실행하지 않는다.
- 문자열, import, 섹션, 디컴파일 코드, 샌드박스 로그를 비실행 데이터로 전달한다.
- 모델 출력은 IOC 후보와 행위 가설로 취급하고 도구로 검증한다.
- 샘플 경로, 해시, 분석 시각, 도구 버전을 기록한다.

## 5.6 프런트엔드·API 보안

**추천:** Qwen3.6-35B-A3B 또는 Qwen3-Coder-Next

검토 범위:

- DOM XSS와 템플릿 이스케이프
- CSP·CORS·CSRF·SameSite
- OAuth/OIDC redirect, PKCE, state/nonce
- 클라이언트 측 권한 검사 오용
- GraphQL/REST BOLA·IDOR
- SSRF로 이어지는 URL 처리
- 업로드·다운로드 경로와 콘텐츠 타입
- 빌드 도구·의존성·소스맵·비밀정보 노출

## 5.7 두 모델을 역할 분담하는 구성

메모리가 허용하면 한 모델에 모든 역할을 맡기기보다 다음 조합이 효율적이다.

| 역할 | 추천 모델 | 출력 |
|---|---|---|
| 보안 도메인 트리아지 | Foundation-Sec-8B Q4 | CWE/CVE 맥락, 위협 모델, 검증 체크리스트 |
| 코드베이스 에이전트 | Devstral Small 2 또는 Qwen3-Coder-Next | 파일 탐색, 패치, 테스트, 리팩터링 |
| 최종 검증 | 정적/동적 분석 도구 + 사람 | 재현 가능성, 오탐 제거, 정책 준수 |

모델 두 개를 동시에 메모리에 올리기 어렵다면 순차 실행하고, JSON/Markdown 산출물을 다음 단계 입력으로 넘긴다.

---

# 6. 다운로드 및 실행 명령

## 6.1 `llama.cpp` 설치

### macOS / Linux

```bash
curl -LsSf https://llama.app/install.sh | sh
```

또는 소스에서 최신 버전을 빌드한다.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j
```

Apple Silicon에서 Metal을 명시하려면 빌드 환경에 따라 `-DGGML_METAL=ON`을 사용한다. CUDA/ROCm/Vulkan 옵션은 최신 `llama.cpp` 빌드 문서를 확인한다.

### Windows

```powershell
winget install llama.cpp
```

설치 방식에 따라 실행 파일 이름은 `llama-cli`/`llama-server` 또는 래퍼 명령인 `llama cli`/`llama serve`일 수 있다.

## 6.2 Hugging Face에서 자동 다운로드하며 실행

### 범용 템플릿

```bash
llama-cli \
  -hf OWNER/REPOSITORY:QUANT \
  -c 8192 \
  -ngl 99 \
  --jinja
```

- `-hf`: Hugging Face 저장소와 양자화를 지정하고 필요한 GGUF 샤드를 자동 다운로드한다.
- `-c 8192`: 먼저 8K 컨텍스트로 시작한다.
- `-ngl 99`: 가능한 레이어를 GPU/Metal에 오프로딩한다.
- CPU 전용은 `-ngl 0`을 사용한다.
- 모델 카드가 다른 채팅 템플릿 옵션을 요구하면 해당 지침을 우선한다.

### Qwen3.5-4B — 8 GB급

```bash
llama-cli \
  -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 99 \
  --jinja
```

### Foundation-Sec-8B — 12–16 GB급 보안 특화

```bash
llama-cli \
  -hf fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 99 \
  --jinja
```

다중 양자화 저장소를 사용할 때:

```bash
llama-cli \
  -hf mradermacher/Foundation-Sec-8B-Reasoning-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 99 \
  --jinja
```

### gpt-oss-20b — 16 GB 최소, 24 GB 권장

```bash
llama-cli \
  -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M \
  -c 8192 \
  -ngl 99 \
  --jinja
```

### Devstral Small 2 24B — 24–32 GB급

```bash
llama-cli \
  -hf unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:Q4_K_M \
  -c 16384 \
  -ngl 99 \
  --jinja
```

### Qwen3.6-35B-A3B — 24 GB Q3 / 32 GB Q4

```bash
llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  -c 16384 \
  -ngl 99 \
  --jinja
```

### Qwen3-Coder-Next — 48 GB Q3 / 64 GB Q4

```bash
llama-cli \
  -hf unsloth/Qwen3-Coder-Next-GGUF:Q3_K_M \
  -c 16384 \
  -ngl 99 \
  --jinja
```

## 6.3 로컬 OpenAI 호환 서버

```bash
llama-server \
  -hf unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:Q4_K_M \
  -c 16384 \
  -ngl 99 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

기본적으로 `127.0.0.1`에만 바인딩한다. 외부 공개가 필요하면 TLS, 인증, 방화벽, 요청 제한, 감사 로그를 먼저 구성한다.

예시 요청:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [
      {
        "role": "system",
        "content": "You assist only with authorized security testing. Cite file paths and mark unverified findings as hypotheses."
      },
      {
        "role": "user",
        "content": "Review the supplied code for authorization boundary failures."
      }
    ],
    "temperature": 0.2
  }'
```

## 6.4 파일만 미리 다운로드

```bash
python -m pip install -U "huggingface_hub[cli]"
```

단일 파일 모델:

```bash
hf download unsloth/Qwen3.5-4B-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir ./models/qwen3.5-4b-q4
```

샤드 모델:

```bash
hf download unsloth/Qwen3-Coder-Next-GGUF \
  --include "*Q3_K_M*.gguf" \
  --local-dir ./models/qwen3-coder-next-q3
```

샤드 파일은 `00001-of-0000N` 중 하나만 받으면 실행되지 않는다. 수동 다운로드보다 `llama-cli -hf ...`가 안전하다.

## 6.5 이미 내려받은 GGUF 실행

```bash
llama-cli \
  -m ./models/model-q4_k_m.gguf \
  -c 8192 \
  -ngl 99 \
  --jinja
```

멀티모달 모델이 별도 projector를 요구하면 모델 카드에 따라 `--mmproj` 등을 추가한다.

## 6.6 KV 캐시도 양자화하기

메모리가 부족한 경우, 지원되는 모델·백엔드에서 다음을 시험한다.

```bash
llama-cli \
  -m ./models/model.gguf \
  -c 16384 \
  -ngl 99 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja
```

더 낮은 KV 양자화는 메모리를 줄이지만 품질, 속도 또는 백엔드 호환성에 영향을 줄 수 있다. 반드시 실제 보안/코딩 평가로 검증한다.

---

# 7. 컨텍스트와 KV 캐시 관리

## 7.1 최대 컨텍스트는 기본값이 아니다

일부 최신 모델은 256K 또는 1M 토큰 컨텍스트를 표방하지만, 최대치를 그대로 지정하면 KV 캐시와 프롬프트 처리 시간이 크게 증가한다.

권장 시작값:

| 작업 | 시작 컨텍스트 | 확장 기준 |
|---|---:|---|
| 짧은 로그·함수 리뷰 | 4K–8K | 잘림이 확인될 때만 확대 |
| 파일 여러 개의 코드 감사 | 8K–16K | 검색/RAG로 필요한 코드만 선택 |
| 저장소 에이전트 | 16K–32K | KV 메모리와 프리필 시간을 측정 |
| 대규모 문서/CVE 코퍼스 | 전체 투입 금지 | RAG·요약 계층·청크 검색 사용 |

## 7.2 KV 캐시의 개념적 계산

일반적인 어텐션 모델에서 KV 캐시는 대략 다음 요소에 비례한다.

```text
KV 메모리 ∝ 레이어 수 × 컨텍스트 토큰 수 × KV 헤드 수 × 헤드 차원 × 데이터 타입 크기
```

GQA, MLA, 슬라이딩 윈도, 하이브리드 어텐션 등 구조에 따라 실제 값은 크게 다르다. 가장 신뢰할 수 있는 방법은 모델을 원하는 `-c` 값으로 로드하고 `llama.cpp` 시작 로그의 KV 캐시 할당량을 확인하는 것이다.

## 7.3 메모리가 부족할 때의 조정 순서

1. 동시 요청/parallel slot 수를 1로 줄인다.
2. 컨텍스트를 32K → 16K → 8K로 줄인다.
3. 비전 projector와 불필요한 모달리티를 로드하지 않는다.
4. KV 캐시를 Q8/Q4 계열로 양자화한다.
5. Q4 → Q3로 내린다.
6. 그래도 부족하면 한 단계 작은 모델의 Q4로 바꾼다.
7. 하이브리드 CPU 오프로딩은 마지막 수단으로 사용한다.

---

# 8. 안전한 로컬 보안 연구 환경

## 8.1 승인 범위 고정

모델의 시스템 프롬프트와 에이전트 설정에 다음을 명시한다.

- 승인된 프로그램/조직/자산만 테스트
- 인스코프 도메인·IP·저장소·브랜치 목록
- 금지 행위: 파괴적 테스트, 데이터 대량 접근, 지속성, 소셜 엔지니어링 등
- 요청 속도와 시간대 제한
- 외부 전송이 금지된 데이터 유형
- 결과를 “가설 / 재현됨 / 반증됨”으로 구분

## 8.2 에이전트 도구 실행 격리

권장 구성:

```text
호스트 OS
└─ 로컬 모델 서버: 127.0.0.1 전용
   └─ 에이전트 오케스트레이터
      └─ 일회성 VM/컨테이너
         ├─ 읽기 전용 소스 마운트
         ├─ 쓰기 가능한 별도 작업 디렉터리
         ├─ 비밀정보 미주입
         ├─ 기본 네트워크 차단 또는 허용 목록
         └─ 명령 승인·감사 로그
```

핵심 원칙:

- 저장소의 `.env`, SSH 키, 클라우드 자격증명, 브라우저 프로필을 컨테이너에 넣지 않는다.
- 모델이 제안한 셸 명령은 신뢰하지 않고 승인 게이트를 둔다.
- 대상 네트워크가 필요한 경우 인스코프 자산만 egress allowlist에 넣는다.
- 위험한 PoC는 별도 스냅샷 VM에서 실행한다.
- 모델 서버를 `0.0.0.0`에 무인증으로 공개하지 않는다.
- 프롬프트, 도구 호출, 출력, 파일 변경 diff를 기록한다.

## 8.3 최신 취약점 데이터 연결

로컬 모델 자체는 배포 시점 이후 정보를 자동으로 알지 못한다. 다음 소스를 주기적으로 동기화한 로컬 RAG를 권장한다.

- NVD / CVE JSON
- GitHub Security Advisories
- OSV
- CISA KEV
- 벤더 보안 권고문
- 사내 SBOM, 자산 목록, 패치 상태
- 버그바운티 프로그램 정책과 공개 보고서

모델 응답에는 출처 ID, 문서 날짜, 영향 버전, 검색 시각을 포함시킨다.

## 8.4 결과 검증 기준

보안 모델의 출력은 다음 중 하나로 라벨링한다.

| 상태 | 의미 |
|---|---|
| `HYPOTHESIS` | 코드상 가능성이 있으나 재현되지 않음 |
| `REPRODUCED` | 통제된 환경에서 최소 재현됨 |
| `NOT_REPRODUCED` | 현재 전제조건에서 재현 실패 |
| `MITIGATED` | 방어 설정 또는 패치로 차단됨 |
| `OUT_OF_SCOPE` | 정책·자산 범위를 벗어남 |
| `NEEDS_HUMAN_REVIEW` | 법적·정책적·업무 영향 판단이 필요 |

---

# 9. 초대형·실험적 모델

이 구간은 일반 워크스테이션보다 서버·연구 환경을 대상으로 한다. 최신 아키텍처는 기본 `llama.cpp` 릴리스에서 즉시 지원되지 않거나 전용 포크가 필요할 수 있다.

| 모델 | 알려진 구조/특성 | 커뮤니티 GGUF 대표 크기 | 현실적 메모리 | 링크·주의사항 |
|---|---|---:|---:|---|
| **DeepSeek-V4-Flash** | 약 284B 총/13B 활성 MoE, 최대 1M 컨텍스트 계열 | Q2_K 약 96 GB, Q3_K_M 약 125 GB, Q4_K_M 약 161 GB | Q2 128 GB, Q3 192 GB, Q4 192–256 GB | [공식 기본 모델](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) · [커뮤니티 GGUF](https://huggingface.co/Preyazz/DeepSeek-V4-Flash-GGUF) · 백엔드 지원 확인 |
| **GLM-5.2** | 초대형 장문·에이전트 계열, 1M 컨텍스트 배포 존재 | 저장소 샤드 합계 확인 | 저비트도 384–512 GB 이상을 예상하고 실측 | [Unsloth GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF) |
| **Kimi-K2.7-Code** | 장기 코드 에이전트·도구 사용 계열 | 저장소 샤드 합계 확인 | 384–768 GB+ 실험 구간 | [Unsloth GGUF](https://huggingface.co/unsloth/Kimi-K2.7-Code-GGUF) |
| **DeepSeek-V4-Pro** | 약 1.6T 총/49B 활성급 초대형 MoE 계열 | 커뮤니티 Q2_K-XL이 약 535 GiB로 보고됨 | 768 GB–1 TB+ 권장 | [공식 기본 모델](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) · [커뮤니티 GGUF](https://huggingface.co/teamblobfish/DeepSeek-V4-Pro-GGUF) · 특정 포크/Metal·CPU 제약 확인 |

### 초대형 모델에서 반드시 확인할 것

1. 필요한 `llama.cpp` 커밋 또는 전용 포크
2. CUDA/ROCm/Metal/Vulkan/CPU 중 실제 지원 백엔드
3. 샤드 전체 다운로드 크기와 SSD 여유 공간
4. 모델 로드 시 피크 메모리
5. NUMA 인터리브와 스레드 고정
6. 4K 컨텍스트, 단일 슬롯에서 먼저 성공한 뒤 확대
7. 일부 전문가를 SSD에서 페이징할 때의 급격한 속도 저하

초대형 최신 모델은 저장소가 빠르게 바뀌므로, 위 숫자는 구매 결정의 확정치가 아니라 **초기 용량 산정용**으로만 사용한다.

---

# 10. 제약·검증 체크리스트

## 모델을 고르기 전

- [ ] 실제 사용 가능한 RAM/VRAM/통합 메모리를 확인했다.
- [ ] 운영체제와 IDE/브라우저에 남길 여유를 뺐다.
- [ ] Q4 가중치와 목표 컨텍스트의 KV 캐시가 함께 들어가는지 계산했다.
- [ ] 모델 라이선스가 개인·연구·상업·서비스 목적에 맞는지 확인했다.
- [ ] GGUF 저장소의 작성자, 모델 카드, 파일 해시, 최근 수정일을 확인했다.
- [ ] 최신 `llama.cpp`가 해당 아키텍처와 멀티모달 projector를 지원하는지 확인했다.

## 모델을 실행한 뒤

- [ ] 시작 로그에서 가중치·KV·계산 버퍼 할당량을 기록했다.
- [ ] 스왑 또는 과도한 unified-memory pressure가 없는지 확인했다.
- [ ] 실제 코드 감사·보안 질의 20–50개로 Q3와 Q4를 비교했다.
- [ ] 허위 CVE, 존재하지 않는 함수/파일, 잘못된 패치가 발생하는지 측정했다.
- [ ] 모델 출력이 자동 실행되기 전에 사람 승인 또는 정책 엔진을 거친다.
- [ ] 서버는 기본적으로 localhost에만 바인딩한다.

## 표의 한계

- 파일 크기는 같은 양자화명이라도 저장소별로 달라질 수 있다.
- 모델 카드의 최대 컨텍스트는 현재 하드웨어에서 실용적이라는 의미가 아니다.
- Vision/audio 모델은 별도 projector와 입력 버퍼가 필요할 수 있다.
- CPU·GPU 혼합 오프로딩은 용량을 늘리지만 속도는 크게 떨어질 수 있다.
- 보안 특화 모델이 최신 범용 코딩 모델보다 항상 우수한 것은 아니다. 도메인 트리아지와 코드 에이전트를 분리해 평가한다.
- 모델은 사실 확인 도구가 아니다. CVE·패치·법적 스코프는 원문 출처로 검증한다.

---

# 11. 주요 출처

## 실행 엔진·다운로드

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Hugging Face Hub CLI 문서](https://huggingface.co/docs/huggingface_hub/guides/cli)

## 소형·중형 GGUF

- [Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- [Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
- [Ministral 3 3B Reasoning GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF)
- [Ministral 3 8B Reasoning GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512-GGUF)
- [Ministral 3 14B Reasoning GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF)
- [Foundation-Sec-8B-Reasoning](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning)
- [Foundation-Sec-8B Q4_K_M GGUF](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF)
- [Foundation-Sec-8B 다중 양자화 GGUF](https://huggingface.co/mradermacher/Foundation-Sec-8B-Reasoning-GGUF)
- [Gemma 4 E2B GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- [Gemma 4 E4B GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
- [Gemma 4 12B GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [Gemma 4 26B-A4B GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [Gemma 4 31B GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

## 코딩·에이전트·대형 GGUF

- [gpt-oss-20b GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
- [gpt-oss-120b GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)
- [OpenAI gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [OpenAI gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [Devstral Small 2 24B GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF)
- [Qwen3.6-27B GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF)
- [Qwen3.6-35B-A3B GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [Qwen3-Coder-Next GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- [Qwen 공식 Qwen3-Coder-Next GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF)
- [Mistral Small 4 119B GGUF](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF)
- [Devstral 2 123B GGUF](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF)

## 초대형·실험적

- [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek-V4-Flash 커뮤니티 GGUF](https://huggingface.co/Preyazz/DeepSeek-V4-Flash-GGUF)
- [GLM-5.2 GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)
- [Kimi-K2.7-Code GGUF](https://huggingface.co/unsloth/Kimi-K2.7-Code-GGUF)
- [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek-V4-Pro 커뮤니티 GGUF](https://huggingface.co/teamblobfish/DeepSeek-V4-Pro-GGUF)

---

## 최종 권고

로컬 보안·코딩 모델을 고를 때는 “가장 큰 모델”보다 다음 조합을 우선한다.

```text
충분한 OS 여유
+ Q4 수준의 안정적인 양자화
+ 8K–16K에서 시작하는 컨텍스트
+ 최신 보안 데이터 RAG
+ 격리된 도구 실행
+ 사람이 검증하는 재현 절차
```

가장 합리적인 시작점은 다음과 같다.

- **8 GB:** Qwen3.5-4B Q4
- **12–16 GB:** Foundation-Sec-8B Q4 또는 Qwen3.5-9B Q4
- **24–32 GB:** Devstral Small 2 24B Q4 또는 Qwen3.6-35B-A3B Q4
- **48–64 GB:** Qwen3-Coder-Next Q3/Q4
- **96 GB 이상:** gpt-oss-120b, Mistral Small 4 119B Q4, Devstral 2 Q4

