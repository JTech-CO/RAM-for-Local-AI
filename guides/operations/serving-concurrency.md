# 로컬 AI 서빙·동시성 메모리 가이드
> `llama.cpp`·Ollama·vLLM·SGLang·TensorRT-LLM·TGI를 RAM·VRAM·Apple 통합 메모리 기준으로 용량 계획하고, KV 캐시·continuous batching·queue·SLO·다중 GPU·다중 사용자를 설계하는 실전 가이드

[← 메인 README](../../README.md) · [생산성·문서·RAG](../domains/productivity-rag.md) · [데이터 분석](../domains/data-analysis.md) · [비전·OCR](../modalities/vision-ocr.md) · [이미지 생성](../modalities/image-generation.md) · [오디오·음성](../modalities/audio-speech.md)

> **최종 검증일:** 2026-07-21 (KST)
> **주요 런타임:** `llama.cpp`, Ollama, vLLM, SGLang, TensorRT-LLM, Hugging Face TGI, MLX-LM, Ray Serve, KServe·llm-d
> **범위:** 생성형 LLM·MoE·VLM·OCR·임베딩·reranker·ASR·TTS의 온라인 추론 메모리, 동시성, 스케줄링, 분산 서빙, 관측성, 보안과 부하 시험
> **관련 문서:** [양자화](./quantization.md) · [파인튜닝 메모리](./fine-tuning-memory.md) · [런타임·하드웨어](./runtime-hardware.md) (예정)

이 문서는 “모델 Q4 파일이 VRAM에 들어가므로 사용자 10명을 동시에 받을 수 있는가?”, “컨텍스트 128K 모델이면 128K 요청 여러 개도 처리할 수 있는가?”, “GPU 두 장이면 처리량이 정확히 두 배가 되는가?” 같은 오해를 피하기 위한 운영 가이드다.

**모델 적재 가능성**과 **서비스 가능성**은 서로 다르다. 모델 가중치를 한 번 적재하는 데 성공해도, 긴 prompt의 prefill, 요청별 KV 캐시, CUDA graph·workspace, tokenizer, vision encoder, 여러 adapter와 동시 요청이 합쳐지면 첫 실제 부하에서 OOM 또는 긴 queue가 발생할 수 있다.

```text
온라인 추론 peak device memory
≈ resident model weights
+ KV cache pool
+ prefill·decode activation/workspace
+ attention·MoE·quantization temporary buffer
+ CUDA graph·compiled graph reserve
+ LoRA·draft model·vision/audio encoder
+ communication buffer
+ allocator fragmentation
+ 운영 안전 여유
```

시스템 전체로는 다음도 포함한다.

```text
전체 serving memory
≈ accelerator memory
+ CPU-side model·KV offload
+ tokenizer·detokenizer worker
+ HTTP·streaming·request queue state
+ image·audio decode buffer
+ prefix cache metadata·remote KV client
+ model download·mmap page cache
+ observability agent·reverse proxy
+ 운영체제·디스플레이·파일 캐시 여유
```

> **핵심 원칙:** 최대 동시 사용자 수는 `VRAM ÷ 모델 파일 크기`로 계산하지 않는다. 실제 입력·출력 길이 분포, 요청 도착률, KV 캐시 bytes/token, prefill peak, 목표 TTFT·TPOT와 queue 상한을 함께 측정해야 한다.

모델 구조, attention backend, runtime scheduler, quantization kernel과 API는 빠르게 변한다. 이 문서의 수치는 **초기 용량 계획용 근사치**다. 최종 값은 대상 모델·runtime·hardware·실제 traffic trace에서 재현 가능한 load test로 결정한다.

---

## 목차

1. [30초 선택표](#1-30초-선택표)
2. [동시성의 의미부터 구분하기](#2-동시성의-의미부터-구분하기)
3. [서빙 메모리 해부](#3-서빙-메모리-해부)
4. [KV 캐시 계산식](#4-kv-캐시-계산식)
5. [Attention 구조별 차이](#5-attention-구조별-차이)
6. [KV 캐시 용량표](#6-kv-캐시-용량표)
7. [실제 context 분포와 token budget](#7-실제-context-분포와-token-budget)
8. [TTFT·TPOT·ITL·throughput·goodput](#8-ttfttpotitlthroughputgoodput)
9. [Prefill과 decode](#9-prefill과-decode)
10. [Continuous batching과 paged KV](#10-continuous-batching과-paged-kv)
11. [스케줄링·공정성·우선순위](#11-스케줄링공정성우선순위)
12. [Chunked prefill](#12-chunked-prefill)
13. [Prefix caching](#13-prefix-caching)
14. [KV 캐시 정밀도·양자화·offload](#14-kv-캐시-정밀도양자화offload)
15. [Queue·backpressure·admission control](#15-queuebackpressureadmission-control)
16. [RAM·VRAM별 30초 구성표](#16-ramvram별-30초-구성표)
17. [개인 PC·워크스테이션 운영](#17-개인-pc워크스테이션-운영)
18. [`llama.cpp` 서버](#18-llamacpp-서버)
19. [Ollama](#19-ollama)
20. [vLLM](#20-vllm)
21. [SGLang](#21-sglang)
22. [TensorRT-LLM](#22-tensorrt-llm)
23. [Hugging Face TGI](#23-hugging-face-tgi)
24. [Apple Silicon·CPU·AMD](#24-apple-siliconcpuamd)
25. [다중 GPU·TP·PP·DP·EP·CP](#25-다중-gputpppdpepcp)
26. [Replica·load balancing·autoscaling](#26-replicaload-balancingautoscaling)
27. [Prefill-decode 분리와 KV 전송](#27-prefill-decode-분리와-kv-전송)
28. [Speculative decoding](#28-speculative-decoding)
29. [Multi-LoRA·다중 모델·model multiplexing](#29-multi-lora다중-모델model-multiplexing)
30. [VLM·OCR·이미지·오디오·임베딩 서빙](#30-vlmocr이미지오디오임베딩-서빙)
31. [관측성·부하 시험·capacity benchmark](#31-관측성부하-시험capacity-benchmark)
32. [보안·다중 tenant·개인정보](#32-보안다중-tenant개인정보)
33. [용량 계획 예제](#33-용량-계획-예제)
34. [문제 해결](#34-문제-해결)
35. [주요 출처와 저장소](#35-주요-출처와-저장소)
36. [최종 권장안과 갱신 주의](#36-최종-권장안과-갱신-주의)

---

## 1. 30초 선택표

### 1.1 목적별 시작점

| 목표 | 권장 시작점 | 먼저 고정할 값 | 피해야 할 시작 방식 |
| --- | --- | --- | --- |
| 개인용 단일 사용자 chat | `llama.cpp` 또는 Ollama | context 4K–16K, 동시 슬롯 1 | 모델 최대 context를 그대로 예약 |
| 가족·소규모 팀 2–8명 | `llama.cpp` 다중 slot 또는 vLLM·SGLang | p95 input/output, queue 상한 | 무제한 queue와 `max_tokens` |
| NVIDIA 단일 GPU API | vLLM 또는 SGLang | KV pool, `max_num_seqs`, prefill budget | `gpu_memory_utilization=1.0`에 가까운 무여유 설정 |
| NVIDIA 최적화 전용 환경 | TensorRT-LLM | model recipe, block reuse, KV fraction | release·GPU 세대와 맞지 않는 engine 재사용 |
| 빠른 모델 설치·교체 | Ollama | loaded model 수, parallel 수, context | 여러 대형 모델을 모두 상주시킴 |
| CPU·AMD·Apple·혼합 offload | `llama.cpp` | thread·NUMA·GPU layer·KV dtype | CPU RAM이 많다는 이유로 긴 context 다중화 |
| Apple Silicon 앱 내부 추론 | MLX-LM 또는 `llama.cpp` | 통합 메모리 여유, batch, thermal | UI·브라우저와 메모리를 끝까지 공유 |
| 긴 RAG prompt | chunked prefill + prefix cache | p95 prompt, shared prefix hit rate | 평균 길이만 보고 capacity 계산 |
| 다중 agent·multi-turn | prefix/radix cache + session affinity | tenant·session cache key | 서로 다른 tenant가 cache를 공유 |
| 높은 QPS batch extraction | continuous batching, 큰 token batch | output 상한, 비스트리밍 | chat latency SLO와 같은 pool 사용 |
| p99 TTFT가 중요한 chat | 작은 batch·reserved headroom·우선순위 | queue time SLO | 처리량 최대점까지 GPU 포화 |
| 매우 긴 prompt와 대규모 traffic | prefill/decode 분리 검토 | KV transfer bandwidth, P/D ratio | 분리하면 처리량이 항상 증가한다고 가정 |

### 1.2 장비만 알고 있을 때

아래 표는 **한 모델을 상주시킨 생성형 text API**의 보수적 출발점이다. 정확한 모델은 각 domain 가이드에서 선택하고, 이 문서에서는 runtime과 동시성만 정한다.

| 실사용 가능 메모리 | 보수적인 모델·동시성 시작점 | context 출발점 | 우선 runtime |
| ---: | --- | ---: | --- |
| 6–8 GB | 1–3B Q4, 1 slot | 4K–8K | `llama.cpp`, Ollama |
| 10–12 GB | 3–7B Q4, 1 slot 또는 3B 2 slots | 4K–8K | `llama.cpp`, Ollama |
| 16 GB | 7–8B Q4, 1–2 slots | 8K | `llama.cpp`, Ollama |
| 20–24 GB | 7–14B Q4, 2–4 slots | 8K–16K | vLLM·SGLang 또는 `llama.cpp` |
| 32 GB | 14B Q4 2–4 slots, 7–8B 4–8 slots | 8K–16K | vLLM·SGLang |
| 40–48 GB | 27–32B Q4 1–4 slots, 14B 4–12 slots | 8K–16K | vLLM·SGLang |
| 64 GB | 32B Q4 2–8 slots, 14B 8–16 slots | 8K–32K | vLLM·SGLang |
| 80 GB | 70B Q4 또는 32B BF16급, workload별 2–16 slots | 8K–32K | vLLM·SGLang·TensorRT-LLM |
| 96–128 GB | 70B Q4/Q5 다중 사용자, 120B 저비트 연구 | 16K–64K | multi-GPU 또는 대용량 unified memory |
| 192 GB 이상 | 70B BF16급 또는 120–235B 저비트 | workload별 산정 | TP·PP·EP·replica 설계 |

> 표의 slot 수는 품질이나 SLO를 보장하지 않는다. 같은 8K slot이라도 MHA·GQA·MLA 구조, KV dtype과 실제 cached token 수에 따라 메모리가 크게 달라진다.

### 1.3 가장 먼저 줄일 값

OOM 또는 p99 지연이 발생하면 다음 순서로 조정한다.

```text
1. 실제 요청의 max_input_tokens·max_new_tokens 제한
2. 동시 running sequence·slot 수 제한
3. prefill token budget·chunk size 축소
4. prefix cache가 재사용하지 못하는 원인 수정
5. KV cache FP16/BF16 → Q8/FP8 검증
6. model Q5/Q6 → Q4 또는 더 작은 Q4 모델
7. CPU KV·weight offload 또는 replica 추가
8. 긴 context 전용 pool 분리
```

가중치 Q4를 Q3로 낮추기 전에, 사용자가 실제로 필요하지 않은 64K·128K context와 과도한 동시 slot을 예약하고 있지 않은지 확인한다.

### 1.4 즉시 판단 규칙

```text
모델이 한 장에 여유 있게 들어감?
  예 → replica 또는 단일 엔진 continuous batching 비교
  아니오 → TP/PP 또는 weight offload

p95 prompt가 짧고 shared prefix가 많음?
  예 → prefix caching·cache-aware routing

긴 prompt가 짧은 chat을 막음?
  예 → chunked prefill·priority·별도 pool·P/D 분리

KV cache가 병목?
  예 → context/sequence cap → Q8/FP8 KV → offload 순서

compute가 병목?
  예 → batching·kernel·quantization·spec decode·replica

queue가 계속 증가?
  예 → overload 상태. queue를 늘리지 말고 admission·scale-out
```

---

## 2. 동시성의 의미부터 구분하기

“동시 사용자 10명”은 기술적으로 모호하다. 최소 다음 값을 분리해야 한다.

### 2.1 동시성 용어

| 용어 | 의미 | 메모리·성능 영향 |
| --- | --- | --- |
| 접속 사용자 | WebSocket·HTTP 연결을 보유한 사용자 | 연결 상태와 proxy 메모리, 반드시 GPU request는 아님 |
| queued request | 도착했지만 아직 실행되지 않은 요청 | GPU KV는 없을 수 있으나 queue time과 RAM 증가 |
| running request | scheduler가 활성화한 요청 | KV 캐시·sequence state 사용 |
| active sequence | beam·`n`·`best_of`까지 포함한 실제 생성 sequence | request 수보다 클 수 있음 |
| batch size | 한 iteration에서 함께 계산되는 sequence/token 묶음 | kernel 효율과 activation peak 영향 |
| token batch | iteration에서 처리하는 총 token 수 | prefill peak와 throughput을 직접 좌우 |
| server slot | `llama.cpp` 등에서 동시에 유지 가능한 context 단위 | slot별 또는 unified KV 용량 사용 |
| loaded model | 메모리에 상주한 서로 다른 모델 | 가중치가 모델마다 중복됨 |
| replica | 동일 모델을 독립적으로 적재한 serving process | throughput·격리 증가, weight 중복 |
| TP rank | 한 모델을 tensor 단위로 나눈 GPU | weight/KV 일부 분할, 통신 발생 |
| tenant | 보안·quota·cache isolation 단위 | cache key와 rate limit을 분리해야 함 |

### 2.2 request와 sequence가 다른 경우

다음 옵션은 하나의 API 요청을 여러 sequence로 늘릴 수 있다.

- `n > 1`
- `best_of > 1`
- beam search
- parallel tool candidate 생성
- speculative draft branch
- reasoning 후보를 여러 개 생성한 뒤 verifier가 선택하는 구조

```text
실제 sequence 수
≈ request 수 × 후보 수 × beam 수
```

따라서 admission control은 HTTP request 개수뿐 아니라 **잠재 sequence 수와 총 token budget**으로 해야 한다.

### 2.3 동시 접속과 동시 추론

100명의 사용자가 UI를 열어도 동시에 생성하는 사용자가 2명이라면 GPU 동시성은 2에 가깝다. 반대로 agent 1명이 병렬 tool call과 후보 생성 16개를 요청하면 active sequence는 16개 이상일 수 있다.

서비스 사양에는 다음처럼 명시한다.

```yaml
capacity_definition:
  connected_users: 100
  accepted_requests_per_second: 0.5
  max_running_requests: 8
  max_active_sequences: 16
  p95_input_tokens: 4000
  p95_output_tokens: 600
  max_total_tokens_per_request: 16384
  queue_limit: 32
```

### 2.4 session 동시성

multi-turn chat은 request가 끝나도 다음 turn에서 이전 prefix를 다시 사용한다. session affinity와 prompt cache를 쓰면 TTFT를 줄일 수 있지만, idle session을 무기한 유지하면 RAM·VRAM·cache metadata가 증가한다.

운영 정책에 다음을 둔다.

- idle session TTL
- 최대 session context
- tenant별 cached token quota
- cache eviction 정책
- 서버 재시작 후 session 복원 여부
- cache 저장 파일의 암호화·권한

---

## 3. 서빙 메모리 해부

### 3.1 Device memory 구성

| 항목 | 고정/가변 | 주요 제어점 |
| --- | --- | --- |
| model weights | 대체로 고정 | weight quantization, TP·PP, CPU offload |
| KV cache pool | traffic에 따라 가변 또는 시작 시 pool 예약 | context, running sequences, KV dtype, page size |
| prefill activation | prompt batch마다 peak | chunked prefill, max batched tokens, flash attention |
| decode workspace | batch·kernel별 | max sequences, CUDA graph sizes, backend |
| logits·sampling | vocabulary·batch에 비례 | logprobs·top-n 제한, distributed sampler |
| quantization workspace | backend·format별 | kernel 지원, dequant buffer |
| MoE workspace | active experts·routing batch별 | EP, expert parallel load balance |
| graph capture | captured batch shape별 | CUDA graph 범위·비활성화 |
| adapter | loaded LoRA마다 증가 | adapter 수·rank·device cache |
| draft model | speculative decoding 시 추가 | draft size·KV dtype·draft tokens |
| modality encoder | image/audio 요청 시 추가 | encoder offload, modality batch limit |
| communication buffer | TP·PP·EP·KV transfer | topology, NCCL, RDMA, rank 수 |
| fragmentation | workload·allocator별 | 운영 여유, process 재시작, pool 조정 |

### 3.2 System RAM 구성

- mmap된 weight page와 파일 cache
- CPU offload weight
- CPU KV cache·secondary cache
- tokenizer·detokenizer worker
- Python·Rust·C++ runtime heap
- request body와 streaming response buffer
- JSON schema·grammar compiler cache
- 이미지·PDF·오디오 decode buffer
- observability queue와 trace exporter
- remote KV connector staging buffer
- 모델 download와 shard 병합 임시 파일

### 3.3 안전한 device memory 예산

```text
usable_device_memory
= physical_device_memory
- display·OS·driver reserve
- runtime fixed overhead
- operational headroom
```

```text
KV budget
= usable_device_memory
- resident weights
- peak prefill/workspace
- graph·adapter·draft reserve
- distributed communication reserve
```

초기 설정에서는 전체 VRAM의 마지막 수백 MiB까지 채우지 않는다. kernel·graph capture·새로운 request shape가 추가 메모리를 요구할 수 있으므로, 실측 후 여유를 줄인다.

### 3.4 “free VRAM”을 그대로 믿기 어려운 이유

- 다른 process와 display가 메모리를 사용한다.
- framework allocator가 reserved와 allocated를 다르게 표시한다.
- 첫 request에서 graph capture와 kernel workspace가 추가된다.
- 처음 보는 긴 prompt·큰 batch에서 peak가 갱신된다.
- NCCL communicator가 multi-GPU 시작 후 생성된다.
- modality encoder가 첫 이미지·오디오 요청에서 lazy load될 수 있다.
- LoRA adapter 또는 grammar cache가 runtime 중 추가된다.

따라서 startup 직후 idle 메모리와 load test 중 peak를 모두 기록한다.

### 3.5 장착 메모리와 사용 가능 메모리

| 환경 | 장착량에서 먼저 제외할 것 |
| --- | --- |
| NVIDIA·AMD discrete GPU | display·driver, 다른 CUDA/ROCm process, graph·workspace |
| Apple Silicon | macOS, WindowServer, 브라우저, IDE, 파일 cache, 다른 앱 |
| CPU-only server | OS, page cache, NUMA 불균형, tokenizer worker, remote KV |
| container | container limit, `/dev/shm`, host daemon, sidecar |
| Kubernetes | pod memory limit, eviction threshold, device plugin overhead |

Apple 통합 메모리는 모델·KV·CPU app이 같은 pool을 사용한다. GPU 전용 VRAM처럼 “모델 적재 후 남은 RAM 전체”를 KV에 쓸 수 있다고 가정하지 않는다.

---

## 4. KV 캐시 계산식

KV 캐시는 autoregressive decoder가 이전 token의 attention key와 value를 재사용하기 위해 저장하는 상태다. 동시성과 긴 context에서 가장 중요한 가변 메모리 중 하나다.

### 4.1 표준 decoder-only transformer

표준 full-attention layer의 근사식은 다음과 같다.

```text
KV bytes per token per sequence
≈ 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
```

- 첫 번째 `2`: K와 V
- `n_layers`: KV를 저장하는 attention layer 수
- `n_kv_heads`: key/value head 수
- `head_dim`: head 하나의 차원
- `bytes_per_element`: FP16/BF16 2, FP8/Q8 계열 약 1, 4-bit 계열 약 0.5에 metadata overhead 추가

전체 KV는 다음과 같다.

```text
active KV bytes
≈ bytes_per_token
× Σ(active sequence의 cached tokens)
× paging·alignment·metadata factor
```

### 4.2 예제 A: GQA 8B형 구조

가상의 구성:

```text
layers = 32
kv_heads = 8
head_dim = 128
KV dtype = FP16 = 2 bytes
```

```text
2 × 32 × 8 × 128 × 2
= 131,072 bytes/token
= 128 KiB/token
```

따라서 한 sequence는 대략 다음 KV를 사용한다.

| cached tokens | FP16 KV |
| ---: | ---: |
| 4K | 0.5 GiB |
| 8K | 1 GiB |
| 16K | 2 GiB |
| 32K | 4 GiB |
| 64K | 8 GiB |
| 128K | 16 GiB |

### 4.3 예제 B: MHA 7B형 구조

가상의 구성:

```text
layers = 32
kv_heads = 32
head_dim = 128
KV dtype = FP16
```

```text
KV = 512 KiB/token
8K sequence 하나 ≈ 4 GiB
```

같은 파라미터 규모라도 GQA 모델보다 동시성이 크게 낮을 수 있다.

### 4.4 예제 C: 대형 GQA 구조

```text
layers = 80
kv_heads = 8
head_dim = 128
KV dtype = FP16
```

```text
KV = 320 KiB/token
8K sequence 하나 ≈ 2.5 GiB
32K sequence 하나 ≈ 10 GiB
```

70B급 모델의 weight가 Q4로 들어가더라도, 긴 context 여러 개는 KV 때문에 별도 GPU 메모리를 빠르게 소비한다.

### 4.5 TP 적용 시

이상적으로 KV head가 tensor parallel rank에 균등 분할되면 per-GPU KV가 줄 수 있다.

```text
ideal per-GPU KV ≈ total KV / TP
```

그러나 다음 이유로 정확히 나누어지지 않을 수 있다.

- KV head 수보다 TP rank가 많음
- MQA·GQA head replication
- backend별 page ownership
- context parallel 또는 sequence parallel 사용
- hybrid attention·MLA 구현 차이
- pipeline stage별 layer 수 불균형

실제 startup log 또는 runtime의 cache block 수로 확인한다.

### 4.6 Python 계산기

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVConfig:
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: float = 2.0

    def bytes_per_token(self) -> float:
        return (
            2
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.bytes_per_element
        )

    def gib(self, cached_tokens: int, sequences: int = 1, overhead: float = 1.10) -> float:
        if cached_tokens < 0 or sequences < 1 or overhead < 1.0:
            raise ValueError("invalid KV cache inputs")
        total = self.bytes_per_token() * cached_tokens * sequences * overhead
        return total / (1024**3)


cfg = KVConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_element=2)
print(f"bytes/token: {cfg.bytes_per_token():,.0f}")
print(f"8K × 4 sequences, 10% overhead: {cfg.gib(8192, 4):.2f} GiB")
```

이 계산은 표준 transformer의 이론값이다. runtime page metadata, padding, graph와 activation은 별도로 더한다.

### 4.7 Hugging Face config에서 읽을 값

일반적으로 `config.json`에서 다음을 찾는다.

```json
{
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "hidden_size": 4096,
  "head_dim": 128
}
```

`head_dim`이 없으면 흔히 다음처럼 계산하지만, 모든 architecture에 보편적이지 않다.

```text
head_dim ≈ hidden_size / num_attention_heads
```

custom architecture, MLA, hybrid recurrent model은 model implementation과 runtime 문서를 확인한다.

### 4.8 총 cached token 기준

paged KV runtime에서는 “slot마다 max context 전체를 즉시 할당”하기보다 token block pool을 공유할 수 있다. 이때 더 유용한 값은 다음이다.

```text
max cached tokens
≈ floor(KV budget bytes / effective KV bytes per token)
```

예를 들어 KV pool이 16 GiB이고 128 KiB/token이면 이론상 약 131K tokens다. page·metadata·여유를 10% 제외하면 약 117K tokens를 capacity planning 상한으로 사용한다.

---

## 5. Attention 구조별 차이

### 5.1 MHA·GQA·MQA

| 구조 | KV head 수 | KV 메모리 | 특징 |
| --- | ---: | --- | --- |
| MHA | attention head와 동일 | 가장 큼 | 오래된 모델에서 흔함 |
| GQA | head group별 KV | 중간 | 최신 범용 LLM에서 흔함 |
| MQA | 보통 1 KV head | 매우 작음 | KV 효율이 높지만 model별 품질·kernel 차이 |

모델 파라미터 수만으로 KV 크기를 추정하지 않는다. 같은 7B·8B라도 MHA와 GQA의 동시성 차이가 수배일 수 있다.

### 5.2 MLA

Multi-head Latent Attention 계열은 K/V를 그대로 저장하지 않고 latent representation을 사용하는 구현이 있다. 이 경우 표준 `n_kv_heads × head_dim` 공식이 맞지 않을 수 있다.

- model card의 latent dimension 확인
- serving runtime의 MLA 전용 cache 계산 확인
- TP·context parallel과의 sharding 방식 확인
- startup log의 실제 cache block 수로 capacity 역산

### 5.3 Sliding Window Attention

SWA layer는 전체 context가 아니라 최근 window만 KV에 유지할 수 있다.

```text
hybrid KV
≈ full-attention layers × total context
+ SWA layers × min(total context, window size)
```

다만 runtime이 full-size SWA cache를 사용하거나 hybrid cache manager 지원이 제한되면 절감 폭이 달라진다. 관련 옵션을 켰다고 가정하지 말고 실제 allocated cache를 확인한다.

### 5.4 Recurrent·SSM hybrid

Mamba·SSM·recurrent layer는 token 길이에 선형인 전통적 KV 대신 sequence별 recurrent state를 유지할 수 있다. hybrid 모델은 다음이 함께 존재한다.

- attention layer KV
- recurrent state
- convolution state
- speculative overlap용 추가 state buffer

sequence 수가 증가할 때 고정 state가 늘 수 있으므로 “KV bytes/token이 작다 = 동시성 무제한”이 아니다.

### 5.5 Encoder-decoder

T5·Whisper·번역·일부 speech model은 다음을 분리한다.

- encoder activation 또는 encoded memory
- decoder self-attention KV
- cross-attention K/V 또는 encoder representation

긴 audio·document 입력은 decoder output이 짧아도 encoder 메모리와 prefill compute가 클 수 있다.

### 5.6 MoE

MoE는 token마다 일부 expert만 활성화해 compute를 줄일 수 있지만, serving memory에는 전체 resident expert weight가 필요할 수 있다.

```text
MoE serving memory
≈ all resident expert weights
+ active-token routing workspace
+ expert communication buffer
+ KV cache
```

동시성이 높아지면 expert load imbalance와 all-to-all 통신이 병목이 될 수 있다. active parameter 수만으로 장착 메모리를 계산하지 않는다.

### 5.7 Multimodal

VLM의 visual token은 LLM KV에 들어갈 수 있으며, vision encoder의 activation·projector도 별도다.

```text
VLM request memory
≈ text KV
+ visual token KV
+ image encoder peak
+ decoded image tensor
+ multimodal cache metadata
```

동일한 “이미지 1장”이라도 해상도·dynamic tiling·crop 수에 따라 visual token이 크게 달라진다.

---

## 6. KV 캐시 용량표

### 6.1 bytes/token별 context 용량

아래는 sequence 하나의 순수 KV 이론값이다. page·metadata·fragmentation과 prefill workspace는 제외한다.

| KV 크기/token | 4K | 8K | 16K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 KiB | 0.125 GiB | 0.25 GiB | 0.5 GiB | 1 GiB | 2 GiB | 4 GiB |
| 64 KiB | 0.25 GiB | 0.5 GiB | 1 GiB | 2 GiB | 4 GiB | 8 GiB |
| 96 KiB | 0.375 GiB | 0.75 GiB | 1.5 GiB | 3 GiB | 6 GiB | 12 GiB |
| 128 KiB | 0.5 GiB | 1 GiB | 2 GiB | 4 GiB | 8 GiB | 16 GiB |
| 160 KiB | 0.625 GiB | 1.25 GiB | 2.5 GiB | 5 GiB | 10 GiB | 20 GiB |
| 256 KiB | 1 GiB | 2 GiB | 4 GiB | 8 GiB | 16 GiB | 32 GiB |
| 320 KiB | 1.25 GiB | 2.5 GiB | 5 GiB | 10 GiB | 20 GiB | 40 GiB |
| 512 KiB | 2 GiB | 4 GiB | 8 GiB | 16 GiB | 32 GiB | 64 GiB |

### 6.2 8K sequence 수의 이론적 상한

다음 표는 KV pool의 90%만 사용하고, 모든 sequence가 정확히 8K cached tokens를 가진다고 가정한다. 다른 메모리는 이미 제외된 **순수 KV budget**이다.

| KV budget | 64 KiB/token | 128 KiB/token | 320 KiB/token | 512 KiB/token |
| ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 7 | 3 | 1 | 0 |
| 8 GiB | 14 | 7 | 2 | 1 |
| 16 GiB | 28 | 14 | 5 | 3 |
| 24 GiB | 43 | 21 | 8 | 5 |
| 32 GiB | 57 | 28 | 11 | 7 |
| 48 GiB | 86 | 43 | 17 | 10 |
| 64 GiB | 115 | 57 | 23 | 14 |

실제 운영에서는 sequence 길이가 서로 다르고, preemption·prefix sharing·page fragmentation이 있으므로 이 표보다 낮거나 높을 수 있다.

### 6.3 FP16·Q8·Q4 상대 비교

| KV dtype | 명목 bytes/value | FP16 대비 순수 데이터 | 일반적 용도 |
| --- | ---: | ---: | --- |
| FP32 | 4 | 2배 | 디버깅·특수 정확도 |
| FP16·BF16 | 2 | 1배 | 품질 기준선 |
| FP8·Q8 | 약 1 | 약 1/2 | 긴 context·동시성의 첫 절감 후보 |
| Q5 | 약 0.625 + metadata | 약 0.31 | 지원 runtime에서 검증 필요 |
| Q4 | 약 0.5 + metadata | 약 1/4 | 극저메모리, 장문 품질 검증 필수 |

명목 비트만큼 정확히 줄지 않는 이유는 block scale, alignment, metadata와 backend layout 때문이다.

### 6.4 평균 길이로 계산하지 않기

다음 두 traffic은 평균 cached token이 같아도 위험도가 다르다.

```text
A: 모든 요청 4K
B: 90% 1K + 10% 31K
```

B는 긴 요청이 들어올 때 prefill peak와 KV fragmentation, head-of-line blocking을 일으킨다. p50·p90·p95·p99와 max를 모두 기록한다.

### 6.5 prefix cache가 있는 경우

shared prefix를 물리 block 단위로 재사용하는 runtime에서는 다음처럼 계산할 수 있다.

```text
physical cached tokens
≈ unique prefix tokens
+ Σ(request별 unique suffix tokens)
```

그러나 cache hit가 발생하지 않으면 다시 전체 token이 필요하다. capacity 산정의 보수적 기준은 낮은 hit-rate 또는 cold-start traffic으로 둔다.

---

## 7. 실제 context 분포와 token budget

### 7.1 모델 최대 context와 서비스 제한

모델이 128K·256K·1M context를 지원해도 모든 사용자에게 동일한 한도를 제공할 필요는 없다.

| 값 | 역할 |
| --- | --- |
| model max context | architecture·RoPE가 허용하는 상한 |
| server max model length | runtime이 허용하는 상한 |
| API max input tokens | 사용자 prompt 상한 |
| API max output tokens | 생성 상한 |
| request max total tokens | input + output budget |
| tenant quota | tenant별 동시 token·일일 token 제한 |
| queue token budget | 대기 중 잠재 작업량 제한 |

서비스 기본값은 실제 workload의 p99보다 약간 높은 값에서 시작하고, 긴 context는 별도 endpoint나 premium pool로 분리한다.

### 7.2 입력·출력 합계

```text
request total token budget
= input tokens
+ max_new_tokens
```

현재 cached token은 생성할수록 증가한다.

```text
decode 시점 cached tokens
≈ input tokens + generated tokens so far
```

`max_new_tokens`를 과도하게 크게 두면 실제 출력이 짧아도 scheduler가 worst-case reserve 또는 admission 판단에 사용할 수 있다.

### 7.3 길이 분포 manifest

```yaml
traffic_profile:
  sampling_window: 7d
  requests: 125000
  input_tokens:
    p50: 620
    p90: 2400
    p95: 4100
    p99: 12200
    max: 63800
  output_tokens:
    p50: 180
    p90: 520
    p95: 840
    p99: 1800
    max: 8192
  shared_prefix_tokens:
    p50: 900
  streaming_ratio: 0.92
  cancellation_ratio: 0.08
```

### 7.4 RAG의 숨은 token

RAG request는 다음이 합쳐진다.

```text
system prompt
+ tool schema
+ conversation history
+ retrieved chunks
+ citation metadata
+ user query
+ output budget
```

검색 chunk 수만 제한하면 안 된다. tokenizer를 통과한 최종 rendered prompt를 기준으로 admission한다.

### 7.5 Agent workload

agent는 turn 하나의 output이 짧아도 system prompt·tool schema·history를 반복한다. 다음을 측정한다.

- turn당 input token 증가량
- 동일 prefix 재사용률
- tool result 크기
- 취소·재시도율
- 한 사용자당 병렬 branch 수
- reasoning token과 visible output token 분리

### 7.6 Tokenizer CPU 병목

긴 prompt와 높은 QPS에서는 GPU가 아니라 tokenization이 병목일 수 있다.

- tokenizer worker 수
- GIL·process model
- chat template rendering
- JSON validation
- Unicode normalization
- image URL download·decode
- detokenization과 streaming flush

GPU utilization이 낮고 queue가 늘면 tokenizer·HTTP·scheduler CPU를 함께 profile한다.

---

## 8. TTFT·TPOT·ITL·throughput·goodput

### 8.1 핵심 지표

| 지표 | 정의 | 주로 영향을 주는 요소 |
| --- | --- | --- |
| TTFT | 요청 도착부터 첫 token까지 | queue, tokenization, prefill, scheduler |
| TPOT | 첫 token 이후 output token당 평균 시간 | decode batch, memory bandwidth, TP 통신 |
| ITL | 인접 output token 사이 지연 | scheduling jitter, prefill interruption, streaming |
| E2EL | 요청 전체 완료 시간 | queue + prefill + decode + network |
| request throughput | 초당 완료 request 수 | 길이 분포와 batch |
| input throughput | 초당 처리 prompt token | prefill compute |
| output throughput | 초당 생성 token | decode compute·memory bandwidth |
| goodput | SLO를 만족한 request/token 처리율 | tail latency와 overload control |
| cache hit rate | 재사용된 prefix·KV 비율 | prompt 일관성·routing |
| queue depth/time | 대기 request 수·시간 | overload·autoscaling 신호 |
| preemption rate | running request 중단·재계산 빈도 | KV 부족·scheduler 압박 |

### 8.2 TTFT 분해

```text
TTFT
≈ client/network
+ gateway queue
+ tokenizer/render
+ engine queue
+ prefill
+ first decode step
+ stream flush
```

서버의 `time_to_first_token`만 보면 gateway queue와 client network가 빠질 수 있다. end-to-end와 engine-level metric을 분리한다.

### 8.3 TPOT와 ITL

TPOT는 전체 decode 시간을 output token 수로 나눈 평균이다. ITL histogram은 간헐적인 긴 멈춤을 더 잘 보여준다. 긴 prefill이 decode iteration을 방해하면 평균 TPOT는 괜찮아도 p99 ITL이 나빠질 수 있다.

### 8.4 처리량 최대점과 SLO 최대점

batch를 키우면 output throughput은 증가할 수 있지만, 각 사용자의 TPOT와 queue time은 악화할 수 있다.

```text
최대 raw throughput ≠ 최대 SLO goodput
```

운영점은 GPU가 완전히 포화되는 지점이 아니라, p95·p99 TTFT/TPOT SLO를 만족하며 goodput이 최대가 되는 지점으로 정한다.

### 8.5 Little의 법칙

안정된 queue에서 대략 다음이 성립한다.

```text
평균 동시 request L ≈ 도착률 λ × 평균 체류시간 W
```

예를 들어 0.5 req/s가 들어오고 평균 E2EL이 12초면 평균 6개 request가 시스템 안에 있다. p99 burst와 길이 분산을 반영하려면 load test가 필요하다.

### 8.6 SLO 예시

```yaml
slo:
  interactive_chat:
    p95_ttft_ms: 1500
    p99_ttft_ms: 3500
    p95_tpot_ms: 70
    p99_e2el_s: 45
  batch_extraction:
    completion_rate: 0.999
    deadline_s: 300
  overload:
    max_queue_time_ms: 2000
    reject_status: 429
```

수치는 제품 요구에 따라 정한다. 문서의 예시를 그대로 production SLO로 사용하지 않는다.

### 8.7 평균만 보면 안 되는 이유

- 긴 request 1개가 평균을 크게 왜곡한다.
- cache hit request와 cold request가 섞인다.
- streaming client가 중간에 취소한다.
- batch extraction과 chat이 다른 SLO를 가진다.
- warmup·graph capture 직후 latency가 다르다.

최소 p50·p90·p95·p99, cold/warm, input/output 길이 bucket별로 나눈다.

---

## 9. Prefill과 decode

### 9.1 두 단계의 성격

| 단계 | 하는 일 | 주 병목 | 중요 제어점 |
| --- | --- | --- | --- |
| Prefill | 전체 input token을 병렬 처리하고 KV 생성 | compute·activation·attention | input length, token batch, chunking |
| Decode | sequence마다 보통 한 token씩 반복 생성 | weight·KV memory bandwidth, scheduler | active sequences, KV, batch |

### 9.2 Prefill peak

긴 prompt 여러 개를 한 번에 prefill하면 다음이 커진다.

- attention activation
- temporary workspace
- prompt logits 요청 시 vocabulary tensor
- multimodal encoder output
- compiled graph shape
- KV block allocation burst

모델이 idle에서 잘 적재되어도 첫 64K prompt에서 OOM이 날 수 있다.

### 9.3 Decode batch

동시 sequence가 증가하면 하나의 decode iteration에서 더 많은 token을 생성할 수 있어 weight reuse가 좋아진다. 그러나 다음이 악화될 수 있다.

- 사용자별 TPOT
- KV read bandwidth
- TP all-reduce latency
- scheduler CPU overhead
- streaming flush overhead
- long-tail sequence가 batch를 점유하는 시간

### 9.4 Prefill-decode 간섭

통합 scheduler에서 긴 prefill이 들어오면 진행 중인 decode가 지연될 수 있다. 해결 후보는 다음이다.

1. chunked prefill
2. prefill token budget 제한
3. 짧은 request 우선 정책
4. 긴 context endpoint 분리
5. prefill/decode disaggregation

### 9.5 Workload별 compute 비율

| workload | prefill 비중 | decode 비중 | 권장 방향 |
| --- | ---: | ---: | --- |
| 장문 요약·짧은 답 | 높음 | 낮음 | chunked prefill, prefix cache |
| code completion | 낮음~중간 | 낮음 | 낮은 TTFT, speculative decode |
| 긴 reasoning 생성 | 중간 | 높음 | decode throughput, output cap |
| multi-turn chat | turn마다 증가 | 중간 | session prefix reuse |
| batch extraction | 중간 | 낮음 | 높은 token batch, throughput 우선 |
| RAG QA | 중간~높음 | 낮음~중간 | retrieved token budget, shared prefix |

### 9.6 용량을 두 자원으로 계산

단일 “tokens/s”보다 다음 두 수치를 별도로 본다.

```text
prefill demand ≈ arrival_rate × E[input_tokens]
decode demand  ≈ arrival_rate × E[output_tokens]
```

서버가 둘 중 하나만 감당하지 못해도 queue가 계속 증가한다.

---

## 10. Continuous batching과 paged KV

### 10.1 Static batch의 한계

전통적 static batch는 모든 sequence가 끝날 때까지 batch shape를 유지한다. 길이가 다른 생성 요청에서는 먼저 끝난 sequence의 자원이 낭비된다.

```text
요청 A: 20 output tokens
요청 B: 200 output tokens

static batch:
A가 끝난 뒤에도 B가 끝날 때까지 slot 낭비 가능
```

### 10.2 Continuous batching

Continuous batching 또는 in-flight batching은 iteration 경계에서 끝난 request를 제거하고 새 request를 넣는다.

장점:

- 가변 길이 output에서 GPU utilization 향상
- queue에서 새 request를 빠르게 합류
- request 수보다 token 단위 scheduling 가능

비용:

- scheduler와 metadata overhead
- batch shape 변화
- 새 prefill이 진행 중 decode를 방해할 가능성
- request별 latency 예측이 복잡

### 10.3 Paged KV cache

Paged KV는 KV를 고정 크기 block/page로 관리해 sequence별 연속 대형 buffer 예약을 줄인다.

```text
logical token sequence
→ KV block table
→ physical KV pages
```

주요 효과:

- 실제 사용 token만 block 단위 할당
- variable length request에서 내부 낭비 감소
- prefix block 공유 가능
- preemption·eviction·offload 구현 용이

### 10.4 Page size trade-off

| 작은 page | 큰 page |
| --- | --- |
| 마지막 block 낭비 감소 | metadata·page table overhead 감소 |
| fine-grained prefix reuse | kernel·memory access가 단순할 수 있음 |
| block 관리 비용 증가 | 짧은 request에서 내부 fragmentation 증가 |

runtime 기본값은 kernel과 model architecture에 맞춰진 경우가 많다. page size를 임의로 바꾸기 전에 실제 hit rate·fragmentation·throughput을 측정한다.

### 10.5 Token budget scheduler

고성능 runtime은 request count만이 아니라 한 iteration의 총 scheduled token을 제한한다.

```text
iteration token budget
= prefill chunk tokens
+ decode tokens
+ speculative verification tokens
```

같은 `max_num_seqs=64`라도 prompt가 100 tokens인지 32K인지에 따라 peak가 다르다. sequence cap과 token cap을 함께 둔다.

### 10.6 Continuous batching이 항상 좋은 것은 아님

다음 workload에서는 작은 고정 batch 또는 별도 pool이 더 예측 가능할 수 있다.

- strict deterministic batch invariance가 필요한 평가
- 아주 낮은 QPS의 개인용 chat
- latency가 매우 중요한 code completion
- 긴 batch job이 많은 interactive traffic과 섞인 경우
- encoder model처럼 output generation이 없는 경우

### 10.7 Batch shape와 CUDA graph

CUDA graph는 특정 shape 범위를 미리 capture해 launch overhead를 줄일 수 있지만, capture한 batch size마다 memory reserve가 추가될 수 있다.

- 너무 많은 graph size: 메모리 증가
- 너무 적은 graph size: eager fallback 증가
- 실제 동시성보다 큰 graph: 불필요한 reserve
- 새 model·adapter·modality: capture 호환성 확인

OOM이 startup이 아니라 첫 특정 batch에서 발생하면 graph capture를 의심한다.

---

## 11. 스케줄링·공정성·우선순위

### 11.1 FCFS의 장단점

First-Come, First-Served는 단순하고 예측 가능하지만, 매우 긴 prompt 하나가 짧은 request를 막는 head-of-line blocking을 만들 수 있다.

```text
64K 문서 요약 1개
→ 뒤의 200-token chat 20개가 TTFT SLO 위반
```

### 11.2 Shortest-job 계열

입력·예상 출력 길이가 짧은 request를 우선하면 평균 latency를 줄일 수 있다. 그러나 긴 request starvation과 tenant 불공정이 생길 수 있다.

사용 가능한 신호:

- input token 수
- `max_new_tokens`
- endpoint class
- tenant priority
- deadline
- cache hit token 수
- 이미 소비한 compute

### 11.3 Priority scheduling

priority는 다음처럼 제한적으로 사용한다.

| 우선순위 | 예시 | 정책 |
| --- | --- | --- |
| 긴급 | 코드 자동완성·interactive voice | 작은 token cap·엄격한 rate limit |
| 기본 | 일반 chat·RAG | 표준 queue |
| 낮음 | batch extraction·offline report | 남는 capacity 사용 |
| 격리 | 64K 이상 문서·대형 multimodal | 별도 pool 권장 |

priority가 높다는 이유로 무제한 output·동시 request를 허용하지 않는다.

### 11.4 Fairness

공정성 단위는 request 수보다 token·GPU time이 적합하다.

```text
tenant cost
≈ input tokens × prefill coefficient
+ output tokens × decode coefficient
+ modality encoder cost
+ candidate multiplier
```

가능한 정책:

- tenant별 token bucket
- weighted fair queue
- max running sequence per tenant
- max cached token per tenant
- 긴 request 별도 queue
- burst credit와 지속 rate 분리

### 11.5 Preemption

KV pool이 부족하면 runtime은 sequence를 중단하고 KV를 버리거나 CPU로 swap한 뒤 나중에 재개할 수 있다.

| 방식 | 장점 | 비용 |
| --- | --- | --- |
| recompute | GPU memory 절약 | 이전 prompt를 다시 prefill |
| CPU swap/offload | compute 재실행 감소 | PCIe·RAM bandwidth·latency |
| eviction + reject | overload 확산 방지 | request 실패 |
| lower-priority pause | interactive SLO 보호 | batch completion 지연 |

preemption이 자주 발생하면 “성공 처리량”이 아니라 재계산으로 GPU를 낭비한다. preemption rate를 핵심 metric으로 둔다.

### 11.6 Cancellation

사용자가 브라우저를 닫거나 agent가 답을 더 이상 필요로 하지 않으면 generation을 즉시 취소해야 한다.

- gateway에서 client disconnect 감지
- engine cancel API로 전파
- KV block 해제
- stream buffer·tool task 정리
- billing·token metric 일관성 유지

취소가 engine에 전달되지 않으면 보이지 않는 zombie generation이 throughput을 소모한다.

### 11.7 Streaming interval

매 token마다 flush하면 체감 latency는 좋지만 CPU·network syscall과 proxy overhead가 커진다. 여러 token을 묶으면 throughput은 좋아질 수 있으나 ITL 체감이 나빠진다.

모바일·WAN·voice client와 LAN UI에 같은 flush interval을 강제하지 않는다.

---

## 12. Chunked prefill

### 12.1 목적

Chunked prefill은 긴 prompt를 한 번에 처리하지 않고 token chunk로 나눠 decode request와 interleave한다.

```text
64K prompt
→ 4K × 16 chunks
```

주요 효과:

- prefill activation peak 제한
- 긴 prompt가 decode를 장시간 막는 현상 완화
- 짧은 request가 중간에 scheduler에 들어올 기회 제공

### 12.2 Chunk가 너무 큰 경우

- prefill peak OOM
- p99 ITL 악화
- 짧은 request의 TTFT 증가
- 한 iteration 시간이 길어짐

### 12.3 Chunk가 너무 작은 경우

- kernel launch·scheduler overhead 증가
- prefill throughput 저하
- KV block update 빈도 증가
- distributed communication overhead 증가

### 12.4 튜닝 순서

```text
1. 실제 p99 prompt로 OOM 없는 최대 chunk 찾기
2. decode traffic과 혼합해 p95/p99 ITL 측정
3. prefill throughput과 TTFT Pareto curve 작성
4. 짧은/긴 prompt pool 분리 여부 결정
```

### 12.5 vLLM 관련 제어점

현재 vLLM은 다음 계열의 설정을 제공한다.

- `--enable-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-partial-prefills`
- `--max-long-partial-prefills`
- `--long-prefill-token-threshold`

긴 prompt와 짧은 prompt를 일부 함께 prefill할 수 있지만, 실제 기본값과 지원 조합은 설치한 release의 `vllm serve --help`에서 확인한다.

### 12.6 SGLang 관련 제어점

현재 SGLang은 `--chunked-prefill-size`로 최대 chunk token 수를 지정하며, 긴 prompt prefill OOM 시 더 작은 값 사용을 공식 문서에서 안내한다. `-1`은 chunked prefill 비활성화다.

함께 보는 값:

- `--prefill-max-requests`
- `--max-prefill-tokens`
- `--max-running-requests`
- `--mem-fraction-static`

### 12.7 Multimodal chunk

image·video token은 임의 경계에서 나누기 어려울 수 있다. runtime에 따라 multimodal item을 부분 schedule하지 않는 옵션이 있다. 다음을 분리 측정한다.

- vision encoder peak
- visual token prefill
- text token prefill
- image별 dynamic token 수

### 12.8 Chunked prefill의 한계

Chunking은 총 prefill compute를 없애지 않는다. 긴 request가 많아 arrival rate가 prefill capacity를 넘으면 queue는 계속 증가한다. 이 경우 긴 context quota·replica·P/D 분리 또는 요청 자체 축소가 필요하다.

---

## 13. Prefix caching

### 13.1 재사용 가능한 것

동일한 token prefix를 공유하는 request는 이미 계산한 KV block을 재사용할 수 있다.

예시:

- 같은 system prompt
- 같은 tool schema
- 같은 few-shot examples
- multi-turn session의 이전 대화
- 동일 문서에 대한 여러 질문
- batch task의 공통 instruction

### 13.2 정확히 token이 같아야 함

다음 차이는 cache miss를 만든다.

- 공백·줄바꿈·JSON key 순서
- timestamp·request ID를 system prompt에 삽입
- chat template 변경
- tokenizer revision 변경
- tool schema 순서 변경
- 다른 LoRA adapter
- image content/hash 차이
- reasoning mode·special token 차이

공통 prefix에는 매 request마다 바뀌는 값을 뒤쪽으로 이동한다.

### 13.3 Cache hit의 이점

- TTFT 감소
- prefill compute 절약
- 장문 system/tool schema 비용 절감
- multi-turn session의 반복 prefill 방지

prefix cache는 decode throughput을 직접 높이지 않을 수 있다. output이 긴 workload에서는 decode가 여전히 병목이다.

### 13.4 Capacity 계산

```text
logical prompt tokens
= 각 request에서 보이는 전체 token

physical prefill tokens
= cold 또는 cache miss token
```

benchmark에는 두 값을 모두 기록한다. hit rate가 높은 synthetic test만으로 cold-start capacity를 결정하지 않는다.

### 13.5 Cache-aware routing

replica가 여러 개면 해당 prefix를 가진 replica로 보내야 cache hit를 유지할 수 있다.

routing key 후보:

```text
hash(
  model_revision,
  tokenizer_revision,
  chat_template,
  adapter_id,
  tenant_cache_domain,
  stable_prefix_tokens,
  multimodal_content_id
)
```

단순 round-robin은 prefix locality를 깨뜨릴 수 있다.

### 13.6 Session affinity의 위험

한 replica에 session을 고정하면 cache hit는 좋아지지만, 특정 replica에 긴 session이 몰려 imbalance가 생길 수 있다.

대응:

- cache load와 active token을 routing 신호로 사용
- idle session cache를 secondary tier로 이동
- session TTL
- hot tenant의 dedicated replica
- cache miss 허용 후 load 재균형

### 13.7 보안 경계

서로 신뢰하지 않는 tenant가 같은 prefix cache block을 공유하면 timing·hit 여부를 통한 정보 추론 위험이 있다.

- tenant별 cache namespace 또는 salt
- ACL·document 권한이 다른 RAG prefix 분리
- private prompt를 global cache에 넣지 않기
- cache metadata·event log에서 prompt hash 노출 제한
- cache 저장·offload 데이터 암호화

TensorRT-LLM의 현재 KV cache system은 request의 `cache_salt`가 같은 경우에만 block reuse를 허용하는 보안 메커니즘을 제공한다. 다른 runtime에서도 같은 원칙을 application layer에서 구현한다.

### 13.8 Prompt cache 무효화

다음이 바뀌면 cache namespace를 변경한다.

- model weight revision
- tokenizer·chat template
- rope/context setting
- KV dtype·layout
- LoRA adapter revision
- system policy
- tool schema
- safety filter version

### 13.9 Prefix cache 평가표

| 항목 | 측정값 |
| --- | --- |
| cold TTFT p95 |  |
| warm TTFT p95 |  |
| cached token ratio |  |
| physical KV GiB |  |
| eviction rate |  |
| cross-replica miss rate |  |
| tenant isolation test | pass/fail |

---

## 14. KV 캐시 정밀도·양자화·offload

### 14.1 선택 순서

```text
1. FP16/BF16 quality baseline
2. Q8/FP8 KV의 장문·task 품질 평가
3. 메모리가 여전히 부족하면 Q4/Q5 지원 backend 실험
4. CPU offload·secondary cache
5. context·concurrency 제한과 비교
```

### 14.2 Q8·FP8

Q8·FP8은 FP16 대비 순수 KV 데이터를 대략 절반으로 줄이는 첫 후보다. 다음을 확인한다.

- GPU architecture와 kernel 지원
- scale 계산 방식
- model별 accuracy sensitivity
- 긴 context에서 누적 오차
- prefix cache·offload와 호환성
- throughput이 실제로 개선되는지

### 14.3 Q4·Q5

낮은 비트의 KV는 메모리를 더 줄일 수 있지만, 장문 retrieval·수학·코드·정확한 복사에서 오류가 누적될 수 있다.

평가해야 할 항목:

- needle retrieval 위치별 정확도
- 긴 문서 entity·숫자 보존
- multi-turn instruction 유지
- JSON·tool call 정확도
- code repository context 참조
- audio/VLM long context

### 14.4 K와 V를 다르게 양자화

일부 runtime은 K와 V dtype을 별도로 선택한다. model과 kernel에 따라 민감도가 다를 수 있으므로 다음 조합을 A/B한다.

```text
K=Q8, V=Q8
K=Q8, V=Q4
K=Q4, V=Q4
```

메모리만 보고 비대칭 조합을 기본값으로 두지 않는다.

### 14.5 CPU KV offload

GPU KV block을 CPU RAM에 옮기면 더 많은 session을 유지할 수 있지만, 다시 decode에 사용할 때 PCIe·CXL·interconnect 전송이 필요하다.

적합한 경우:

- idle session이 많음
- 동일 prefix가 재사용됨
- latency보다 capacity가 중요
- CPU RAM과 pinned memory가 충분

부적합한 경우:

- 모든 active sequence가 매 token KV를 읽음
- PCIe bandwidth가 낮음
- p99 TPOT가 엄격함
- CPU memory pressure와 NUMA remote access가 큼

### 14.6 Remote KV·secondary cache

remote cache는 replica 재시작·scale-out·P/D 분리에서 KV를 공유할 수 있다. 그러나 다음 비용이 추가된다.

- serialization·deserialization
- network bandwidth
- checksum·metadata
- cache consistency
- 암호화
- eviction와 storage cost

remote KV hit가 local recompute보다 빠른 길이 구간을 benchmark로 찾는다.

### 14.7 vLLM 제어점

현재 vLLM은 다음 계열을 제공한다.

- `--kv-cache-dtype`
- `--kv-cache-memory-bytes`
- `--gpu-memory-utilization`
- `--kv-offloading-size`
- `--kv-offloading-backend`
- KV connector·LMCache integration

`--kv-cache-memory-bytes`를 명시하면 자동 GPU memory utilization 계산보다 직접적인 KV pool 제어가 가능하며, 현재 문서상 이 값이 설정되면 `gpu_memory_utilization`을 무시한다.

### 14.8 `llama.cpp` 제어점

현재 `llama-server`는 K·V cache dtype을 개별 지정할 수 있다.

```bash
-ctk q8_0 \
-ctv q8_0
```

지원 값은 build·backend에 따라 확인하며, 공식 server README에는 `f32`, `f16`, `bf16`, `q8_0`, 여러 Q4·Q5 계열이 기재되어 있다.

### 14.9 Ollama 제어점

Ollama는 `OLLAMA_KV_CACHE_TYPE`으로 cache type을 설정할 수 있다. 공식 FAQ는 `q8_0`을 FP16 대비 약 절반, `q4_0`을 약 1/4 수준의 메모리 선택지로 설명하지만, 정확도 trade-off와 model별 지원을 검증해야 한다.

### 14.10 Offload는 실패를 숨길 수 있음

GPU OOM이 사라져도 다음이 발생할 수 있다.

- TPOT 급증
- PCIe utilization 포화
- host page fault
- NUMA node 간 traffic
- kernel이 offloaded cache를 지원하지 않아 recompute
- CPU RAM OOM 또는 swap

“실행 성공”과 “SLO 충족”을 분리한다.

---

## 15. Queue·backpressure·admission control

### 15.1 Queue는 capacity가 아님

queue를 크게 하면 reject가 늦어질 뿐, 처리 capacity가 증가하지 않는다.

```text
arrival rate > service rate
→ queue가 계속 증가
→ TTFT 폭증
→ timeout·retry storm
```

### 15.2 Admission 기준

request를 받을 때 다음을 검사한다.

- input token 상한
- input + max output 상한
- 후보 수·beam·`best_of`
- image·audio 수와 크기
- tenant별 running·queued token
- 현재 KV pool 여유
- deadline과 예상 queue time
- adapter·model availability

### 15.3 Request 수가 아닌 token 기준

```text
admission_cost
= input_tokens
+ output_reservation_weight × max_new_tokens
+ modality_tokens
+ candidate_multiplier
```

짧은 request 20개보다 128K request 1개가 더 비쌀 수 있다.

### 15.4 Queue 상한

| 조건 | 응답 후보 |
| --- | --- |
| tenant rate 초과 | `429 Too Many Requests` |
| 전체 queue 포화 | `503 Service Unavailable` |
| request token 상한 초과 | `400` 또는 `413` |
| deadline 내 처리 불가 | 빠른 reject·retry-after |
| 필요한 model 미적재 | queue 또는 명시적 cold-start 응답 |

API 계약에 status code와 retry 정책을 명시한다.

### 15.5 Retry storm 방지

client는 지수 backoff와 jitter를 사용하고, server는 `Retry-After` 또는 명시적 재시도 힌트를 제공한다.

```text
즉시 무한 retry
→ overload 증폭
```

### 15.6 Max output 제어

다음은 비용이 크므로 server-side cap을 둔다.

- `max_tokens`
- `n`
- `best_of`
- beam width
- logprobs·top logprobs
- prompt logprobs
- reasoning budget
- tool call loop 횟수

client 입력을 그대로 신뢰하지 않는다.

### 15.7 Deadline-aware admission

request deadline이 2초인데 현재 queue estimate가 5초면 queue에 넣기보다 즉시 reject하는 편이 전체 시스템에 낫다.

```text
estimated_finish
≈ queue_work / service_rate
+ request_prefill_estimate
+ request_decode_estimate
```

### 15.8 Load shedding

overload 시 다음 순서로 품질을 보존할 수 있다.

1. 낮은 priority batch reject
2. 긴 context 요청 별도 queue
3. max output 축소가 허용된 endpoint 적용
4. cache miss 대형 request 제한
5. fallback 작은 모델
6. 전체 reject로 cascade failure 방지

모델을 자동 축소할 때 사용자에게 model·quality 변경을 명시한다.

### 15.9 Queue metric

최소 다음을 수집한다.

- queued request 수
- queued token estimate
- queue time p50/p95/p99
- reject count와 이유
- timeout·client cancel
- retry count
- tenant별 queue share

---

## 16. RAM·VRAM별 30초 구성표

이 표는 모델별 정확한 weight 크기 대신 **동시성 전략**을 제시한다. 모델 파일과 실제 runtime memory는 각 모델 저장소와 [양자화 가이드](./quantization.md)를 함께 확인한다.

### 16.1 단일 장비

| 실사용 메모리 | 생성 모델 예시 | 기본 동시성 | KV 전략 | 운영 형태 |
| ---: | --- | ---: | --- | --- |
| 6–8 GB | 1–3B Q4 | 1 | FP16 4K 또는 Q8 8K | 개인 chat |
| 10–12 GB | 3B Q4/Q5, 7B Q3/Q4 | 1 | 4K–8K, slot 고정 | 개인·개발 API |
| 16 GB | 7–8B Q4 | 1–2 | 8K, Q8 검토 | 1–2명 |
| 24 GB | 7–14B Q4 | 2–4 | paged Q8/FP8 또는 8K cap | 소규모 팀 |
| 32 GB | 14B Q4 또는 7B 고정밀 | 4–8 | prefix cache·chunking | 팀 API |
| 48 GB | 27–32B Q4 또는 14B 다중 | 2–12 | workload별 pool | 중형 워크스테이션 |
| 64 GB | 32B Q4 또는 14B high concurrency | 4–16 | 16K 이하 기본, 긴 요청 분리 | 사내 서비스 |
| 80 GB | 70B Q4 또는 32B 고정밀 | 2–16 | GQA/MLA 구조별 산정 | 단일 datacenter GPU |
| 96–128 GB | 70B Q4/Q5·여러 작은 replica | 4–32 | cache-aware routing | 고성능 Mac·GPU server |
| 192 GB 이상 | 대형 BF16·120B+ low-bit | 설계별 | TP/PP/EP + KV tier | 연구·cluster |

### 16.2 “큰 모델 1개”와 “작은 모델 여러 replica”

| 선택 | 장점 | 단점 |
| --- | --- | --- |
| 큰 모델 1개 | 품질·긴 reasoning | weight가 KV budget을 압박, 장애 domain 큼 |
| 작은 모델 1개 고동시성 | weight reuse·throughput | task 품질 한계 |
| 작은 모델 여러 replica | 격리·autoscale·tail 개선 | weight 중복 |
| router + 크기별 model | 비용·품질 tier | routing·cache 복잡성 |

### 16.3 메모리 여유 권장

고정 비율 하나를 보편적으로 적용할 수 없지만 초기에는 다음 항목을 명시적으로 예약한다.

```yaml
memory_budget:
  weights_gib: null
  kv_pool_gib: null
  prefill_workspace_gib: null
  graph_and_kernel_gib: null
  adapters_and_draft_gib: null
  communication_gib: null
  safety_headroom_gib: null
```

실제 peak를 측정한 뒤 KV pool을 늘린다.

### 16.4 긴 context 전용 pool

```text
/default-chat     max total 8K, 높은 동시성
/long-context     max total 64K, 낮은 동시성
/batch            deadline 기반, streaming 없음
/realtime         짧은 output, 우선순위 높음
```

한 engine에 모두 섞는 것보다 SLO가 예측 가능하다.

---

## 17. 개인 PC·워크스테이션 운영

### 17.1 단일 사용자

개인용은 raw throughput보다 다음이 중요하다.

- 첫 실행 load time
- 단일 stream TPOT
- UI와 시스템 반응성
- idle 메모리 반환
- 모델 전환 시간
- suspend·sleep 복구

동시 slot을 1로 두고 남는 메모리를 context·품질에 배정하는 편이 일반적이다.

### 17.2 2–4명 공유

```text
모델 weight가 메모리의 대부분을 사용
  → 한 모델 + continuous batching

작은 모델이 여러 번 들어감
  → replica 2개와 단일 큰 batch를 비교
```

집·사무실 LAN에서도 API key, bind address와 firewall을 설정한다.

### 17.3 Display GPU

데스크톱 display와 inference가 같은 GPU를 쓰면 다음을 고려한다.

- 브라우저·영상·게임의 VRAM 변동
- desktop compositor
- driver reset 위험
- 그래픽 앱 시작 시 OOM

운영 안전 여유를 서버용 headless GPU보다 크게 둔다.

### 17.4 Sleep·thermal

노트북·Mac은 장시간 serving에서 thermal throttling과 memory pressure가 발생할 수 있다.

- 전원 연결 상태
- fan·온도
- sustained tokens/s
- memory compression·swap
- background app

짧은 benchmark의 peak 수치를 24시간 capacity로 사용하지 않는다.

### 17.5 모델 자동 unload

idle unload는 RAM을 회수하지만 다음 request의 cold-start를 만든다.

평가 항목:

- 모델 load time
- mmap page cache hit
- first request latency
- 여러 모델의 사용 빈도
- SSD read bandwidth와 endurance

### 17.6 Local reverse proxy

서버를 직접 외부에 노출하지 않고 reverse proxy에서 다음을 처리한다.

- TLS
- authentication
- request body limit
- rate limit
- timeout
- CORS
- access log redaction

inference server는 `127.0.0.1` 또는 private network에 bind하는 것을 기본으로 한다.

---

## 18. `llama.cpp` 서버

[`llama.cpp` server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)는 OpenAI-compatible API, parallel slots, continuous batching, prompt cache와 Prometheus endpoint를 제공한다. GGUF와 광범위한 CPU·GPU backend를 사용하는 로컬·혼합 offload 환경에 적합하다.

### 18.1 기본 구성 예제

```bash
llama-server \
  -hf OWNER/MODEL-GGUF:Q4_K_M \
  -c 16384 \
  -np 2 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0 \
  --cache-prompt \
  --metrics \
  --host 127.0.0.1 \
  --port 8080
```

> 현재 flag와 model별 지원은 사용 중인 build의 `llama-server --help`로 확인한다. `-c`와 slot별 실제 `n_ctx` 관계는 version·unified KV 설정에 따라 달라질 수 있으므로 startup log와 `/slots`를 기준으로 검증한다.

### 18.2 핵심 옵션

| 옵션 | 의미 | 튜닝 방향 |
| --- | --- | --- |
| `-c`, `--ctx-size` | prompt context 크기 | 필요 이상 크게 두지 않음 |
| `-np`, `--parallel` | server slot 수 | 동시성 증가, KV 증가 |
| `-b`, `--batch-size` | logical max batch | prefill throughput·RAM |
| `-ub`, `--ubatch-size` | physical micro-batch | peak memory·kernel |
| `-cb` | continuous batching | 현재 기본 활성화 |
| `-ctk`, `-ctv` | K/V cache dtype | Q8부터 검증 |
| `--cache-prompt` | prompt cache | 반복 prefix에 유리 |
| `--cache-reuse` | KV shifting 재사용 최소 chunk | workload별 실험 |
| `--cache-ram` | RAM-side prompt/context cache 상한 | 개인정보·RAM 고려 |
| `--kv-unified` | sequence가 unified KV buffer 공유 | build 기본·slot 동작 확인 |
| `--metrics` | Prometheus endpoint | production 필수 |
| `--slots` | slot 상태 endpoint | 현재 기본 활성화 |
| `--fit` | memory에 맞춰 자동 조정 | 결과 context·offload를 반드시 확인 |

### 18.3 Slot 확인

```bash
curl -s http://127.0.0.1:8080/slots | jq
```

확인할 값:

- slot별 `n_ctx`
- 처리 중 여부
- prompt·generation 속도
- cached token 수
- speculative 활성화

`/slots?fail_on_no_slot=1`은 사용 가능한 slot이 없을 때 503을 반환하도록 확인하는 health/admission 신호로 사용할 수 있다.

### 18.4 Prometheus metric

현재 공식 README에는 다음과 같은 metric이 기재되어 있다.

- `llamacpp:prompt_tokens_total`
- `llamacpp:prompt_tokens_seconds`
- `llamacpp:tokens_predicted_total`
- `llamacpp:predicted_tokens_seconds`
- `llamacpp:requests_processing`
- `llamacpp:requests_deferred`
- `llamacpp:n_tokens_max`
- `llamacpp:n_busy_slots_per_decode`

```bash
curl -s http://127.0.0.1:8080/metrics
```

### 18.5 Context checkpoint와 RAM cache

현재 server에는 slot context checkpoint와 `--cache-ram` 기능이 있다. multi-agent·multi-turn에서 유용할 수 있으나 다음을 측정한다.

- cache RAM 실제 사용량
- idle slot 저장 빈도
- tenant 간 isolation
- restart 후 cache persistence
- unique prompt가 많은 장시간 부하에서 안정성

### 18.6 CPU·GPU split

`llama.cpp`는 일부 layer·MoE expert를 CPU에 두는 구성이 가능하다. 이때 동시성은 GPU compute뿐 아니라 다음에 제한된다.

- CPU memory bandwidth
- PCIe transfer
- NUMA remote memory
- CPU thread contention
- weight page fault

한 request benchmark가 아니라 동시 2·4·8 request에서 TPOT를 측정한다.

### 18.7 권장 튜닝 순서

```text
1. -np 1, 실제 p95 context, FP16 KV 기준선
2. -np 2로 throughput·TPOT 측정
3. Q8 KV 품질·메모리 비교
4. batch/ubatch 조정
5. prompt cache hit 검증
6. slot 4 이상은 SLO goodput 기준으로 결정
```

---

## 19. Ollama

[Ollama FAQ](https://docs.ollama.com/faq)는 모델 lifecycle과 간단한 API 운영에 유용한 환경 변수를 제공한다. 고급 scheduler 세부 제어보다 설치·모델 교체·일반 애플리케이션 통합이 우선일 때 적합하다.

### 19.1 핵심 환경 변수

| 변수 | 역할 | 현재 공식 기본 설명 |
| --- | --- | --- |
| `OLLAMA_MAX_LOADED_MODELS` | 동시에 적재할 모델 수 | GPU 수의 3배 또는 CPU에서 3을 기본으로 설명 |
| `OLLAMA_NUM_PARALLEL` | 모델당 parallel request 수 | 기본 1 |
| `OLLAMA_MAX_QUEUE` | busy 시 queue 상한 | 기본 512 |
| `OLLAMA_CONTEXT_LENGTH` | context 길이 | parallel과 함께 메모리 증가 |
| `OLLAMA_KV_CACHE_TYPE` | KV dtype | FP16·Q8·Q4 계열 |
| `OLLAMA_FLASH_ATTENTION` | Flash Attention | 지원 시 메모리·성능에 영향 |

### 19.2 예제

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_QUEUE=32
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

ollama serve
```

서비스 관리자가 환경 변수를 실제 `ollama serve` process에 전달했는지 확인한다. systemd·launchd·Docker에서는 shell의 export가 daemon에 적용되지 않을 수 있다.

### 19.3 Parallel과 context

공식 FAQ는 parallel request 처리 시 필요한 RAM이 대략 다음에 비례한다고 설명한다.

```text
OLLAMA_NUM_PARALLEL × OLLAMA_CONTEXT_LENGTH
```

예를 들어 2K context와 parallel 4는 총 8K context에 해당하는 추가 메모리를 예약할 수 있다. model architecture와 KV dtype에 따라 실제 bytes는 달라진다.

### 19.4 여러 모델 동시 적재

여러 모델이 모두 VRAM에 들어가면 concurrent load가 가능하지만, 그렇지 않으면 request가 queue되거나 기존 모델이 unload될 수 있다.

운영 선택:

- 한 모델 상주 + endpoint 통합
- 모델별 전용 Ollama instance
- 작은 embedding model은 별도 process
- 사용 빈도가 낮은 모델은 cold load 허용

### 19.5 Queue를 줄여야 하는 이유

기본 queue가 애플리케이션 SLO에 비해 너무 크면 사용자는 긴 시간 대기한 뒤 timeout될 수 있다. front proxy의 timeout보다 작은 queue-time 목표를 두고 overload를 빨리 알린다.

### 19.6 KV cache dtype

```text
f16: 품질 기준선
q8_0: 메모리 절감의 일반적 첫 후보
q4_0: 더 큰 절감, 긴 context 품질 검증 필수
```

Flash Attention과 KV quantization 지원은 GPU·model·Ollama version별로 확인한다.

### 19.7 Ollama가 적합한 경우

- 로컬 개발과 desktop 앱
- 모델 설치·태그 관리가 중요
- 복잡한 distributed serving이 필요 없음
- 1–4 parallel request 수준
- 사용자가 runtime 세부 flag를 최소화하고 싶음

### 19.8 다른 runtime을 검토할 신호

- queue·batch·KV pool을 세밀하게 제어해야 함
- 높은 QPS와 p99 SLO
- TP·EP·P/D disaggregation
- cache-aware routing
- multi-node
- per-request scheduling priority

---

## 20. vLLM

[vLLM](https://docs.vllm.ai/)은 paged KV, continuous batching, prefix caching, quantized KV, distributed serving와 OpenAI-compatible API를 제공한다. NVIDIA·AMD 등 지원 환경에서 고동시성 text·VLM serving의 주요 선택지다.

### 20.1 보수적 시작 예제

```bash
MODEL="OWNER/MODEL"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.88 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-per-request-metrics
```

flag 기본값과 이름은 release마다 변할 수 있다. production에서는 version·container digest를 고정하고 `vllm serve --help` 출력도 manifest에 보존한다.

### 20.2 핵심 메모리 옵션

| 옵션 | 의미 | 주의 |
| --- | --- | --- |
| `--gpu-memory-utilization` | model·KV pool 등을 위한 device memory 비율 | 다른 process·graph peak 여유 |
| `--kv-cache-memory-bytes` | GPU별 KV cache 크기를 직접 지정 | 현재 문서상 설정 시 utilization보다 우선 |
| `--kv-cache-dtype` | KV dtype | FP8 지원 hardware·model 검증 |
| `--max-model-len` | 최대 sequence 길이 | 높이면 KV capacity 감소 |
| `--max-num-seqs` | iteration당 최대 sequence | 메모리와 TPOT trade-off |
| `--max-num-batched-tokens` | iteration token cap | prefill peak·throughput |
| `--cpu-offload-gb` | GPU별 weight CPU offload | 빠른 interconnect 필요 |
| `--kv-offloading-size` | CPU KV offload 공간 | latency·RAM bandwidth 측정 |

### 20.3 자동 KV 계산 확인

vLLM worker는 model을 적재하고 memory profile을 수행해 KV cache용 free memory를 산정한다. startup log에서 다음을 기록한다.

- model weight memory
- non-KV memory
- KV block 수
- 최대 cache token 수
- estimated maximum concurrency

자동 추정값은 실제 SLO concurrency가 아니라 메모리 상한에 가깝다.

### 20.4 `max_num_seqs`

`max_num_seqs`를 크게 하면 decode throughput이 증가할 수 있으나 다음이 늘어난다.

- scheduler state
- active KV
- output sampling
- TPOT·ITL
- preemption 가능성

1, 2, 4, 8, 16, 32처럼 단계적으로 sweep한다.

### 20.5 `max_num_batched_tokens`

작으면:

- prefill peak 감소
- 긴 prompt가 여러 iteration으로 분할
- overhead 증가 가능

크면:

- prefill throughput 증가 가능
- activation·workspace peak 증가
- decode 간섭 가능

interactive와 batch traffic에 같은 값을 쓰지 않을 수 있다.

### 20.6 Prefix caching

반복 system prompt·multi-turn·RAG에 `--enable-prefix-caching`을 검토한다. 다음 조건을 검증한다.

- tokenizer와 rendered prompt 완전 일치
- LoRA·multimodal cache key
- tenant isolation
- cache hit metric
- replica routing

### 20.7 Per-request metrics

현재 최신 vLLM 문서는 `--enable-per-request-metrics`로 response usage에 request별 timing 정보를 포함하는 기능을 안내한다. 예시 metric은 다음 계열이다.

- `time_to_first_token_ms`
- `generation_time_ms`
- `queue_time_ms`
- `mean_itl_ms`
- `tokens_per_second`

높은 concurrency에서는 response 생성 CPU overhead가 추가될 수 있으므로 production에서 필요 범위를 평가한다.

### 20.8 Load test

```bash
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 1000 \
  --request-rate 4
```

설치 version의 help에서 정확한 dataset 인자명을 확인한다. synthetic fixed-length test 외에 실제 trace 또는 길이 분포를 재현한다.

### 20.9 측정 metric

vLLM benchmark는 TTFT, TPOT, ITL, E2EL과 percentile·goodput SLO 평가를 지원한다. 다음 sweep을 만든다.

```text
request_rate: 0.5, 1, 2, 4, 8, 16
max_num_seqs: 4, 8, 16, 32
max_num_batched_tokens: 2K, 4K, 8K, 16K
KV dtype: BF16/FP16, FP8
prefix cache: off, on
```

### 20.10 Disaggregated prefill

vLLM의 현재 문서는 disaggregated prefilling을 experimental 기능으로 설명한다. 주 목적은 prefill과 decode를 분리해 tail ITL을 제어하는 것이며, 공식 문서상 **처리량을 자동으로 개선하는 기능은 아니다**. KV transfer 비용과 P/D sizing을 포함해 검증한다.

### 20.11 Production 체크리스트

```text
[ ] model·tokenizer revision 고정
[ ] max model/input/output tokens 고정
[ ] KV pool과 graph peak 실측
[ ] prefix cache namespace 설계
[ ] queue·rate limit은 gateway에도 설정
[ ] p95/p99 TTFT·TPOT·ITL 기록
[ ] cancel propagation 테스트
[ ] OOM 후 worker recovery 테스트
[ ] multi-GPU topology·NCCL 검증
[ ] metrics label에 prompt·tenant PII 없음
```

---

## 21. SGLang

[SGLang](https://docs.sglang.ai/)은 RadixAttention 기반 prefix reuse, continuous batching, paged attention, structured output, speculative decoding, TP·PP·DP·EP, prefill-decode disaggregation과 multi-LoRA를 제공한다. 반복 prefix가 많은 agent·RAG·structured generation과 대규모 MoE serving에서 주요 선택지다.

### 21.1 단일 GPU 시작 예제

```bash
MODEL="OWNER/MODEL"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 \
  --port 30000 \
  --context-length 16384 \
  --mem-fraction-static 0.82 \
  --max-running-requests 16 \
  --max-queued-requests 64 \
  --chunked-prefill-size 4096 \
  --enable-metrics
```

현재 정확한 flag와 기본값은 [`Server Arguments`](https://docs.sglang.io/docs/advanced_features/server_arguments)와 설치 version의 `--help`를 확인한다.

### 21.2 핵심 메모리 옵션

| 옵션 | 의미 | 튜닝 방향 |
| --- | --- | --- |
| `--mem-fraction-static` | model weights와 KV pool의 static allocation 비율 | OOM 시 낮춤 |
| `--max-running-requests` | running request 상한 | KV와 TPOT 제어 |
| `--max-queued-requests` | queued request 상한 | overload fail-fast |
| `--max-total-tokens` | memory pool의 최대 token 수 | 개발·명시적 capacity 제어 |
| `--chunked-prefill-size` | prefill chunk 상한 | 긴 prompt peak 제어 |
| `--prefill-max-requests` | prefill batch request 상한 | prefill burst 제한 |
| `--max-prefill-tokens` | prefill token budget | TTFT·peak trade-off |
| `--kv-cache-dtype` | KV dtype | `fp8_e4m3`·`fp8_e5m2` 등 지원 확인 |
| `--stream-interval` | streaming flush 간격 | 체감 ITL과 CPU overhead |

공식 문서는 memory OOM 시 `--mem-fraction-static`을 줄이고, 긴 prompt prefill OOM 시 `--chunked-prefill-size 4096` 같은 더 작은 chunk를 시도하도록 안내한다.

### 21.3 Radix cache

RadixAttention은 공통 token prefix를 radix tree로 관리해 multi-turn·few-shot·agent prompt를 재사용한다.

잘 맞는 workload:

- 동일 tool schema를 쓰는 agent
- 같은 문서에 대한 반복 질문
- 공통 긴 system prompt
- branching reasoning
- 다수 사용자의 동일 public prefix

다음은 cache miss를 일으킨다.

- timestamp가 prefix 앞부분에 있음
- tool schema 순서가 request마다 다름
- tenant별 policy가 섞임
- adapter·model revision 차이

### 21.4 Scheduler

현재 server argument에는 FCFS 등 scheduling policy와 priority 관련 설정이 있다. priority를 활성화할 때 다음을 추가한다.

- tenant별 허용 priority 범위
- 낮은 priority starvation 방지
- priority 요청의 max token cap
- queue time과 preemption metric

### 21.5 Data parallel router

여러 DP replica를 사용할 때 단순 request count보다 다음을 고려하는 router가 유리하다.

- active token 수
- KV cache utilization
- prefix cache locality
- prefill/decode 부하
- LoRA adapter locality
- tenant·session affinity

SGLang Model Gateway·DP-aware router 문서를 확인해 version에 맞는 배치를 선택한다.

### 21.6 Expert parallelism

MoE에서는 expert를 GPU 사이에 분산해 resident weight 병목을 줄일 수 있다. 그러나 all-to-all, expert load imbalance와 hot expert가 throughput을 제한할 수 있다.

측정값:

- expert별 token count
- all-to-all 시간
- communication/compute overlap
- rank별 memory imbalance
- EPLB 또는 rebalance 효과

### 21.7 HiCache·secondary cache

SGLang의 HiCache 계열은 GPU 외 계층의 KV cache를 활용하는 구성을 제공한다. 적용 전 다음을 분리 평가한다.

```text
L1: GPU KV
L2: CPU RAM
L3: remote·distributed storage
```

- tier별 hit rate
- promote·evict bandwidth
- cold recompute 대비 latency
- tenant isolation
- cache invalidation

### 21.8 Metrics

현재 [`Production Metrics`](https://docs.sglang.io/docs/references/production_metrics)는 `--enable-metrics`와 `/metrics` endpoint를 안내한다. 대표 metric은 다음과 같다.

- `sglang:time_to_first_token_seconds`
- `sglang:time_per_output_token_seconds`
- `sglang:e2e_request_latency_seconds`
- `sglang:num_running_reqs`
- `sglang:num_queue_reqs`
- `sglang:num_used_tokens`
- `sglang:gen_throughput`
- `sglang:cache_hit_rate`
- prompt·generation token counters

### 21.9 Benchmark

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 1000 \
  --random-input 1024 \
  --random-output 256 \
  --random-range-ratio 0.5
```

실제 CLI는 설치 version에서 확인하고, fixed length뿐 아니라 length range·cache hit/cold·burst traffic을 포함한다.

### 21.10 권장 튜닝 순서

```text
1. mem-fraction-static을 보수적으로 시작
2. startup에서 max token pool 확인
3. max-running-requests sweep
4. 긴 prompt 혼합 후 chunked-prefill-size 조정
5. radix cache hit rate 확인
6. FP8 KV A/B
7. DP·EP·P/D 분리는 단일 node 기준선 후
```

---

## 22. TensorRT-LLM

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)은 NVIDIA GPU에 특화된 LLM inference stack이다. in-flight batching, paged attention, KV block reuse, quantization, speculative decoding, TP·PP·EP와 disaggregated serving을 제공한다.

### 22.1 적합한 경우

- NVIDIA datacenter GPU 중심
- model·GPU 조합을 고정할 수 있음
- build·recipe·kernel tuning 비용을 감수 가능
- 높은 throughput 또는 엄격한 latency 목표
- multi-GPU·multi-node MoE serving

### 22.2 주의할 경우

- GPU 세대가 다양함
- 모델이 자주 바뀜
- Mac·CPU·AMD까지 같은 runtime이 필요
- custom architecture가 빠르게 변함
- 작은 팀이 build artifact를 지속 관리하기 어려움

### 22.3 현재 workflow

TensorRT-LLM 최신 문서는 신규 project에 `trtllm-serve` 또는 LLM Python API 사용을 안내하며, legacy engine-build 경로는 문서상 제거·이전 대상으로 구분한다. 사용 중인 release의 Quick Start와 model recipe를 기준으로 한다.

개념적 시작 흐름:

```text
release container 고정
→ supported model recipe 확인
→ model·tokenizer revision 고정
→ trtllm-serve 또는 LLM API 실행
→ KV cache·batch·parallelism 설정
→ AIPerf/benchmark
→ engine·config·container digest 보존
```

### 22.4 KV cache system

현재 [`KV Cache System`](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)은 다음 기능을 설명한다.

- paged KV block pool
- MHA·MQA·GQA 지원
- block reuse와 prioritized LRU
- GPU free memory fraction 기반 pool
- token 수 기반 상한
- host cache offload
- sliding/limited attention window
- cache salting
- multimodal cache identification

### 22.5 Memory fraction

현재 문서는 KV cache의 `free_gpu_memory_fraction` 기본값을 free memory의 0.9로 설명하며, `max_tokens`로 상한을 둘 수 있다. 이 값은 physical GPU 전체가 아니라 model load 후 free memory에 적용되는 설정이므로, 실제 release config를 확인한다.

### 22.6 Block reuse

block reuse는 동일 prefix KV를 재사용한다. 현재 문서상 기본 활성화된 구성에서 다음을 검증한다.

- tenant별 `cache_salt`
- priority·duration과 eviction
- request가 실제 동일 token prefix인지
- multimodal UUID/content hash
- host cache hit latency

### 22.7 Cache salting

```text
cache_salt = tenant 또는 security domain별 비밀/불투명 식별자
```

동일 salt 요청끼리만 cache block을 재사용하게 해 cross-tenant prompt theft 위험을 줄인다. user-visible ID를 그대로 쓰기보다 server가 관리하는 namespace를 사용한다.

### 22.8 In-flight batching

TensorRT-LLM의 in-flight batching은 request가 끝나는 대로 새 request를 batch에 넣는다. 조정 항목:

- max batch size
- max num tokens
- scheduler policy
- context chunking
- CUDA graph batch sizes
- KV cache block 수

### 22.9 Quantization과 GPU 세대

FP8·FP4·INT4 kernel은 GPU 세대와 model recipe에 크게 의존한다. 다음을 artifact에 기록한다.

```yaml
trtllm_artifact:
  release: null
  container_digest: null
  gpu_model: null
  compute_capability: null
  model_revision: null
  quantization: null
  backend: pytorch_or_other
  world_size: null
  recipe_commit: null
```

### 22.10 Release note 검증

TensorRT-LLM은 feature와 지원 조합이 빠르게 변하므로 다음을 배포 전 확인한다.

- known issue
- removed legacy backend
- model-specific limitation
- KV reuse·offload 호환성
- speculative decoding 제한
- multi-node topology
- driver·CUDA·NCCL matrix

### 22.11 운영 체크리스트

```text
[ ] release container digest 고정
[ ] model recipe와 GPU 세대 일치
[ ] cold/warm KV reuse benchmark
[ ] cache salt tenant isolation
[ ] host cache bandwidth 측정
[ ] max batch/token sweep
[ ] P/D·EP 통신 profile
[ ] worker crash·restart·cache loss 테스트
```

---

## 23. Hugging Face TGI

[Text Generation Inference](https://huggingface.co/docs/text-generation-inference/)는 Hugging Face model 배포, continuous batching, tensor parallelism, quantization, Prometheus·OpenTelemetry와 OpenAI-compatible integration을 제공한다. 기존 TGI 표준 배포와 Hugging Face 생태계 통합이 중요할 때 유용하다.

> **현재 상태:** Hugging Face 공식 문서는 TGI가 **maintenance mode**이며 앞으로 minor bug fix, 문서 개선과 경량 유지보수 중심으로 운영된다고 명시한다. 신규 배포는 공식 권고에 따라 vLLM·SGLang을 우선 비교하고, 로컬 상호운용성이 중요하면 `llama.cpp`·MLX도 평가한다. 이 장은 기존 TGI 설치의 안정 운영과 마이그레이션 판단을 위해 유지한다.

### 23.1 기본 container 예제

```bash
MODEL="OWNER/MODEL"
DATA="$PWD/tgi-data"

mkdir -p "$DATA"

docker run --rm --gpus all \
  --shm-size 1g \
  -p 127.0.0.1:8080:80 \
  -v "$DATA:/data" \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id "$MODEL" \
  --max-concurrent-requests 32 \
  --max-input-tokens 8192 \
  --max-total-tokens 9216 \
  --max-batch-prefill-tokens 8192
```

production에서는 `latest` 대신 검증한 version·digest를 고정한다.

### 23.2 핵심 launcher 옵션

| 옵션 | 역할 | 현재 공식 설명의 핵심 |
| --- | --- | --- |
| `--max-concurrent-requests` | deployment 동시 request 상한 | 낮은 값으로 오래 기다리기보다 reject 가능, 기본 128 |
| `--max-input-tokens` | 사용자 input 상한 | 길수록 memory 영향 증가 |
| `--max-total-tokens` | request당 input+output memory budget | 가장 중요한 값 중 하나 |
| `--max-batch-prefill-tokens` | prefill token 상한 | prefill은 compute·memory가 큼 |
| `--max-batch-total-tokens` | batch 전체 잠재 token 상한 | 남은 memory 활용의 핵심 제어 |
| `--waiting-served-ratio` | waiting/running 합류 조건 | prefill과 decode 간섭 |
| `--max-waiting-tokens` | 기다리는 request를 batch에 합류시키는 간격 | 너무 작거나 크면 latency 악화 |
| `--kv-cache-dtype` | CUDA FP8 KV | 지원 조합 확인 |
| `--cuda-memory-fraction` | visible CUDA memory 제한 | sidecar·다중 process 여유 |
| `--max-best-of` | 후보 생성 상한 | sequence 폭증 방지 |
| `--max-client-batch-size` | client batch 상한 | abuse·peak 방지 |

### 23.3 Backpressure

TGI 공식 launcher 설명은 `max-concurrent-requests`를 낮게 두면 request가 오래 기다리는 대신 거절되어 backpressure를 올바르게 처리하는 데 유리할 수 있다고 설명한다. gateway queue와 engine queue를 모두 무제한으로 두지 않는다.

### 23.4 Total token budget

```text
max_total_tokens
= input tokens + max_new_tokens
```

TGI 공식 문서는 이 값을 running client의 memory budget을 정의하는 핵심 값으로 설명한다. 사용자에게 model max context 전체를 기본 제공하지 않는다.

### 23.5 Prefill logprobs

prompt logprobs는 긴 prompt에서 큰 VRAM을 사용할 수 있어 현재 launcher 문서상 기본 비활성화되어 있다. 필요 endpoint만 별도로 허용한다.

### 23.6 기존 TGI 환경을 유지할 이유

- 검증된 production image·monitoring 표준이 이미 있음
- Hugging Face private/gated model workflow가 운영 절차에 통합되어 있음
- validation worker·payload limit·API key 정책이 안정화되어 있음
- 현재 model·hardware 조합의 SLO와 장애 대응이 충분히 검증됨
- 즉시 교체 비용이 기대 성능·기능 개선보다 큼

### 23.7 신규 배포·마이그레이션 판단

| 상황 | 권장 판단 | 반드시 비교할 항목 |
| --- | --- | --- |
| 신규 NVIDIA text API | vLLM·SGLang 우선 평가 | model 지원, TTFT·TPOT·goodput, KV 기능 |
| NVIDIA 고정 recipe·최적화 | TensorRT-LLM 추가 평가 | GPU 세대, build·release 운용 비용 |
| 개인·로컬·Apple 환경 | `llama.cpp`·MLX 우선 평가 | GGUF·MLX 생태계, 메모리, 앱 통합 |
| 안정적인 기존 TGI 서비스 | version·digest를 고정해 유지 가능 | security fix, 신규 model 지원, 운영 인력 |
| 기능·model 지원이 막힘 | 단계적 migration | API 호환, metric 변환, cache warmup, rollback |

새 deployment에서는 동일 model·hardware·traffic trace에서 vLLM·SGLang·TensorRT-LLM과 SLO goodput을 비교한다. 기존 TGI를 유지할 때도 maintenance mode의 범위를 고려해 security advisory, dependency와 container digest를 별도로 추적한다.

---

## 24. Apple Silicon·CPU·AMD

### 24.1 Apple Silicon 통합 메모리

Mac에서는 CPU·GPU·Neural Engine·앱이 unified memory를 공유한다.

```text
available unified memory
= physical unified memory
- macOS·WindowServer
- active applications
- model weights
- KV cache
- Metal graph·temporary buffer
- file cache·memory compression 여유
```

32GB Mac에 30GB model file이 들어간다고 안정적인 server가 되는 것은 아니다.

### 24.2 Apple runtime 선택

| runtime | 강점 | 서빙 관점 |
| --- | --- | --- |
| MLX-LM | Apple Silicon 최적화, quantized model, batch generation | custom app·worker에 적합, production gateway는 별도 설계 |
| `llama.cpp` Metal | GGUF·HTTP server·slots·metrics | 범용 local API에 적합 |
| Ollama | 설치·model lifecycle | 간단한 desktop/team API |

MLX-LM은 batch generation과 distributed inference 기능을 제공하지만, multi-tenant queue·rate limit·metrics를 포함한 완성형 production control plane으로 가정하지 않는다.

### 24.3 Mac 동시성

- 작은 model과 2–4 sequence부터 시작
- memory pressure·swap·compressed memory 확인
- GPU history와 CPU package power 확인
- 브라우저·IDE가 동시에 실행되는 실제 환경에서 측정
- thermal steady state에서 30–60분 sustained test

### 24.4 CPU-only serving

CPU inference는 RAM보다 memory bandwidth와 NUMA가 병목인 경우가 많다.

체크 항목:

- physical core와 SMT
- memory channel 수·속도
- NUMA node별 model placement
- first-touch allocation
- thread pinning
- mmap·mlock
- batch size와 cache locality
- tokenizer thread 경쟁

### 24.5 NUMA

model weight가 여러 socket에 분산되면 remote memory access가 늘 수 있다.

```text
single replica per NUMA node
vs
one large replica across sockets
```

두 구성을 실제 동시 request에서 비교한다. 단일 request 최고 tok/s만으로 결정하지 않는다.

### 24.6 AMD GPU

AMD ROCm에서 runtime·model·quantization 지원은 release별로 확인한다.

- vLLM·SGLang ROCm image/version
- attention backend
- AWQ·GPTQ·FP8 kernel 지원
- multi-GPU RCCL topology
- Windows와 Linux 차이
- monitoring metric

지원되지 않는 quantization은 fallback kernel로 실행되어 메모리는 줄어도 속도가 크게 낮을 수 있다.

### 24.7 Mixed CPU·GPU offload

weight 일부를 CPU에 두면 큰 model을 실행할 수 있지만 concurrent decode에서 PCIe가 병목이 된다.

```text
single-user capacity 모드
→ offload 허용

multi-user latency 모드
→ weight를 GPU에 완전히 적재하거나 더 작은 model
```

### 24.8 Storage

mmap model은 load time과 RAM 사용에 유리할 수 있으나 다음이 중요하다.

- NVMe random/read throughput
- page cache
- filesystem compression
- network filesystem latency
- 여러 replica 동시 cold start
- model shard count

autoscaling 시 수십 pod가 동시에 Hub에서 download하지 않도록 local cache·image·model cache를 설계한다.

---

## 25. 다중 GPU·TP·PP·DP·EP·CP

### 25.1 어떤 parallelism을 쓰는가

```text
모델이 GPU 한 장에 들어감
  → 먼저 단일 GPU 또는 DP replica

모델 weight가 한 장에 안 들어감
  → TP 또는 PP

MoE expert weight·compute가 병목
  → EP

아주 긴 context KV·attention이 병목
  → CP/SP 또는 KV sharding 지원 검토

throughput을 늘리고 모델이 각 replica에 들어감
  → DP replica
```

### 25.2 Tensor parallelism

TP는 layer 내부 tensor를 GPU에 나눈다.

장점:

- per-GPU weight 감소
- 큰 model 실행
- 한 request latency 개선 가능

비용:

- layer마다 collective communication
- PCIe·NVLink·InfiniBand topology 의존
- 작은 batch에서 통신 비중 증가
- KV head replication 가능

### 25.3 TP rank 수를 늘릴 때

| 효과 | 기대 | 현실의 제한 |
| --- | --- | --- |
| weight/GPU | 대략 감소 | 일부 tensor replication·padding |
| KV/GPU | 감소 가능 | GQA/MQA head replication |
| compute/GPU | 감소 | collective overhead |
| throughput | 증가 가능 | latency·batch·network에 따라 감소 가능 |
| fault domain | rank 수 증가 | 한 rank 실패로 전체 replica 중단 |

모델이 1 GPU에 들어가면 TP=2보다 GPU별 replica 2개가 throughput과 격리에 더 나을 수 있다.

### 25.4 Pipeline parallelism

PP는 layer를 stage로 나눈다.

- model이 단일 node TP로도 안 들어갈 때
- stage별 weight 불균형 주의
- micro-batch가 작으면 pipeline bubble
- first token latency 증가 가능
- stage 간 activation transfer

### 25.5 Data parallelism

DP는 model weight를 replica마다 복제한다.

장점:

- request isolation
- scale-out·rolling update 단순
- cache·fault domain 분리
- model이 한 GPU에 들어가면 효율적

비용:

- weight memory 중복
- cache-aware routing 필요
- replica별 load imbalance

### 25.6 Expert parallelism

EP는 MoE expert를 rank에 분산한다.

- all-to-all bandwidth
- token routing skew
- expert load balancing
- shared expert 처리
- DP attention과 조합

active parameter 수가 적어도 expert weight를 rank 전체에 배치해야 하므로 topology가 중요하다.

### 25.7 Context·sequence parallelism

긴 sequence의 attention·KV를 여러 GPU에 나누는 기능은 runtime·model별 지원이 다르다.

적용 신호:

- weight는 들어가지만 128K+ KV가 안 들어감
- 단일 request latency보다 context capacity가 중요
- high-bandwidth interconnect 사용 가능

### 25.8 GPU topology

```bash
nvidia-smi topo -m
```

확인할 것:

- NVLink/NVSwitch
- PCIe switch
- CPU socket affinity
- NIC proximity
- GPU Direct RDMA
- MIG partition

TP·EP rank를 topology를 무시한 순서로 배치하지 않는다.

### 25.9 Multi-node

multi-node에서는 다음을 별도 측정한다.

- model load synchronization
- NCCL init time
- collective p95
- network congestion
- node failure recovery
- rolling deployment 중 version skew
- clock synchronization

### 25.10 Parallelism manifest

```yaml
parallelism:
  world_size: 8
  tensor_parallel: 4
  pipeline_parallel: 1
  data_parallel: 2
  expert_parallel: 1
  context_parallel: 1
  topology: "2 nodes x 4 GPUs, NVLink intra-node, IB inter-node"
```

---

## 26. Replica·load balancing·autoscaling

### 26.1 Replica 수

```text
필요 replica
≈ peak goodput demand / replica당 SLO goodput
```

raw maximum throughput이 아니라 목표 TTFT·TPOT를 만족하는 goodput을 사용한다.

### 26.2 Round-robin의 한계

request 수가 같아도 다음이 다르다.

- input token 길이
- output token 길이
- active KV token
- cache hit
- LoRA adapter
- multimodal encoder cost

token-aware·KV-aware routing이 load balance를 개선할 수 있다.

### 26.3 Routing 신호

| 신호 | 장점 | 위험 |
| --- | --- | --- |
| running request 수 | 단순 | 길이 차이 무시 |
| queued token | 작업량 반영 | output 길이 추정 필요 |
| active KV token | memory 압박 반영 | metric freshness |
| prefix cache hit | TTFT 절감 | replica 고착·imbalance |
| adapter locality | adapter load 절감 | hot adapter 편중 |
| estimated finish time | SLO 지향 | estimator 복잡 |

### 26.4 Autoscaling 지표

GPU utilization만으로 scale하지 않는다. decode는 memory-bound이고, queue가 늘어도 utilization이 이미 높거나 반대로 CPU 병목으로 낮을 수 있다.

권장 신호 조합:

- queue time·depth
- running request/token
- TTFT SLO violation
- KV utilization
- prefill·decode token rate
- reject rate
- replica warmup state

### 26.5 Ray Serve

[Ray Serve autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)은 ongoing request 수와 queue를 기반으로 replica를 조정할 수 있다. 현재 `num_replicas="auto"` 예시는 `max_ongoing_requests=5`, `target_ongoing_requests=2`, `min_replicas=1`, `max_replicas=100` 기본 구성을 설명하지만, LLM request 길이에 따라 값을 직접 튜닝해야 한다.

긴 request일수록 target ongoing request를 낮게 두는 것이 일반적이다. `max_ongoing_requests`를 너무 크게 두면 한 replica에 traffic이 몰려 tail latency가 악화할 수 있다.

### 26.6 예시 Ray Serve 개념 구성

```yaml
num_replicas: auto
max_ongoing_requests: 4
autoscaling_config:
  target_ongoing_requests: 2
  min_replicas: 1
  max_replicas: 8
  upscale_delay_s: 5
  downscale_delay_s: 300
```

정확한 field는 설치한 Ray version 문서를 확인한다. scale-down은 model cold start와 cache loss를 고려해 느리게 설정하는 경우가 많다.

### 26.7 Cold start

autoscaling latency:

```text
pod scheduling
+ image pull
+ model artifact load
+ kernel compile·graph capture
+ health check
+ cache warmup
```

몇 초가 아니라 수분이 걸릴 수 있으므로 spike가 오기 전에 predictive scale 또는 minimum replica를 둔다.

### 26.8 Scale-down과 cache

replica를 줄이면 해당 replica의 prefix·session KV가 사라질 수 있다.

- drain 후 종료
- 새 request 차단
- active request 완료
- session route 이동
- remote KV 또는 recompute
- cache-aware router 업데이트

### 26.9 KServe·llm-d

KServe generative inference stack과 llm-d 계열은 Kubernetes에서 model serving, KV-aware routing, disaggregation과 autoscaling을 구성할 수 있다. cluster 도입 전 단일 runtime 기준선을 확보하고, control plane overhead와 KV network cost를 별도 측정한다.

### 26.10 HPA만으로 부족한 이유

CPU·GPU utilization HPA는 다음을 보지 못할 수 있다.

- queued token 폭증
- cache hit 감소
- 긴 request arrival
- p99 TTFT
- model cold loading
- KV memory saturation

custom metric이나 serving-aware autoscaler를 사용한다.

### 26.11 Overprovisioning

peak 예측 오차, node failure와 rolling update를 위해 headroom replica를 둔다. 비율은 traffic 변동과 cold start에 따라 결정하고, 고정 20% 규칙을 보편적으로 적용하지 않는다.

---

## 27. Prefill-decode 분리와 KV 전송

### 27.1 왜 분리하는가

Prefill은 compute-heavy, decode는 memory bandwidth·KV-heavy 성격이 강하다. 같은 worker에서 혼합하면 긴 prefill이 decode ITL을 방해할 수 있다.

```text
Prefill workers
  prompt 처리 + KV 생성
        ↓ KV transfer
Decode workers
  token generation
```

### 27.2 기대 효과

- decode ITL isolation
- prefill·decode GPU 비율 독립 조정
- 서로 다른 GPU 세대 사용 가능
- 긴 prompt burst를 별도 흡수
- P/D별 kernel·batch 최적화

### 27.3 추가 비용

- KV transfer bandwidth
- serialization·metadata
- request orchestration
- retry·failure recovery
- cache consistency
- extra queue
- network topology

### 27.4 처리량이 자동 증가하지 않음

분리는 tail latency 제어에 유리할 수 있지만, KV 전송과 자원 불균형 때문에 total throughput이 감소할 수 있다. vLLM 공식 experimental 문서도 disaggregated prefilling이 throughput을 자동 개선하지 않는다고 명시한다.

### 27.5 KV 전송량 근사

```text
KV transfer bytes
≈ prompt cached tokens
× KV bytes/token
```

128 KiB/token 구조에서 16K prompt의 KV는 약 2 GiB다. request당 이 데이터를 전송하면 100Gbps network에서도 serialization·contention을 포함해 무시할 수 없는 시간이 든다.

### 27.6 P/D 비율

```text
prefill worker demand
≈ arrival_rate × E[uncached_input_tokens]

decode worker demand
≈ arrival_rate × E[output_tokens]
```

cache hit가 높아지면 prefill demand가 줄고 필요한 P worker 비율도 달라진다.

### 27.7 Network

확인할 항목:

- NVLink/NVSwitch
- InfiniBand·RoCE
- GPUDirect RDMA
- PCIe staging 여부
- topology-aware placement
- NIC/GPU NUMA affinity
- concurrent transfer

### 27.8 Failure 처리

```text
Prefill 완료 후 decode worker 실패
→ KV 재전송 또는 prefill 재실행

KV transfer 중 timeout
→ idempotent request ID와 retry policy

worker version 불일치
→ KV layout 호환성 검증 후 reject
```

### 27.9 Remote KV cache

[KServe KV cache offloading](https://kserve.github.io/website/docs/model-serving/generative-inference/kvcache-offloading)은 vLLM과 LMCache를 연결해 CPU RAM·Redis·LMCache server 등 remote backend를 사용하는 예를 제공한다. remote cache는 재사용과 scale-out에 유용하지만, 저장소 자체의 보안·capacity·eviction·network SLO가 필요하다.

### 27.10 도입 기준

P/D 분리를 검토할 신호:

- 통합 scheduler에서 p99 ITL이 긴 prefill에 의해 악화
- prefill와 decode demand가 시간대별로 다름
- 충분한 고속 network
- 단일 node chunked prefill로 해결되지 않음
- cluster 운영 역량이 있음

---

## 28. Speculative decoding

### 28.1 원리

작은 draft model 또는 MTP·EAGLE·n-gram 방식이 여러 token을 제안하고 target model이 한 번에 검증한다.

```text
draft: t1, t2, t3, t4 제안
main: 4개를 병렬 검증
accepted prefix만 사용
```

### 28.2 기대 효과

- 낮은 batch에서 single-request latency 감소 가능
- code·반복 문구에서 높은 acceptance
- output token당 main model forward 횟수 감소 가능

### 28.3 추가 메모리

- draft model weights
- draft KV cache
- draft activation
- verification token batch
- speculative metadata
- 추가 graph capture

```text
spec serving memory
≈ main serving memory
+ draft weights/KV/workspace
```

### 28.4 높은 concurrency에서의 trade-off

main model이 이미 큰 batch로 효율적으로 동작하면 speculation이 verification overhead와 memory를 추가해 throughput을 악화할 수 있다.

따라서 다음을 분리 측정한다.

- concurrency 1·2·4·8·16
- acceptance rate
- accepted tokens/step
- TTFT·TPOT
- total output throughput
- power·GPU utilization

### 28.5 Draft model 방식

작은 model을 추가로 적재한다. draft와 main tokenizer·vocabulary·architecture 호환성을 확인한다.

적합:

- single-user latency
- 작은 draft가 충분히 정확
- weight 여유가 있음

### 28.6 MTP·EAGLE 계열

model-specific head·draft artifact를 사용한다. release와 checkpoint 호환성이 중요하며, 다음을 고정한다.

- target revision
- draft revision
- speculative method
- draft token 수
- top-k·threshold
- KV dtype

### 28.7 N-gram speculation

별도 neural draft weight 없이 prompt·generated text의 반복 패턴을 이용할 수 있다. code completion·문서 복사에서 유리할 수 있으나 창의적 생성에서는 acceptance가 낮을 수 있다.

### 28.8 Draft token 수

너무 적으면 speedup이 작고, 너무 많으면 rejected token compute가 늘어난다.

```text
1, 2, 4, 8 draft tokens sweep
```

workload별 acceptance curve로 결정한다.

### 28.9 품질

올바르게 구현된 speculative decoding은 target distribution을 보존하도록 설계되지만, sampling·quantization·implementation 조합에 따라 차이가 생길 수 있다. deterministic test와 output distribution regression을 수행한다.

### 28.10 적용 순서

```text
1. 기본 runtime·batch·KV tuning
2. latency-bound인지 확인
3. n-gram 또는 supported draft baseline
4. concurrency별 A/B
5. memory 증가 대비 SLO goodput 개선 여부로 채택
```

---

## 29. Multi-LoRA·다중 모델·model multiplexing

### 29.1 Multi-LoRA

하나의 base model에 여러 LoRA adapter를 요청별로 적용하면 weight 중복을 줄일 수 있다.

```text
resident memory
≈ base model
+ loaded adapter weights
+ adapter workspace/cache
+ KV
```

### 29.2 Adapter가 동시성을 줄이는 경우

- adapter rank가 큼
- 수백 개 adapter를 device에 상주
- request마다 adapter가 달라 batch fusion이 어려움
- hot adapter와 cold adapter load가 반복
- adapter별 cache namespace가 분리

### 29.3 Adapter lifecycle

```yaml
adapter_policy:
  max_gpu_adapters: 8
  max_cpu_adapters: 64
  load_timeout_s: 10
  idle_ttl_s: 1800
  revision_required: true
  checksum_required: true
```

### 29.4 Batch grouping

같은 adapter request를 묶으면 kernel 효율이 좋아질 수 있지만, queue wait가 늘어난다. micro-batching window를 짧게 두고 SLO로 결정한다.

### 29.5 Adapter 보안

- tenant가 임의 Hub repo를 load하지 못하게 allowlist
- safetensors 우선
- remote code 금지 또는 revision 고정·sandbox
- base model 호환성 검사
- adapter ID에 tenant ACL
- adapter prompt·data provenance 기록

### 29.6 여러 모델 상주

모델 A·B를 동시에 적재할 때:

```text
total memory
≈ weights_A + weights_B
+ KV_A + KV_B
+ runtime_A + runtime_B
```

모델별 QPS가 낮으면 한 GPU에서 multiplexing보다 모델별 time-sharing 또는 cold load가 유리할 수 있다.

### 29.7 Model router

작은 model과 큰 model을 routing한다.

| route | 예시 |
| --- | --- |
| 작은 model | 분류·간단 요약·짧은 chat |
| 큰 model | 복잡한 코드·수학·긴 reasoning |
| 전용 model | embedding·reranker·OCR·ASR |
| fallback | overload·장애 시 제한적 품질 |

routing 자체의 정확도와 실패 비용을 평가한다.

### 29.8 Model swap

VRAM이 부족하면 idle model을 unload하고 새 model을 load한다.

측정:

- unload synchronization
- memory가 실제 반환되는 시간
- load·warmup·graph capture
- SSD cache hit
- in-flight request drain
- adapter와 KV 무효화

### 29.9 Process isolation

다른 model·modality를 한 process에 넣으면 한 OOM이 전체를 중단할 수 있다. 다음은 별도 worker가 안전하다.

- embedding/reranker
- vision encoder/OCR parser
- ASR/TTS
- code execution tool
- image generation
- untrusted adapter

### 29.10 Base revision 일관성

같은 이름의 base model이라도 revision·tokenizer·chat template가 다르면 adapter 결과와 prefix cache가 달라진다. deployment ID에 모두 포함한다.

---

## 30. VLM·OCR·이미지·오디오·임베딩 서빙

### 30.1 VLM·OCR

VLM serving memory는 text LLM 외에 vision pipeline이 추가된다.

```text
VLM peak
≈ LLM weights + text/visual KV
+ vision encoder weights
+ image tensor·tiles
+ projector
+ prefill workspace
```

동시성 제한을 request 수가 아니라 다음으로 둔다.

- images/request
- total pixels
- dynamic tiles
- visual tokens
- PDF pages
- image download bytes

### 30.2 VLM pool 분리

text-only request와 image request를 같은 engine이 지원해도, vision encoder peak 때문에 text chat의 p99가 흔들릴 수 있다.

```text
/text  → 높은 동시성
/vision → 낮은 동시성, pixel quota
/ocr-long-pdf → asynchronous batch pool
```

자세한 모델·메모리는 [비전·OCR 가이드](../modalities/vision-ocr.md)를 참조한다.

### 30.3 PDF·OCR pipeline

```text
PDF upload
→ parser sandbox
→ page render queue
→ layout/OCR worker
→ VLM validation
→ artifact store/RAG
```

페이지를 모두 GPU에 올리지 않고 bounded queue와 backpressure를 사용한다.

### 30.4 Image generation

diffusion·flow model은 autoregressive KV 대신 다음이 동시성을 좌우한다.

- resolution·latent size
- batch size
- denoising step
- DiT/UNet activation
- text encoder·VAE
- ControlNet·LoRA·IP-Adapter
- image decode·upscale

```text
동시 job 수
≈ available VRAM / job peak
```

단, memory-efficient attention과 sequential offload로 peak가 시간에 따라 달라진다. interactive LLM과 같은 GPU를 공유하지 않는 것이 예측 가능하다.

자세한 내용은 [이미지 생성 가이드](../modalities/image-generation.md)를 참조한다.

### 30.5 Image job queue

- job별 pixel-step budget
- 최대 batch
- ControlNet 수
- LoRA 수
- priority
- cancellation
- intermediate preview 빈도

preview를 자주 전송하면 VAE decode와 network overhead가 증가한다.

### 30.6 ASR

ASR은 audio duration·chunk·sample rate가 핵심이다.

```text
ASR demand
≈ incoming audio seconds / wall-clock second
```

real-time factor(RTF)가 0.25면 단일 worker가 이론상 4배 real-time audio를 처리하지만, VAD·diarization·queue·burst를 포함하면 낮아진다.

### 30.7 Streaming ASR

- stream별 encoder state
- ring buffer
- VAD state
- partial hypothesis
- endpointing
- diarization state

connection 수와 active speech stream 수를 구분한다.

### 30.8 TTS

TTS serving은 text token뿐 아니라 output audio duration과 codec/vocoder가 병목이다.

- characters 또는 text tokens
- target audio seconds
- voice cloning reference
- speaker encoder
- codec/vocoder
- streaming chunk

voice clone model은 tenant별 reference audio cache와 개인정보 정책이 필요하다.

자세한 내용은 [오디오·음성 가이드](../modalities/audio-speech.md)를 참조한다.

### 30.9 Embedding

embedding model은 KV cache가 없거나 생성형 decode가 없으므로 높은 batch가 유리할 수 있다.

주요 제어:

- texts/request
- total tokens/batch
- max sequence length
- pooling output dtype
- tokenizer CPU
- output vector response size

생성 API와 같은 scheduler에 섞지 않고 별도 worker를 권장한다.

### 30.10 Reranker

cross-encoder reranker의 비용은 query-document pair 수에 비례한다.

```text
pair count = queries × candidates
```

candidate 100개를 한 request에 허용하면 짧은 text라도 batch memory가 커질 수 있다. top-K·total pair token cap을 둔다.

### 30.11 Data analysis code execution

LLM serving과 Python·SQL 실행은 별도 resource pool로 둔다.

- LLM GPU queue
- sandbox CPU/RAM queue
- database connection pool
- result serialization

LLM request가 끝나도 tool job이 계속되는 zombie task를 취소한다. 자세한 내용은 [데이터 분석 가이드](../domains/data-analysis.md)를 참조한다.

### 30.12 Modality별 admission unit

| modality | 기본 admission 단위 |
| --- | --- |
| Text LLM | input+output token, sequence |
| VLM | text token + visual token + pixel |
| OCR/PDF | page + pixel + output token |
| Image generation | pixel × steps × batch |
| ASR | audio seconds × channels |
| TTS | input token + target audio seconds |
| Embedding | total input tokens |
| Reranker | pair tokens × candidate 수 |

---

## 31. 관측성·부하 시험·capacity benchmark

### 31.1 관측성의 네 계층

| 계층 | 핵심 신호 |
| --- | --- |
| Client·gateway | end-to-end TTFT/E2EL, retry, timeout, disconnect |
| Inference engine | queue, running sequence, token throughput, KV utilization, preemption |
| Accelerator | VRAM, SM activity, memory bandwidth, PCIe/NVLink, power·thermal |
| Host·cluster | CPU, RAM, NUMA, disk, network, pod scheduling, replica count |

engine metric만 보고 client 체감 latency를 추정하지 않는다.

### 31.2 최소 dashboard

```text
Row 1: traffic
  request rate, accepted/rejected, active/queued requests

Row 2: latency
  TTFT p50/p95/p99, TPOT/ITL p95/p99, E2EL p95/p99

Row 3: tokens
  input/output tokens/s, cached tokens, avg·p99 lengths

Row 4: memory
  weight, KV pool used, device free, CPU offload, cache tier

Row 5: scheduler
  batch size, scheduled tokens, preemption, cancellation

Row 6: hardware
  GPU compute, memory bandwidth, PCIe/NVLink, power, temperature

Row 7: errors
  OOM, 429/503, timeout, worker restart, cache transfer failure
```

### 31.3 Metric cardinality

Prometheus label에 다음을 넣지 않는다.

- raw prompt
- user ID·email
- request ID를 무제한 label로 사용
- document ID 전체
- arbitrary model parameter
- URL 전체

높은 cardinality는 monitoring system 자체를 장애 상태로 만들 수 있다. tenant는 제한된 internal tier 또는 hashed bucket으로 집계한다.

### 31.4 Trace

trace span 예시:

```text
request.receive
request.validate
prompt.render
tokenize
engine.queue
prefill
kv.transfer
first_token
stream
request.complete
```

span attribute에는 token count·model revision·cache hit·adapter ID·status를 넣고, prompt·response 원문은 기본 제외한다.

### 31.5 GPU metric

NVIDIA에서는 DCGM exporter 등으로 다음을 수집한다.

- framebuffer memory used/free
- SM active
- tensor core activity
- DRAM utilization
- PCIe/NVLink throughput
- power·temperature
- clock throttle reason
- ECC/XID error

GPU utilization 100%가 반드시 좋은 상태는 아니다. memory bandwidth 포화와 compute 포화를 분리한다.

### 31.6 CPU·RAM metric

- tokenizer CPU time
- scheduler event-loop lag
- RSS·page cache·swap
- pinned memory
- NUMA remote access
- file descriptor
- connection count
- network transmit queue
- JSON serialization time

### 31.7 Load test 유형

| 유형 | 목적 |
| --- | --- |
| 단일 request | latency·품질 기준선 |
| 고정 concurrency | batch 효율·TPOT 변화 |
| open-loop request rate | 실제 overload·queue 관찰 |
| Poisson arrival | 일반적인 비동기 도착 근사 |
| burst | 회의 시작·batch fan-out·agent 폭주 |
| trace replay | 실제 길이·cache·tenant 분포 |
| soak test | leak·fragmentation·thermal·cache churn |
| fault injection | worker·network·cache 장애 복구 |

### 31.8 Closed-loop와 open-loop

closed-loop client는 한 요청이 끝나야 다음을 보내므로 server가 느려지면 자동으로 arrival rate가 낮아진다. overload 지점을 숨길 수 있다.

open-loop test는 정해진 arrival rate로 요청을 보내 queue와 reject를 측정한다. production capacity는 open-loop 또는 실제 trace로 결정한다.

### 31.9 Warmup

benchmark 전 다음을 분리한다.

```text
cold:
  model load·page fault·kernel compile·graph capture·empty cache

warm:
  model resident·graph ready·prefix cache 상태 명시
```

warmup request를 결과에서 제외하되, cold-start SLO도 별도 보고한다.

### 31.10 길이 bucket

최소 다음 bucket을 만든다.

| Input | Output |
| --- | --- |
| 0–512 | 0–128 |
| 513–2K | 129–512 |
| 2K–8K | 513–2K |
| 8K–32K | 2K–8K |
| 32K+ | 8K+ |

각 bucket의 TTFT·TPOT·error를 따로 본다.

### 31.11 Cache scenario

```text
A. cold unique prompts
B. 50% common system prefix
C. multi-turn session affinity
D. cache-aware routing
E. cache eviction 후 재접근
F. cross-tenant isolation
```

### 31.12 Concurrency sweep

```yaml
sweep:
  concurrency: [1, 2, 4, 8, 16, 32, 64]
  request_rate: [0.25, 0.5, 1, 2, 4, 8, 16]
  input_tokens: [512, 2048, 8192, 32768]
  output_tokens: [128, 512, 2048]
  kv_dtype: [baseline, q8_or_fp8]
  prefix_cache: [false, true]
```

모든 조합을 무작정 실행하기보다 실제 workload 주변에서 parameter sweep을 한다.

### 31.13 Saturation knee

그래프에 다음을 함께 그린다.

- x축: offered request rate
- y축 1: SLO goodput
- y축 2: p95/p99 TTFT
- y축 3: reject·queue

arrival rate 증가에 비해 goodput이 더 이상 늘지 않고 tail latency가 급상승하는 지점이 saturation knee다. production limit은 그보다 낮게 둔다.

### 31.14 vLLM benchmark

[vLLM `bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/)는 online endpoint의 TTFT·TPOT·ITL·E2EL과 percentile·goodput 관련 측정을 제공한다.

```bash
vllm bench serve \
  --backend vllm \
  --model OWNER/MODEL \
  --dataset-name random \
  --num-prompts 2000 \
  --request-rate 4
```

실제 input/output 길이 옵션은 설치 version의 help와 benchmark dataset schema를 확인한다.

### 31.15 SGLang benchmark

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024 \
  --random-range-ratio 0.5
```

SGLang metrics endpoint와 함께 queue·used token·cache hit를 수집한다.

### 31.16 `llama.cpp` benchmark

`llama-bench`는 kernel·prompt·generation 성능 기준선을 제공하지만 HTTP queue·streaming·slot behavior는 `llama-server`에 실제 concurrent client를 보내 별도로 측정한다.

```bash
curl -s http://127.0.0.1:8080/metrics > before.prom
# load generator 실행
curl -s http://127.0.0.1:8080/metrics > after.prom
```

### 31.17 결과 manifest

```yaml
benchmark:
  timestamp_kst: "2026-07-21T00:00:00+09:00"
  git_commit: null
  runtime:
    name: vllm
    version: null
    container_digest: null
  model:
    repo: null
    revision: null
    quantization: null
    tokenizer_revision: null
  hardware:
    gpu: null
    gpu_count: 1
    driver: null
    cuda_or_rocm: null
    cpu: null
    ram_gib: null
  engine:
    max_model_len: 16384
    max_num_seqs: 16
    max_num_batched_tokens: 8192
    kv_dtype: null
    prefix_cache: true
  traffic:
    source: trace_or_synthetic
    request_rate: null
    concurrency: null
    input_distribution: null
    output_distribution: null
  results:
    accepted_rps: null
    output_tokens_per_second: null
    p95_ttft_ms: null
    p99_ttft_ms: null
    p95_tpot_ms: null
    p99_e2el_ms: null
    reject_rate: null
    peak_device_memory_gib: null
```

### 31.18 Go/no-go gate

```text
[ ] p95·p99 SLO 만족
[ ] 30–60분 soak에서 memory 증가 없음
[ ] max-length request에서 OOM 없음
[ ] queue cap과 429/503 동작
[ ] client cancellation 후 resource 회수
[ ] worker crash 후 자동 복구
[ ] rolling update 중 error budget 충족
[ ] cache isolation 테스트 통과
[ ] cold-start와 cache-loss 시나리오 통과
```

---

## 32. 보안·다중 tenant·개인정보

### 32.1 기본 network 경계

```text
Internet/LAN client
→ TLS reverse proxy/API gateway
→ authentication·quota·request validation
→ private inference network
→ model server
```

model server를 `0.0.0.0`으로 직접 공개하지 않는다. 필요 시 private subnet과 firewall·mTLS를 사용한다.

### 32.2 Authentication·authorization

- API key 또는 OIDC
- tenant·user별 model 권한
- adapter 권한
- max context·priority tier
- RAG document ACL
- admin endpoint 분리
- `/metrics`, `/slots`, health endpoint 접근 통제

### 32.3 Prompt cache isolation

cache key에 최소 다음을 포함한다.

```text
model revision
+ tokenizer/chat template revision
+ adapter revision
+ tenant security domain
+ policy version
+ stable token prefix
+ modality content identifier
```

공용 system prompt만 global cache로 공유하고, private document·conversation prefix는 tenant namespace를 사용한다.

### 32.4 Cache timing side channel

공격자가 cache hit로 TTFT가 빨라지는지 관찰해 다른 사용자의 prefix 존재를 추론할 수 있다.

대응:

- tenant별 salt·namespace
- private prefix cross-tenant reuse 금지
- cache-hit detail을 client에 노출하지 않기
- 민감 endpoint의 timing 분석
- cache event log 권한 제한

### 32.5 Prompt·response logging

기본 정책:

```text
운영 metric: token count·latency·status만
원문 log: 기본 비활성화
debug 원문: 명시적 승인·짧은 TTL·암호화
```

민감 정보:

- source code·secret
- 의료·금융·법률 문서
- 음성·얼굴 reference
- RAG chunk
- tool output
- system prompt·policy

### 32.6 Denial of service 입력

다음 요청은 compute·memory amplification을 일으킬 수 있다.

- 매우 긴 prompt
- 큰 `max_tokens`
- `n`·`best_of`·beam
- prompt logprobs·top logprobs
- 복잡한 JSON schema·regex grammar
- 수백 개 tool definition
- 다수 고해상도 이미지·긴 video
- 매우 긴 audio
- 반복 retry·connection churn

server-side hard cap과 tenant quota를 둔다.

### 32.7 Structured output

복잡한 grammar·schema는 CPU compile과 constrained decoding overhead를 추가한다.

- schema byte limit
- nesting depth
- regex complexity
- compile cache size
- timeout
- untrusted schema sandbox

### 32.8 Multimodal SSRF와 parser

remote image·audio·PDF URL을 model server가 직접 가져오게 하지 않는다.

- gateway fetch proxy
- scheme allowlist
- private IP·metadata endpoint 차단
- redirect 제한
- content length·pixel·duration 상한
- decoder sandbox
- malware scan

### 32.9 Tool execution 분리

LLM inference worker에 shell·filesystem·database write secret을 주지 않는다.

```text
model output
→ schema validator
→ policy engine
→ least-privilege tool worker
→ result sanitizer
```

model server의 concurrency와 tool worker의 concurrency를 별도 제한한다.

### 32.10 Remote code·model supply chain

- `trust_remote_code` 기본 비활성화
- 필요 시 commit revision 고정
- safetensors 우선
- model·tokenizer·adapter checksum
- container image digest
- SBOM·vulnerability scan
- gated model token 최소 권한
- download cache 읽기 전용

### 32.11 Admin endpoint

다음 endpoint는 public API와 분리한다.

- model load/unload
- adapter load
- cache reset·save·restore
- runtime property 변경
- detailed slot state
- profile·debug dump
- metrics with internal labels

### 32.12 Queue와 tenant 격리

한 tenant의 긴 request가 전체 queue를 점유하지 않도록 한다.

- tenant별 running cap
- tenant별 queued token cap
- weighted fairness
- per-tier queue
- global emergency cap
- retry budget

### 32.13 Model extraction·abuse

완전한 방지는 어렵지만 다음을 적용할 수 있다.

- rate·token quota
- suspicious sampling pattern 탐지
- massive logprob access 제한
- output length 제한
- commercial license·acceptable use 정책
- abuse response process

### 32.14 Memory remanence

request 종료 후 KV block이 logical free되어도 physical memory가 즉시 zeroize되지 않을 수 있다. 신뢰 경계가 강한 환경에서는 다음을 검토한다.

- tenant별 process/replica
- cache salting만으로 충분한지 threat model 검토
- GPU memory reset 정책
- worker recycle
- encrypted remote cache
- crash dump 비활성화

### 32.15 Observability 보안

- Prometheus를 public 노출하지 않음
- trace exporter TLS
- metric label PII 제거
- error stack의 prompt fragment 제거
- dashboard RBAC
- audit log와 inference log 분리

### 32.16 Incident 대응

```text
1. affected model·revision·replica 식별
2. traffic drain·new request 차단
3. cache·adapter·remote storage 격리
4. logs와 artifact hash 보존
5. credential rotation
6. clean image로 redeploy
7. tenant 통지·postmortem
```

### 32.17 Security checklist

```text
[ ] inference server private bind
[ ] TLS·auth·rate limit
[ ] input/output/candidate hard cap
[ ] tenant별 cache namespace·salt
[ ] remote code·adapter allowlist
[ ] parser·URL fetch sandbox
[ ] tool worker 분리
[ ] prompt 원문 log 기본 off
[ ] metrics/admin endpoint 접근 통제
[ ] cancel·timeout·queue cap
[ ] model/container checksum·revision
[ ] incident drain·rollback 절차
```

---

## 33. 용량 계획 예제

아래는 계산 방법을 보여 주는 **가상 예제**다. 특정 model의 실제 구조·weight 크기를 의미하지 않는다.

### 33.1 24GB GPU·8B Q4·소규모 chat

가정:

```yaml
physical_vram_gib: 24
weights_gib: 5.5
prefill_workspace_gib: 2.5
graph_runtime_gib: 1.0
safety_headroom_gib: 2.0
kv_fp16_kib_per_token: 128
kv_dtype: q8
p95_cached_tokens_per_request: 6000
```

KV budget:

```text
24 - 5.5 - 2.5 - 1.0 - 2.0
= 13 GiB
```

Q8 KV를 FP16의 약 절반인 64 KiB/token으로 근사한다.

```text
usable KV tokens with 10% reserve
≈ 13 GiB × 0.9 / 64 KiB
≈ 191,000 tokens
```

p95 6K 기준 memory-only 상한:

```text
191K / 6K ≈ 31 sequences
```

그러나 8B model의 compute·TPOT SLO가 먼저 제한할 수 있으므로 실제 시작값은 `max running 8` 또는 `16`으로 두고 sweep한다.

### 33.2 48GB GPU·32B Q4·8K chat

가정:

```yaml
physical_vram_gib: 48
weights_gib: 20
workspace_graph_gib: 6
safety_gib: 4
kv_fp16_kib_per_token: 320
kv_dtype: q8
```

```text
KV budget = 48 - 20 - 6 - 4 = 18 GiB
Q8 KV ≈ 160 KiB/token
8K sequence ≈ 1.25 GiB
```

10% reserve 후 memory-only 상한:

```text
18 × 0.9 / 1.25 ≈ 12 sequences
```

실제 운영은 4·8·12 sequence를 비교하고, 32B decode compute 때문에 p95 TPOT가 먼저 SLO를 넘을 수 있다.

### 33.3 128GB Apple 통합 메모리·70B Q8

가정:

```yaml
physical_unified_memory_gib: 128
macos_apps_reserve_gib: 18
weights_gib: 75
runtime_workspace_gib: 8
safety_gib: 10
kv_fp16_kib_per_token: 320
kv_dtype: q8
p95_context: 16000
```

```text
KV budget
= 128 - 18 - 75 - 8 - 10
= 17 GiB

Q8 16K sequence
≈ 160 KiB/token × 16K
≈ 2.5 GiB
```

10% reserve 후 약 6 sequence가 memory-only 상한이다. 실제 Mac에서는 memory compression·thermal·GPU bandwidth 때문에 2–4 concurrent request부터 평가한다.

### 33.4 2 × 80GB GPU·70B BF16·TP=2

가정:

```text
70B BF16 weights ≈ 130.4 GiB total
ideal per GPU ≈ 65.2 GiB
```

GPU별 예산:

```yaml
physical_per_gpu_gib: 80
weights_per_gpu_gib: 65.2
runtime_communication_gib: 4
safety_gib: 5
kv_budget_per_gpu_gib: 5.8
```

70B형 GQA KV를 전체 FP16 320 KiB/token, TP=2에서 이상적으로 GPU당 160 KiB/token으로 가정하면:

```text
8K sequence per GPU KV ≈ 1.25 GiB
5.8 × 0.9 / 1.25 ≈ 4 sequences
```

BF16 weight가 두 GPU 대부분을 점유해 동시성이 낮다. Q8/FP8 weight, 4 GPU TP, 작은 model replica와 품질·SLO를 비교한다.

### 33.5 Prefix cache가 높은 RAG

가정:

```yaml
requests: 10
common_system_and_tools: 4000
unique_retrieval_and_query_per_request: 2000
```

cache 없음:

```text
10 × (4K + 2K) = 60K logical/physical prompt tokens
```

공통 prefix block 공유:

```text
4K + 10 × 2K = 24K physical unique prompt tokens
```

prefill compute와 physical cache가 크게 줄 수 있다. 단, tenant·policy가 같고 token prefix가 정확히 일치해야 한다.

### 33.6 Autoscaling

실측 결과 replica 하나의 SLO goodput이 2.5 req/s이고 peak demand가 8 req/s라면:

```text
minimum replicas
= ceil(8 / 2.5)
= 4
```

다음도 추가한다.

- node failure 여유
- rolling update
- traffic burst
- cold-start 시간
- cache warmup 손실

따라서 steady 4개, peak target 5–6개처럼 load test와 비용 정책으로 결정할 수 있다.

### 33.7 Little의 법칙 예제

```yaml
arrival_rate_rps: 1.2
mean_e2el_s: 9
```

```text
평균 in-system requests
≈ 1.2 × 9
≈ 10.8
```

평균만으로 p99 queue를 보장하지 않으므로 burst·length distribution test가 필요하다.

### 33.8 P/D network 예제

가정:

```text
KV = 128 KiB/token
uncached prompt = 16K tokens
```

```text
KV transfer/request ≈ 2 GiB
```

초당 2 request면 payload만 약 4 GiB/s다. protocol·contention·bidirectional traffic을 더하면 25GbE는 부족할 수 있으며, 100GbE도 latency를 실측해야 한다.

### 33.9 Image generation concurrency

가정:

```yaml
usable_vram_gib: 24
resident_weights_gib: 12
single_1024_job_peak_extra_gib: 7
safety_gib: 3
```

```text
24 - 12 - 3 = 9 GiB job budget
```

동시 2 job은 14GiB extra가 필요해 OOM 가능성이 높다. queue 1로 시작하고 sequential offload·resolution tier를 별도 평가한다.

### 33.10 계산 결과를 검증하는 절차

```text
계산상 max sequences 도출
→ 25% 수준에서 시작
→ p95 traffic load test
→ memory·SLO 확인
→ 25%씩 증가
→ saturation knee 이전에서 production cap
```

---

## 34. 문제 해결

### 34.1 Startup OOM

가능한 원인:

- weight가 예상보다 큼
- quantization fallback·dequantized load
- 다른 GPU process
- TP rank 배치 오류
- graph capture·warmup
- vision encoder·draft model 추가

대응:

```text
1. 실제 process별 VRAM 확인
2. model·quant config와 runtime kernel 확인
3. KV pool fraction 축소
4. graph·spec·adapter 비활성 기준선
5. weight quant 또는 TP/offload
```

### 34.2 첫 긴 request에서 OOM

- prefill activation peak
- max batched token 과다
- multimodal token 폭증
- prompt logprobs
- CUDA graph shape

대응:

- input hard cap
- chunked prefill
- prefill batch 축소
- image pixel·tile cap
- prompt logprobs 비활성

### 34.3 동시성 증가 시 OOM

```text
KV pool 부족
→ max running sequences ↓
→ context/output cap ↓
→ Q8/FP8 KV
→ prefix cache·offload
```

weight quantization만 낮춰도 KV가 그대로면 개선이 제한적일 수 있다.

### 34.4 GPU utilization은 낮지만 queue 증가

가능한 원인:

- tokenizer CPU
- scheduler event loop
- network·serialization
- TP communication stall
- CPU offload
- database/tool worker
- 작은 batch·낮은 arrival grouping

CPU profile과 GPU memory bandwidth·PCIe metric을 함께 본다.

### 34.5 Throughput은 높지만 TTFT 악화

- batch/token budget이 너무 큼
- 긴 prefill이 decode를 방해
- queue가 과도함
- interactive와 batch traffic 혼합

대응:

- max batched token 축소
- chunked prefill
- priority·별도 pool
- queue cap·admission

### 34.6 TTFT는 좋지만 TPOT 악화

- running sequence 과다
- decode batch가 너무 큼
- KV bandwidth 포화
- TP communication
- CPU KV offload

`max_num_seqs`·slot을 줄이고 output throughput과 p95 TPOT Pareto를 본다.

### 34.7 Cache hit rate가 낮음

- prompt serialization 차이
- timestamp·ID가 prefix 앞에 있음
- tokenizer·template revision 불일치
- round-robin으로 replica가 바뀜
- adapter·tenant namespace 차이
- cache eviction

rendered token ID prefix를 샘플링해 비교한다. 원문 string만 비교하지 않는다.

### 34.8 Prefix cache는 hit인데 메모리 절감이 작음

- logical block 공유와 physical layout 차이
- suffix가 매우 김
- output KV가 병목
- cache metadata·duplicate replica
- model architecture상 KV가 작고 weight가 병목

prefill time·physical block count를 함께 확인한다.

### 34.9 Q8/FP8 KV에서 품질 저하

- model 민감도
- scale calibration
- 긴 context 누적 오차
- backend bug
- K/V 비대칭

FP16 기준선과 동일 prompt·seed·장문 test를 실행하고, 문제가 있으면 FP16 또는 다른 dtype으로 rollback한다.

### 34.10 CPU offload 후 급격히 느림

- PCIe bandwidth
- pinned memory 부족
- NUMA remote RAM
- active KV를 매 step 이동
- CPU memory bandwidth

capacity 모드로 인정하거나 더 작은 model·GPU memory 증설을 선택한다.

### 34.11 Multi-GPU가 단일 GPU보다 느림

- model이 이미 한 GPU에 들어감
- PCIe-only topology
- 작은 batch
- TP communication overhead
- rank NUMA/NIC 배치

TP 1·2·4와 DP replica를 동일 workload에서 비교한다.

### 34.12 Replica load imbalance

- request 길이 차이
- session affinity
- cache-aware routing 고착
- adapter locality
- metric stale

active token·estimated work와 cache hit를 함께 사용하는 router를 검토한다.

### 34.13 Autoscaler가 늦음

- model cold start
- metric scrape·window 지연
- queue가 gateway에 있어 replica metric에 안 보임
- min replica 과소

predictive scale, faster model cache, gateway queue metric과 minimum replica를 적용한다.

### 34.14 Scale-down 후 TTFT 급증

- prefix cache loss
- session reroute
- cold graph
- weight page fault

drain·cache transfer·slow scale-down과 warm replica를 사용한다.

### 34.15 P/D 분리 후 throughput 감소

- KV transfer 병목
- P/D ratio 불균형
- network topology
- extra queue
- small prompt라 분리 overhead가 더 큼

통합 chunked prefill 기준선과 비교하고, 긴 prompt bucket만 P/D 경로로 보낸다.

### 34.16 Streaming이 끊기거나 burst로 옴

- proxy buffering
- stream interval
- HTTP/2·SSE 설정
- gzip buffering
- client read loop

reverse proxy에서 streaming buffering을 비활성화하고 engine ITL과 client ITL을 비교한다.

### 34.17 Client가 취소했는데 GPU는 계속 계산

- disconnect가 gateway에서 종료
- cancel API 미전파
- batch engine가 cancellation을 늦게 확인

request ID로 gateway→engine cancel trace를 만든다.

### 34.18 429·503이 없고 timeout만 증가

queue·admission 상한이 없거나 proxy가 모든 request를 받아 둔 상태다. 명시적 queue cap과 예상 queue-time reject를 구현한다.

### 34.19 모델 전환 후 memory가 반환되지 않음

- allocator reserved memory
- mmap page cache
- graph·adapter cache
- process 내부 reference

model lifecycle이 잦으면 process-per-model 또는 worker restart가 더 예측 가능할 수 있다.

### 34.20 장시간 후 OOM

- memory leak
- cache·checkpoint 무제한
- unique schema·adapter cache
- fragmentation
- client connection leak

soak test에서 RSS·device reserved·cache entry·FD를 시간축으로 기록한다.

### 34.21 MoE throughput 변동

- expert routing skew
- hot expert
- EP all-to-all congestion
- workload domain 변화

expert load metric과 traffic category를 연계하고 load balancing 기능을 검토한다.

### 34.22 VLM request가 text chat을 멈춤

- vision encoder peak
- 대형 image prefill
- dynamic tile 폭증

vision queue·pixel cap·별도 worker를 둔다.

### 34.23 ASR stream 수는 적은데 CPU가 포화

- resampling·decode
- VAD·diarization
- audio format 변환
- Python callback

audio preprocessing worker와 model worker를 분리한다.

### 34.24 Benchmark가 재현되지 않음

다음이 누락되었는지 확인한다.

- runtime commit·container digest
- model/tokenizer revision
- exact prompt tokens
- cache warm/cold state
- driver·GPU clock·power
- quantization·KV dtype
- concurrency model
- arrival distribution
- background process

### 34.25 빠른 진단표

| 증상 | 먼저 볼 metric | 첫 조치 |
| --- | --- | --- |
| OOM | weight·KV·prefill peak | token/sequence cap |
| 높은 TTFT | queue·prefill | queue cap·chunking |
| 높은 TPOT | active sequence·KV bandwidth | concurrency 축소 |
| 낮은 cache hit | rendered token prefix | prompt canonicalization |
| 낮은 GPU util | CPU·PCIe·batch | tokenizer/profile |
| 높은 reject | offered load·goodput | replica 또는 quota |
| 높은 preemption | KV utilization | KV pool·context cap |
| 높은 p99만 발생 | long request·burst | workload 분리 |

---

## 35. 주요 출처와 저장소

이 문서는 가능한 한 공식 문서·원 저장소를 기준으로 작성했다. `latest` 문서는 developer preview일 수 있으므로 production에서는 설치한 release의 versioned 문서를 우선한다.

### 35.1 vLLM

- [vLLM documentation](https://docs.vllm.ai/)
- [Engine arguments](https://docs.vllm.ai/en/latest/api/vllm/engine/arg_utils/)
- [vLLM launch·render CLI arguments](https://docs.vllm.ai/en/latest/cli/launch/render/)
- [GPU worker memory profiling](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)
- [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [Per-Request Metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
- [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM repository](https://github.com/vllm-project/vllm)

### 35.2 SGLang

- [SGLang documentation](https://docs.sglang.ai/)
- [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
- [Production Metrics](https://docs.sglang.io/docs/references/production_metrics)
- [Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
- [Prefill-Decode Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [HiCache Design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [DP·DPA Router Guide](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
- [SGLang Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)
- [Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [SGLang repository](https://github.com/sgl-project/sglang)
- [SGLang paper](https://arxiv.org/abs/2312.07104)

### 35.3 TensorRT-LLM

- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
- [KV Cache System](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM)
- [Release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [Triton TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend)

### 35.4 `llama.cpp`·Ollama·MLX

- [`llama.cpp` repository](https://github.com/ggml-org/llama.cpp)
- [`llama-server` README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Ollama FAQ](https://docs.ollama.com/faq)
- [Ollama repository](https://github.com/ollama/ollama)
- [MLX-LM repository](https://github.com/ml-explore/mlx-lm)
- [MLX repository](https://github.com/ml-explore/mlx)

### 35.5 Hugging Face TGI

- [Text Generation Inference documentation](https://huggingface.co/docs/text-generation-inference/)
- [Launcher arguments](https://huggingface.co/docs/text-generation-inference/en/reference/launcher)
- [Exported metrics](https://huggingface.co/docs/text-generation-inference/en/reference/metrics)
- [PagedAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention)
- [Tensor Parallelism](https://huggingface.co/docs/text-generation-inference/en/conceptual/tensor_parallelism)
- [TGI repository](https://github.com/huggingface/text-generation-inference)

### 35.6 Orchestration·autoscaling·KV tier

- [Ray Serve autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)
- [Ray Serve advanced autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
- [Ray Serve deployment configuration](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
- [Ray Serve custom request router](https://docs.ray.io/en/latest/serve/advanced-guides/custom-request-router.html)
- [KServe KV Cache Offloading](https://kserve.github.io/website/docs/model-serving/generative-inference/kvcache-offloading)
- [KServe generative inference overview](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview)
- [LMCache documentation](https://docs.lmcache.ai/)
- [llm-d repository](https://github.com/llm-d/llm-d)

### 35.7 Monitoring·benchmark

- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Prometheus](https://prometheus.io/docs/)
- [OpenTelemetry](https://opentelemetry.io/docs/)
- [NVIDIA GenAI-Perf](https://github.com/NVIDIA/GenAI-Perf)
- [GuideLLM](https://github.com/vllm-project/guidellm)

### 35.8 기반 논문·개념

- [PagedAttention·vLLM paper](https://arxiv.org/abs/2309.06180)
- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [Orca: Iteration-level Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII)

### 35.9 관련 레포지토리 문서

- [생산성·문서·RAG](../domains/productivity-rag.md)
- [데이터 분석](../domains/data-analysis.md)
- [비전·OCR](../modalities/vision-ocr.md)
- [이미지 생성](../modalities/image-generation.md)
- [오디오·음성](../modalities/audio-speech.md)
- [양자화](./quantization.md)
- [파인튜닝 메모리](./fine-tuning-memory.md)
- [런타임·하드웨어](./runtime-hardware.md) (예정)

---

## 36. 최종 권장안과 갱신 주의

### 36.1 대부분의 개인용 서버

```text
llama.cpp 또는 Ollama
→ slot/parallel 1
→ context 4K–8K
→ 실제 필요 시 2 slots
→ Q8 KV 검증
→ private bind + reverse proxy
```

### 36.2 단일 NVIDIA GPU·팀 API

```text
vLLM 또는 SGLang
→ max model length 제한
→ KV pool 보수적 설정
→ max sequences와 batched tokens sweep
→ prefix cache + chunked prefill
→ queue cap + load test
```

### 36.3 NVIDIA 최적화 고정 환경

```text
TensorRT-LLM
→ release·GPU·recipe 고정
→ KV block reuse·cache salt
→ batch/token/parallelism tuning
→ release note와 artifact manifest
```

### 36.4 메모리가 부족할 때

```text
context·output 상한
→ running sequence 상한
→ chunked prefill
→ Q8/FP8 KV
→ 더 작은 Q4 model
→ offload·scale-out
```

Q2·Q3 weight로 바로 내려가기 전에 KV와 동시 slot이 실제 요구인지 확인한다.

### 36.5 Tail latency가 나쁠 때

```text
queue time 확인
→ 긴 prompt bucket 확인
→ prefill/decode 간섭 확인
→ batch/token budget 축소
→ priority·pool 분리
→ P/D 분리는 마지막 단계에서 검토
```

### 36.6 Throughput이 낮을 때

```text
GPU compute·memory bandwidth·CPU·PCIe profile
→ continuous batching
→ max sequences/token budget sweep
→ supported quantization kernel
→ prefix cache
→ speculative decoding A/B
→ replica·TP·EP
```

### 36.7 Multi-GPU

```text
model이 한 GPU에 들어감
  → DP replica와 TP를 비교

model이 안 들어감
  → TP/PP

MoE
  → EP + load balance

긴 context
  → KV dtype·CP·KV tier
```

### 36.8 Production 최소 기준

```text
[ ] 실제 p95/p99 input·output 분포
[ ] KV bytes/token 또는 runtime max token pool
[ ] cold/warm load test
[ ] p95/p99 TTFT·TPOT·E2EL
[ ] queue·reject·retry policy
[ ] cancellation resource 회수
[ ] cache namespace·tenant isolation
[ ] model/runtime/container revision·checksum
[ ] 30–60분 soak와 fault test
[ ] rollback 가능한 config
```

### 36.9 한 문장 기준

> **서빙 capacity는 모델이 메모리에 들어가는지보다, 실제 traffic의 cached token과 prefill peak를 감당하면서 목표 TTFT·TPOT를 만족하는 goodput이 얼마인지로 결정한다.**

### 36.10 갱신 주의

이 문서는 2026-07-21 KST 기준으로 공식 문서와 원 저장소를 확인해 작성했다. serving runtime은 빠르게 변하므로 배포 직전에 다음을 다시 검증한다.

- CLI flag·기본값
- model architecture·KV layout
- quantized KV 지원
- attention backend
- prefix cache·multimodal key
- TP·PP·EP·P/D 지원 조합
- GPU·driver·CUDA·ROCm matrix
- security advisory
- metrics 이름
- release note의 known issue

### 36.11 레포지토리에 남길 측정값

```yaml
required_serving_measurements:
  runtime_and_version: null
  model_revision_and_quant: null
  hardware_and_topology: null
  resident_weight_gib: null
  kv_pool_gib: null
  kv_dtype: null
  max_cached_tokens: null
  max_running_sequences: null
  max_batched_tokens: null
  traffic_length_distribution: null
  prefix_cache_hit_rate: null
  p95_ttft_ms: null
  p99_ttft_ms: null
  p95_tpot_ms: null
  output_tokens_per_second: null
  reject_rate: null
  peak_device_memory_gib: null
  soak_duration_minutes: null
```

계산표보다 실제 측정 결과와 재현 가능한 manifest가 이 repository의 가장 가치 있는 기여다.

---

**문서 종료**
