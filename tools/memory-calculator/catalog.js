/* Generated from the RAM-for-Local-AI guide set. No network request is made. */
window.RAMAI_CATALOG = {
  "schemaVersion": 1,
  "verified": "2026-07-21",
  "project": "RAM-for-Local-AI",
  "guides": {
    "cybersecurity": {
      "label": "버그바운티·사이버보안",
      "path": "../../local_ai_models_for_bug_bounty_cybersecurity_by_memory_2026-07.md",
      "group": "domains"
    },
    "programming-stem": {
      "label": "범용 프로그래밍·수학·과학",
      "path": "../../local_ai_models_for_programming_math_science_research_by_memory_2026-07.md",
      "group": "domains"
    },
    "productivity-rag": {
      "label": "생산성·문서·RAG",
      "path": "../../guides/domains/productivity-rag.md",
      "group": "domains"
    },
    "data-analysis": {
      "label": "데이터 분석·BI·SQL",
      "path": "../../guides/domains/data-analysis.md",
      "group": "domains"
    },
    "vision-ocr": {
      "label": "비전·OCR·문서 이해",
      "path": "../../guides/modalities/vision-ocr.md",
      "group": "modalities"
    },
    "image-generation": {
      "label": "이미지 생성·편집",
      "path": "../../guides/modalities/image-generation.md",
      "group": "modalities"
    },
    "audio-speech": {
      "label": "오디오·음성",
      "path": "../../guides/modalities/audio-speech.md",
      "group": "modalities"
    },
    "quantization": {
      "label": "양자화",
      "path": "../../guides/operations/quantization.md",
      "group": "operations"
    },
    "fine-tuning-memory": {
      "label": "파인튜닝 메모리",
      "path": "../../guides/operations/fine-tuning-memory.md",
      "group": "operations"
    },
    "serving-concurrency": {
      "label": "서빙·동시성",
      "path": "../../guides/operations/serving-concurrency.md",
      "group": "operations"
    },
    "runtime-hardware": {
      "label": "런타임·하드웨어",
      "path": "../../guides/operations/runtime-hardware.md",
      "group": "operations"
    }
  },
  "domains": [
    {
      "id": "cybersecurity",
      "label": "버그바운티·사이버보안",
      "guide": "cybersecurity",
      "tasks": [
        {
          "id": "security-triage",
          "label": "CVE·위협 인텔리전스·트리아지"
        },
        {
          "id": "secure-code",
          "label": "코드 감사·보안 프로그래밍"
        },
        {
          "id": "security-agent",
          "label": "승인된 저장소·보안 에이전트"
        }
      ],
      "defaultTask": "secure-code"
    },
    {
      "id": "programming-stem",
      "label": "프로그래밍·수학·과학",
      "guide": "programming-stem",
      "tasks": [
        {
          "id": "general-coding",
          "label": "범용 코딩·테스트·리팩터링"
        },
        {
          "id": "repo-agent",
          "label": "저장소 수준 코딩 에이전트"
        },
        {
          "id": "math-science",
          "label": "자연어 수학·과학 추론"
        },
        {
          "id": "lean-proof",
          "label": "Lean 4 형식증명"
        }
      ],
      "defaultTask": "general-coding"
    },
    {
      "id": "productivity-rag",
      "label": "생산성·문서·RAG",
      "guide": "productivity-rag",
      "tasks": [
        {
          "id": "document-work",
          "label": "요약·추출·번역·문서 비교"
        },
        {
          "id": "personal-rag",
          "label": "개인 지식베이스 RAG"
        },
        {
          "id": "team-rag",
          "label": "팀·조직 지식베이스 RAG"
        }
      ],
      "defaultTask": "personal-rag"
    },
    {
      "id": "data-analysis",
      "label": "데이터 분석·BI·SQL",
      "guide": "data-analysis",
      "tasks": [
        {
          "id": "text-to-sql",
          "label": "Text-to-SQL·BI 질의"
        },
        {
          "id": "eda-code",
          "label": "EDA·Python/R 코드 분석"
        },
        {
          "id": "tabular-ml",
          "label": "표형 머신러닝"
        },
        {
          "id": "time-series",
          "label": "시계열 예측·이상 탐지"
        }
      ],
      "defaultTask": "eda-code"
    },
    {
      "id": "vision-ocr",
      "label": "비전·OCR",
      "guide": "vision-ocr",
      "tasks": [
        {
          "id": "document-ocr",
          "label": "PDF·문서 OCR·Markdown 변환"
        },
        {
          "id": "ui-vision",
          "label": "스크린샷·UI·일반 이미지 이해"
        },
        {
          "id": "chart-math",
          "label": "표·차트·수식·구조화 추출"
        }
      ],
      "defaultTask": "document-ocr"
    },
    {
      "id": "image-generation",
      "label": "이미지 생성·편집",
      "guide": "image-generation",
      "tasks": [
        {
          "id": "text-to-image",
          "label": "Text-to-Image"
        },
        {
          "id": "image-edit",
          "label": "편집·인페인팅·참조 이미지"
        },
        {
          "id": "image-control",
          "label": "ControlNet·LoRA·구조 제어"
        }
      ],
      "defaultTask": "text-to-image"
    },
    {
      "id": "audio-speech",
      "label": "오디오·음성",
      "guide": "audio-speech",
      "tasks": [
        {
          "id": "asr",
          "label": "오프라인 음성 인식"
        },
        {
          "id": "streaming-asr",
          "label": "실시간·스트리밍 자막"
        },
        {
          "id": "tts",
          "label": "TTS·보이스 클로닝"
        },
        {
          "id": "audio-agent",
          "label": "오디오 QA·음성 에이전트"
        }
      ],
      "defaultTask": "asr"
    }
  ],
  "operations": [
    {
      "id": "inference",
      "label": "단일 사용자 추론"
    },
    {
      "id": "rag",
      "label": "RAG·검색 파이프라인"
    },
    {
      "id": "serving",
      "label": "API 서빙·동시 사용자"
    },
    {
      "id": "fine-tuning",
      "label": "LoRA·QLoRA·Full FT"
    }
  ],
  "quantizations": {
    "q2": {
      "label": "Q2 / IQ2",
      "bits": 2.7,
      "overhead": 1.04,
      "quality": 1,
      "family": "gguf",
      "note": "기능 확인·극저메모리 실험용"
    },
    "q3": {
      "label": "Q3",
      "bits": 3.5,
      "overhead": 1.04,
      "quality": 2,
      "family": "gguf",
      "note": "Q4가 들어가지 않을 때"
    },
    "q4": {
      "label": "Q4",
      "bits": 4.7,
      "overhead": 1.04,
      "quality": 3,
      "family": "gguf",
      "note": "일반적인 기본값"
    },
    "q5": {
      "label": "Q5",
      "bits": 5.6,
      "overhead": 1.04,
      "quality": 4,
      "family": "gguf",
      "note": "품질 여유가 필요할 때"
    },
    "q6": {
      "label": "Q6",
      "bits": 6.6,
      "overhead": 1.03,
      "quality": 5,
      "family": "gguf",
      "note": "정확도 우선 로컬 추론"
    },
    "q8": {
      "label": "Q8 / INT8",
      "bits": 8.5,
      "overhead": 1.02,
      "quality": 6,
      "family": "int8",
      "note": "고정밀 기준선"
    },
    "int8": {
      "label": "INT8",
      "bits": 8.0,
      "overhead": 1.04,
      "quality": 5,
      "family": "int8",
      "note": "ONNX·CTranslate2·OpenVINO 등"
    },
    "nf4": {
      "label": "NF4 / 4-bit",
      "bits": 4.5,
      "overhead": 1.06,
      "quality": 3,
      "family": "nf4",
      "note": "이미지·QLoRA·bitsandbytes 계열"
    },
    "awq": {
      "label": "AWQ / GPTQ W4A16",
      "bits": 4.5,
      "overhead": 1.06,
      "quality": 3,
      "family": "w4a16",
      "note": "CUDA 서버용 4-bit weight-only"
    },
    "mxfp4": {
      "label": "MXFP4 / FP4",
      "bits": 4.2,
      "overhead": 1.05,
      "quality": 3,
      "family": "fp4",
      "note": "네이티브 지원 모델·GPU에서만"
    },
    "fp8": {
      "label": "FP8",
      "bits": 8.0,
      "overhead": 1.04,
      "quality": 5,
      "family": "fp8",
      "note": "지원 GPU·커널 확인"
    },
    "bf16": {
      "label": "BF16",
      "bits": 16.0,
      "overhead": 1.01,
      "quality": 7,
      "family": "float",
      "note": "고정밀 기준선"
    },
    "fp16": {
      "label": "FP16",
      "bits": 16.0,
      "overhead": 1.01,
      "quality": 7,
      "family": "float",
      "note": "고정밀 기준선"
    }
  },
  "kvDtypes": {
    "fp16": {
      "label": "FP16/BF16 KV",
      "factor": 1.0
    },
    "q8": {
      "label": "Q8/FP8 KV",
      "factor": 0.5
    },
    "q4": {
      "label": "Q4 KV",
      "factor": 0.25
    }
  },
  "runtimes": [
    {
      "id": "auto",
      "label": "자동 추천",
      "overheadRatio": 0.07,
      "overheadMin": 0.8
    },
    {
      "id": "llama.cpp",
      "label": "llama.cpp",
      "overheadRatio": 0.06,
      "overheadMin": 0.7
    },
    {
      "id": "ollama",
      "label": "Ollama",
      "overheadRatio": 0.08,
      "overheadMin": 1.0
    },
    {
      "id": "mlx",
      "label": "MLX-LM / MLX-VLM",
      "overheadRatio": 0.06,
      "overheadMin": 0.8
    },
    {
      "id": "vllm",
      "label": "vLLM",
      "overheadRatio": 0.1,
      "overheadMin": 1.8
    },
    {
      "id": "sglang",
      "label": "SGLang",
      "overheadRatio": 0.1,
      "overheadMin": 1.8
    },
    {
      "id": "tensorrt",
      "label": "TensorRT-LLM",
      "overheadRatio": 0.12,
      "overheadMin": 2.2
    },
    {
      "id": "diffusers",
      "label": "Diffusers / ComfyUI",
      "overheadRatio": 0.08,
      "overheadMin": 1.0
    },
    {
      "id": "whisper.cpp",
      "label": "whisper.cpp",
      "overheadRatio": 0.05,
      "overheadMin": 0.35
    },
    {
      "id": "onnx",
      "label": "ONNX Runtime / OpenVINO",
      "overheadRatio": 0.08,
      "overheadMin": 0.5
    }
  ],
  "addons": {
    "embeddings": [
      {
        "id": "none",
        "label": "없음",
        "memoryGiB": 0,
        "hf": ""
      },
      {
        "id": "granite97",
        "label": "Granite Embedding 97M R2",
        "memoryGiB": 0.2,
        "hf": "https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2"
      },
      {
        "id": "granite311",
        "label": "Granite Embedding 311M R2",
        "memoryGiB": 0.65,
        "hf": "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2"
      },
      {
        "id": "qwen06",
        "label": "Qwen3 Embedding 0.6B",
        "memoryGiB": 1.2,
        "hf": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"
      },
      {
        "id": "qwen4q4",
        "label": "Qwen3 Embedding 4B Q4",
        "memoryGiB": 2.6,
        "hf": "https://huggingface.co/Qwen/Qwen3-Embedding-4B"
      },
      {
        "id": "qwen8q4",
        "label": "Qwen3 Embedding 8B Q4",
        "memoryGiB": 5.1,
        "hf": "https://huggingface.co/Qwen/Qwen3-Embedding-8B"
      }
    ],
    "rerankers": [
      {
        "id": "none",
        "label": "없음",
        "memoryGiB": 0,
        "hf": ""
      },
      {
        "id": "qwen06",
        "label": "Qwen3 Reranker 0.6B",
        "memoryGiB": 1.25,
        "hf": "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B"
      },
      {
        "id": "bgev2m3",
        "label": "BGE-reranker-v2-m3",
        "memoryGiB": 1.15,
        "hf": "https://huggingface.co/BAAI/bge-reranker-v2-m3"
      },
      {
        "id": "qwen4q4",
        "label": "Qwen3 Reranker 4B Q4",
        "memoryGiB": 2.7,
        "hf": "https://huggingface.co/Qwen/Qwen3-Reranker-4B"
      },
      {
        "id": "qwen8q4",
        "label": "Qwen3 Reranker 8B Q4",
        "memoryGiB": 5.2,
        "hf": "https://huggingface.co/Qwen/Qwen3-Reranker-8B"
      }
    ],
    "parsers": [
      {
        "id": "none",
        "label": "없음",
        "systemGiB": 0,
        "deviceGiB": 0
      },
      {
        "id": "classic",
        "label": "PDF parser + 고전 OCR",
        "systemGiB": 1.0,
        "deviceGiB": 0.3
      },
      {
        "id": "vlm",
        "label": "OCR·layout VLM worker",
        "systemGiB": 1.5,
        "deviceGiB": 2.0
      }
    ]
  },
  "gpuMemoryHints": [
    {
      "pattern": "RTX 5090 Laptop",
      "values": [
        24
      ],
      "label": "GeForce RTX 5090 Laptop GPU"
    },
    {
      "pattern": "RTX 5090",
      "values": [
        32
      ],
      "label": "GeForce RTX 5090"
    },
    {
      "pattern": "RTX 5080 Laptop",
      "values": [
        16
      ],
      "label": "GeForce RTX 5080 Laptop GPU"
    },
    {
      "pattern": "RTX 5080",
      "values": [
        16
      ],
      "label": "GeForce RTX 5080"
    },
    {
      "pattern": "RTX 5070 Ti Laptop",
      "values": [
        12
      ],
      "label": "GeForce RTX 5070 Ti Laptop GPU"
    },
    {
      "pattern": "RTX 5070 Ti",
      "values": [
        16
      ],
      "label": "GeForce RTX 5070 Ti"
    },
    {
      "pattern": "RTX 5070 Laptop",
      "values": [
        8,
        12
      ],
      "label": "GeForce RTX 5070 Laptop GPU"
    },
    {
      "pattern": "RTX 5070",
      "values": [
        12
      ],
      "label": "GeForce RTX 5070"
    },
    {
      "pattern": "RTX 5060 Ti",
      "values": [
        8,
        16
      ],
      "label": "GeForce RTX 5060 Ti"
    },
    {
      "pattern": "RTX 5060 Laptop",
      "values": [
        8
      ],
      "label": "GeForce RTX 5060 Laptop GPU"
    },
    {
      "pattern": "RTX 5060",
      "values": [
        8
      ],
      "label": "GeForce RTX 5060"
    },
    {
      "pattern": "RTX 5050 Laptop",
      "values": [
        8
      ],
      "label": "GeForce RTX 5050 Laptop GPU"
    },
    {
      "pattern": "RTX 5050",
      "values": [
        8
      ],
      "label": "GeForce RTX 5050"
    },
    {
      "pattern": "RTX 4090 Laptop",
      "values": [
        16
      ],
      "label": "GeForce RTX 4090 Laptop GPU"
    },
    {
      "pattern": "RTX 4090",
      "values": [
        24
      ],
      "label": "GeForce RTX 4090"
    },
    {
      "pattern": "RTX 4080 Laptop",
      "values": [
        12
      ],
      "label": "GeForce RTX 4080 Laptop GPU"
    },
    {
      "pattern": "RTX 4080",
      "values": [
        16
      ],
      "label": "GeForce RTX 4080"
    },
    {
      "pattern": "RTX 4070 Ti SUPER",
      "values": [
        16
      ],
      "label": "GeForce RTX 4070 Ti SUPER"
    },
    {
      "pattern": "RTX 4070 Ti",
      "values": [
        12
      ],
      "label": "GeForce RTX 4070 Ti"
    },
    {
      "pattern": "RTX 4070 SUPER",
      "values": [
        12
      ],
      "label": "GeForce RTX 4070 SUPER"
    },
    {
      "pattern": "RTX 4070 Laptop",
      "values": [
        8
      ],
      "label": "GeForce RTX 4070 Laptop GPU"
    },
    {
      "pattern": "RTX 4070",
      "values": [
        12
      ],
      "label": "GeForce RTX 4070"
    },
    {
      "pattern": "RTX 4060 Ti",
      "values": [
        8,
        16
      ],
      "label": "GeForce RTX 4060 Ti"
    },
    {
      "pattern": "RTX 4060",
      "values": [
        8
      ],
      "label": "GeForce RTX 4060"
    },
    {
      "pattern": "RTX 3090 Ti",
      "values": [
        24
      ],
      "label": "GeForce RTX 3090 Ti"
    },
    {
      "pattern": "RTX 3090",
      "values": [
        24
      ],
      "label": "GeForce RTX 3090"
    },
    {
      "pattern": "RTX 3080 Ti",
      "values": [
        12
      ],
      "label": "GeForce RTX 3080 Ti"
    },
    {
      "pattern": "RTX 3080",
      "values": [
        10,
        12
      ],
      "label": "GeForce RTX 3080"
    },
    {
      "pattern": "RTX PRO 6000 Blackwell",
      "values": [
        96
      ],
      "label": "NVIDIA RTX PRO 6000 Blackwell"
    },
    {
      "pattern": "RTX A6000",
      "values": [
        48
      ],
      "label": "RTX A6000"
    },
    {
      "pattern": "RTX 6000 Ada",
      "values": [
        48
      ],
      "label": "RTX 6000 Ada"
    },
    {
      "pattern": "RTX A5000",
      "values": [
        24
      ],
      "label": "RTX A5000"
    },
    {
      "pattern": "L40S",
      "values": [
        48
      ],
      "label": "NVIDIA L40S"
    },
    {
      "pattern": "RX 9070 XT",
      "values": [
        16
      ],
      "label": "Radeon RX 9070 XT"
    },
    {
      "pattern": "RX 9070 GRE",
      "values": [
        12,
        16
      ],
      "label": "Radeon RX 9070 GRE"
    },
    {
      "pattern": "RX 9070",
      "values": [
        16
      ],
      "label": "Radeon RX 9070"
    },
    {
      "pattern": "RX 9060 XT",
      "values": [
        8,
        16
      ],
      "label": "Radeon RX 9060 XT"
    },
    {
      "pattern": "RX 9060",
      "values": [
        8
      ],
      "label": "Radeon RX 9060"
    },
    {
      "pattern": "Radeon PRO W7900",
      "values": [
        48
      ],
      "label": "Radeon PRO W7900"
    },
    {
      "pattern": "Radeon PRO W7800",
      "values": [
        32,
        48
      ],
      "label": "Radeon PRO W7800"
    },
    {
      "pattern": "Radeon PRO W7700",
      "values": [
        16
      ],
      "label": "Radeon PRO W7700"
    },
    {
      "pattern": "RX 7900 XTX",
      "values": [
        24
      ],
      "label": "Radeon RX 7900 XTX"
    },
    {
      "pattern": "RX 7900 XT",
      "values": [
        20
      ],
      "label": "Radeon RX 7900 XT"
    },
    {
      "pattern": "RX 7800 XT",
      "values": [
        16
      ],
      "label": "Radeon RX 7800 XT"
    },
    {
      "pattern": "RX 7700 XT",
      "values": [
        12
      ],
      "label": "Radeon RX 7700 XT"
    },
    {
      "pattern": "RX 7600 XT",
      "values": [
        16
      ],
      "label": "Radeon RX 7600 XT"
    },
    {
      "pattern": "RX 7600",
      "values": [
        8
      ],
      "label": "Radeon RX 7600"
    },
    {
      "pattern": "RX 6900 XT",
      "values": [
        16
      ],
      "label": "Radeon RX 6900 XT"
    },
    {
      "pattern": "RX 6800 XT",
      "values": [
        16
      ],
      "label": "Radeon RX 6800 XT"
    },
    {
      "pattern": "RX 6800",
      "values": [
        16
      ],
      "label": "Radeon RX 6800"
    },
    {
      "pattern": "Arc B580",
      "values": [
        12
      ],
      "label": "Intel Arc B580"
    },
    {
      "pattern": "Arc B570",
      "values": [
        10
      ],
      "label": "Intel Arc B570"
    },
    {
      "pattern": "Arc A770",
      "values": [
        8,
        16
      ],
      "label": "Intel Arc A770"
    },
    {
      "pattern": "Arc A750",
      "values": [
        8
      ],
      "label": "Intel Arc A750"
    },
    {
      "pattern": "Arc A380",
      "values": [
        6
      ],
      "label": "Intel Arc A380"
    }
  ],
  "models": [
    {
      "id": "qwen35-08b",
      "name": "Qwen3.5 0.8B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "security-triage",
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 0.8,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 0.54
      },
      "kvMiBPerTokenFp16": 0.025,
      "runtimes": [],
      "quality": 28,
      "notes": [
        "4GB급 초경량 기준선",
        "코드·수학 결과는 외부 도구로 검증"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen35-2b",
      "name": "Qwen3.5 2B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "security-triage",
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 2.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF",
      "guide": "cybersecurity",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 1.29
      },
      "kvMiBPerTokenFp16": 0.045,
      "runtimes": [],
      "quality": 34,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "granite41-3b",
      "name": "Granite 4.1 3B",
      "kind": "llm",
      "domains": [
        "productivity-rag",
        "data-analysis",
        "cybersecurity"
      ],
      "tasks": [
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "security-triage"
      ],
      "paramsB": 3.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF",
      "guide": "productivity-rag",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q2": 1.37,
        "q3": 1.73,
        "q4": 2.1
      },
      "kvMiBPerTokenFp16": 0.06,
      "runtimes": [],
      "quality": 40,
      "notes": [
        "요약·추출·RAG·function calling 중심"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen35-4b",
      "name": "Qwen3.5 4B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "security-triage",
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 4.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q3": 2.11,
        "q4": 2.78
      },
      "kvMiBPerTokenFp16": 0.08,
      "runtimes": [],
      "quality": 44,
      "notes": [
        "8GB급 범용 기준선"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "foundation-sec-8b",
      "name": "Foundation-Sec 8B Reasoning",
      "kind": "llm",
      "domains": [
        "cybersecurity"
      ],
      "tasks": [
        "security-triage",
        "secure-code"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF",
      "guide": "cybersecurity",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q4": 4.92
      },
      "kvMiBPerTokenFp16": 0.125,
      "runtimes": [],
      "quality": 58,
      "notes": [
        "CVE·위협 인텔리전스·보안 용어 특화"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "granite41-8b",
      "name": "Granite 4.1 8B",
      "kind": "llm",
      "domains": [
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF",
      "guide": "productivity-rag",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q2": 3.41,
        "q3": 4.35,
        "q4": 5.35
      },
      "kvMiBPerTokenFp16": 0.125,
      "runtimes": [],
      "quality": 55,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen35-9b",
      "name": "Qwen3.5 9B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "security-triage",
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 9.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 5.84
      },
      "kvMiBPerTokenFp16": 0.14,
      "runtimes": [],
      "quality": 60,
      "notes": [
        "12GB급 코딩·도구 호출·문서 분석 균형형"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "ministral3-14b",
      "name": "Ministral 3 14B Reasoning",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "secure-code",
        "general-coding",
        "math-science",
        "document-work",
        "personal-rag",
        "eda-code"
      ],
      "paramsB": 14.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 8.24
      },
      "kvMiBPerTokenFp16": 0.16,
      "runtimes": [],
      "quality": 67,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gptoss-20b",
      "name": "gpt-oss 20B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem"
      ],
      "tasks": [
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science"
      ],
      "paramsB": 20.0,
      "architecture": "moe",
      "activeParamsB": 3.6,
      "hf": "https://huggingface.co/unsloth/gpt-oss-20b-GGUF",
      "guide": "cybersecurity",
      "formats": [
        "q2",
        "q3",
        "q4",
        "mxfp4"
      ],
      "exactSizes": {
        "q2": 11.5,
        "q3": 11.5,
        "q4": 11.6,
        "mxfp4": 11.6
      },
      "kvMiBPerTokenFp16": 0.18,
      "runtimes": [],
      "quality": 70,
      "notes": [
        "네이티브 MXFP4 계열은 Q2·Q3·Q4 크기 차이가 작음"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "devstral-small2-24b",
      "name": "Devstral Small 2 24B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "data-analysis"
      ],
      "tasks": [
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "eda-code"
      ],
      "paramsB": 24.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q3": 11.5,
        "q4": 14.9
      },
      "kvMiBPerTokenFp16": 0.2,
      "runtimes": [],
      "quality": 76,
      "notes": [
        "저장소 수준 소프트웨어 엔지니어링 에이전트"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen36-27b",
      "name": "Qwen3.6 27B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 27.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q3": 13.6,
        "q4": 17.6
      },
      "kvMiBPerTokenFp16": 0.24,
      "runtimes": [],
      "quality": 80,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "granite41-30b",
      "name": "Granite 4.1 30B",
      "kind": "llm",
      "domains": [
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 30.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ibm-granite/granite-4.1-30b-GGUF",
      "guide": "productivity-rag",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q2": 10.7,
        "q3": 14.0,
        "q4": 17.5
      },
      "kvMiBPerTokenFp16": 0.24,
      "runtimes": [],
      "quality": 76,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen36-35b-a3b",
      "name": "Qwen3.6 35B-A3B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis"
      ],
      "tasks": [
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "math-science",
        "document-work",
        "personal-rag",
        "team-rag",
        "text-to-sql",
        "eda-code",
        "tabular-ml",
        "time-series"
      ],
      "paramsB": 35.0,
      "architecture": "moe",
      "activeParamsB": 3.0,
      "hf": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q3": 16.8,
        "q4": 22.1
      },
      "kvMiBPerTokenFp16": 0.25,
      "runtimes": [],
      "quality": 84,
      "notes": [
        "활성 파라미터가 작아도 전체 가중치를 적재"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gemma4-31b",
      "name": "Gemma 4 31B",
      "kind": "vlm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "vision-ocr"
      ],
      "tasks": [
        "secure-code",
        "general-coding",
        "math-science",
        "document-work",
        "ui-vision",
        "chart-math"
      ],
      "paramsB": 31.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 18.3
      },
      "kvMiBPerTokenFp16": 0.25,
      "runtimes": [],
      "quality": 81,
      "notes": [
        "멀티모달 입력은 projector·이미지 버퍼 추가"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen3-coder-next",
      "name": "Qwen3-Coder-Next 80B-A3B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "data-analysis"
      ],
      "tasks": [
        "secure-code",
        "security-agent",
        "general-coding",
        "repo-agent",
        "eda-code"
      ],
      "paramsB": 80.0,
      "architecture": "moe",
      "activeParamsB": 3.0,
      "hf": "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 23.3,
        "q3": 33.3,
        "q4": 48.5
      },
      "kvMiBPerTokenFp16": 0.3,
      "runtimes": [],
      "quality": 90,
      "notes": [
        "Q4 크기는 배포 방식에 따라 큰 차이가 있어 보수적으로 48.5GB 사용"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "mistral-small4-119b",
      "name": "Mistral Small 4 119B-A6.5B",
      "kind": "vlm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "vision-ocr"
      ],
      "tasks": [
        "security-agent",
        "repo-agent",
        "math-science",
        "team-rag",
        "ui-vision",
        "chart-math"
      ],
      "paramsB": 119.0,
      "architecture": "moe",
      "activeParamsB": 6.5,
      "hf": "https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 40.2,
        "q3": 54.4,
        "q4": 59.2
      },
      "kvMiBPerTokenFp16": 0.42,
      "runtimes": [],
      "quality": 91,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gptoss-120b",
      "name": "gpt-oss 120B",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem"
      ],
      "tasks": [
        "security-agent",
        "repo-agent",
        "math-science"
      ],
      "paramsB": 120.0,
      "architecture": "moe",
      "activeParamsB": 5.1,
      "hf": "https://huggingface.co/unsloth/gpt-oss-120b-GGUF",
      "guide": "cybersecurity",
      "formats": [
        "q2",
        "q3",
        "q4",
        "mxfp4"
      ],
      "exactSizes": {
        "q2": 62.6,
        "q3": 62.6,
        "q4": 62.8,
        "mxfp4": 62.8
      },
      "kvMiBPerTokenFp16": 0.42,
      "runtimes": [],
      "quality": 92,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "mistral-medium35-128b",
      "name": "Mistral Medium 3.5 128B",
      "kind": "vlm",
      "domains": [
        "cybersecurity",
        "programming-stem",
        "productivity-rag",
        "data-analysis",
        "vision-ocr"
      ],
      "tasks": [
        "security-agent",
        "repo-agent",
        "math-science",
        "team-rag",
        "eda-code",
        "ui-vision",
        "chart-math"
      ],
      "paramsB": 128.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q3": 63.3,
        "q4": 78.4,
        "q5": 91.1,
        "q6": 107.8,
        "q8": 132.9
      },
      "kvMiBPerTokenFp16": 0.46,
      "runtimes": [],
      "quality": 94,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "deepseek-v4-flash",
      "name": "DeepSeek V4 Flash",
      "kind": "llm",
      "domains": [
        "cybersecurity",
        "programming-stem"
      ],
      "tasks": [
        "security-agent",
        "repo-agent",
        "math-science"
      ],
      "paramsB": 236.0,
      "architecture": "moe",
      "activeParamsB": 20.0,
      "hf": "https://huggingface.co/Preyazz/DeepSeek-V4-Flash-GGUF",
      "guide": "cybersecurity",
      "formats": [
        "q3",
        "q4"
      ],
      "exactSizes": {
        "q3": 125.0,
        "q4": 161.0
      },
      "kvMiBPerTokenFp16": 0.65,
      "runtimes": [],
      "quality": 96,
      "notes": [
        "최신 아키텍처와 런타임 지원을 먼저 확인"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "deepseek-prover-v2-7b",
      "name": "DeepSeek-Prover-V2 7B",
      "kind": "llm",
      "domains": [
        "programming-stem"
      ],
      "tasks": [
        "lean-proof"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/DeepSeek-Prover-V2-7B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 2.9,
        "q3": 3.46,
        "q4": 4.22
      },
      "kvMiBPerTokenFp16": 0.12,
      "runtimes": [],
      "quality": 70,
      "notes": [
        "Lean 4 compiler feedback loop 전제"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "goedel-prover-v2-8b",
      "name": "Goedel-Prover-V2 8B",
      "kind": "llm",
      "domains": [
        "programming-stem"
      ],
      "tasks": [
        "lean-proof"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/Goedel-Prover-V2-8B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 3.28,
        "q3": 4.12,
        "q4": 5.03
      },
      "kvMiBPerTokenFp16": 0.13,
      "runtimes": [],
      "quality": 73,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "su01-31b-a3b",
      "name": "SU-01 31B-A3B",
      "kind": "llm",
      "domains": [
        "programming-stem"
      ],
      "tasks": [
        "math-science"
      ],
      "paramsB": 31.0,
      "architecture": "moe",
      "activeParamsB": 3.0,
      "hf": "https://huggingface.co/axi0mX/SU-01-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6"
      ],
      "exactSizes": {
        "q2": 11.8,
        "q3": 14.7,
        "q4": 18.6
      },
      "kvMiBPerTokenFp16": 0.24,
      "runtimes": [],
      "quality": 83,
      "notes": [
        "자연어 수학·과학 올림피아드형 장기 추론"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "goedel-prover-v2-32b",
      "name": "Goedel-Prover-V2 32B",
      "kind": "llm",
      "domains": [
        "programming-stem"
      ],
      "tasks": [
        "lean-proof"
      ],
      "paramsB": 32.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6"
      ],
      "exactSizes": {
        "q2": 12.4,
        "q3": 16.1,
        "q4": 19.9
      },
      "kvMiBPerTokenFp16": 0.25,
      "runtimes": [],
      "quality": 86,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "leanstral15-119b",
      "name": "Leanstral 1.5 119B-A6.5B",
      "kind": "llm",
      "domains": [
        "programming-stem"
      ],
      "tasks": [
        "lean-proof"
      ],
      "paramsB": 119.0,
      "architecture": "moe",
      "activeParamsB": 6.5,
      "hf": "https://huggingface.co/Abiray/Leanstral-1.5-119B-A6B-Q4KM-GGUF",
      "guide": "programming-stem",
      "formats": [
        "q4"
      ],
      "exactSizes": {
        "q4": 72.2
      },
      "kvMiBPerTokenFp16": 0.42,
      "runtimes": [],
      "quality": 95,
      "notes": [
        "Lean 4 증명공학·자동형식화 전용"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "xiyansql-3b",
      "name": "XiYanSQL QwenCoder 3B",
      "kind": "sql",
      "domains": [
        "data-analysis"
      ],
      "tasks": [
        "text-to-sql"
      ],
      "paramsB": 3.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/XiYanSQL-QwenCoder-3B-2504-GGUF",
      "guide": "data-analysis",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 1.27,
        "q4": 1.93
      },
      "kvMiBPerTokenFp16": 0.06,
      "runtimes": [],
      "quality": 52,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "xiyansql-7b",
      "name": "XiYanSQL QwenCoder 7B",
      "kind": "sql",
      "domains": [
        "data-analysis"
      ],
      "tasks": [
        "text-to-sql"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/XiYanSQL-QwenCoder-7B-2504-GGUF",
      "guide": "data-analysis",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q3": 3.81,
        "q4": 4.68
      },
      "kvMiBPerTokenFp16": 0.12,
      "runtimes": [],
      "quality": 65,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "arctic-text2sql-7b",
      "name": "Arctic-Text2SQL-R1 7B",
      "kind": "sql",
      "domains": [
        "data-analysis"
      ],
      "tasks": [
        "text-to-sql"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/Arctic-Text2SQL-R1-7B-GGUF",
      "guide": "data-analysis",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q4": 4.68
      },
      "kvMiBPerTokenFp16": 0.12,
      "runtimes": [],
      "quality": 67,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "omnisql-14b",
      "name": "OmniSQL 14B",
      "kind": "sql",
      "domains": [
        "data-analysis"
      ],
      "tasks": [
        "text-to-sql"
      ],
      "paramsB": 14.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/OmniSQL-14B-GGUF",
      "guide": "data-analysis",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8"
      ],
      "exactSizes": {
        "q2": 5.9,
        "q3": 7.4,
        "q4": 9.1
      },
      "kvMiBPerTokenFp16": 0.16,
      "runtimes": [],
      "quality": 75,
      "notes": [
        "영어·SQLite 중심이므로 대상 dialect 평가 필요"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "xiyansql-32b",
      "name": "XiYanSQL QwenCoder 32B",
      "kind": "sql",
      "domains": [
        "data-analysis"
      ],
      "tasks": [
        "text-to-sql"
      ],
      "paramsB": 32.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mradermacher/XiYanSQL-QwenCoder-32B-2504-GGUF",
      "guide": "data-analysis",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6"
      ],
      "exactSizes": {
        "q2": 12.4,
        "q3": 16.0,
        "q4": 20.0
      },
      "kvMiBPerTokenFp16": 0.25,
      "runtimes": [],
      "quality": 84,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "glm-ocr",
      "name": "GLM-OCR",
      "kind": "ocr",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "document-ocr",
        "chart-math",
        "personal-rag"
      ],
      "paramsB": 0.9,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggml-org/GLM-OCR-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q8": 1.434,
        "bf16": 2.274
      },
      "kvMiBPerTokenFp16": 0.025,
      "runtimes": [],
      "quality": 57,
      "notes": [
        "OCR 전용 모델은 저비트보다 Q8 우선"
      ],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "paddleocr-vl16",
      "name": "PaddleOCR-VL 1.6",
      "kind": "ocr",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "document-ocr",
        "chart-math",
        "personal-rag"
      ],
      "paramsB": 0.9,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q8"
      ],
      "exactSizes": {
        "q8": 1.818
      },
      "kvMiBPerTokenFp16": 0.025,
      "runtimes": [],
      "quality": 60,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "unlimited-ocr",
      "name": "Unlimited-OCR",
      "kind": "ocr",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "document-ocr",
        "chart-math",
        "personal-rag"
      ],
      "paramsB": 3.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q2",
        "q3",
        "q4",
        "q8"
      ],
      "exactSizes": {
        "q2": 2.042,
        "q3": 2.362,
        "q4": 2.762,
        "q8": 3.942
      },
      "kvMiBPerTokenFp16": 0.06,
      "runtimes": [],
      "quality": 64,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gemma4-e2b-vlm",
      "name": "Gemma 4 E2B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "ui-vision",
        "document-ocr",
        "document-work"
      ],
      "paramsB": 4.5,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 3.397
      },
      "kvMiBPerTokenFp16": 0.08,
      "runtimes": [],
      "quality": 63,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gemma4-e4b-vlm",
      "name": "Gemma 4 E4B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "ui-vision",
        "document-ocr",
        "chart-math",
        "document-work"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 5.15,
        "q8": 8.59
      },
      "kvMiBPerTokenFp16": 0.11,
      "runtimes": [],
      "quality": 69,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen3vl-8b",
      "name": "Qwen3-VL 8B",
      "kind": "vlm",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "ui-vision",
        "document-ocr",
        "chart-math",
        "document-work"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 5.782,
        "q8": 9.46
      },
      "kvMiBPerTokenFp16": 0.13,
      "runtimes": [],
      "quality": 75,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gemma4-12b-vlm",
      "name": "Gemma 4 12B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "ui-vision",
        "document-ocr",
        "chart-math",
        "document-work"
      ],
      "paramsB": 12.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 7.379,
        "q8": 12.86
      },
      "kvMiBPerTokenFp16": 0.15,
      "runtimes": [],
      "quality": 79,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "gemma4-26b-a4b-vlm",
      "name": "Gemma 4 26B-A4B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr",
        "productivity-rag"
      ],
      "tasks": [
        "ui-vision",
        "document-ocr",
        "chart-math",
        "team-rag"
      ],
      "paramsB": 26.0,
      "architecture": "moe",
      "activeParamsB": 4.0,
      "hf": "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 15.406,
        "bf16": 51.69
      },
      "kvMiBPerTokenFp16": 0.22,
      "runtimes": [],
      "quality": 84,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen36-27b-vlm",
      "name": "Qwen3.6 27B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr"
      ],
      "tasks": [
        "ui-vision",
        "chart-math",
        "document-ocr"
      ],
      "paramsB": 27.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 19.729,
        "q8": 29.23
      },
      "kvMiBPerTokenFp16": 0.24,
      "runtimes": [],
      "quality": 87,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen36-35b-vlm",
      "name": "Qwen3.6 35B-A3B VLM",
      "kind": "vlm",
      "domains": [
        "vision-ocr"
      ],
      "tasks": [
        "ui-vision",
        "chart-math",
        "document-ocr"
      ],
      "paramsB": 35.0,
      "architecture": "moe",
      "activeParamsB": 3.0,
      "hf": "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 21.014,
        "q8": 37.51,
        "bf16": 70.3
      },
      "kvMiBPerTokenFp16": 0.25,
      "runtimes": [],
      "quality": 90,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "qwen3vl-235b",
      "name": "Qwen3-VL 235B-A22B Thinking",
      "kind": "vlm",
      "domains": [
        "vision-ocr"
      ],
      "tasks": [
        "ui-vision",
        "chart-math",
        "document-ocr"
      ],
      "paramsB": 235.0,
      "architecture": "moe",
      "activeParamsB": 22.0,
      "hf": "https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-GGUF",
      "guide": "vision-ocr",
      "formats": [
        "q4",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 143.0,
        "q8": 250.0,
        "bf16": 470.0
      },
      "kvMiBPerTokenFp16": 0.7,
      "runtimes": [],
      "quality": 98,
      "notes": [],
      "minDeviceGiB": {},
      "basePeakGiB": {},
      "sizeMode": "weights",
      "projectorIncluded": true,
      "systemRamRecommendedGiB": null
    },
    {
      "id": "sd15",
      "name": "Stable Diffusion 1.5",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 0.86,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
      "guide": "image-generation",
      "formats": [
        "fp16"
      ],
      "exactSizes": {
        "fp16": 2.1
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 45,
      "notes": [],
      "minDeviceGiB": {
        "fp16": 4
      },
      "basePeakGiB": {
        "fp16": 3.4
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 16
    },
    {
      "id": "sdxl",
      "name": "Stable Diffusion XL 1.0",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 3.5,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
      "guide": "image-generation",
      "formats": [
        "fp16",
        "bf16",
        "int8"
      ],
      "exactSizes": {
        "fp16": 6.9
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 62,
      "notes": [],
      "minDeviceGiB": {
        "fp16": 6,
        "int8": 6
      },
      "basePeakGiB": {
        "fp16": 6.0,
        "bf16": 6.0,
        "int8": 5.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 24
    },
    {
      "id": "sd35-medium",
      "name": "Stable Diffusion 3.5 Medium",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit"
      ],
      "paramsB": 2.5,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
      "guide": "image-generation",
      "formats": [
        "fp16",
        "bf16",
        "nf4",
        "int8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 68,
      "notes": [],
      "minDeviceGiB": {
        "fp16": 12,
        "nf4": 8,
        "int8": 8
      },
      "basePeakGiB": {
        "fp16": 10.5,
        "bf16": 10.5,
        "nf4": 7.0,
        "int8": 7.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 32
    },
    {
      "id": "flux1-dev",
      "name": "FLUX.1 dev",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 12.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/city96/FLUX.1-dev-gguf",
      "guide": "image-generation",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 72,
      "notes": [],
      "minDeviceGiB": {
        "q4": 8,
        "q5": 10,
        "q6": 12,
        "q8": 16,
        "bf16": 24
      },
      "basePeakGiB": {
        "q4": 7.5,
        "q5": 9.0,
        "q6": 10.5,
        "q8": 14.0,
        "bf16": 22.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 32
    },
    {
      "id": "flux2-klein4",
      "name": "FLUX.2 Klein 4B",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 4.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF",
      "guide": "image-generation",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8",
        "fp8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 2.6
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 80,
      "notes": [
        "공식 full pipeline 약 13GB VRAM 기준"
      ],
      "minDeviceGiB": {
        "q4": 12,
        "q5": 12,
        "q6": 14,
        "q8": 16,
        "fp8": 16,
        "bf16": 16
      },
      "basePeakGiB": {
        "q4": 10.0,
        "q5": 11.0,
        "q6": 12.0,
        "q8": 13.0,
        "fp8": 13.0,
        "bf16": 13.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 48
    },
    {
      "id": "z-image-turbo",
      "name": "Z-Image Turbo",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit"
      ],
      "paramsB": 6.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
      "guide": "image-generation",
      "formats": [
        "nf4",
        "fp8",
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 79,
      "notes": [],
      "minDeviceGiB": {
        "nf4": 12,
        "fp8": 16,
        "bf16": 16
      },
      "basePeakGiB": {
        "nf4": 10.0,
        "fp8": 13.0,
        "bf16": 14.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 48
    },
    {
      "id": "ideogram4-nf4",
      "name": "Ideogram 4 NF4",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit"
      ],
      "paramsB": 12.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ideogram-ai/ideogram-4-nf4",
      "guide": "image-generation",
      "formats": [
        "nf4"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 87,
      "notes": [
        "타이포그래피·레이아웃 중심"
      ],
      "minDeviceGiB": {
        "nf4": 24
      },
      "basePeakGiB": {
        "nf4": 21.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "hidream-o1",
      "name": "HiDream O1 Image",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/HiDream-ai/HiDream-O1-Image",
      "guide": "image-generation",
      "formats": [
        "fp8",
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 86,
      "notes": [],
      "minDeviceGiB": {
        "fp8": 24,
        "bf16": 32
      },
      "basePeakGiB": {
        "fp8": 21.0,
        "bf16": 28.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "qwen-image-q4",
      "name": "Qwen-Image",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit"
      ],
      "paramsB": 20.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/city96/Qwen-Image-gguf",
      "guide": "image-generation",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8",
        "bf16"
      ],
      "exactSizes": {
        "q4": 13.0
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 88,
      "notes": [
        "텍스트 인코더·VAE는 별도 구성요소"
      ],
      "minDeviceGiB": {
        "q4": 24,
        "q5": 32,
        "q6": 32,
        "q8": 48,
        "bf16": 64
      },
      "basePeakGiB": {
        "q4": 20.0,
        "q5": 24.0,
        "q6": 27.0,
        "q8": 40.0,
        "bf16": 58.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "flux2-klein9",
      "name": "FLUX.2 Klein 9B",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 9.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
      "guide": "image-generation",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8",
        "fp8",
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 89,
      "notes": [
        "공식 full pipeline 약 29GB 기준"
      ],
      "minDeviceGiB": {
        "q4": 24,
        "q5": 28,
        "q6": 32,
        "q8": 32,
        "fp8": 32,
        "bf16": 32
      },
      "basePeakGiB": {
        "q4": 22.0,
        "q5": 25.0,
        "q6": 27.0,
        "q8": 29.0,
        "fp8": 29.0,
        "bf16": 29.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 80
    },
    {
      "id": "flux2-dev",
      "name": "FLUX.2 dev",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image",
        "image-edit",
        "image-control"
      ],
      "paramsB": 32.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/unsloth/FLUX.2-dev-GGUF",
      "guide": "image-generation",
      "formats": [
        "q4",
        "q5",
        "q6",
        "q8",
        "fp8",
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 94,
      "notes": [],
      "minDeviceGiB": {
        "q4": 48,
        "q5": 48,
        "q6": 64,
        "q8": 64,
        "fp8": 64,
        "bf16": 96
      },
      "basePeakGiB": {
        "q4": 42.0,
        "q5": 46.0,
        "q6": 54.0,
        "q8": 60.0,
        "fp8": 60.0,
        "bf16": 88.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 128
    },
    {
      "id": "hunyuanimage3-base",
      "name": "HunyuanImage 3.0 Base",
      "kind": "image",
      "domains": [
        "image-generation"
      ],
      "tasks": [
        "text-to-image"
      ],
      "paramsB": 80.0,
      "architecture": "moe",
      "activeParamsB": 13.0,
      "hf": "https://huggingface.co/tencent/HunyuanImage-3.0",
      "guide": "image-generation",
      "formats": [
        "bf16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 99,
      "notes": [
        "공식 최소 3×80GB 경로"
      ],
      "minDeviceGiB": {
        "bf16": 240
      },
      "basePeakGiB": {
        "bf16": 240.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 512
    },
    {
      "id": "whisper-small-q5",
      "name": "Whisper small Q5",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "asr"
      ],
      "paramsB": 0.24,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggerganov/whisper.cpp",
      "guide": "audio-speech",
      "formats": [
        "q5",
        "q8"
      ],
      "exactSizes": {
        "q5": 0.5,
        "q8": 0.8
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 48,
      "notes": [],
      "minDeviceGiB": {
        "q5": 2,
        "q8": 2
      },
      "basePeakGiB": {
        "q5": 1.2,
        "q8": 1.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 8
    },
    {
      "id": "whisper-large-turbo",
      "name": "Whisper large-v3-turbo",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "asr"
      ],
      "paramsB": 0.81,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/ggerganov/whisper.cpp",
      "guide": "audio-speech",
      "formats": [
        "q5",
        "q8",
        "fp16"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 70,
      "notes": [],
      "minDeviceGiB": {
        "q5": 4,
        "q8": 4,
        "fp16": 6
      },
      "basePeakGiB": {
        "q5": 2.8,
        "q8": 3.5,
        "fp16": 5.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 12
    },
    {
      "id": "qwen3-asr-06",
      "name": "Qwen3-ASR 0.6B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "asr",
        "streaming-asr"
      ],
      "paramsB": 0.6,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {
        "bf16": 1.88,
        "fp16": 1.88
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 72,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 4,
        "fp16": 4,
        "int8": 3
      },
      "basePeakGiB": {
        "bf16": 3.0,
        "fp16": 3.0,
        "int8": 2.2
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 12
    },
    {
      "id": "parakeet-06",
      "name": "Parakeet TDT 0.6B v3",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "asr",
        "streaming-asr"
      ],
      "paramsB": 0.6,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 73,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 6,
        "fp16": 6,
        "int8": 4
      },
      "basePeakGiB": {
        "bf16": 4.5,
        "fp16": 4.5,
        "int8": 3.2
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 16
    },
    {
      "id": "nemotron-asr-06",
      "name": "Nemotron 3.5 ASR Streaming 0.6B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "streaming-asr"
      ],
      "paramsB": 0.6,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 75,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 6,
        "fp16": 6,
        "int8": 4
      },
      "basePeakGiB": {
        "bf16": 4.5,
        "fp16": 4.5,
        "int8": 3.2
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 16
    },
    {
      "id": "qwen3-tts-06",
      "name": "Qwen3-TTS 0.6B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "tts"
      ],
      "paramsB": 0.6,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 73,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 6,
        "fp16": 6,
        "int8": 4
      },
      "basePeakGiB": {
        "bf16": 4.5,
        "fp16": 4.5,
        "int8": 3.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 16
    },
    {
      "id": "qwen3-asr-17",
      "name": "Qwen3-ASR 1.7B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "asr",
        "streaming-asr"
      ],
      "paramsB": 1.7,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {
        "bf16": 4.7,
        "fp16": 4.7
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 80,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 8,
        "fp16": 8,
        "int8": 6
      },
      "basePeakGiB": {
        "bf16": 6.5,
        "fp16": 6.5,
        "int8": 4.8
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 24
    },
    {
      "id": "qwen3-tts-17",
      "name": "Qwen3-TTS 1.7B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "tts"
      ],
      "paramsB": 1.7,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 82,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 12,
        "fp16": 12,
        "int8": 8
      },
      "basePeakGiB": {
        "bf16": 9.5,
        "fp16": 9.5,
        "int8": 6.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 32
    },
    {
      "id": "voxtral-realtime4b",
      "name": "Voxtral Mini 4B Realtime",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "streaming-asr"
      ],
      "paramsB": 4.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "fp8"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 86,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 16,
        "fp16": 16,
        "fp8": 12
      },
      "basePeakGiB": {
        "bf16": 14.0,
        "fp16": 14.0,
        "fp8": 10.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 48
    },
    {
      "id": "phi4-multimodal",
      "name": "Phi-4 Multimodal",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "audio-agent",
        "asr"
      ],
      "paramsB": 5.6,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {
        "bf16": 12.9,
        "fp16": 12.9
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 86,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 24,
        "fp16": 24,
        "int8": 16
      },
      "basePeakGiB": {
        "bf16": 20.0,
        "fp16": 20.0,
        "int8": 14.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "audio-flamingo-next",
      "name": "Audio Flamingo Next",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "audio-agent"
      ],
      "paramsB": 8.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/nvidia/audio-flamingo-next-hf",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16",
        "int8"
      ],
      "exactSizes": {
        "bf16": 16.5,
        "fp16": 16.5
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 89,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 24,
        "fp16": 24,
        "int8": 18
      },
      "basePeakGiB": {
        "bf16": 21.0,
        "fp16": 21.0,
        "int8": 15.5
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "qwen25-omni7-gptq",
      "name": "Qwen2.5-Omni 7B GPTQ-Int4",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "audio-agent"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
      "guide": "audio-speech",
      "formats": [
        "awq"
      ],
      "exactSizes": {},
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 88,
      "notes": [],
      "minDeviceGiB": {
        "awq": 24
      },
      "basePeakGiB": {
        "awq": 20.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 64
    },
    {
      "id": "qwen25-omni7",
      "name": "Qwen2.5-Omni 7B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "audio-agent"
      ],
      "paramsB": 7.0,
      "architecture": "dense",
      "activeParamsB": null,
      "hf": "https://huggingface.co/Qwen/Qwen2.5-Omni-7B",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp16"
      ],
      "exactSizes": {
        "bf16": 22.4,
        "fp16": 22.4
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 91,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 32,
        "fp16": 32
      },
      "basePeakGiB": {
        "bf16": 28.0,
        "fp16": 28.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 80
    },
    {
      "id": "qwen3-omni30-a3b",
      "name": "Qwen3-Omni 30B-A3B",
      "kind": "audio",
      "domains": [
        "audio-speech"
      ],
      "tasks": [
        "audio-agent"
      ],
      "paramsB": 30.0,
      "architecture": "moe",
      "activeParamsB": 3.0,
      "hf": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct",
      "guide": "audio-speech",
      "formats": [
        "bf16",
        "fp8"
      ],
      "exactSizes": {
        "bf16": 70.5
      },
      "kvMiBPerTokenFp16": 0.0,
      "runtimes": [],
      "quality": 98,
      "notes": [],
      "minDeviceGiB": {
        "bf16": 80,
        "fp8": 64
      },
      "basePeakGiB": {
        "bf16": 76.0,
        "fp8": 58.0
      },
      "sizeMode": "pipeline",
      "projectorIncluded": false,
      "systemRamRecommendedGiB": 192
    }
  ]
};
