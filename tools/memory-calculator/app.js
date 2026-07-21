(() => {
  "use strict";

  const catalog = window.RAMAI_CATALOG;
  if (!catalog) {
    document.body.innerHTML = '<p style="padding:1rem;color:#fff">catalog.js를 불러오지 못했습니다.</p>';
    return;
  }

  const $ = (id) => document.getElementById(id);
  const form = $("calculatorForm");
  const modelById = new Map(catalog.models.map((model) => [model.id, model]));
  const guideById = catalog.guides;
  const repository = catalog.repository || {};
  const runtimeById = new Map(catalog.runtimes.map((runtime) => [runtime.id, runtime]));
  const quantById = catalog.quantizations;
  const storageKey = `ram-for-local-ai-memory-calculator-v${catalog.schemaVersion}`;

  function normalizeRepositoryPath(path) {
    return String(path || "")
      .trim()
      .replace(/\\/g, "/")
      .replace(/^(?:\.\/)+/, "")
      .replace(/^(?:\.\.\/)+/, "")
      .replace(/^\/+/, "");
  }

  function inferRepositoryUrl() {
    const pagesMatch = window.location.hostname.match(/^([^.]+)\.github\.io$/i);
    const pathSegments = window.location.pathname.split("/").filter(Boolean);
    if (pagesMatch && pathSegments.length > 0) {
      return `https://github.com/${pagesMatch[1]}/${pathSegments[0]}`;
    }
    return String(repository.url || "").replace(/\/+$/, "");
  }

  const repositoryUrl = inferRepositoryUrl();
  const repositoryBranch = String(repository.defaultBranch || repository.branch || "main");

  function repositoryFileUrl(path) {
    const value = String(path || "").trim();
    if (/^https?:\/\//i.test(value)) return value;

    const normalized = normalizeRepositoryPath(value);
    if (!normalized) return repositoryUrl || "#";

    if (repositoryUrl) {
      const encodedPath = normalized
        .split("/")
        .filter(Boolean)
        .map((segment) => encodeURIComponent(segment))
        .join("/");
      return `${repositoryUrl}/blob/${encodeURIComponent(repositoryBranch)}/${encodedPath}`;
    }

    return new URL(`../../${normalized}`, window.location.href).href;
  }

  function repositoryReadmeUrl() {
    if (repositoryUrl) return `${repositoryUrl}#readme`;
    const readmePath = normalizeRepositoryPath(repository.readmePath || "README.md");
    return new URL(`../../${readmePath}`, window.location.href).href;
  }

  function markExternalLink(anchor, href) {
    if (!href || !/^https?:\/\//i.test(href)) return;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
  }

  function applyRepositoryLinks() {
    const readmeLink = $("repositoryReadmeLink");
    if (!readmeLink) return;
    const href = repositoryReadmeUrl();
    readmeLink.href = href;
    markExternalLink(readmeLink, href);
  }

  const defaultModelByTask = {
    "security-triage": "foundation-sec-8b",
    "secure-code": "qwen35-9b",
    "security-agent": "devstral-small2-24b",
    "general-coding": "qwen35-9b",
    "repo-agent": "devstral-small2-24b",
    "math-science": "qwen35-9b",
    "lean-proof": "deepseek-prover-v2-7b",
    "document-work": "granite41-8b",
    "personal-rag": "granite41-8b",
    "team-rag": "qwen36-27b",
    "text-to-sql": "xiyansql-7b",
    "eda-code": "qwen35-9b",
    "tabular-ml": "qwen35-4b",
    "time-series": "qwen35-4b",
    "document-ocr": "glm-ocr",
    "ui-vision": "qwen3vl-8b",
    "chart-math": "qwen3vl-8b",
    "text-to-image": "flux2-klein4",
    "image-edit": "flux2-klein4",
    "image-control": "sdxl",
    "asr": "qwen3-asr-06",
    "streaming-asr": "nemotron-asr-06",
    "tts": "qwen3-tts-06",
    "audio-agent": "qwen25-omni7-gptq"
  };

  const kindLabels = {
    llm: "LLM",
    vlm: "VLM",
    ocr: "OCR",
    sql: "Text-to-SQL",
    image: "이미지",
    audio: "오디오"
  };

  const fitMeta = {
    excellent: { label: "여유", headline: "안정적인 시작점입니다." },
    good: { label: "적합", headline: "권장 여유 안에서 실행 가능합니다." },
    tight: { label: "경계", headline: "실행 가능성이 있으나 peak 측정이 필요합니다." },
    offload: { label: "조정 필요", headline: "오프로딩·컨텍스트·동시성 조정이 필요합니다." },
    insufficient: { label: "부족", headline: "현재 설정은 메모리 예산을 초과합니다." },
    neutral: { label: "대기", headline: "입력값을 확인하십시오." }
  };

  let latestEstimate = null;
  let latestDetection = null;
  let calculationTimer = null;

  function numberValue(id, fallback = 0) {
    const value = Number($(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function ceilDiv(a, b) {
    return Math.ceil(a / Math.max(1, b));
  }

  function formatMemory(value) {
    if (!Number.isFinite(value)) return "—";
    if (value < 0.01) return "<0.01 GB";
    if (value < 10) return `${value.toFixed(2)} GB`;
    if (value < 100) return `${value.toFixed(1)} GB`;
    return `${Math.round(value)} GB`;
  }

  function formatPercent(value) {
    if (!Number.isFinite(value)) return "—";
    return `${Math.round(value * 100)}%`;
  }

  function formatTokens(value) {
    if (!Number.isFinite(value) || value <= 0) return "0";
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`;
    return `${Math.round(value)}`;
  }

  function formatNumber(value, digits = 2) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
  }

  function populateSelect(select, items, valueKey = "id", labelKey = "label") {
    select.replaceChildren();
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item[valueKey];
      option.textContent = item[labelKey];
      select.append(option);
    }
  }

  function initializeBaseSelects() {
    populateSelect($("domain"), catalog.domains);
    populateSelect($("operation"), catalog.operations);
    populateSelect($("runtime"), catalog.runtimes);

    $("kvDtype").replaceChildren();
    for (const [id, kv] of Object.entries(catalog.kvDtypes)) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = kv.label;
      $("kvDtype").append(option);
    }

    populateSelect($("embeddingModel"), catalog.addons.embeddings);
    populateSelect($("rerankerModel"), catalog.addons.rerankers);
    populateSelect($("parserModel"), catalog.addons.parsers);

    $("domain").value = "programming-stem";
    $("operation").value = "inference";
    $("runtime").value = "auto";
    $("kvDtype").value = "q8";
    $("embeddingModel").value = "granite311";
    $("rerankerModel").value = "qwen06";
    $("parserModel").value = "classic";
  }

  function currentDomain() {
    return catalog.domains.find((domain) => domain.id === $("domain").value) || catalog.domains[0];
  }

  function populateTasks(preferredValue) {
    const domain = currentDomain();
    populateSelect($("task"), domain.tasks);
    const candidate = preferredValue && domain.tasks.some((task) => task.id === preferredValue)
      ? preferredValue
      : domain.defaultTask;
    $("task").value = candidate;
  }

  function filteredModels() {
    const domain = $("domain").value;
    const task = $("task").value;
    return catalog.models
      .filter((model) => model.domains.includes(domain) && (model.tasks.includes(task) || model.tasks.length === 0))
      .sort((a, b) => {
        const aSize = a.paramsB || 0;
        const bSize = b.paramsB || 0;
        return aSize - bSize || a.name.localeCompare(b.name, "ko");
      });
  }

  function populateModels(preferredValue) {
    const select = $("model");
    const models = filteredModels();
    select.replaceChildren();

    const groups = new Map();
    for (const model of models) {
      const label = kindLabels[model.kind] || "기타";
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(model);
    }

    for (const [label, entries] of groups) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = label;
      for (const model of entries) {
        const option = document.createElement("option");
        option.value = model.id;
        const architecture = model.architecture === "moe" && model.activeParamsB
          ? `${model.paramsB}B-A${model.activeParamsB}B`
          : `${model.paramsB}B`;
        option.textContent = `${model.name} · ${architecture}`;
        optgroup.append(option);
      }
      select.append(optgroup);
    }

    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "사용자 정의 모델";
    select.append(custom);

    const defaultId = defaultModelByTask[$("task").value];
    if (preferredValue && [...select.options].some((option) => option.value === preferredValue)) {
      select.value = preferredValue;
    } else if (defaultId && [...select.options].some((option) => option.value === defaultId)) {
      select.value = defaultId;
    } else if (models[0]) {
      select.value = models[0].id;
    } else {
      select.value = "custom";
    }
  }

  function preferredQuantForModel(model) {
    const orderByKind = {
      image: ["q4", "nf4", "fp8", "int8", "fp16", "bf16", "q5", "q6", "q8"],
      audio: ["int8", "q5", "q8", "awq", "fp8", "bf16", "fp16"],
      ocr: ["q8", "q4", "bf16", "q3", "q2"],
      default: ["q4", "q5", "q6", "q3", "q8", "mxfp4", "awq", "bf16", "fp16", "q2"]
    };
    const order = orderByKind[model.kind] || orderByKind.default;
    return order.find((format) => model.formats.includes(format)) || model.formats[0];
  }

  function populateQuants(preferredValue) {
    const model = getSelectedModel();
    const formats = model.formats?.length ? model.formats : Object.keys(quantById);
    $("quant").replaceChildren();
    for (const id of formats) {
      const quant = quantById[id];
      if (!quant) continue;
      const option = document.createElement("option");
      option.value = id;
      const exact = model.exactSizes?.[id];
      option.textContent = exact ? `${quant.label} · ${exact} GB` : quant.label;
      $("quant").append(option);
    }
    if (preferredValue && formats.includes(preferredValue)) {
      $("quant").value = preferredValue;
    } else {
      $("quant").value = preferredQuantForModel(model);
    }
  }

  function getSelectedModel() {
    const id = $("model").value;
    if (id !== "custom" && modelById.has(id)) return modelById.get(id);

    const domain = $("domain").value;
    const kind = domain === "image-generation"
      ? "image"
      : domain === "audio-speech"
        ? "audio"
        : domain === "vision-ocr"
          ? "vlm"
          : "llm";
    return {
      id: "custom",
      name: "사용자 정의 모델",
      kind,
      domains: [domain],
      tasks: [$("task").value],
      paramsB: Math.max(0.05, numberValue("customParams", 8)),
      architecture: $("customArchitecture").value,
      activeParamsB: null,
      hf: "",
      guide: currentDomain().guide,
      formats: Object.keys(quantById),
      exactSizes: {},
      kvMiBPerTokenFp16: Math.max(0, numberValue("customKv", 0.13)),
      runtimes: [],
      quality: 50,
      notes: ["사용자 입력 파라미터 기반 추정"],
      minDeviceGiB: {},
      basePeakGiB: {},
      sizeMode: "weights",
      projectorIncluded: false,
      systemRamRecommendedGiB: null
    };
  }

  function updateModelMeta() {
    const model = getSelectedModel();
    const custom = model.id === "custom";
    $("customModelFields").classList.toggle("is-hidden", !custom);
    $("customModelFields").setAttribute("aria-hidden", custom ? "false" : "true");

    const architecture = model.architecture === "moe" && model.activeParamsB
      ? `MoE ${model.paramsB}B total / ${model.activeParamsB}B active`
      : `${model.architecture === "moe" ? "MoE" : "Dense"} ${model.paramsB}B`;
    const exactCount = Object.keys(model.exactSizes || {}).length;
    const suffix = exactCount ? ` · 가이드 실측 크기 ${exactCount}개 형식` : " · 비트 기반 추정";
    $("modelMeta").textContent = `${kindLabels[model.kind] || model.kind} · ${architecture}${suffix}`;

    const quant = quantById[$("quant").value];
    const exact = model.exactSizes?.[$("quant").value];
    $("quantMeta").textContent = exact
      ? `${quant?.note || ""} · 가이드 대표값 ${exact} GB`
      : `${quant?.note || ""} · 모델 구조와 block metadata에 따라 실제값이 달라집니다.`;
  }

  function updateConditionalPanels() {
    const mode = $("hardwareMode").value;
    document.querySelectorAll("[data-hardware-visible]").forEach((element) => {
      const visible = element.dataset.hardwareVisible.split(/\s+/).includes(mode);
      element.classList.toggle("is-hidden", !visible);
    });

    const domain = $("domain").value;
    const operation = $("operation").value;
    document.querySelectorAll(".conditional-panel").forEach((panel) => {
      const domainMatch = !panel.dataset.domain || panel.dataset.domain === domain;
      const operationMatch = !panel.dataset.operation || panel.dataset.operation === operation;
      panel.classList.toggle("is-hidden", !(domainMatch && operationMatch));
    });

    const offload = $("gpuOffload");
    if (mode === "cpu") {
      offload.value = "0";
      offload.disabled = true;
    } else if (mode === "unified") {
      offload.value = "100";
      offload.disabled = true;
    } else {
      offload.disabled = false;
    }

    if (domain === "image-generation") {
      $("contextTokens").disabled = true;
      $("outputTokens").disabled = true;
      $("kvDtype").disabled = true;
    } else if (domain === "audio-speech" && $("task").value !== "audio-agent") {
      $("contextTokens").disabled = true;
      $("outputTokens").disabled = true;
      $("kvDtype").disabled = true;
    } else {
      $("contextTokens").disabled = false;
      $("outputTokens").disabled = false;
      $("kvDtype").disabled = false;
    }
  }

  function recommendedRuntime(config, model) {
    if (config.operation === "fine-tuning") return "auto";
    if (model.kind === "image") return "diffusers";
    if (model.kind === "audio") {
      if (model.id.startsWith("whisper")) return "whisper.cpp";
      if (config.vendor === "nvidia" && (config.operation === "serving" || model.id.includes("voxtral"))) return "vllm";
      return "onnx";
    }
    if (config.vendor === "apple" || config.hardwareMode === "unified") return "mlx";
    if (config.hardwareMode === "cpu" || config.vendor === "cpu") return "llama.cpp";
    if (config.vendor === "intel") return "onnx";
    if (config.vendor === "nvidia" && config.operation === "serving") return "vllm";
    return "llama.cpp";
  }

  function deriveHardware(config) {
    const mode = config.hardwareMode;
    const replicas = Math.max(1, config.replicas);
    let deviceTotal = 0;
    let groupCount = 1;
    let label = "장치 메모리";

    if (mode === "dedicated") {
      deviceTotal = config.vramPerGpu;
      label = "VRAM";
    } else if (mode === "multi") {
      if (config.multiGpuMode === "shard") {
        deviceTotal = config.vramPerGpu * config.gpuCount;
        groupCount = 1;
        label = `분할 VRAM ${config.gpuCount}장 합계`;
      } else {
        deviceTotal = config.vramPerGpu;
        groupCount = config.gpuCount;
        label = "replica당 VRAM";
      }
    } else if (mode === "unified") {
      deviceTotal = config.unifiedMemory;
      label = "통합 메모리";
    } else {
      deviceTotal = 0;
      label = "CPU 전용";
    }

    const replicaDeviceFactor = mode === "multi" && config.multiGpuMode === "replica"
      ? ceilDiv(replicas, groupCount)
      : replicas;
    const sequencesOnGroup = mode === "multi" && config.multiGpuMode === "replica"
      ? ceilDiv(config.concurrency, groupCount)
      : config.concurrency;
    const streamsOnGroup = mode === "multi" && config.multiGpuMode === "replica"
      ? ceilDiv(config.audioStreams, groupCount)
      : config.audioStreams;

    const deviceReserve = clamp(config.deviceReserve / 100, 0, 0.9);
    const systemReserve = clamp(config.systemReserve / 100, 0, 0.9);
    const unifiedReserve = Math.max(deviceReserve, systemReserve);

    return {
      mode,
      label,
      isUnified: mode === "unified",
      isCpu: mode === "cpu",
      deviceTotal,
      deviceAvailable: mode === "unified"
        ? deviceTotal * (1 - unifiedReserve)
        : deviceTotal * (1 - deviceReserve),
      systemTotal: config.systemRam,
      systemAvailable: config.systemRam * (1 - systemReserve),
      groupCount,
      replicaDeviceFactor,
      sequencesOnGroup: Math.max(1, sequencesOnGroup),
      streamsOnGroup: Math.max(1, streamsOnGroup),
      totalClusterVram: mode === "multi" ? config.vramPerGpu * config.gpuCount : deviceTotal
    };
  }

  function collectConfig() {
    return {
      hardwareMode: $("hardwareMode").value,
      vendor: $("hardwareVendor").value,
      systemRam: Math.max(1, numberValue("systemRam", 32)),
      vramPerGpu: Math.max(0, numberValue("vramPerGpu", 16)),
      gpuCount: Math.max(1, Math.round(numberValue("gpuCount", 2))),
      multiGpuMode: $("multiGpuMode").value,
      unifiedMemory: Math.max(1, numberValue("unifiedMemory", 32)),
      gpuOffload: clamp(numberValue("gpuOffload", 100), 0, 100),
      deviceReserve: clamp(numberValue("deviceReserve", 12), 0, 90),
      systemReserve: clamp(numberValue("systemReserve", 20), 0, 90),
      domain: $("domain").value,
      task: $("task").value,
      operation: $("operation").value,
      runtime: $("runtime").value,
      quant: $("quant").value,
      weightOverride: Math.max(0, numberValue("weightOverride", 0)),
      contextTokens: Math.max(0, Math.round(numberValue("contextTokens", 8192))),
      outputTokens: Math.max(0, Math.round(numberValue("outputTokens", 2048))),
      concurrency: Math.max(1, Math.round(numberValue("concurrency", 1))),
      kvDtype: $("kvDtype").value,
      prefixShare: clamp(numberValue("prefixShare", 0), 0, 95),
      replicas: Math.max(1, Math.round(numberValue("replicas", 1))),
      sequentialAddons: $("sequentialAddons").checked,
      embeddingModel: $("embeddingModel").value,
      rerankerModel: $("rerankerModel").value,
      parserModel: $("parserModel").value,
      ragChunks: Math.max(0, Math.round(numberValue("ragChunks", 100000))),
      embeddingDims: Math.max(1, Math.round(numberValue("embeddingDims", 768))),
      vectorBytes: Math.max(1, numberValue("vectorDtype", 2)),
      indexOverhead: Math.max(1, numberValue("indexOverhead", 1.6)),
      documentCache: Math.max(0, numberValue("documentCache", 1)),
      datasetSize: Math.max(0, numberValue("datasetSize", 2)),
      datasetExpansion: Math.max(0, numberValue("datasetExpansion", 2.5)),
      workingMultiplier: Math.max(0, numberValue("workingMultiplier", 1.5)),
      notebookReserve: Math.max(0, numberValue("notebookReserve", 2)),
      imageCount: Math.max(1, Math.round(numberValue("imageCount", 1))),
      megapixels: Math.max(0.01, numberValue("megapixels", 2)),
      visionWorkerRam: Math.max(0, numberValue("visionWorkerRam", 1)),
      imageWidth: Math.max(64, Math.round(numberValue("imageWidth", 1024))),
      imageHeight: Math.max(64, Math.round(numberValue("imageHeight", 1024))),
      imageBatch: Math.max(1, Math.round(numberValue("imageBatch", 1))),
      referenceImages: Math.max(0, Math.round(numberValue("referenceImages", 0))),
      controlNets: Math.max(0, Math.round(numberValue("controlNets", 0))),
      loraCount: Math.max(0, Math.round(numberValue("loraCount", 0))),
      upscaler: $("upscaler").checked,
      imageCpuOffload: $("imageCpuOffload").checked,
      audioStreams: Math.max(1, Math.round(numberValue("audioStreams", 1))),
      audioMinutes: Math.max(0.1, numberValue("audioMinutes", 10)),
      diarization: $("diarization").checked,
      audioCascade: $("audioCascade").checked,
      trainMethod: $("trainMethod").value,
      trainSequence: Math.max(128, Math.round(numberValue("trainSequence", 2048))),
      microBatch: Math.max(1, Math.round(numberValue("microBatch", 1))),
      loraRank: Math.max(1, Math.round(numberValue("loraRank", 16))),
      loraTarget: $("loraTarget").value,
      gradientCheckpointing: $("gradientCheckpointing").checked,
      trainingDataRam: Math.max(0, numberValue("trainingDataRam", 4)),
      checkpointStaging: Math.max(0, numberValue("checkpointStaging", 2))
    };
  }

  function estimateWeight(model, quantId, override = 0) {
    if (override > 0) {
      return { size: override, source: "수동 재정의", exact: true };
    }
    const exact = model.exactSizes?.[quantId];
    if (Number.isFinite(exact)) {
      return { size: exact, source: "가이드 대표값", exact: true };
    }
    const quant = quantById[quantId] || quantById.q4;
    const size = model.paramsB * (quant.bits / 8) * quant.overhead;
    return { size, source: "파라미터·유효 비트 추정", exact: false };
  }

  function resolveRuntime(config, model) {
    const id = config.runtime === "auto" ? recommendedRuntime(config, model) : config.runtime;
    return runtimeById.get(id) || runtimeById.get("auto");
  }

  function fitFromRatios(deviceRatio, systemRatio, hardware) {
    const activeRatios = [];
    if (!hardware.isCpu && hardware.deviceAvailable > 0) activeRatios.push(deviceRatio);
    if (!hardware.isUnified && hardware.systemAvailable > 0) activeRatios.push(systemRatio);
    if (hardware.isCpu) activeRatios.push(systemRatio);
    if (hardware.isUnified) activeRatios.push(deviceRatio);
    const ratio = Math.max(...activeRatios.filter(Number.isFinite), 0);
    if (ratio <= 0.70) return { id: "excellent", ratio };
    if (ratio <= 0.86) return { id: "good", ratio };
    if (ratio <= 1.00) return { id: "tight", ratio };
    if (ratio <= 1.15) return { id: "offload", ratio };
    return { id: "insufficient", ratio };
  }

  function calculateFineTuning(config, model, quantId, options = {}) {
    const hardware = deriveHardware(config);
    const components = [];
    const warnings = [];

    function add(label, device = 0, system = 0, dynamic = false) {
      if (hardware.isUnified) {
        device += system;
        system = 0;
      } else if (hardware.isCpu) {
        system += device;
        device = 0;
      }
      components.push({ label, device: Math.max(0, device), system: Math.max(0, system), dynamic });
    }

    const p = model.paramsB;
    const activeP = model.architecture === "moe" && model.activeParamsB ? model.activeParamsB : p;
    let modelState = 0;
    let stateLabel = "학습 모델 상태";
    let trainableFraction = 0;

    if (config.trainMethod === "qlora") {
      modelState = p * 0.62;
      stateLabel = "QLoRA frozen base · 0.62 B/P";
      trainableFraction = { attention: 0.0012, selective: 0.0025, all: 0.0055 }[config.loraTarget] * (config.loraRank / 16);
    } else if (config.trainMethod === "lora" || config.trainMethod === "dora") {
      modelState = p * 2.0;
      stateLabel = "BF16/FP16 frozen base · 2 B/P";
      trainableFraction = { attention: 0.0012, selective: 0.0025, all: 0.0055 }[config.loraTarget] * (config.loraRank / 16);
    } else if (config.trainMethod === "full8") {
      modelState = p * 6.0;
      stateLabel = "Full FT + 8-bit optimizer · 6 B/P";
    } else if (config.trainMethod === "fullbf16") {
      modelState = p * 12.0;
      stateLabel = "Native BF16 Adam · 12 B/P";
    } else {
      modelState = p * 18.0;
      stateLabel = "Classic mixed AdamW · 18 B/P";
    }

    const trainableState = trainableFraction > 0
      ? p * trainableFraction * (config.trainMethod === "dora" ? 19 : 16)
      : 0;
    const checkpointFactor = config.gradientCheckpointing ? 0.45 : 1.0;
    let activation = Math.max(0.5, activeP * 0.55 * (config.trainSequence / 2048) * config.microBatch * checkpointFactor);

    if (model.kind === "image") {
      const pixels = (config.imageWidth * config.imageHeight) / (1024 * 1024);
      activation *= 1.4 * pixels * config.imageBatch;
    } else if (model.kind === "vlm" || model.kind === "ocr") {
      activation *= 1 + config.imageCount * config.megapixels * 0.08;
    } else if (model.kind === "audio") {
      activation *= 1 + Math.min(4, config.audioMinutes / 10) * 0.18;
    }

    const workspace = Math.max(1.0, modelState * 0.08);
    const pinnedAndStaging = Math.max(1.5, modelState * 0.12);

    add(stateLabel, modelState, 0, false);
    if (trainableState > 0) add("LoRA·DoRA parameter/gradient/optimizer", trainableState, 0, true);
    add("Saved activation", activation, 0, true);
    add("Kernel·quantization·communication workspace", workspace, 0, true);
    add("Pinned memory·CPU staging", 0, pinnedAndStaging, false);
    add("Dataset·전처리 캐시", 0, config.trainingDataRam, false);
    add("Checkpoint staging", 0, config.checkpointStaging, false);

    const deviceRequired = components.reduce((sum, item) => sum + item.device, 0);
    const systemRequired = components.reduce((sum, item) => sum + item.system, 0);
    const deviceRatio = hardware.deviceAvailable > 0 ? deviceRequired / hardware.deviceAvailable : 0;
    const systemRatio = hardware.systemAvailable > 0 ? systemRequired / hardware.systemAvailable : Infinity;
    const fit = fitFromRatios(deviceRatio, systemRatio, hardware);

    if (hardware.isCpu) warnings.push({ level: "danger", text: "CPU 전용 파인튜닝은 계산상 가능해도 학습 시간이 매우 길 수 있습니다." });
    if (config.hardwareMode === "multi" && config.multiGpuMode === "replica") warnings.push({ level: "danger", text: "Full/adapter 학습의 모델 상태 sharding에는 replica 배치가 아니라 FSDP2·ZeRO·TP 구성이 필요합니다." });
    if (config.trainSequence > 8192) warnings.push({ level: "warn", text: "긴 sequence는 activation을 선형 이상으로 늘릴 수 있습니다. 2K–4K에서 peak를 먼저 측정하십시오." });
    if (!config.gradientCheckpointing) warnings.push({ level: "warn", text: "Gradient checkpointing이 꺼져 있어 activation peak가 크게 증가합니다." });
    if (config.trainMethod.startsWith("full") && p > 7) warnings.push({ level: "warn", text: "7B 초과 Full FT는 단일 장치보다 FSDP2·ZeRO-3를 우선 검토하십시오." });
    if (config.trainMethod === "qlora") warnings.push({ level: "ok", text: "QLoRA 계산은 NF4 frozen base 약 0.62 B/P와 adapter 상태를 분리한 계획값입니다." });

    return {
      model,
      quantId,
      hardware,
      runtime: { id: "training", label: "학습 프레임워크" },
      weightInfo: { size: modelState, source: stateLabel, exact: false },
      components,
      deviceRequired,
      systemRequired,
      deviceRatio,
      systemRatio,
      fit,
      dynamicMemory: trainableState + activation + workspace,
      dynamicDetail: `activation ${formatMemory(activation)}`,
      maxContext: null,
      maxConcurrency: null,
      warnings,
      capacityType: "training",
      config,
      exactWeight: false
    };
  }

  function calculateInference(config, model, quantId, options = {}) {
    const hardware = deriveHardware(config);
    const components = [];
    const warnings = [];
    const weightInfo = estimateWeight(model, quantId, options.ignoreOverride ? 0 : config.weightOverride);
    const weight = weightInfo.size;
    const runtime = resolveRuntime(config, model);
    const replicasPerGroup = hardware.replicaDeviceFactor;
    const seqOnGroup = hardware.sequencesOnGroup;
    const totalReplicas = Math.max(1, config.replicas);
    const offloadFraction = hardware.isCpu ? 0 : hardware.isUnified ? 1 : clamp(config.gpuOffload / 100, 0, 1);

    function add(label, device = 0, system = 0, dynamic = false) {
      if (hardware.isUnified) {
        device += system;
        system = 0;
      } else if (hardware.isCpu) {
        system += device;
        device = 0;
      }
      components.push({ label, device: Math.max(0, device), system: Math.max(0, system), dynamic });
    }

    const weightDevice = weight * offloadFraction * replicasPerGroup;
    const weightSystem = weight * (1 - offloadFraction) * 1.08 * totalReplicas;
    add("모델 가중치·pipeline resident", weightDevice, weightSystem, false);

    let kvMemory = 0;
    let capacityUnitPerSequence = 0;
    let dynamicDeviceForCapacity = 0;

    if (model.kind === "image") {
      const basePeak = model.basePeakGiB?.[quantId] || (weight + Math.max(1.5, weight * 0.24));
      const baselineExtra = Math.max(0, basePeak - weight);
      let baselineDevice = baselineExtra * replicasPerGroup;
      let offloadedSystem = 0;
      if (config.imageCpuOffload && !hardware.isCpu && !hardware.isUnified) {
        const movable = Math.min(weightDevice + baselineDevice, basePeak * 0.25 * replicasPerGroup);
        baselineDevice = Math.max(0, baselineDevice - movable * 0.55);
        offloadedSystem = movable * 1.15;
      }
      add("텍스트 인코더·VAE·기본 activation", baselineDevice, offloadedSystem, true);

      const pixelScale = (config.imageWidth * config.imageHeight) / (1024 * 1024);
      const jobsOnGroup = Math.max(1, seqOnGroup);
      const activationBase = Math.max(1.0, basePeak * 0.18);
      const workUnits = pixelScale * config.imageBatch * jobsOnGroup;
      const resolutionExtra = Math.max(0, activationBase * (workUnits - 1));
      const controlMemory = config.controlNets * Math.max(0.6, basePeak * 0.07) * pixelScale * jobsOnGroup;
      const loraMemory = config.loraCount * Math.max(0.08, weight * 0.006) * replicasPerGroup;
      const referenceMemory = config.referenceImages * Math.max(0.25, basePeak * 0.025) * pixelScale * jobsOnGroup;
      const upscalerMemory = config.upscaler ? Math.max(1.5, 2.0 * pixelScale * jobsOnGroup) : 0;

      add("해상도·batch activation 증가", resolutionExtra, 0, true);
      if (controlMemory > 0) add("ControlNet·adapter", controlMemory, 0, true);
      if (loraMemory > 0) add("LoRA", loraMemory, 0, false);
      if (referenceMemory > 0) add("참조 이미지 activation", referenceMemory, 0, true);
      if (upscalerMemory > 0) add("업스케일러·복원", upscalerMemory, 0, true);
      add("호스트 UI·preview·파일 캐시", 0, Math.max(1.0, weight * 0.08) + config.imageBatch * 0.2, false);

      capacityUnitPerSequence = Math.max(0.3, activationBase * pixelScale * config.imageBatch + controlMemory / jobsOnGroup + referenceMemory / jobsOnGroup);
      dynamicDeviceForCapacity = resolutionExtra + controlMemory + referenceMemory + upscalerMemory;

      if (pixelScale > 4) warnings.push({ level: "warn", text: "2K 이상 생성은 attention·VAE peak가 모델별로 비선형 증가할 수 있습니다." });
      if (config.imageBatch > 1) warnings.push({ level: "warn", text: "이미지 batch는 activation을 거의 비례해 늘립니다. batch=1 기준 peak부터 확인하십시오." });
      if (config.controlNets > 1) warnings.push({ level: "warn", text: "여러 ControlNet·adapter는 동시에 올리지 않고 순차 적용하는 구성을 비교하십시오." });
      if (config.imageCpuOffload) warnings.push({ level: "ok", text: "CPU offload는 VRAM을 줄이지만 시스템 RAM과 생성 지연을 늘립니다." });
    } else if (model.kind === "audio") {
      const basePeak = model.basePeakGiB?.[quantId] || (weight + Math.max(0.8, weight * 0.22));
      const baselineExtra = Math.max(0, basePeak - weight) * replicasPerGroup;
      add("Audio encoder·codec·vocoder·기본 state", baselineExtra, 0, true);

      const streams = hardware.streamsOnGroup;
      const perAdditionalStream = 0.18 + model.paramsB * 0.025;
      const streamMemory = Math.max(0, streams - 1) * perAdditionalStream;
      const audioBufferRam = config.audioMinutes * streams * 0.004 + 0.25;
      add("추가 streaming session state", streamMemory, 0, true);
      add("Decode·resample·ring buffer", 0, audioBufferRam, true);

      if (config.diarization) add("VAD·화자 분리·alignment", 1.2, 1.0, true);
      if (config.audioCascade) add("ASR→LLM→TTS 보조 worker", 6.0, 6.0, false);
      add("호스트 runtime·tokenizer", 0, Math.max(0.6, weight * 0.06) * totalReplicas, false);

      capacityUnitPerSequence = Math.max(0.2, perAdditionalStream);
      dynamicDeviceForCapacity = streamMemory + (config.diarization ? 1.2 : 0);

      if (config.audioStreams > 1) warnings.push({ level: "warn", text: "실시간 음성은 메모리보다 first-partial latency와 real-time factor를 함께 부하 시험해야 합니다." });
      if (config.audioCascade) warnings.push({ level: "warn", text: "ASR·LLM·TTS 동시 상주 예산은 소형 worker 기준이며 실제 선택 모델에 따라 크게 달라집니다." });
    } else {
      const runtimeBase = Math.max(runtime.overheadMin, weight * runtime.overheadRatio) * replicasPerGroup;
      const contextScale = Math.sqrt(Math.max(0.125, config.contextTokens / 8192));
      const sequenceScale = Math.min(2.5, Math.sqrt(seqOnGroup));
      const workspace = Math.max(0.15, model.paramsB * 0.012) * contextScale * sequenceScale;
      const runtimeDevice = offloadFraction > 0 ? (runtimeBase + workspace) * Math.max(0.35, offloadFraction) : 0;
      const runtimeSystem = offloadFraction < 1 ? (runtimeBase + workspace) * Math.max(0.35, 1 - offloadFraction) : 0;
      add("런타임·그래프·prefill workspace", runtimeDevice, runtimeSystem, true);

      const hostRuntime = Math.max(0.45, weight * 0.06) * totalReplicas + config.concurrency * 0.015;
      add("호스트 tokenizer·queue·mmap metadata", 0, hostRuntime, false);

      const kvBaseMiB = Math.max(0, model.kvMiBPerTokenFp16 || 0);
      const kvFactor = catalog.kvDtypes[config.kvDtype]?.factor ?? 1;
      const prefix = clamp(config.prefixShare / 100, 0, 0.95);
      const cachedTokens = (config.contextTokens + config.outputTokens) * seqOnGroup
        - config.contextTokens * prefix * Math.max(0, seqOnGroup - 1);
      kvMemory = kvBaseMiB * kvFactor * cachedTokens / 1024;
      const kvDevice = offloadFraction > 0 ? kvMemory : 0;
      const kvSystem = offloadFraction === 0 ? kvMemory : 0;
      add("KV 캐시", kvDevice, kvSystem, true);

      if (config.operation === "serving") {
        const graphReserve = Math.max(0.45, weight * 0.035) * replicasPerGroup;
        add("서빙 graph·scheduler·allocator reserve", graphReserve, 0, false);
      }

      if (config.domain === "vision-ocr" || model.kind === "ocr") {
        const imageCount = config.domain === "vision-ocr" ? config.imageCount : 1;
        const megapixels = config.domain === "vision-ocr" ? config.megapixels : 2;
        const visualPerMp = 0.14 + Math.min(model.paramsB, 40) * 0.004;
        const visualMemory = imageCount * megapixels * visualPerMp * seqOnGroup;
        const rasterRam = imageCount * megapixels * 0.025 * seqOnGroup;
        add("Vision encoder·visual token activation", visualMemory, 0, true);
        add("이미지 decode·PDF raster buffer", 0, rasterRam + config.visionWorkerRam, true);
        dynamicDeviceForCapacity += visualMemory;
        if (imageCount > 4) warnings.push({ level: "warn", text: "다중 이미지·페이지는 visual token과 KV를 함께 늘립니다. 페이지 streaming을 우선하십시오." });
        if (megapixels > 4) warnings.push({ level: "warn", text: "고해상도 OCR은 150–200 DPI 기준선과 비교해 peak를 측정하십시오." });
      }

      capacityUnitPerSequence = kvBaseMiB * kvFactor * (config.contextTokens + config.outputTokens) / 1024;
      dynamicDeviceForCapacity += kvMemory + workspace;
    }

    const baseDeviceBeforeAddons = components.reduce((sum, item) => sum + item.device, 0);

    if (config.operation === "rag") {
      const embedding = catalog.addons.embeddings.find((item) => item.id === config.embeddingModel) || catalog.addons.embeddings[0];
      const reranker = catalog.addons.rerankers.find((item) => item.id === config.rerankerModel) || catalog.addons.rerankers[0];
      const parser = catalog.addons.parsers.find((item) => item.id === config.parserModel) || catalog.addons.parsers[0];
      const addonDeviceValues = [embedding.memoryGiB, reranker.memoryGiB, parser.deviceGiB].filter((value) => value > 0);

      if (config.sequentialAddons) {
        const addonPeak = Math.max(0, ...addonDeviceValues);
        const adjustment = Math.max(0, addonPeak - baseDeviceBeforeAddons);
        if (adjustment > 0) add("보조 모델 순차 peak 보정", adjustment, 0, false);
        warnings.push({ level: "ok", text: "임베딩·reranker·OCR worker는 생성 모델과 순차 실행하는 peak 방식으로 계산했습니다." });
      } else {
        add("임베딩 모델 상주", embedding.memoryGiB, 0, false);
        add("Reranker 상주", reranker.memoryGiB, 0, false);
        add("문서 parser·OCR device", parser.deviceGiB, 0, false);
      }

      const vectorPayload = config.ragChunks * config.embeddingDims * config.vectorBytes / 1e9;
      const vectorIndex = vectorPayload * config.indexOverhead;
      const metadata = config.ragChunks * 1024 / 1e9;
      add("벡터 인덱스·그래프·metadata", 0, vectorIndex + metadata, false);
      add("문서·청크 캐시", 0, config.documentCache + parser.systemGiB, false);

      if (config.ragChunks > 1_000_000) warnings.push({ level: "warn", text: "백만 청크 이상에서는 벡터 payload뿐 아니라 HNSW graph·metadata·복제본을 실측하십시오." });
    }

    if (config.domain === "data-analysis") {
      const workingSet = config.datasetSize * config.datasetExpansion * config.workingMultiplier;
      add("데이터 working set·조인·정렬", 0, workingSet, true);
      add("DuckDB·Polars·Python/R 커널", 0, config.notebookReserve, false);
      if (workingSet > hardware.systemAvailable * 0.5) warnings.push({ level: "warn", text: "데이터 working set이 가용 RAM의 절반을 넘습니다. Parquet·lazy scan·spill을 사용하십시오." });
    }

    const deviceRequired = components.reduce((sum, item) => sum + item.device, 0);
    const systemRequired = components.reduce((sum, item) => sum + item.system, 0);
    const deviceRatio = hardware.deviceAvailable > 0 ? deviceRequired / hardware.deviceAvailable : 0;
    const systemRatio = hardware.systemAvailable > 0 ? systemRequired / hardware.systemAvailable : Infinity;
    const fit = fitFromRatios(deviceRatio, systemRatio, hardware);

    let maxContext = null;
    let maxConcurrency = null;
    if (!["image", "audio"].includes(model.kind) && model.kvMiBPerTokenFp16 > 0) {
      const kvFactor = catalog.kvDtypes[config.kvDtype]?.factor ?? 1;
      const perToken = model.kvMiBPerTokenFp16 * kvFactor / 1024;
      const poolAvailable = hardware.isCpu ? hardware.systemAvailable : hardware.deviceAvailable;
      const kvPoolUsed = hardware.isCpu
        ? components.find((item) => item.label === "KV 캐시")?.system || 0
        : components.find((item) => item.label === "KV 캐시")?.device || 0;
      const poolRequired = hardware.isCpu ? systemRequired : deviceRequired;
      const fixed = Math.max(0, poolRequired - kvPoolUsed);
      const remaining = Math.max(0, poolAvailable - fixed);
      if (perToken > 0) {
        const maxPerSequenceTokens = remaining / (perToken * Math.max(1, hardware.sequencesOnGroup));
        maxContext = Math.max(0, Math.floor((maxPerSequenceTokens - config.outputTokens) / 256) * 256);
        maxContext = Math.min(1_048_576, maxContext);
        maxConcurrency = Math.max(0, Math.floor(remaining / (perToken * Math.max(1, config.contextTokens + config.outputTokens))));
        maxConcurrency = Math.min(9999, maxConcurrency);
      }
    } else if (model.kind === "image" || model.kind === "audio") {
      const poolAvailable = hardware.isCpu ? hardware.systemAvailable : hardware.deviceAvailable;
      const poolRequired = hardware.isCpu ? systemRequired : deviceRequired;
      const currentUnits = model.kind === "image" ? Math.max(1, hardware.sequencesOnGroup) : Math.max(1, hardware.streamsOnGroup);
      const dynamic = Math.max(0.01, dynamicDeviceForCapacity);
      const perUnit = Math.max(0.1, dynamic / currentUnits || capacityUnitPerSequence);
      const fixed = Math.max(0, poolRequired - dynamic);
      maxConcurrency = Math.max(0, Math.floor(Math.max(0, poolAvailable - fixed) / perUnit));
      maxConcurrency = Math.min(9999, maxConcurrency);
    }

    if (!weightInfo.exact) warnings.push({ level: "warn", text: "선택 형식의 파일 크기는 유효 비트 기반 추정입니다. Hugging Face 전체 shard 합계로 재정의하십시오." });
    if (["q2", "q3"].includes(quantId) && ["secure-code", "security-agent", "math-science", "lean-proof", "text-to-sql", "chart-math"].includes(config.task)) {
      warnings.push({ level: "warn", text: "정밀 코드·수학·SQL·OCR 작업에서 Q2/Q3는 누락과 형식 오류가 늘 수 있습니다. 한 단계 작은 Q4와 비교하십시오." });
    }
    if (offloadFraction < 1 && offloadFraction > 0) warnings.push({ level: "warn", text: "부분 GPU offload는 용량 우회이며 PCIe 전송으로 생성 속도가 크게 낮아질 수 있습니다." });
    if (hardware.isUnified && deviceRequired > hardware.deviceTotal * 0.75) warnings.push({ level: "warn", text: "통합 메모리의 75% 이상을 사용합니다. macOS·UI·브라우저 memory pressure와 swap을 확인하십시오." });
    if (config.contextTokens > 32768) warnings.push({ level: "warn", text: "32K 초과 context는 advertised maximum과 별개로 KV·prefill peak를 실제 prompt 분포에서 측정해야 합니다." });
    if (config.operation === "serving" && config.concurrency > 1) warnings.push({ level: "warn", text: "메모리상 sequence 상한은 TTFT·TPOT·goodput SLO를 보장하지 않습니다. open-loop 부하 시험이 필요합니다." });
    if (config.hardwareMode === "multi" && config.multiGpuMode === "shard") warnings.push({ level: "warn", text: "다중 GPU 합산값은 TP·PP backend, 최대 layer, P2P·PCIe·NVLink 통신을 고려하지 않은 용량 상한입니다." });
    if (config.hardwareMode === "multi" && config.multiGpuMode === "replica" && config.replicas < config.gpuCount) warnings.push({ level: "ok", text: "GPU replica 모드는 GPU별 독립 용량으로 계산했습니다. 총 VRAM을 한 모델에 합산하지 않습니다." });
    if (model.architecture === "moe") warnings.push({ level: "ok", text: "MoE 활성 파라미터가 작아도 계산은 전체 저장 파라미터·가중치 크기를 기준으로 합니다." });
    if (model.systemRamRecommendedGiB && config.systemRam < model.systemRamRecommendedGiB) warnings.push({ level: "warn", text: `가이드의 편안한 시스템 RAM 권장은 약 ${model.systemRamRecommendedGiB}GB입니다. CPU offload·캐시·UI 여유가 부족할 수 있습니다.` });
    if (model.minDeviceGiB?.[quantId] && hardware.deviceTotal > 0 && hardware.deviceTotal < model.minDeviceGiB[quantId]) warnings.push({ level: "warn", text: `가이드상 ${quantById[quantId]?.label || quantId} 시작점은 장치 메모리 약 ${model.minDeviceGiB[quantId]}GB입니다.` });
    for (const note of model.notes || []) warnings.push({ level: "ok", text: note });

    return {
      model,
      quantId,
      hardware,
      runtime,
      weightInfo,
      components,
      deviceRequired,
      systemRequired,
      deviceRatio,
      systemRatio,
      fit,
      dynamicMemory: components.filter((item) => item.dynamic).reduce((sum, item) => sum + item.device + item.system, 0),
      dynamicDetail: model.kind === "image"
        ? `${config.imageWidth}×${config.imageHeight}, batch ${config.imageBatch}`
        : model.kind === "audio"
          ? `${config.audioStreams} stream`
          : `KV ${formatMemory(kvMemory)}`,
      maxContext,
      maxConcurrency,
      warnings,
      capacityType: model.kind,
      config,
      exactWeight: weightInfo.exact
    };
  }

  function calculateEstimate(config, model, quantId, options = {}) {
    if (config.operation === "fine-tuning") return calculateFineTuning(config, model, quantId, options);
    return calculateInference(config, model, quantId, options);
  }

  function renderEstimate(estimate) {
    latestEstimate = estimate;
    const meta = fitMeta[estimate.fit.id] || fitMeta.neutral;
    const detailParts = [];
    if (!estimate.hardware.isCpu) detailParts.push(`${estimate.hardware.label} ${formatPercent(estimate.deviceRatio)} 사용`);
    if (!estimate.hardware.isUnified) detailParts.push(`RAM ${formatPercent(estimate.systemRatio)} 사용`);
    if (estimate.hardware.isUnified) detailParts.push(`공유 pool ${formatPercent(estimate.deviceRatio)} 사용`);

    $("fitBanner").dataset.fit = estimate.fit.id;
    $("fitLabel").textContent = meta.label;
    $("fitHeadline").textContent = meta.headline;
    $("fitDetail").textContent = detailParts.join(" · ");

    if (estimate.hardware.isCpu) {
      $("deviceRequired").textContent = "CPU 전용";
      $("deviceAvailable").textContent = "별도 VRAM 없음";
      $("deviceMeterRow").classList.add("is-hidden");
    } else {
      $("deviceRequired").textContent = formatMemory(estimate.deviceRequired);
      $("deviceAvailable").textContent = `가용 ${formatMemory(estimate.hardware.deviceAvailable)}`;
      $("deviceMeterRow").classList.remove("is-hidden");
    }

    if (estimate.hardware.isUnified) {
      $("systemRequired").textContent = "공유 pool";
      $("systemAvailable").textContent = "장치 필요량에 합산";
      $("systemMeterRow").classList.add("is-hidden");
    } else {
      $("systemRequired").textContent = formatMemory(estimate.systemRequired);
      $("systemAvailable").textContent = `가용 ${formatMemory(estimate.hardware.systemAvailable)}`;
      $("systemMeterRow").classList.remove("is-hidden");
    }

    $("weightSize").textContent = formatMemory(estimate.weightInfo.size);
    $("weightSource").textContent = estimate.weightInfo.source;
    $("dynamicMemory").textContent = formatMemory(estimate.dynamicMemory);
    $("dynamicDetail").textContent = estimate.dynamicDetail;

    updateMeter("device", estimate.deviceRatio, estimate.deviceRequired, estimate.hardware.deviceAvailable);
    updateMeter("system", estimate.systemRatio, estimate.systemRequired, estimate.hardware.systemAvailable);

    $("maxContext").textContent = estimate.maxContext == null
      ? "해당 없음"
      : estimate.maxContext > 0 ? `${formatTokens(estimate.maxContext)} token` : "여유 없음";
    $("maxConcurrency").textContent = estimate.maxConcurrency == null
      ? "해당 없음"
      : estimate.maxConcurrency > 0
        ? `${estimate.maxConcurrency.toLocaleString("ko-KR")} ${estimate.capacityType === "image" ? "작업" : estimate.capacityType === "audio" ? "stream" : "sequence"}`
        : "여유 없음";

    renderBreakdown(estimate.components);
    renderWarnings(estimate.warnings, estimate.fit.id);
    renderCommand(estimate);
    renderSelectedLinks(estimate);
    updateRuntimeHint(estimate);
  }

  function updateMeter(prefix, ratio, required, available) {
    const bar = $(`${prefix}Meter`);
    const text = $(`${prefix}MeterText`);
    const meter = bar.parentElement;
    const normalized = Number.isFinite(ratio) ? Math.max(0, ratio) : 1.5;
    const width = clamp(normalized * 100, 0, 100);
    bar.style.width = `${width}%`;
    bar.dataset.level = normalized > 1 ? "over" : normalized > 0.86 ? "tight" : "ok";
    meter.setAttribute("aria-valuenow", String(Math.round(normalized * 100)));
    text.textContent = `${formatMemory(required)} / ${formatMemory(available)}`;
  }

  function renderBreakdown(components) {
    const body = $("breakdownBody");
    body.replaceChildren();
    for (const item of components) {
      if (item.device < 0.005 && item.system < 0.005) continue;
      const row = document.createElement("tr");
      const label = document.createElement("td");
      const device = document.createElement("td");
      const system = document.createElement("td");
      label.textContent = item.label;
      if (item.dynamic) {
        const marker = document.createElement("span");
        marker.textContent = " · peak";
        marker.style.color = "var(--subtle)";
        marker.style.fontSize = "0.68rem";
        label.append(marker);
      }
      device.textContent = item.device > 0 ? formatMemory(item.device) : "—";
      system.textContent = item.system > 0 ? formatMemory(item.system) : "—";
      device.className = "numeric";
      system.className = "numeric";
      row.append(label, device, system);
      body.append(row);
    }
  }

  function renderWarnings(warnings, fitId) {
    const list = $("warningList");
    list.replaceChildren();
    if (fitId === "excellent" || fitId === "good") {
      warnings = [{ level: "ok", text: "예약 메모리를 제외한 추정치가 현재 용량 안에 들어옵니다." }, ...warnings];
    } else if (fitId === "insufficient") {
      warnings = [{ level: "danger", text: "가중치만 적재되는지 보지 말고 Q4→Q3, context, 동시성, batch, 보조 모델 상주 여부 순으로 줄이십시오." }, ...warnings];
    }
    for (const warning of warnings.slice(0, 12)) {
      const item = document.createElement("li");
      item.className = warning.level || "warn";
      item.textContent = warning.text;
      list.append(item);
    }
  }

  function renderCommand(estimate) {
    const c = estimate.config;
    const h = estimate.hardware;
    const lines = [
      "[profile]",
      `hardware.mode=${c.hardwareMode}`,
      `hardware.vendor=${c.vendor}`,
      `hardware.system_ram_gb=${c.systemRam}`,
      `hardware.device_capacity_gb=${h.deviceTotal}`,
      `hardware.device_reserve_pct=${c.deviceReserve}`,
      `hardware.system_reserve_pct=${c.systemReserve}`,
      "",
      "[workload]",
      `domain=${c.domain}`,
      `task=${c.task}`,
      `operation=${c.operation}`,
      `runtime=${estimate.runtime.label}`,
      `model=${estimate.model.name}`,
      `quant=${estimate.quantId}`,
      `context_tokens=${c.contextTokens}`,
      `output_tokens=${c.outputTokens}`,
      `concurrency=${c.concurrency}`,
      `replicas=${c.replicas}`,
      "",
      "[estimate]",
      `fit=${estimate.fit.id}`,
      `device_required_gb=${formatNumber(estimate.deviceRequired)}`,
      `device_available_gb=${formatNumber(h.deviceAvailable)}`,
      `system_required_gb=${formatNumber(estimate.systemRequired)}`,
      `system_available_gb=${formatNumber(h.systemAvailable)}`,
      `weight_or_pipeline_gb=${formatNumber(estimate.weightInfo.size)}`,
      `weight_source=${estimate.weightInfo.source}`,
      `catalog_verified=${catalog.verified}`
    ];
    $("commandOutput").textContent = lines.join("\n");
  }

  function renderSelectedLinks(estimate) {
    const wrap = $("selectedLinks");
    wrap.replaceChildren();
    const links = [];
    if (estimate.model.hf) links.push({ label: "Hugging Face", href: estimate.model.hf });
    const modelGuide = guideById[estimate.model.guide];
    if (modelGuide) links.push({ label: modelGuide.label, href: repositoryFileUrl(modelGuide.path) });
    links.push({ label: "양자화 가이드", href: repositoryFileUrl(guideById.quantization.path) });
    if (estimate.config.operation === "fine-tuning") links.push({ label: "파인튜닝 가이드", href: repositoryFileUrl(guideById["fine-tuning-memory"].path) });
    if (estimate.config.operation === "serving") links.push({ label: "서빙 가이드", href: repositoryFileUrl(guideById["serving-concurrency"].path) });
    links.push({ label: "런타임·하드웨어", href: repositoryFileUrl(guideById["runtime-hardware"].path) });

    for (const link of links) {
      const anchor = document.createElement("a");
      anchor.href = link.href;
      anchor.textContent = link.label;
      if (/^https:\/\//.test(link.href)) {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
      }
      wrap.append(anchor);
    }
  }

  function updateRuntimeHint(estimate) {
    const c = estimate.config;
    const recommended = recommendedRuntime(c, estimate.model);
    const recommendedProfile = runtimeById.get(recommended);
    if (c.operation === "fine-tuning") {
      $("runtimeHint").textContent = "학습은 Transformers·TRL·PEFT, Axolotl, LLaMA-Factory, MLX-LM 등의 실제 설정을 별도 검증하십시오.";
    } else if (c.runtime === "auto") {
      $("runtimeHint").textContent = `자동 선택: ${recommendedProfile?.label || recommended}. 형식·하드웨어 지원표를 확인하십시오.`;
    } else if (c.runtime !== recommended) {
      $("runtimeHint").textContent = `현재 선택: ${runtimeById.get(c.runtime)?.label}. 이 구성의 일반적인 시작점은 ${recommendedProfile?.label || recommended}입니다.`;
    } else {
      $("runtimeHint").textContent = `${recommendedProfile?.label || recommended}가 현재 하드웨어·작업의 일반적인 시작점입니다.`;
    }
  }

  function calculateAndRender() {
    const config = collectConfig();
    const model = getSelectedModel();
    const quant = $("quant").value;
    const estimate = calculateEstimate(config, model, quant);
    renderEstimate(estimate);
    renderRecommendations(config);
    saveState();
  }

  function scheduleCalculation() {
    window.clearTimeout(calculationTimer);
    calculationTimer = window.setTimeout(calculateAndRender, 40);
  }

  function candidateQuantOrder(model) {
    return [...model.formats].sort((a, b) => {
      const qa = quantById[a]?.quality ?? 0;
      const qb = quantById[b]?.quality ?? 0;
      return qb - qa;
    });
  }

  function recommendationFitRank(fitId) {
    return { excellent: 0, good: 1, tight: 2, offload: 3, insufficient: 4 }[fitId] ?? 5;
  }

  function renderRecommendations(config) {
    const models = filteredModels();
    const candidates = [];

    for (const model of models) {
      const results = candidateQuantOrder(model).map((quantId) => ({
        quantId,
        estimate: calculateEstimate({ ...config, weightOverride: 0 }, model, quantId, { ignoreOverride: true })
      }));
      const chosen = results.find(({ estimate }) => estimate.fit.ratio <= 0.86)
        || results.find(({ estimate }) => estimate.fit.ratio <= 1.0)
        || results[results.length - 1];
      if (!chosen) continue;
      const utilizationScore = Math.abs(0.68 - Math.min(1.5, chosen.estimate.fit.ratio));
      candidates.push({ model, ...chosen, utilizationScore });
    }

    candidates.sort((a, b) => {
      const fitDiff = recommendationFitRank(a.estimate.fit.id) - recommendationFitRank(b.estimate.fit.id);
      if (fitDiff !== 0) return fitDiff;
      const qualityDiff = b.model.quality - a.model.quality;
      if (qualityDiff !== 0) return qualityDiff;
      return a.utilizationScore - b.utilizationScore;
    });

    const body = $("recommendationBody");
    body.replaceChildren();
    for (const candidate of candidates.slice(0, 10)) {
      const row = document.createElement("tr");
      const fitCell = document.createElement("td");
      const modelCell = document.createElement("td");
      const quantCell = document.createElement("td");
      const deviceCell = document.createElement("td");
      const systemCell = document.createElement("td");
      const reasonCell = document.createElement("td");
      const actionCell = document.createElement("td");

      const chip = document.createElement("span");
      chip.className = `fit-chip ${candidate.estimate.fit.id}`;
      chip.textContent = fitMeta[candidate.estimate.fit.id]?.label || candidate.estimate.fit.id;
      fitCell.append(chip);

      const name = document.createElement("strong");
      name.textContent = candidate.model.name;
      modelCell.append(name);
      const sub = document.createElement("div");
      sub.style.color = "var(--subtle)";
      sub.style.fontSize = "0.7rem";
      sub.textContent = candidate.model.architecture === "moe" && candidate.model.activeParamsB
        ? `${candidate.model.paramsB}B total / ${candidate.model.activeParamsB}B active`
        : `${candidate.model.paramsB}B`;
      modelCell.append(sub);

      quantCell.textContent = quantById[candidate.quantId]?.label || candidate.quantId;
      deviceCell.textContent = formatMemory(candidate.estimate.deviceRequired);
      systemCell.textContent = formatMemory(candidate.estimate.systemRequired);
      reasonCell.textContent = candidate.model.notes?.[0] || `${kindLabels[candidate.model.kind] || candidate.model.kind} · 품질 점수 ${candidate.model.quality}`;

      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "select-model-button";
      selectButton.dataset.modelId = candidate.model.id;
      selectButton.dataset.quantId = candidate.quantId;
      selectButton.textContent = "적용";
      actionCell.append(selectButton);

      row.append(fitCell, modelCell, quantCell, deviceCell, systemCell, reasonCell, actionCell);
      body.append(row);
    }

    if (!candidates.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 7;
      cell.textContent = "현재 분야·작업에 등록된 추천 모델이 없습니다. 사용자 정의 모델을 사용하십시오.";
      row.append(cell);
      body.append(row);
    }
  }

  function renderGuides() {
    const groupLabels = { domains: "domain", modalities: "modality", operations: "operation" };
    const grid = $("guideGrid");
    grid.replaceChildren();
    for (const [id, guide] of Object.entries(guideById)) {
      const anchor = document.createElement("a");
      const href = repositoryFileUrl(guide.path);
      anchor.className = "guide-card";
      anchor.href = href;
      markExternalLink(anchor, href);
      anchor.dataset.guideId = id;
      anchor.setAttribute("aria-label", `${guide.label} 가이드를 GitHub에서 열기`);
      const group = document.createElement("span");
      group.textContent = groupLabels[guide.group] || guide.group;
      const title = document.createElement("strong");
      title.textContent = guide.label;
      anchor.append(group, title);
      grid.append(anchor);
    }
  }

  function serializeForm() {
    const state = {};
    form.querySelectorAll("[name]").forEach((element) => {
      state[element.name] = element.type === "checkbox" ? element.checked : element.value;
    });
    return state;
  }

  function saveState() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(serializeForm()));
    } catch (_) {
      // Storage can be disabled; calculator remains fully functional.
    }
  }

  function loadSavedState() {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function applyState(state) {
    if (!state || typeof state !== "object") return;

    if (state.domain && catalog.domains.some((domain) => domain.id === state.domain)) $("domain").value = state.domain;
    populateTasks(state.task);
    populateModels(state.model);
    populateQuants(state.quant);

    form.querySelectorAll("[name]").forEach((element) => {
      if (!(element.name in state)) return;
      if (element.type === "checkbox") {
        element.checked = Boolean(state[element.name]);
      } else if ([...element.options || []].length && ![...element.options].some((option) => option.value === String(state[element.name]))) {
        return;
      } else {
        element.value = String(state[element.name]);
      }
    });
  }

  function resetCalculator() {
    try { localStorage.removeItem(storageKey); } catch (_) { /* no-op */ }
    form.reset();
    initializeBaseSelects();
    populateTasks("general-coding");
    populateModels("qwen35-9b");
    populateQuants("q4");
    $("systemRam").value = "32";
    $("vramPerGpu").value = "16";
    $("unifiedMemory").value = "32";
    $("deviceReserve").value = "12";
    $("systemReserve").value = "20";
    $("ramSource").textContent = "수동 입력값";
    $("vramSource").textContent = "수동 입력값";
    $("detectedLine").innerHTML = '<span class="status-dot status-neutral" aria-hidden="true"></span><span>자동 탐지를 실행하지 않았습니다.</span>';
    latestDetection = null;
    updateConditionalPanels();
    updateModelMeta();
    calculateAndRender();
  }

  async function copyResults() {
    const text = $("commandOutput").textContent;
    try {
      await navigator.clipboard.writeText(text);
      $("copyButton").textContent = "복사됨";
    } catch (_) {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
      $("copyButton").textContent = "복사됨";
    }
    window.setTimeout(() => { $("copyButton").textContent = "결과 복사"; }, 1200);
  }

  function exportJson() {
    if (!latestEstimate) return;
    const payload = {
      generatedAt: new Date().toISOString(),
      catalogVerified: catalog.verified,
      configuration: serializeForm(),
      result: {
        fit: latestEstimate.fit.id,
        deviceRequiredGiB: latestEstimate.deviceRequired,
        deviceAvailableGiB: latestEstimate.hardware.deviceAvailable,
        systemRequiredGiB: latestEstimate.systemRequired,
        systemAvailableGiB: latestEstimate.hardware.systemAvailable,
        weightOrPipelineGiB: latestEstimate.weightInfo.size,
        weightSource: latestEstimate.weightInfo.source,
        maxContextTokens: latestEstimate.maxContext,
        maxConcurrency: latestEstimate.maxConcurrency,
        breakdown: latestEstimate.components,
        warnings: latestEstimate.warnings.map((warning) => warning.text)
      }
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `ram-for-local-ai-estimate-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function getWebGLInfo() {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2", { powerPreference: "high-performance" })
        || canvas.getContext("webgl", { powerPreference: "high-performance" });
      if (!gl) return null;
      const debug = gl.getExtension("WEBGL_debug_renderer_info");
      return {
        vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
        renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE)
      };
    } catch (_) {
      return null;
    }
  }

  async function getWebGPUInfo() {
    if (!navigator.gpu?.requestAdapter) return null;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) return null;
      let info = adapter.info || null;
      if (!info && typeof adapter.requestAdapterInfo === "function") {
        info = await adapter.requestAdapterInfo();
      }

      // memoryHeaps is a Chromium developer feature, not a standard WebGPU API.
      // Read it only when the browser exposes both the heap list and the
      // DEVICE_LOCAL flag. Never infer total VRAM from maxBufferSize limits.
      const heapFlag = globalThis.GPUHeapProperty?.DEVICE_LOCAL;
      const heaps = Array.from(info?.memoryHeaps || []).map((heap) => ({
        size: Number(heap?.size || 0),
        properties: Number(heap?.properties || 0)
      })).filter((heap) => Number.isFinite(heap.size) && heap.size > 0);
      const deviceLocalBytes = heapFlag
        ? heaps.filter((heap) => (heap.properties & heapFlag) !== 0).reduce((sum, heap) => sum + heap.size, 0)
        : 0;
      const deviceLocalGiB = deviceLocalBytes > 0
        ? Math.round((deviceLocalBytes / (1024 ** 3)) * 2) / 2
        : null;

      return {
        vendor: info?.vendor || "",
        architecture: info?.architecture || "",
        device: info?.device || "",
        description: info?.description || "",
        driver: info?.driver || "",
        backend: info?.backend || "",
        type: info?.type || "",
        isFallbackAdapter: Boolean(adapter.isFallbackAdapter),
        maxBufferSize: Number(adapter.limits?.maxBufferSize || 0),
        maxStorageBufferBindingSize: Number(adapter.limits?.maxStorageBufferBindingSize || 0),
        memoryHeaps: heaps,
        deviceLocalGiB
      };
    } catch (_) {
      return null;
    }
  }

  function inferPlatform() {
    const uaPlatform = navigator.userAgentData?.platform || navigator.platform || "";
    const ua = navigator.userAgent || "";
    if (/Mac|macOS|iPhone|iPad/i.test(`${uaPlatform} ${ua}`)) return "macOS";
    if (/Win/i.test(`${uaPlatform} ${ua}`)) return "Windows";
    if (/Linux|X11/i.test(`${uaPlatform} ${ua}`)) return "Linux";
    if (/Android/i.test(ua)) return "Android";
    return uaPlatform || "미확인";
  }

  function inferVendor(text) {
    if (/NVIDIA|GeForce|RTX|Quadro|Tesla/i.test(text)) return "nvidia";
    if (/AMD|Radeon|ATI/i.test(text)) return "amd";
    if (/Apple/i.test(text)) return "apple";
    if (/Intel|Arc/i.test(text)) return "intel";
    return "other";
  }

  function matchGpuMemoryHint(text) {
    const normalized = text || "";
    return catalog.gpuMemoryHints.find((hint) => new RegExp(hint.pattern, "i").test(normalized)) || null;
  }

  function exactCommandsForPlatform(platform) {
    if (platform === "Windows") {
      return `Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
nvidia-smi --query-gpu=name,memory.total --format=csv`;
    }
    if (platform === "macOS") {
      return `system_profiler SPHardwareDataType | grep -E "Chip|Memory"
system_profiler SPDisplaysDataType | grep -E "Chipset Model|VRAM|Metal"
vm_stat`;
    }
    return `free -h
nvidia-smi --query-gpu=name,memory.total --format=csv
rocm-smi --showproductname --showmeminfo vram`;
  }

  async function detectDevice() {
    $("detectButton").disabled = true;
    $("detectButton").textContent = "탐지 중";
    const [webgpu, webgl] = await Promise.all([getWebGPUInfo(), Promise.resolve(getWebGLInfo())]);
    const platform = inferPlatform();
    const ramGiB = Number(navigator.deviceMemory || 0) || null;
    const cores = Number(navigator.hardwareConcurrency || 0) || null;
    const gpuText = [webgpu?.description, webgpu?.device, webgpu?.vendor, webgl?.renderer, webgl?.vendor].filter(Boolean).join(" · ");
    const hint = matchGpuMemoryHint(gpuText);
    const vendor = inferVendor(gpuText);
    const fallback = Boolean(webgpu?.isFallbackAdapter || /SwiftShader|llvmpipe|software/i.test(gpuText));
    const heapVramGiB = !fallback && vendor !== "apple" && Number.isFinite(webgpu?.deviceLocalGiB)
      ? webgpu.deviceLocalGiB
      : null;
    const vramCandidates = [...new Set([
      ...(heapVramGiB ? [heapVramGiB] : []),
      ...(hint?.values || [])
    ])].filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);

    latestDetection = {
      platform,
      ramGiB,
      cores,
      webgpu,
      webgl,
      gpuText: gpuText || "GPU 정보가 공개되지 않음",
      vendor,
      hint,
      heapVramGiB,
      vramCandidates,
      fallback,
      selectedVram: heapVramGiB || (vramCandidates.length === 1 ? vramCandidates[0] : null)
    };

    renderDeviceReport(latestDetection);
    const dialog = $("deviceDialog");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");

    $("detectButton").disabled = false;
    $("detectButton").textContent = "장치 자동 탐지";
  }

  function appendReportRow(dl, term, value) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    dl.append(dt, dd);
  }

  function renderDeviceReport(report) {
    const wrap = $("deviceReport");
    wrap.replaceChildren();
    const dl = document.createElement("dl");
    dl.className = "report-grid";
    appendReportRow(dl, "플랫폼", report.platform);
    appendReportRow(dl, "CPU logical cores", report.cores ? String(report.cores) : "공개되지 않음");
    appendReportRow(dl, "브라우저 RAM", report.ramGiB ? `${report.ramGiB} GiB · 근사·상한 가능` : "지원되지 않음");
    appendReportRow(dl, "GPU 식별 문자열", report.gpuText);
    appendReportRow(dl, "GPU vendor 추정", report.vendor);
    appendReportRow(dl, "WebGPU", report.webgpu
      ? [report.webgpu.isFallbackAdapter ? "fallback adapter" : "사용 가능", report.webgpu.type, report.webgpu.backend].filter(Boolean).join(" · ")
      : "사용 불가·차단");
    appendReportRow(dl, "WebGL", report.webgl ? `사용 가능 · max texture ${report.webgl.maxTextureSize}` : "사용 불가·차단");
    appendReportRow(dl, "WebGPU device-local heap", report.heapVramGiB
      ? `${report.heapVramGiB} GiB · 비표준 개발 기능 보고값`
      : "공개되지 않음");
    appendReportRow(dl, "GPU 이름 기반 사양", report.hint
      ? `${report.hint.label}: ${report.hint.values.join(" / ")} GB`
      : "매칭되지 않음");
    appendReportRow(dl, "적용 가능한 VRAM 후보", report.vramCandidates.length
      ? `${report.vramCandidates.join(" / ")} GB`
      : "정확한 총량을 표준 웹 API로 확인할 수 없음");
    wrap.append(dl);

    const note = document.createElement("p");
    note.className = "report-note";
    if (report.fallback) {
      note.textContent = "브라우저가 소프트웨어 렌더러를 사용하고 있어 실제 GPU를 식별하지 못했습니다. 아래 운영체제 명령으로 확인하십시오.";
    } else if (report.heapVramGiB) {
      note.textContent = "WebGPU memoryHeaps는 Chromium의 비표준 개발 기능이며 일반 브라우저에서는 보통 제공되지 않습니다. 장치 관리자·시스템 명령의 값과 일치하는지 확인하십시오.";
    } else {
      note.textContent = "브라우저 RAM은 반올림·상한 적용값일 수 있고 WebGPU·WebGL의 buffer limit은 총 VRAM이 아닙니다. 제품 변형이 여러 개면 후보를 선택한 뒤 운영체제 값으로 확인하십시오.";
    }
    wrap.append(note);

    const command = document.createElement("pre");
    command.className = "report-command";
    command.textContent = exactCommandsForPlatform(report.platform);
    wrap.append(command);

    const candidateWrap = $("vramCandidateWrap");
    const candidate = $("vramCandidate");
    candidate.replaceChildren();
    if (report.vramCandidates.length > 1) {
      candidateWrap.classList.remove("is-hidden");
      for (const value of report.vramCandidates) {
        const option = document.createElement("option");
        option.value = String(value);
        const sources = [];
        if (value === report.heapVramGiB) sources.push("WebGPU heap");
        if (report.hint?.values?.includes(value)) sources.push("제품 사양");
        option.textContent = `${value} GB${sources.length ? ` · ${sources.join("+")}` : ""}`;
        candidate.append(option);
      }
      if (report.selectedVram && report.vramCandidates.includes(report.selectedVram)) {
        candidate.value = String(report.selectedVram);
      }
      report.selectedVram = Number(candidate.value);
    } else {
      candidateWrap.classList.add("is-hidden");
      report.selectedVram = report.vramCandidates[0] || null;
    }
  }

  function applyDetection() {
    const report = latestDetection;
    if (!report) return;
    if (report.ramGiB) {
      $("systemRam").value = String(report.ramGiB);
      $("ramSource").textContent = `브라우저 근사값 ${report.ramGiB} GiB · 수동 확인 필요`;
    }

    if (report.vendor !== "other") $("hardwareVendor").value = report.vendor;

    if (report.vendor === "apple") {
      $("hardwareMode").value = "unified";
      if (report.ramGiB) $("unifiedMemory").value = String(report.ramGiB);
    } else if (report.vramCandidates.length && !report.fallback) {
      $("hardwareMode").value = "dedicated";
      const chosen = report.vramCandidates.length > 1
        ? Number($("vramCandidate").value)
        : report.vramCandidates[0];
      if (chosen) {
        $("vramPerGpu").value = String(chosen);
        const source = chosen === report.heapVramGiB
          ? "WebGPU memoryHeaps 비표준 보고값"
          : `${report.hint?.label || "GPU 이름"} 제품 사양 기반`;
        $("vramSource").textContent = `${source} ${chosen}GB · 수동 확인 필요`;
      }
    }

    const line = $("detectedLine");
    line.replaceChildren();
    const dot = document.createElement("span");
    dot.className = `status-dot ${report.fallback ? "status-warn" : "status-ok"}`;
    dot.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = `${report.platform} · ${report.gpuText}${report.ramGiB ? ` · RAM 보고값 ${report.ramGiB} GiB` : ""}`;
    line.append(dot, text);

    updateConditionalPanels();
    calculateAndRender();
    $("deviceDialog").close?.();
  }

  function handleDomainChange() {
    const previousTask = $("task").value;
    populateTasks(previousTask);
    if ($("domain").value === "productivity-rag" && $("operation").value === "inference") {
      $("operation").value = "rag";
    } else if ($("domain").value !== "productivity-rag" && $("operation").value === "rag") {
      $("operation").value = "inference";
    }
    populateModels();
    populateQuants();
    updateConditionalPanels();
    updateModelMeta();
    scheduleCalculation();
  }

  function handleTaskChange() {
    const previous = $("model").value;
    populateModels(previous);
    populateQuants();
    updateConditionalPanels();
    updateModelMeta();
    scheduleCalculation();
  }

  function handleModelChange() {
    populateQuants();
    updateModelMeta();
    updateConditionalPanels();
    scheduleCalculation();
  }

  function bindEvents() {
    $("domain").addEventListener("change", handleDomainChange);
    $("task").addEventListener("change", handleTaskChange);
    $("model").addEventListener("change", handleModelChange);
    $("quant").addEventListener("change", () => { updateModelMeta(); scheduleCalculation(); });
    $("hardwareMode").addEventListener("change", () => { updateConditionalPanels(); scheduleCalculation(); });
    $("operation").addEventListener("change", () => { updateConditionalPanels(); scheduleCalculation(); });

    form.addEventListener("input", (event) => {
      if (["domain", "task", "model", "quant", "hardwareMode", "operation"].includes(event.target.id)) return;
      if (["customParams", "customArchitecture", "customKv"].includes(event.target.id)) updateModelMeta();
      scheduleCalculation();
    });
    form.addEventListener("change", (event) => {
      if (["domain", "task", "model", "quant", "hardwareMode", "operation"].includes(event.target.id)) return;
      scheduleCalculation();
    });

    $("detectButton").addEventListener("click", detectDevice);
    $("applyDetectionButton").addEventListener("click", applyDetection);
    $("vramCandidate").addEventListener("change", () => {
      if (latestDetection) latestDetection.selectedVram = Number($("vramCandidate").value);
    });
    $("copyButton").addEventListener("click", copyResults);
    $("exportButton").addEventListener("click", exportJson);
    $("resetButton").addEventListener("click", resetCalculator);

    $("recommendationBody").addEventListener("click", (event) => {
      const button = event.target.closest(".select-model-button");
      if (!button) return;
      const modelId = button.dataset.modelId;
      const quantId = button.dataset.quantId;
      if (![...$("model").options].some((option) => option.value === modelId)) return;
      $("model").value = modelId;
      populateQuants(quantId);
      updateModelMeta();
      calculateAndRender();
      $("model-title").scrollIntoView({ block: "center" });
    });
  }

  function initialize() {
    initializeBaseSelects();
    populateTasks("general-coding");
    populateModels("qwen35-9b");
    populateQuants("q4");
    const saved = loadSavedState();
    if (saved) applyState(saved);
    updateConditionalPanels();
    updateModelMeta();
    applyRepositoryLinks();
    renderGuides();
    bindEvents();
    calculateAndRender();
  }

  initialize();
})();
