import exec from "k6/execution";
import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const VUS = Number(__ENV.VUS || 60);
const DURATION = __ENV.DURATION || "2m";
const BACKPRESSURE_VUS = Number(__ENV.BACKPRESSURE_VUS || 200);
const BACKPRESSURE_DURATION = __ENV.BACKPRESSURE_DURATION || "30s";
const RUN_BACKPRESSURE = (__ENV.RUN_BACKPRESSURE || "false").toLowerCase() === "true";

const MODELS = [
  "gateway-echo",
  "gateway-echo-fast",
  "gateway-echo-balanced",
  "gateway-echo-large",
];

export const queueFull503 = new Counter("queue_full_503");
export const accepted2xx = new Rate("accepted_2xx");

const scenarios = {
  steady_traffic: {
    executor: "constant-vus",
    vus: VUS,
    duration: DURATION,
    exec: "steadyTraffic",
    tags: {
      scenario_type: "steady",
      batch_size: __ENV.BATCH_SIZE || __ENV.BATCH_MAX_SIZE || "server-default",
      max_wait_ms: __ENV.MAX_WAIT_MS || __ENV.BATCH_MAX_WAIT_MS || "server-default",
    },
  },
};

if (RUN_BACKPRESSURE) {
  scenarios.backpressure_probe = {
    executor: "constant-vus",
    vus: BACKPRESSURE_VUS,
    duration: BACKPRESSURE_DURATION,
    exec: "backpressureProbe",
    startTime: DURATION,
    tags: { scenario_type: "backpressure" },
  };
}

export const options = {
  scenarios,
  thresholds: {
    "http_req_duration{scenario_type:steady}": [
      `p(95)<${Number(__ENV.P95_TARGET_MS || 250)}`,
      `p(99)<${Number(__ENV.P99_TARGET_MS || 500)}`,
    ],
    "http_req_failed{scenario_type:steady}": ["rate<0.01"],
    "accepted_2xx{scenario_type:steady}": ["rate>0.99"],
    ...(RUN_BACKPRESSURE
      ? {
          "queue_full_503{scenario_type:backpressure}": ["count>0"],
        }
      : {}),
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

export function steadyTraffic() {
  const response = postCompletion("steady");
  accepted2xx.add(response.status >= 200 && response.status < 300, {
    scenario_type: "steady",
  });
  check(response, {
    "steady status is 200": (r) => r.status === 200,
    "steady response has choices": (r) => {
      try {
        return Array.isArray(r.json("choices"));
      } catch {
        return false;
      }
    },
  });
  sleep(0.05);
}

export function backpressureProbe() {
  const response = postCompletion("backpressure");
  if (response.status === 503) {
    queueFull503.add(1, { scenario_type: "backpressure" });
  }
  check(response, {
    "backpressure returns 200 or 503": (r) => r.status === 200 || r.status === 503,
    "queue full response is explicit": (r) => {
      if (r.status !== 503) {
        return true;
      }
      try {
        return r.json("detail") === "Request queue is full";
      } catch {
        return false;
      }
    },
  });
}

function postCompletion(scenarioType) {
  const iteration = exec.scenario.iterationInTest;
  const model = MODELS[iteration % MODELS.length];
  const payload = JSON.stringify({
    model,
    messages: [
      {
        role: "system",
        content: "Return a concise answer.",
      },
      {
        role: "user",
        content: `${scenarioType} request ${iteration} from vu ${exec.vu.idInTest}`,
      },
    ],
    temperature: 0,
    max_tokens: 32,
    stream: false,
  });

  return http.post(`${BASE_URL}/v1/chat/completions`, payload, {
    headers: {
      "content-type": "application/json",
      "x-trace-id": `k6-${scenarioType}-${exec.vu.idInTest}-${iteration}`,
      ...(apiKeyHeader()),
    },
    tags: {
      name: "chat_completions",
      scenario_type: scenarioType,
      model,
    },
    timeout: __ENV.REQUEST_TIMEOUT || "10s",
  });
}

function apiKeyHeader() {
  if (!__ENV.API_KEY) {
    return {};
  }

  return { "x-api-key": __ENV.API_KEY };
}

export function handleSummary(data) {
  const duration = data.metrics.http_req_duration?.values || {};
  const failed = data.metrics.http_req_failed?.values || {};
  const queueFull = data.metrics.queue_full_503?.values || {};
  const lines = [
    "# k6 batching report",
    "",
    `- base_url: ${BASE_URL}`,
    `- vus: ${VUS}`,
    `- duration: ${DURATION}`,
    `- batch_size: ${__ENV.BATCH_SIZE || __ENV.BATCH_MAX_SIZE || "server-default"}`,
    `- max_wait_ms: ${__ENV.MAX_WAIT_MS || __ENV.BATCH_MAX_WAIT_MS || "server-default"}`,
    `- p95_ms: ${formatMetric(duration["p(95)"])}`,
    `- p99_ms: ${formatMetric(duration["p(99)"])}`,
    `- failed_rate: ${formatMetric(failed.rate)}`,
    `- queue_full_503_count: ${formatMetric(queueFull.count || 0)}`,
    "",
    "Use the JSON report for exact metric dimensions and thresholds.",
    "",
  ].join("\n");

  return {
    stdout: lines,
    "reports/k6-batching-summary.json": JSON.stringify(data, null, 2),
    "reports/k6-batching-summary.md": lines,
  };
}

function formatMetric(value) {
  if (typeof value !== "number") {
    return "n/a";
  }

  return value.toFixed(3);
}
