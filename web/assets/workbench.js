"use strict";

const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 30;
const state = { documentOffset: 0, documentRequest: 0, chunkDocument: null, chunkOffset: 0 };
const labelFields = ["ground_truth_document_ids", "ground_truth_chunk_ids"];
const kinds = { text: "文本", section: "主题", parent: "父块", child: "子块", table: "表格" };
const metricNames = {
  context_recall_knn: "向量候选召回率",
  context_recall_bm25: "BM25 候选召回率",
  context_recall_rrf: "RRF 候选召回率（截尾后）",
  context_recall_topk: "重排 Top-K 召回率",
  context_precision: "上下文精确率",
  context_relevancy: "上下文相关性",
  faithfulness: "忠实性",
  answer_relevancy: "答案相关性",
  relevancy: "相关性", answer_correctness: "答案正确性", context_recall: "上下文召回率",
};
const metricGroups = [
  ["分阶段检索", ["context_recall_knn", "context_recall_bm25", "context_recall_rrf", "context_recall_topk", "context_precision"]],
  ["上下文与答案质量", ["context_relevancy", "faithfulness", "answer_relevancy"]],
];

function node(tag, text = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  return element;
}

function action(text, click, style = "secondary") {
  const button = node("button", text, `btn btn-${style} btn-xs`);
  button.type = "button";
  button.addEventListener("click", click);
  return button;
}

function link(text, url) {
  const anchor = node("a", text);
  anchor.href = url;
  anchor.target = "_blank";
  anchor.rel = "noopener";
  return anchor;
}

function notice(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle("error", error);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    let message = detail?.message || `请求失败（HTTP ${response.status}）`;
    if (Array.isArray(detail)) message = detail.map((item) => item.msg).join("；");
    if (typeof detail === "string") message = detail;
    if (detail?.missing?.length) message += `（${detail.missing.join("、")}）`;
    throw new Error(detail?.stage ? `${detail.stage} · ${message}` : message);
  }
  return data;
}

function postJson(url, payload) {
  return request(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
}

function selectEvalTab(active) {
  for (const name of ["run", "runs", "report"]) {
    const selected = name === active;
    $(`${name}-tab`).classList.toggle("active", selected);
    $(`${name}-tab`).setAttribute("aria-pressed", String(selected));
    $(`panel-${name}`).hidden = !selected;
  }
}

function splitLabels(value) {
  return value.trim().split(/[\s,，]+/).filter(Boolean);
}

function labelAction(field, value, status) {
  const label = field === "ground_truth_chunk_ids" ? "标为标准分块" : "标为标准文档";
  return action(label, () => {
    try {
      const input = $("evaluate-form")?.elements.namedItem(field);
      const labels = new Set(splitLabels(input?.value ?? sessionStorage.getItem(field) ?? ""));
      labels.add(value);
      const updated = [...labels].join("\n");
      sessionStorage.setItem(field, updated);
      if (input) input.value = updated;
      notice(status, "已将标签添加到 RAG 评测。");
    } catch {
      notice(status, "浏览器无法保存标签，请在评测页手动填写。", true);
    }
  });
}

function sourceLocation(source) {
  const parts = [];
  if (source.sheet) parts.push(source.sheet);
  if (source.heading_path?.length) parts.push(source.heading_path.join(" › "));
  if (source.row_start != null) parts.push(`第 ${source.row_start}–${source.row_end} 行`);
  else if (source.start_line != null) parts.push(`第 ${source.start_line}–${source.end_line} 行`);
  else if (source.start_char != null) parts.push(`字符 ${source.start_char}–${source.end_char}`);
  return parts.join(" · ");
}

function pageButtons(prefix, offset, total) {
  $(`${prefix}-prev`).disabled = offset === 0;
  $(`${prefix}-next`).disabled = offset + PAGE_SIZE >= total;
  $(`${prefix}-page`).textContent = `${Math.floor(offset / PAGE_SIZE) + 1} / ${Math.max(1, Math.ceil(total / PAGE_SIZE))}`;
}

async function loadDocuments(offset = state.documentOffset) {
  state.documentOffset = offset;
  const call = ++state.documentRequest;
  const status = $("catalog-status");
  notice(status, "正在读取文档…");
  const filters = new FormData($("document-filter"));
  const params = new URLSearchParams({offset, limit: PAGE_SIZE});
  if (filters.get("title").trim()) params.set("title", filters.get("title").trim());
  if (filters.get("status")) params.set("status", filters.get("status"));
  try {
    const page = await request(`/api/documents?${params}`);
    if (state.documentRequest !== call) return;
    const rows = page.items.map((doc) => {
      const row = node("tr");
      const id = node("td");
      const code = node("code", doc.id, "file-id");
      code.title = doc.id;
      id.append(code);
      const statusCell = node("td");
      statusCell.append(node("span", doc.status === "indexed" ? "向量存储完成" : "待入库", "tag"));
      const tools = node("td");
      const actions = node("div", "", "actions");
      actions.append(action("查看分段", () => loadChunks(doc, 0), "primary"), link("下载原件", doc.original_url));
      if (doc.status === "indexed") actions.append(labelAction("ground_truth_document_ids", doc.id, status));
      tools.append(actions);
      row.append(id, node("td", doc.filename), statusCell, node("td", doc.chunk_count), node("td", doc.vector_count ?? "—"), tools);
      return row;
    });
    $("document-rows").replaceChildren(...rows);
    pageButtons("docs", offset, page.total);
    notice(status, page.total ? `共 ${page.total} 份文档` : "当前还没有文档。");
  } catch (error) {
    if (state.documentRequest === call) notice(status, error.message, true);
  }
}

async function loadChunks(doc, offset) {
  state.chunkDocument = doc;
  state.chunkOffset = offset;
  if (!$("chunk-panel").open) $("chunk-panel").showModal();
  $("chunk-title").textContent = `分段列表 · ${doc.filename}`;
  const status = $("chunk-status");
  notice(status, "正在读取切片…");
  try {
    const page = await request(`${doc.chunks_url}?offset=${offset}&limit=${PAGE_SIZE}`);
    if (state.chunkDocument !== doc || state.chunkOffset !== offset) return;
    $("chunk-items").replaceChildren(...page.items.map((chunk) => {
      const item = node("article", "", "item");
      const label = `${doc.id}#${chunk.index}`;
      item.append(node("h3", `#${chunk.index} · ${kinds[chunk.kind] ?? chunk.kind}`));
      item.append(node("p", sourceLocation(chunk.source), "muted"), node("p", chunk.text, "excerpt"));
      const meta = node("details");
      meta.append(node("summary", "Meta 信息"), node("pre", JSON.stringify(chunk.source, null, 2), "excerpt"));
      item.append(meta);
      if (doc.status === "indexed") item.append(labelAction("ground_truth_chunk_ids", label, status));
      return item;
    }));
    pageButtons("chunks", offset, page.total);
    notice(status, `共 ${page.total} 个切片，编号从 0 开始。`);
  } catch (error) {
    if (state.chunkDocument === doc && state.chunkOffset === offset) notice(status, error.message, true);
  }
}

function stats(values) {
  const list = node("div", "", "stats");
  for (const [title, value] of values) {
    const card = node("div", "", "stat");
    card.append(node("span", title), node("strong", value));
    list.append(card);
  }
  return list;
}

function score(value) {
  return value == null ? "—" : value.toFixed(3);
}

function table(headers, rows) {
  const wrap = node("div", "", "table-wrap");
  const element = node("table");
  const head = element.createTHead().insertRow();
  head.append(...headers.map((name) => node("th", name)));
  const body = element.createTBody();
  for (const values of rows) body.insertRow().append(...values.map((value) => node("td", value)));
  wrap.append(element);
  return wrap;
}

function renderRetrieval(result, status) {
  const panel = node("section", "", "panel");
  panel.append(node("h2", "检索证据"), stats([
    ["向量候选", result.vector_count], ["关键词候选", result.keyword_count], ["命中块", result.hits.length],
  ]));
  if (!result.evidence.length) panel.append(node("p", "当前检索没有可用证据。", "muted"));
  for (const [position, evidence] of result.evidence.entries()) {
    const item = node("article", "", "item");
    item.append(node("h3", `${position + 1}. ${evidence.filename} · ${kinds[evidence.kind] ?? evidence.kind}`));
    const location = [sourceLocation(evidence.source), `BGE ${score(evidence.score)}`].filter(Boolean).join(" · ");
    item.append(node("p", location, "muted"), node("p", evidence.text, "excerpt"));
    const actions = node("div", "", "actions");
    actions.append(link("打开原件", evidence.original_url));
    if (evidence.markdown_url) actions.append(link("Markdown", evidence.markdown_url));
    actions.append(labelAction("ground_truth_chunk_ids", `${evidence.document_id}#${evidence.chunk_index}`, status));
    item.append(actions);
    panel.append(item);
  }
  if (result.candidates.length) {
    const details = node("details");
    const rows = result.candidates.map((candidate) => [
      `${candidate.hit.document_id.slice(0, 16)}… #${candidate.hit.chunk_index}`,
      candidate.vector_rank ?? "—", candidate.keyword_rank ?? "—",
      score(candidate.rrf_score), score(candidate.rerank_score), candidate.hit.text.slice(0, 80),
    ]);
    details.append(node("summary", "查看候选轨迹"), table(["切片", "向量名次", "关键词名次", "RRF", "BGE", "片段"], rows));
    panel.append(details);
  }
  return panel;
}

function renderMetricGroups(ratings, headers, metricCells, legacy = false) {
  const groups = legacy ? [["历史模型评分", Object.keys(ratings)]] : metricGroups;
  const output = node("div", "", "stack");
  for (const [title, keys] of groups) {
    const rows = keys.filter((key) => Object.hasOwn(ratings, key)).map((key) => [
      metricNames[key] ?? key, ...metricCells(ratings[key]),
    ]);
    if (!rows.length) continue;
    const section = node("section", "", "metric-group");
    section.append(node("h3", title), table(headers, rows));
    output.append(section);
  }
  return output;
}

function renderMetricRatings(ratings, legacy = false) {
  return renderMetricGroups(ratings, ["指标", "状态", "分数", "理由"], (item) => [
    { ok: "已计算", skipped: "跳过", error: "失败" }[item.status], score(item.score), item.reason,
  ], legacy);
}

function renderEvaluation(result, status) {
  const panel = node("section", "", "panel");
  const metrics = result.retrieval_metrics;
  panel.append(node("h2", result.metrics ? "辅助检索指标" : "检索指标"), stats([
    [`Hit@${metrics.k}`, metrics.hit_at_k ?? "—"], [`MRR@${metrics.k}`, score(metrics.mrr)],
    [`Recall@${metrics.k}`, score(metrics.recall_at_k)],
  ]));
  panel.append(node("p", metrics.status === "missing_labels"
    ? "本次缺少标准标签，检索指标留空。"
    : `按${metrics.level === "chunk" ? "分块" : "文档"}评测，命中 ${metrics.matched_ids.length} / ${metrics.relevant_count} 个标准标签。`, "muted"));
  if (result.metrics) {
    panel.append(node("p", "辅助指标支持父块匹配和文档标签。分阶段指标按实际命中的分块 ID 核对。", "hint"));
  }
  const timings = result.timings_ms;
  const durations = [["检索", timings.retrieval], ["生成", timings.generation], ["评分", timings.judge]];
  panel.append(node("p", durations.filter(([, ms]) => ms != null).map(([name, ms]) => `${name} ${ms} ms`).join(" · "), "muted"));

  const output = [];
  const ratings = result.metrics ?? result.judgement?.metrics;
  if (ratings && Object.keys(ratings).length) {
    const section = node("section", "", "panel");
    section.append(node("h2", "逐题评测指标"), renderMetricRatings(ratings, !result.metrics));
    output.push(section);
  }
  output.push(panel);
  if (result.answer) {
    const answer = result.answer;
    const section = node("section", "", "panel");
    section.append(node("h2", "生成答案"));
    section.append(node("p", answer.error || answer.answer, answer.error ? "error" : "excerpt"));
    const citations = node("div", "", "actions");
    for (const item of answer.contexts) {
      if (answer.citation_ids.includes(item.id)) citations.append(link(`[${item.id}] ${item.filename}`, item.original_url));
    }
    if (citations.childElementCount) section.append(citations);
    output.push(section);
  }
  output.push(renderRetrieval(result.retrieval, status));
  return output;
}

function searchPayload(form) {
  const data = new FormData(form);
  return {
    query: data.get("query").trim(), mode: data.get("mode"), k: Number(data.get("k")),
    document_id: data.get("document_id").trim() || null,
  };
}

function onSubmit(id, statusId, pending, submit) {
  $(id)?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const status = $(statusId);
    button.disabled = true;
    notice(status, pending);
    try {
      await submit(form, status);
    } catch (error) {
      notice(status, error.message, true);
    } finally {
      button.disabled = false;
    }
  });
}

onSubmit("upload-form", "upload-status", "正在解析、切片并写入向量索引…", async (form, status) => {
  const output = $("upload-result");
  output.hidden = true;
  $("upload-progress").hidden = false;
  try {
    const result = await request("/api/documents", { method: "POST", body: new FormData(form) });
    notice(status, "上传成功，向量存储完成。");
    output.replaceChildren(
      node("p", `${result.filename} · ${result.chunk_count} 个分段 · ${result.vector_count} 个向量`),
      link("查看分段", `/document.html?document_id=${encodeURIComponent(result.id)}`),
    );
    output.hidden = false;
    $("split-panel").hidden = false;
    $("upload-grid").classList.add("has-split");
    form.reset();
    $("file-name").textContent = "";
  } finally {
    $("upload-progress").hidden = true;
  }
});
onSubmit("evaluate-form", "evaluate-status", "正在运行单题评测…", async (form, status) => {
  const values = new FormData(form);
  const result = await postJson("/api/eval/evaluate", {
    ...searchPayload(form),
    ground_truth_document_ids: splitLabels(values.get("ground_truth_document_ids")),
    ground_truth_chunk_ids: splitLabels(values.get("ground_truth_chunk_ids")),
    reference_answer: values.get("reference_answer"), generate: values.has("generate"), judge: values.has("judge"),
  });
  const reportStatus = $("report-status");
  notice(reportStatus, "");
  document.dispatchEvent(new Event("single-evaluation-report"));
  $("evaluate-result").hidden = false;
  $("evaluate-result").replaceChildren(...renderEvaluation(result, reportStatus));
  $("report-empty").hidden = true;
  notice(status, "评测完成。");
  selectEvalTab("report");
});

function message(role, text) {
  const item = node("article", "", `message ${role}`);
  const content = node("div", "", "message-content");
  const body = node("p", text, "message-text");
  content.append(body);
  item.append(node("div", role === "user" ? "我" : "AI", "message-avatar"), content);
  $("messages").append(item);
  return {content, body};
}

function scrollMessages() {
  const container = $("messages");
  container.scrollTop = container.scrollHeight;
}

onSubmit("chat-form", "chat-status", "正在检索并生成回答…", async (form, status) => {
  const payload = searchPayload(form);
  if (!payload.query) throw new Error("请输入问题。");
  $("welcome-message").hidden = true;
  const title = $("conversation-title");
  if (!title.classList.contains("has-chat")) {
    title.textContent = payload.query.slice(0, 30);
    title.classList.add("has-chat");
  }
  message("user", payload.query);
  const reply = message("assistant", "正在生成回答…");
  form.elements.namedItem("query").value = "";
  $("new-chat").disabled = true;
  scrollMessages();
  try {
    const result = await postJson("/api/eval/evaluate", {...payload, generate: true});
    const answer = result.answer;
    reply.body.textContent = answer.error || answer.answer;
    reply.body.classList.toggle("error", answer.status === "error");
    const citations = answer.contexts.filter((item) => answer.citation_ids.includes(item.id));
    if (citations.length) {
      const refs = node("div", "", "message-references");
      refs.append(node("p", "参考来源", "muted"));
      refs.append(...citations.map((item) => link(`[${item.id}] ${item.filename}`, item.original_url)));
      reply.content.append(refs);
    }
    const details = node("details", "", "message-references");
    details.append(node("summary", "查看检索证据"), renderRetrieval(result.retrieval, status));
    reply.content.append(details);
    notice(status, "");
  } catch (error) {
    reply.body.textContent = error.message;
    reply.body.classList.add("error");
    throw error;
  } finally {
    $("new-chat").disabled = false;
    scrollMessages();
  }
});

async function loadOptions() {
  const page = await request("/api/documents?limit=100&status=indexed");
  $("document-options").replaceChildren(...page.items.map((doc) => {
    const option = node("option", doc.filename);
    option.value = doc.id;
    return option;
  }));
}

$("upload-file")?.addEventListener("change", (event) => {
  $("file-name").textContent = event.target.files[0]?.name ?? "";
});
if (location.pathname === "/upload.html") {
  const formerTabs = {documents: "/document.html", retrieve: "/chat.html", evaluate: "/eval-report.html"};
  const page = formerTabs[location.hash.slice(1)];
  if (typeof page === "string") location.replace(page);
}

if ($("document-filter")) {
  $("document-filter").addEventListener("submit", (event) => {
    event.preventDefault();
    loadDocuments(0);
  });
  $("document-filter").addEventListener("reset", () => requestAnimationFrame(() => loadDocuments(0)));
  $("docs-prev").addEventListener("click", () => loadDocuments(Math.max(0, state.documentOffset - PAGE_SIZE)));
  $("docs-next").addEventListener("click", () => loadDocuments(state.documentOffset + PAGE_SIZE));
  $("chunks-prev").addEventListener("click", () => loadChunks(state.chunkDocument, Math.max(0, state.chunkOffset - PAGE_SIZE)));
  $("chunks-next").addEventListener("click", () => loadChunks(state.chunkDocument, state.chunkOffset + PAGE_SIZE));
  loadDocuments();
  const initialDocument = new URLSearchParams(location.search).get("document_id");
  if (initialDocument) {
    request(`/api/documents/${encodeURIComponent(initialDocument)}`)
      .then((doc) => loadChunks(doc, 0))
      .catch((error) => notice($("catalog-status"), error.message, true));
  }
}

if ($("chat-form")) {
  $("new-chat").addEventListener("click", () => {
    for (const message of $("messages").querySelectorAll(".message")) message.remove();
    $("welcome-message").hidden = false;
    $("conversation-title").textContent = "暂无对话记录";
    $("conversation-title").classList.remove("has-chat");
    notice($("chat-status"), "");
    $("chat-form").elements.namedItem("query").focus();
  });
  $("chat-form").elements.namedItem("query").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    const send = $("chat-form").querySelector('button[type="submit"]');
    if (!send.disabled) $("chat-form").requestSubmit(send);
  });
}

if ($("evaluate-form")) {
  const fields = $("evaluate-form").elements;
  $("run-tab").addEventListener("click", () => selectEvalTab("run"));
  $("report-tab").addEventListener("click", () => selectEvalTab("report"));
  fields.namedItem("judge").addEventListener("change", (event) => {
    if (event.target.checked) fields.namedItem("generate").checked = true;
  });
  fields.namedItem("generate").addEventListener("change", (event) => {
    if (!event.target.checked) fields.namedItem("judge").checked = false;
  });
  try {
    for (const name of labelFields) {
      const field = fields.namedItem(name);
      field.value = sessionStorage.getItem(name) ?? "";
      field.addEventListener("change", () => sessionStorage.setItem(name, field.value));
    }
    if (labelFields.some((name) => fields.namedItem(name).value)) $("single-evaluation").open = true;
  } catch {
    notice($("evaluate-status"), "浏览器无法读取已标注的标签，可在此手动填写。", true);
  }
}

if ($("document-options")) {
  loadOptions().catch((error) => notice($("chat-status") ?? $("evaluate-status"), error.message, true));
}
