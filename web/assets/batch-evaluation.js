"use strict";

(() => {
  const RUN_PAGE_SIZE = 15;
  const CASE_PAGE_SIZE = 10;
  const statusNames = { queued: "排队中", running: "运行中", completed: "已完成", failed: "失败" };
  const modeNames = { hybrid: "混合检索", vector: "向量检索", keyword: "关键词检索" };
  let datasets = [];
  let selectedRun = null;
  let runOffset = 0;
  let caseOffset = 0;
  let pollTimer = null;

  function syncDatasetLink() {
    const dataset = datasets.find((item) => item.id === $("batch-dataset").value);
    const anchor = $("dataset-download");
    anchor.hidden = !dataset;
    if (dataset) anchor.href = dataset.download_url;
  }

  async function loadDatasets(preferred) {
    datasets = await request("/api/eval/datasets");
    const selector = $("batch-dataset");
    const selected = preferred || selector.value || datasets[0]?.id;
    selector.replaceChildren(...datasets.map((item) => {
      const option = node("option", `${item.filename}（${item.case_count} 题${item.builtin ? "，内置" : ""}）`);
      option.value = item.id;
      return option;
    }));
    if (datasets.some((item) => item.id === selected)) selector.value = selected;
    syncDatasetLink();
  }

  function paginate(prefix, offset, total, size) {
    $(`${prefix}-prev`).disabled = offset === 0;
    $(`${prefix}-next`).disabled = offset + size >= total;
    $(`${prefix}-page`).textContent = `${Math.floor(offset / size) + 1} / ${Math.max(1, Math.ceil(total / size))}`;
  }

  async function loadRuns(offset = 0) {
    runOffset = offset;
    const status = $("runs-status");
    notice(status, "正在读取评测记录…");
    try {
      const page = await request(`/api/eval/runs?offset=${offset}&limit=${RUN_PAGE_SIZE}`);
      if (offset !== runOffset) return;
      const rows = page.items.map((run) => {
        const tr = node("tr");
        const idCell = node("td");
        const id = action(run.id.slice(0, 12), () => openRun(run.id));
        id.title = run.id;
        idCell.append(id);
        tr.append(idCell, node("td", run.dataset_name), node("td", statusNames[run.status]),
          node("td", `${run.completed_cases} / ${run.total_cases}`),
          node("td", modeNames[run.options.mode]), node("td", run.runtime.metrics_version ?? "历史口径"),
          node("td", new Date(run.created_at).toLocaleString("zh-CN")));
        return tr;
      });
      $("run-rows").replaceChildren(...rows);
      paginate("runs", offset, page.total, RUN_PAGE_SIZE);
      notice(status, page.total ? `共 ${page.total} 次运行` : "暂无评测记录。");
    } catch (error) {
      notice(status, error.message, true);
    }
  }

  function renderRun(run) {
    $("batch-run-id").textContent = `${run.dataset_name} · ${run.id}`;
    $("batch-progress").max = run.total_cases;
    $("batch-progress").value = run.completed_cases;
    const progress = `${statusNames[run.status]} · ${run.completed_cases} / ${run.total_cases} 题`;
    notice($("batch-progress-text"), run.error ? `${progress} · ${run.error}` : progress, Boolean(run.error));
    $("batch-manifest").textContent = JSON.stringify({
      dataset_id: run.dataset_id, options: run.options, runtime: run.runtime, corpus: run.corpus,
    }, null, 2);
    if (!run.summary) return;

    const values = run.summary.retrieval;
    const k = run.options.k;
    const summary = $("batch-summary");
    const ratings = run.summary.metrics ?? run.summary.judge;
    summary.replaceChildren();
    if (Object.keys(ratings).length) {
      summary.append(renderMetricGroups(ratings, ["指标", "均分", "有效", "跳过", "失败"], (item) => [
        score(item.mean), item.count, item.skipped, item.errors,
      ], !run.summary.metrics));
    }
    const retrieval = node("section", "", "metric-group");
    retrieval.append(node("h3", run.summary.metrics ? "辅助检索指标" : "检索指标"), stats([
      [`Hit@${k}`, score(values.hit_at_k.mean)], [`MRR@${k}`, score(values.mrr.mean)],
      [`Recall@${k}`, score(values.recall_at_k.mean)],
    ]), node("p", `检索指标有效 ${values.hit_at_k.count} 题 · 检索失败 ${run.summary.retrieval_errors} 题 · 缺少标签 ${run.summary.missing_labels} 题`, "muted"));
    if (run.summary.metrics) {
      retrieval.append(node("p", "辅助指标支持父块匹配和文档标签。分阶段指标按实际命中的分块 ID 核对。", "hint"));
    }
    summary.append(retrieval);
    if (run.options.generate) {
      const generation = run.summary.generation;
      summary.append(node("p", `生成回答 ${generation.answered ?? 0} 题 · 证据不足 ${generation.insufficient_evidence ?? 0} 题 · 生成失败 ${generation.error ?? 0} 题`, "muted"));
    }
  }

  function renderCase(item) {
    const details = node("details", "", "item");
    const metric = item.result?.retrieval_metrics;
    const outcome = item.error ? "失败" : metric.status === "missing_labels" ? "缺少标签" : metric.hit_at_k ? "命中" : "未命中";
    details.append(node("summary", `${item.ordinal + 1}. ${item.case_id} · ${outcome} · ${item.question}`));
    details.addEventListener("toggle", () => {
      if (!details.open || details.childElementCount > 1) return;
      const body = node("div", "", "stack");
      if (item.reference_answer) body.append(node("p", `标准答案：${item.reference_answer}`));
      const labels = item.ground_truth_chunk_ids.length ? item.ground_truth_chunk_ids : item.ground_truth_document_ids;
      if (labels.length) body.append(node("pre", labels.join("\n"), "excerpt"));
      if (item.error) {
        body.append(node("p", `${item.error.stage} · ${item.error.message}`, "error"));
        if (item.metrics) body.append(renderMetricRatings(item.metrics));
      } else {
        body.append(...renderEvaluation(item.result, $("report-status")));
      }
      details.append(body);
    });
    return details;
  }

  async function loadCases(id, offset = 0) {
    caseOffset = offset;
    try {
      const page = await request(`/api/eval/runs/${id}/cases?offset=${offset}&limit=${CASE_PAGE_SIZE}`);
      if (id !== selectedRun || offset !== caseOffset) return;
      $("batch-cases").hidden = !page.total;
      $("batch-case-items").replaceChildren(...page.items.map(renderCase));
      paginate("cases", offset, page.total, CASE_PAGE_SIZE);
    } catch (error) {
      if (id === selectedRun) notice($("report-status"), error.message, true);
    }
  }

  async function pollRun(id) {
    try {
      const run = await request(`/api/eval/runs/${id}`);
      if (id !== selectedRun) return;
      renderRun(run);
      if (run.status === "queued" || run.status === "running") {
        pollTimer = setTimeout(() => pollRun(id), 1500);
      } else {
        await loadCases(id);
      }
    } catch (error) {
      if (id === selectedRun) {
        notice($("report-status"), error.message, true);
        pollTimer = setTimeout(() => pollRun(id), 4000);
      }
    }
  }

  function setReportURL(id) {
    const url = new URL(location.href);
    if (id) url.searchParams.set("run", id);
    else url.searchParams.delete("run");
    history.replaceState(null, "", url);
  }

  function openRun(id) {
    clearTimeout(pollTimer);
    selectedRun = id;
    setReportURL(id);
    $("report-empty").hidden = true;
    $("evaluate-result").hidden = true;
    $("batch-report").hidden = false;
    $("batch-cases").hidden = true;
    $("batch-summary").replaceChildren();
    $("batch-manifest").textContent = "";
    $("batch-run-id").textContent = id;
    $("batch-progress").value = 0;
    notice($("report-status"), "");
    notice($("batch-progress-text"), "正在读取运行进度…");
    selectEvalTab("report");
    pollRun(id);
  }

  onSubmit("dataset-form", "dataset-status", "正在校验评测集…", async (form, status) => {
    const result = await request("/api/eval/datasets", {method: "POST", body: new FormData(form)});
    await loadDatasets(result.id);
    form.reset();
    notice(status, `${result.filename} 已就绪，共 ${result.case_count} 题。`);
  });
  onSubmit("batch-form", "batch-status", "正在提交批量评测…", async (form, status) => {
    const values = new FormData(form);
    const run = await postJson("/api/eval/runs", {
      dataset_id: values.get("dataset_id"), mode: values.get("mode"), k: Number(values.get("k")),
      generate: values.has("generate"), judge: values.has("judge"),
      expand_parents: values.has("expand_parents"),
    });
    notice(status, `已提交 ${run.total_cases} 题。`);
    openRun(run.id);
  });

  const fields = $("batch-form").elements;
  fields.namedItem("judge").addEventListener("change", (event) => {
    if (event.target.checked) fields.namedItem("generate").checked = true;
  });
  fields.namedItem("generate").addEventListener("change", (event) => {
    if (!event.target.checked) fields.namedItem("judge").checked = false;
  });
  $("batch-dataset").addEventListener("change", syncDatasetLink);
  $("runs-tab").addEventListener("click", () => { selectEvalTab("runs"); loadRuns(0); });
  $("refresh-runs").addEventListener("click", () => loadRuns(runOffset));
  $("runs-prev").addEventListener("click", () => loadRuns(Math.max(0, runOffset - RUN_PAGE_SIZE)));
  $("runs-next").addEventListener("click", () => loadRuns(runOffset + RUN_PAGE_SIZE));
  $("cases-prev").addEventListener("click", () => loadCases(selectedRun, Math.max(0, caseOffset - CASE_PAGE_SIZE)));
  $("cases-next").addEventListener("click", () => loadCases(selectedRun, caseOffset + CASE_PAGE_SIZE));
  document.addEventListener("single-evaluation-report", () => {
    clearTimeout(pollTimer);
    selectedRun = null;
    setReportURL(null);
    $("batch-report").hidden = true;
  });

  loadDatasets().catch((error) => notice($("dataset-status"), error.message, true));
  const run = new URLSearchParams(location.search).get("run");
  if (/^[0-9a-f]{32}$/.test(run ?? "")) openRun(run);
})();
