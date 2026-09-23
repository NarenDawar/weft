let allRuns = [];
const selectedForDiff = new Set();

function findRun(runId) {
  return allRuns.find((run) => run.run_id === runId);
}

async function loadRuns() {
  const response = await fetch("/api/runs");
  allRuns = await response.json();
  renderRunList();
}

function renderRunList() {
  const container = document.getElementById("runs");
  container.innerHTML = "";
  const list = document.createElement("ul");

  allRuns.forEach((run) => {
    const item = document.createElement("li");

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedForDiff.has(run.run_id);
    checkbox.title = "Select for diff";
    checkbox.onchange = () => {
      if (checkbox.checked) {
        selectedForDiff.add(run.run_id);
      } else {
        selectedForDiff.delete(run.run_id);
      }
      renderCompareButton();
    };
    item.appendChild(checkbox);

    const link = document.createElement("a");
    link.href = "#";
    link.textContent = `${run.run_id} — ${run.task} (${run.status})`;
    link.onclick = (event) => {
      event.preventDefault();
      loadTimeline(run.run_id);
    };
    item.appendChild(link);

    if (run.forked_from_run_id) {
      const forkNote = document.createElement("span");
      forkNote.className = "fork-note";
      forkNote.textContent = ` ↳ forked from ${run.forked_from_run_id} @ step ${run.forked_from_step}`;
      item.appendChild(forkNote);
    }

    list.appendChild(item);
  });

  container.appendChild(list);
  renderCompareButton();
}

function renderCompareButton() {
  const existing = document.getElementById("compare-button");
  if (existing) {
    existing.remove();
  }
  if (selectedForDiff.size !== 2) {
    return;
  }
  const [runIdA, runIdB] = Array.from(selectedForDiff);
  const button = document.createElement("button");
  button.id = "compare-button";
  button.textContent = `Compare ${runIdA} vs ${runIdB}`;
  button.onclick = () => loadDiff(runIdA, runIdB);
  document.getElementById("runs").appendChild(button);
}

function renderStepList(steps, divergenceIndex) {
  const list = document.createElement("ol");
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    const classes = [step.is_error ? "step-error" : "step"];
    if (divergenceIndex !== null && divergenceIndex !== undefined && index >= divergenceIndex) {
      classes.push("step-diverged");
    }
    item.className = classes.join(" ");
    item.textContent = `${step.tool_name}(${JSON.stringify(step.args)}) -> ${JSON.stringify(step.result)}`;
    list.appendChild(item);
  });
  return list;
}

async function loadTimeline(runId) {
  document.getElementById("diff").innerHTML = "";
  const response = await fetch(`/api/runs/${runId}`);
  const steps = await response.json();
  const run = findRun(runId);
  const container = document.getElementById("timeline");
  container.innerHTML = "";

  const heading = document.createElement("h2");
  heading.textContent = `Run ${runId}`;
  container.appendChild(heading);

  if (run && run.forked_from_run_id) {
    const forkNote = document.createElement("p");
    forkNote.className = "fork-note";
    forkNote.textContent = `↳ forked from ${run.forked_from_run_id} at step ${run.forked_from_step}`;
    container.appendChild(forkNote);
  }

  const forkedFromStep = run ? run.forked_from_step : null;
  const list = document.createElement("ol");
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    const classes = [step.is_error ? "step-error" : "step"];
    if (forkedFromStep !== null && forkedFromStep !== undefined && index < forkedFromStep) {
      classes.push("step-inherited");
    }
    item.className = classes.join(" ");
    item.textContent = `${step.tool_name}(${JSON.stringify(step.args)}) -> ${JSON.stringify(step.result)}`;
    list.appendChild(item);
  });
  container.appendChild(list);
}

async function loadDiff(runIdA, runIdB) {
  document.getElementById("timeline").innerHTML = "";
  const [diffResponse, stepsAResponse, stepsBResponse] = await Promise.all([
    fetch(`/api/diff/${runIdA}/${runIdB}`),
    fetch(`/api/runs/${runIdA}`),
    fetch(`/api/runs/${runIdB}`),
  ]);
  const diffResult = await diffResponse.json();
  const stepsA = await stepsAResponse.json();
  const stepsB = await stepsBResponse.json();

  const container = document.getElementById("diff");
  container.innerHTML = "";

  const heading = document.createElement("h2");
  heading.textContent = `Diff: ${runIdA} vs ${runIdB}`;
  container.appendChild(heading);

  const summary = document.createElement("p");
  summary.className = "diff-summary";
  if (diffResult.divergence_reason === "decision") {
    summary.textContent = `Shared prefix: ${diffResult.shared_prefix_length} step(s). Diverges at step ${diffResult.divergence_index}: different action taken.`;
  } else if (diffResult.divergence_reason === "observation") {
    summary.textContent = `Shared prefix: ${diffResult.shared_prefix_length} step(s). Diverges at step ${diffResult.divergence_index}: same action, different result.`;
  } else if (diffResult.divergence_index !== null && diffResult.divergence_index !== undefined) {
    summary.textContent = `Shared prefix: ${diffResult.shared_prefix_length} step(s). Runs share a common prefix but differ in length from step ${diffResult.divergence_index}.`;
  } else {
    summary.textContent = "No divergence: runs are identical.";
  }
  container.appendChild(summary);

  const columns = document.createElement("div");
  columns.className = "diff-columns";

  const columnA = document.createElement("div");
  columnA.className = "diff-column";
  const headingA = document.createElement("h3");
  headingA.textContent = runIdA;
  columnA.appendChild(headingA);
  columnA.appendChild(renderStepList(stepsA, diffResult.divergence_index));

  const columnB = document.createElement("div");
  columnB.className = "diff-column";
  const headingB = document.createElement("h3");
  headingB.textContent = runIdB;
  columnB.appendChild(headingB);
  columnB.appendChild(renderStepList(stepsB, diffResult.divergence_index));

  columns.appendChild(columnA);
  columns.appendChild(columnB);
  container.appendChild(columns);
}

loadRuns();
