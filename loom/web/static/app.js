async function loadRuns() {
  const response = await fetch("/api/runs");
  const runs = await response.json();
  const container = document.getElementById("runs");
  container.innerHTML = "";
  const list = document.createElement("ul");
  runs.forEach((run) => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = `${run.run_id} — ${run.task} (${run.status})`;
    link.onclick = (event) => {
      event.preventDefault();
      loadTimeline(run.run_id);
    };
    item.appendChild(link);
    list.appendChild(item);
  });
  container.appendChild(list);
}

async function loadTimeline(runId) {
  const response = await fetch(`/api/runs/${runId}`);
  const steps = await response.json();
  const container = document.getElementById("timeline");
  container.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent = `Run ${runId}`;
  container.appendChild(heading);
  const list = document.createElement("ol");
  steps.forEach((step) => {
    const item = document.createElement("li");
    item.className = step.is_error ? "step step-error" : "step";
    item.textContent = `${step.tool_name}(${JSON.stringify(step.args)}) -> ${JSON.stringify(step.result)}`;
    list.appendChild(item);
  });
  container.appendChild(list);
}

loadRuns();
