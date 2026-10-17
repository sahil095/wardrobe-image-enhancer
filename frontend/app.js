const $ = (id) => document.getElementById(id);

let selectedFile = null;

// ---- live slider value labels ----
const sliders = {
  strength: "strengthVal",
  steps: "stepsVal",
  guidance: "guidanceVal",
  controlnet_scale: "cnVal",
  ip_scale: "ipVal",
};
for (const [id, valId] of Object.entries(sliders)) {
  const el = $(id);
  el.addEventListener("input", () => ($(valId).textContent = el.value));
}

// ---- health badge ----
async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.cuda) {
      const vram = h.vram_gb ? ` · ${h.vram_gb.free}/${h.vram_gb.total} GB free` : "";
      $("health").textContent = `${h.gpu || "CUDA"}${vram}`;
      $("health").style.color = "var(--ok)";
    } else {
      $("health").textContent = "Running on CPU (slow — no CUDA detected)";
      $("health").style.color = "var(--err)";
    }
  } catch {
    $("health").textContent = "backend unreachable";
    $("health").style.color = "var(--err)";
  }
}
loadHealth();

// ---- file selection + drag/drop ----
const drop = $("drop");
const fileInput = $("file");

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  $("thumb").src = url;
  $("thumb").hidden = false;
  $("dropText").hidden = true;
  $("originalImg").src = url;
}

fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("dragover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

// ---- enhance ----
function setBusy(busy, msg) {
  $("enhanceBtn").disabled = busy;
  $("rerunBtn").disabled = busy || !selectedFile;
  $("spinner").hidden = !busy;
  $("status").className = "status";
  $("status").textContent = msg || "";
}

async function runEnhance(randomizeSeed) {
  if (!selectedFile) {
    setBusy(false);
    $("status").className = "status err";
    $("status").textContent = "Please choose an image first.";
    return;
  }

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("category", $("category").value);
  fd.append("color", $("color").value);
  fd.append("pattern", $("pattern").value);
  fd.append("notes", $("notes").value);
  fd.append("strength", $("strength").value);
  fd.append("steps", $("steps").value);
  fd.append("guidance", $("guidance").value);
  fd.append("controlnet_scale", $("controlnet_scale").value);
  fd.append("ip_scale", $("ip_scale").value);
  fd.append("seed", randomizeSeed ? "-1" : "-1"); // always new seed; rerun just re-rolls

  setBusy(true, "Enhancing… (first run loads models, can take a while)");
  const t0 = performance.now();

  try {
    const r = await fetch("/api/enhance", { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    const data = await r.json();

    // cache-bust in case filenames ever repeat
    $("originalImg").src = data.original_url + "?t=" + Date.now();
    $("resultImg").src = data.result_url + "?t=" + Date.now();

    const dl = $("downloadLink");
    dl.href = data.result_url;
    dl.download = `enhanced_${data.job_id}.png`;
    dl.hidden = false;

    const p = data.params;
    $("meta").textContent =
      `seed ${data.seed} · strength ${p.strength} · steps ${p.steps} · ` +
      `guidance ${p.guidance} · cn ${p.controlnet_scale} · ip ${p.ip_scale} · ` +
      `${data.elapsed_sec}s`;

    setBusy(false);
    $("status").className = "status ok";
    $("status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  } catch (e) {
    setBusy(false);
    $("status").className = "status err";
    $("status").textContent = e.message;
  }
}

$("enhanceBtn").addEventListener("click", () => runEnhance(false));
$("rerunBtn").addEventListener("click", () => runEnhance(true));
