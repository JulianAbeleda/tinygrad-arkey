import {codeBlock, colored, displaySelection, fetchValue, isExpanded, metadata, rect, saveToHistory, state, tabulate, toggleCls, traceBlock} from "./ui.js";
import {configureGraph, graphWorkerReady, initWorker, renderDag} from "./graph.js";
import {configureProfiler, formatMicroseconds, formatUnit, moveProfilerFocus, profilerVisible, renderProfiler, restoreProfilerFocus} from "./profiler.js";

var ret = [];
var cache = {};
var ctxs = null;
const evtSources = [];
// VIZ displays graph rewrites in 3 levels, from bottom-up:
// rewrite: a single UOp transformation
// step: collection of rewrites
// context: collection of steps
export function setState(ns) {
  saveToHistory(state);
  const { ctx:prevCtx, step:prevStep } = select(state.currentCtx, state.currentStep);
  const prevRewrite = state.currentRewrite;
  Object.assign(state, ns);
  // update element styles if needed
  const { ctx, step } = select(state.currentCtx, state.currentStep);
  toggleCls(prevCtx, ctx, "expanded", state.expandSteps);
  if (ctx?.id !== prevCtx?.id) {
    toggleCls(prevCtx, ctx, "active");
  }
  if (ctx?.id !== prevCtx?.id || step?.id !== prevStep?.id) {
    toggleCls(prevStep, step, "active");
    // walk the tree back until all parents expanded so that the child is visible
    let e = step;
    while (e?.parentElement?.id.startsWith("step")) {
      e.parentElement.classList.add("expanded");
      e = e.parentElement;
    }
  }
  // re-render
  main();
}

const getSubrewrites = (ul) => ul.querySelectorAll(":scope > ul");

// switch to the start of a new graph and expand all the steps
export const switchCtx = (newCtx, step) => setState({ expandSteps:true, currentCtx:newCtx+1, currentStep:step ?? 0, currentRewrite:0 });

window.addEventListener("popstate", (e) => {
  if (e.state?.shape != null) return restoreProfilerFocus(e.state?.shape);
  if (e.state != null) setState(e.state);
});

const createToggle = (id, text, checked=true) => {
  const label = d3.create("label").text(text).node(), toggle = d3.create("input").attr("type", "checkbox").attr("id", id).property("checked", checked).node();
  label.prepend(toggle);
  return { toggle, label };
}
const showIndexing = createToggle("show-indexing", "Show indexing (r)");
const showCallSrc = createToggle("show-call-src", "Show all CALL src (c)", false);
const showSink = createToggle("show-sink", "Show SINK (s)", false);
const showGraph = createToggle("show-graph", "Show graph (g)");
showGraph.toggle.onchange = () => displaySelection(rect("#graph").width > 0 ? "#custom" : "#graph");

function appendSteps(root, idx, steps) {
  const stack = [];
  for (const [j,u] of steps.entries()) {
    while (stack.length && stack.at(-1).depth >= u.depth) stack.pop();
    const list = stack.length > 0 ? stack.at(-1).li : root;
    u.li = list.appendChild(document.createElement("ul"));
    u.li.id = `step-${idx}-${j}`
    const p = u.li.appendChild(document.createElement("p"));
    p.appendChild(colored(`${u.name}`+(u.match_count ? ` - ${u.match_count}` : '')));
    p.onclick = (e) => {
      e.stopPropagation();
      const subrewrites = getSubrewrites(e.currentTarget.parentElement);
      if (subrewrites.length) { e.currentTarget.parentElement.classList.toggle("expanded"); }
      setState({ currentStep:j, currentCtx:idx, currentRewrite:0 });
    }
    stack.push(u);
  }
  for (const l of root.querySelectorAll("ul > ul > p")) {
    const subrewrites = getSubrewrites(l.parentElement);
    if (subrewrites.length > 0) { l.appendChild(d3.create("span").text(` (${subrewrites.length})`).node()); l.parentElement.classList.add("has-children"); }
  }
}

async function main() {
  // ** left sidebar context list
  if (ctxs == null) {
    ctxs = [{ name:"Profiler", steps:[] }];
    for (const r of await fetchValue("/ctxs")) ctxs.push(r);
    const ctxList = document.querySelector(".ctx-list");
    for (const [i,{name, steps}] of ctxs.entries()) {
      const ul = ctxList.appendChild(document.createElement("ul"));
      ul.id = `ctx-${i}`;
      const p = ul.appendChild(document.createElement("p"));
      p.appendChild(colored(name));
      p.onclick = () => {
        setState(i === state.currentCtx ? { expandSteps:!state.expandSteps } : { expandSteps:true, currentCtx:i, currentStep:0, currentRewrite:0 });
      }
      appendSteps(ul, i, steps);
    }
    return setState({ currentCtx:-1 });
  }
  // ** center graph
  const { currentCtx, currentStep, currentRewrite, expandSteps } = state;
  if (currentCtx == -1) return;
  const ctx = ctxs[currentCtx];
  const step = ctx.steps[currentStep];
  const ckey = step?.query;
  // close any pending event sources
  let activeSrc = null;
  for (const e of evtSources) {
    const url = new URL(e.url);
    if (url.pathname+url.search !== ckey) e.close();
    else if (e.readyState === EventSource.OPEN) activeSrc = e;
  }
  if (ctx.name === "Profiler") return renderProfiler("/get_profile", {unit:"ms", width:"132px"});
  if (!graphWorkerReady()) await initWorker();
  if (ckey in cache) {
    ret = cache[ckey];
  }
  if (!ckey.startsWith("/graph")) {
    if (!(ckey in cache)) cache[ckey] = ret = await fetchValue(ckey);
    if (ret.steps?.length > 0) {
      const el = select(state.currentCtx, state.currentStep);
      if (el.step.querySelectorAll("ul").length === ret.steps.length) return;
      // re render the list with new items
      ctx.steps.push(...ret.steps);
      while (el.ctx.children.length > 1) el.ctx.children[1].remove();
      appendSteps(el.ctx, state.currentCtx, ctx.steps);
      return setState({ currentStep:state.currentStep+1, expandSteps:true });
    }
    // timeline with cycles on the x axis
    if (ret instanceof ArrayBuffer) {
      const pkts = step.query.includes("sqtt");
      return renderProfiler(ckey, {unit:"clk", heightScale:0.5, hideLabels:true, colorByName:pkts});
    }
    metadata.replaceChildren(...((ret.metadata ?? []).map((m) => {
      return tabulate(m.map((e) => [e.label.trim(), typeof e.value === "string" ? e.value : formatUnit(e.value)]));
    })));
    // graph render
    if (ret.data != null) {
      metadata.prepend(showGraph.label);
      renderDag(ret, { recenter:true });
    } else displaySelection("#custom");
    // table / plaintext render
    const root = d3.create("div").classed("raw-text", true);
    function renderTable(root, ret) {
      const table = root.append("table");
      const thead = table.append("thead");
      for (const c of ret.cols) thead.append("th").text(c.title ?? c);
      for (const r of ret.rows) {
        const tr = table.append("tr").classed("main-row", true);
        for (const [i,value] of r.entries()) {
          // nested table
          if (value.cols != null) {
            tr.classed("has-children", true);
            tr.on("click", () => {
              const el = tr.node().nextElementSibling;
              if (el?.classList.contains("nested-row")) { tr.classed("expanded", false); return el.remove(); }
              tr.classed("expanded", true);
              const td = table.insert("tr", () => tr.node().nextSibling).classed("nested-row", true).append("td");
              td.attr("colSpan", ret.cols.length);
              renderTable(td, value);
            });
            continue;
          }
          const td = tr.append("td").classed(ret.cols[i], true);
          // string format scalar values
          td.append(() => typeof value === "string" ? colored(value) : d3.create("p").text(ret.cols[i] === "Duration" ? formatMicroseconds(value) : formatUnit(value)).node());
        }
      }
      return table;
    }
    if (ret.ref != null) {
      const disasmIdx = ctxs[ret.ref+1].steps.findIndex(s => s.name === "View Disassembly")
      metadata.appendChild(d3.create("a").text("View Disassembly").on("click", () => switchCtx(ret.ref, disasmIdx)).node());
    }
    if (ret.cols != null) renderTable(root, ret);
    else if (ret.src != null) root.append(() => codeBlock(ret.src, ret.lang));
    return document.querySelector("#custom").replaceChildren(root.node());
  }
  // ** Graph view
  // if we don't have a complete cache yet we start streaming graphs in this step
  if (!(ckey in cache) || (cache[ckey].length !== step.match_count+1 && activeSrc == null)) {
    ret = [];
    cache[ckey] = ret;
    const eventSource = new EventSource(ckey);
    evtSources.push(eventSource);
    eventSource.onmessage = (e) => {
      if (e.data === "[DONE]") return eventSource.close();
      const chunk = JSON.parse(e.data);
      ret.push(chunk);
      // if it's the first one render this new rgaph
      if (ret.length === 1) return main();
      // otherwise just enable the graph selector
      const ul = document.getElementById(`rewrite-${ret.length-1}`);
      if (ul != null) ul.classList.remove("disabled");
    };
  }
  if (ret.length === 0) return;
  // ** center graph
  const data = ret[currentRewrite];
  const render = (layoutOpts, renderOpts) => renderDag({ data, opts:layoutOpts }, renderOpts);
  const getOpts = () => ({ showIndexing:showIndexing.toggle.checked, showCallSrc:showCallSrc.toggle.checked, showSink:showSink.toggle.checked,
    callSrcMask:state.callSrcMask, expandedNodes:state.expandedNodes });
  render(getOpts(), { recenter:currentRewrite === 0 });
  showIndexing.toggle.onchange = () => render(getOpts(), { recenter:true });
  showCallSrc.toggle.onchange = () => { state.callSrcMask.clear(); render(getOpts(), { recenter:true }); }
  showSink.toggle.onchange = () => render(getOpts(), { recenter:true });
  // ** right sidebar metadata
  metadata.innerHTML = "";
  if (ckey.includes("rewrites")) metadata.append(showIndexing.label, showCallSrc.label, showSink.label);
  if (step.code_line != null) metadata.appendChild(codeBlock(step.code_line, "python", { loc:step.loc, wrap:true }));
  if (step.trace) metadata.appendChild(traceBlock(step.trace));
  if (data.uop != null) metadata.appendChild(codeBlock(data.uop, "python", { wrap:false })).classList.toggle("full-height", step.match_count === 0);
  // ** multi graph in one page
  if (!step.match_count) return;
  const rewriteList = metadata.appendChild(document.createElement("div"));
  rewriteList.className = "rewrite-list";
  for (let s=0; s<=step.match_count; s++) {
    const ul = rewriteList.appendChild(document.createElement("ul"));
    ul.id = `rewrite-${s}`;
    const p = ul.appendChild(document.createElement("p"));
    p.innerText = s;
    ul.onclick = () => setState({ currentRewrite:s });
    ul.className = s > ret.length-1 ? "disabled" : s === currentRewrite ? "active" : "";
    if (s > 0 && s === currentRewrite) {
      const { upat, diff } = ret[s];
      metadata.appendChild(codeBlock(upat[1], "python", { loc:upat[0], wrap:true }));
      const diffCode = metadata.appendChild(document.createElement("pre")).appendChild(document.createElement("code"));
      for (const line of diff) {
        diffCode.appendChild(colored([{st:line, color:line.startsWith("+") ? "#3aa56d" : line.startsWith("-") ? "#d14b4b" : "#f0f0f5"}]));
        diffCode.appendChild(document.createElement("br"));
      }
      diffCode.className = "wrap";
    }
  }
}

// **** collapse/expand

let isCollapsed = false;
document.querySelector(".collapse-btn").addEventListener("click", (e) => {
  isCollapsed = !isCollapsed;
  document.querySelector(".main-container").classList.toggle("collapsed", isCollapsed);
  e.currentTarget.blur();
  e.currentTarget.style.transform = isCollapsed ? "rotate(180deg)" : "rotate(0deg)";
  window.dispatchEvent(new Event("resize"));
});

// **** resizer

function appendResizer(element, { minWidth, maxWidth }, left=false) {
  const handle = Object.assign(document.createElement("div"), { className: "resize-handle", style: left ? "right: 0" : "left: 0; margin-top: 0" });
  element.appendChild(handle);
  const resize = (e) => {
    const change = e.clientX - element.dataset.startX;
    let newWidth = ((Number(element.dataset.startWidth)+(left ? change : -change))/Number(element.dataset.containerWidth))*100;
    element.style.width = `${Math.max(minWidth, Math.min(maxWidth, newWidth))}%`;
  };
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    element.dataset.startX = e.clientX;
    element.dataset.containerWidth = rect(".main-container").width;
    element.dataset.startWidth = element.getBoundingClientRect().width;
    document.documentElement.addEventListener("mousemove", resize, false);
    document.documentElement.addEventListener("mouseup", () => {
      document.documentElement.removeEventListener("mousemove", resize, false);
      element.style.userSelect = "initial";
    }, { once: true });
  });
}
appendResizer(document.querySelector(".ctx-list-parent"), { minWidth: 15, maxWidth: 50 }, left=true);
appendResizer(document.querySelector(".metadata-parent"), { minWidth: 20, maxWidth: 50 });

// **** keyboard shortcuts

const select = (ctx, step) => ({ ctx:document.getElementById(`ctx-${ctx}`), step:document.getElementById(`step-${ctx}-${step}`) });
const deselect = (element) => {
  const parts = element?.id.split("-").map(Number);
  return element?.id.startsWith("ctx") ? { ctx:parts[1], step:null } : element?.id.startsWith("step") ? {ctx:parts[1], step:parts[2]} : {};
}
document.addEventListener("keydown", (event) => {
  const { currentCtx, currentStep, currentRewrite, expandSteps } = state;
  // up and down change the step or context from the list
  const changeStep = expandSteps && ctxs[currentCtx].steps?.length;
  const { step, ctx } = select(currentCtx, currentStep);
  if (event.key == "ArrowUp") {
    event.preventDefault();
    if (changeStep) {
      let prev = deselect(step.previousElementSibling);
      if (prev.step == null && isExpanded(step.parentElement)) prev = deselect(step.parentElement);
      return prev.step != null && !isExpanded(step) && setState({ currentRewrite:0, currentStep:prev.step });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.max(0, currentCtx-1), expandSteps:false });
  }
  if (event.key == "ArrowDown") {
    event.preventDefault();
    if (changeStep) {
      const next = deselect(isExpanded(step) ? step.children[1] : step.nextElementSibling);
      return next.step != null && setState({ currentRewrite:0, currentStep:next.step });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.min(ctxs.length-1, currentCtx+1), expandSteps:false });
  }
  // enter toggles focus on a single rewrite stage
  if (event.key == "Enter") {
    event.preventDefault()
    if (currentCtx === -1) {
      return setState({ currentCtx:0, expandSteps:true });
    }
    if (expandSteps && getSubrewrites(step).length) return step.children[0].click();
    return setState({ expandSteps:!expandSteps });
  }
  // left and right go through rewrites in a single UOp, in profiler go forward/backward in time
  if (event.key == "ArrowLeft" || event.key == "ArrowRight") {
    event.preventDefault()
    if (profilerVisible() && moveProfilerFocus(event.key == "ArrowLeft" ? -1 : 1)) return;
    if (event.key == "ArrowLeft") return setState({ currentRewrite:Math.max(0, currentRewrite-1) });
    const totalRewrites = ret.length-1;
    return setState({ currentRewrite:Math.min(totalRewrites, currentRewrite+1) });
  }
  // space recenters the graph
  if (event.key == " ") {
    event.preventDefault()
    document.getElementById("zoom-to-fit-btn").click();
  }
  // r key toggles indexing
  if (event.key === "r") showIndexing.toggle.click();
  // c key toggles CALL src
  if (event.key === "c" && !event.ctrlKey && !event.metaKey && !event.altKey) showCallSrc.toggle.click();
  // s key toggles SINK
  if (event.key === "s") showSink.toggle.click();
  // g key toggles graph
  if (event.key === "g") showGraph.toggle.click();
});

export function startRewriteBrowser(svgZoom) {
  configureGraph({state, setState, switchCtx, showCallSrc, svgZoom});
  configureProfiler({saveToHistory, switchCtx, getContexts:() => ctxs, state});
  main();
}
