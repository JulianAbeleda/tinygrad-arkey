// ** graph helpers

export const displaySelection = (sel) => {
  for (const e of document.getElementsByClassName("view")) e.style.display = e.matches(sel) ? "flex" : "none";
}
export const metadata = document.querySelector(".metadata");

export const darkenHex = (h, p = 0) => {
  const c = parseInt(h.slice(1), 16), f = 1-p/100;
  const rgb = ((c >> 16 & 255)*f | 0) << 16 | ((c >> 8 & 255)*f | 0) << 8 | ((c & 255)*f | 0);
  return `#${rgb.toString(16).padStart(6, "0")}`;
};

const ANSI_COLORS = ["#b3b3b3", "#ff6666", "#66b366", "#ffff66", "#6666ff", "#ff66ff", "#66ffff", "#ffffff"],
  ANSI_COLORS_LIGHT = ["#d9d9d9","#ff9999","#99cc99","#ffff99","#9999ff","#ff99ff","#ccffff","#ffffff"];
export const parseColors = (name, defaultColor="#ffffff") => Array.from(name.matchAll(/(?:\u001b\[(\d+)m([\s\S]*?)\u001b\[0m)|([^\u001b]+)/g),
  ([_, code, colored_st, st]) => ({ st: colored_st ?? st, color: code != null ? (code>=90 ? ANSI_COLORS_LIGHT : ANSI_COLORS)[(parseInt(code)-30+60)%60] : defaultColor }));

export const colored = n => d3.create("span").call(s => s.selectAll("span").data(typeof n === "string" ? parseColors(n) : n).join("span")
                       .style("color", d => d.color).text(d => d.st)).node();

export const rect = (s) => (typeof s === "string" ? document.querySelector(s) : s).getBoundingClientRect();
export const isExpanded = (el) => el?.classList.contains("expanded");
export const vizZoomFilter = e => (!e.ctrlKey || e.type === "wheel" || e.type === "mousedown") && !e.button && e.type !== "dblclick";
export const fetchValue = async path => fetch(path).then(res => res.headers.get("content-type") === "application/json" ? res.json() : res.arrayBuffer());
export const tabulate = rows => {
  const root = d3.create("div").style("display", "grid").style("grid-template-columns", `${Math.max(...rows.map(x => x[0].length), 0)}ch 1fr`).style("gap", "0.2em").style("white-space", "nowrap");
  for (const [k,v] of rows) { root.append("div").text(k); root.append("div").node().append(v); }
  return root.node();
};

export const state = {currentCtx:-1, currentStep:0, currentRewrite:0, expandSteps:false, callSrcMask:new Set(), expandedNodes:new Set()};
export const saveToHistory = value => (history.replaceState(value, ""), history.pushState(value, ""));

let timeout = null;
export const Status = {STARTED:0, COMPLETE:1, ERR:2}
export const updateProgress = (st, msg) => {
  clearTimeout(timeout);
  const msgEl = d3.select("#progress-message").style("display", "none"), customEl = d3.select("#custom").style("display", "none");
  if (st === Status.STARTED) {
    msgEl.text(msg);
    timeout = setTimeout(() => msgEl.style("display", "block"), 2000);
  } else if (st === Status.ERR) {
    displaySelection("#custom");
    customEl.html("").append("div").classed("raw-text", true).append(() => codeBlock(msg));
  }
}

const pathLink = (fp, lineno) => d3.create("a").attr("href", "vscode://file/"+fp+":"+lineno).text(`${fp.split("/").at(-1)}:${lineno}`);
export function codeBlock(st, language, { loc, wrap }={}) {
  const code = document.createElement("code");
  // plaintext renders like a terminal print, otherwise render with syntax highlighting
  if (!language || language === "txt") code.appendChild(colored(st));
  else code.innerHTML = hljs.highlight(st, { language }).value;
  code.className = "hljs";
  const ret = document.createElement("pre");
  if (wrap) ret.className = "wrap";
  if (loc != null) ret.appendChild(pathLink(loc[0], loc[1]).style("margin-bottom", "4px").node());
  ret.appendChild(code);
  return ret;
}
export function traceBlock(trace) {
  const root = d3.create("pre").append("code").classed("hljs", true);
  for (let i=trace.length-1; i>=0; i--) {
    const [fp, lineno, fn, code] = trace[i];
    root.append("div").style("margin-bottom", "2px").style("display","flex").text(fn+" ").append(() => pathLink(fp, lineno).node());
    root.append("div").html(hljs.highlight(code, { language: "python" }).value).style("margin-bottom", "1ex");
  }
  return root.node().parentNode;
}

export function toggleCls(prev, next, cls, value) {
  prev?.classList.remove(cls);
  next?.classList.toggle(cls, value ?? true);
  requestAnimationFrame(() => next?.scrollIntoView({ behavior: "auto", block: "nearest" }));
}

// ** hljs extra definitions for UOps and float4
hljs.registerLanguage("python", (hljs) => ({
  ...hljs.getLanguage("python"),
  case_insensitive: false,
  contains: [
    { begin: 'dtypes\\.[a-zA-Z_][a-zA-Z0-9_-]*(\\.[a-zA-Z_][a-zA-Z0-9_-]*)*' + '(?=[.\\s\\n[:,(])', className: "type" },
    { begin: 'dtypes\\.[a-zA-Z_][a-zA-Z0-9_-].vec*' + '(?=[.\\s\\n[:,(])', className: "type" },
    { begin: '[a-zA-Z_][a-zA-Z0-9_-]*\\.[a-zA-Z_][a-zA-Z0-9_-]*' + '(?=[.\\s\\n[:,()])',  className: "operator" },
    { begin: '[A-Z][a-zA-Z0-9_]*(?=\\()', className: "section", ignoreEnd: true },
    ...hljs.getLanguage("python").contains,
  ]
}));
hljs.registerLanguage("cpp", (hljs) => ({
  ...hljs.getLanguage('cpp'),
  contains: [{ begin: '\\b(?:float|half)[0-9]+\\b', className: 'type' }, ...hljs.getLanguage('cpp').contains]
}));
