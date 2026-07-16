// ** profiler graph

import {Status, colored, darkenHex, displaySelection, fetchValue, isExpanded, metadata, parseColors, rect, traceBlock, updateProgress, vizZoomFilter} from "./ui.js";

const cache = {};
let browser = null;
export const configureProfiler = callbacks => browser = callbacks;

function formatParts(value, labels, last=true) {
  const hi = Math.floor(value / 1e6), mid = Math.floor((value % 1e6) / 1e3), low = Math.round(value % 1e3);
  const parts = [];
  if (hi) parts.push(`${hi}${labels[0]}`);
  if (mid || (!last && !hi)) parts.push(`${mid}${labels[1]}`);
  if (last && (low || (!mid && !hi))) parts.push(`${low}${labels[2]}`);
  return parts.join(' ');
}
export const formatMicroseconds = (ts, showUs=true) => formatParts(ts, ["s", "ms", "us"], showUs);
const formatCycles = cycles => formatParts(cycles, ["M", "K", ""]);

export const formatUnit = (d, unit="") => d3.format(".3~s")(d)+unit;
const skipFmt = new Set(["tb", "pc", "link"]);
const formatData = (fmt) => Object.entries(fmt ?? {}).filter(([k]) => !skipFmt.has(k)).map(([k, v]) => typeof(v) === "string" ? `${k} ${v}` : formatUnit(v, k));

const waveColor = (op) => {
  let ret = data.waveColors.find(([pattern]) => op.includes(pattern))?.[1] ?? "#ffffff";
  if (op.includes("OTHER_") || op.includes("_ALT")) { ret = darkenHex(ret, 75) }
  if (op.includes("LDS_")) { ret = darkenHex(ret, 25) }
  return ret
};
const colorScheme = {TINY:new Map([["Schedule","#1b5745"],["precompile","#1d2e62"],["compile","#63b0cd"],["DEFAULT","#354f52"]]),
  DEFAULT:["#2b2e39", "#2c2f3a", "#31343f", "#323544", "#2d303a", "#2e313c", "#343746", "#353847", "#3c4050", "#404459", "#444862", "#4a4e65"],
  BUFFER:["#342483", "#3E2E94", "#4938A4", "#5442B4", "#5E4CC2", "#674FCA"], SIMD:new Map([["OCC", "#101725"], ["INST", "#0A2042"]]),
  GPC:new Map([["NONE","#1a7a2e"],["MEMORY_DEPENDENCY","#8b1a00"],["EXEC_DEPENDENCY","#006b6b"],["INST_FETCH","#7a7a00"],["SYNC","#6b006b"],
    ["PIPE_BUSY","#7a4a00"],["MEMORY_THROTTLE","#5c0000"],["CONSTANT_MEMORY","#1a3d7a"],["NOT_SELECTED","#2e2e3a"],["OTHER","#4a4a55"],
    ["SLEEPING","#1a1a2a"],["DEFAULT","#3a3a45"]]), WAVE:waveColor, VMEMEXEC:waveColor, ALUEXEC:waveColor}
const cycleColors = (lst, i) => lst[i%lst.length];

const rescaleTrack = (source, tid, k) => {
  for (const shapes of source.views)
    for (const e of shapes) {
      for (let i=0; i<e.y0.length; i++) {
        e.y0[i] = e.y0[i]*k;
        e.y1[i] = e.y1[i]*k;
      }
    }
  const change = (source.height*k)-source.height;
  const div = document.getElementById(tid);
  div.style.height = rect(div).height+change+"px";
  source.height = source.height*k;
  return change;
}

const drawLine = (ctx, x, y, opts) => {
  ctx.beginPath();
  ctx.moveTo(x[0], y[0]);
  ctx.lineTo(x[1], y[1]);
  ctx.fillStyle = ctx.strokeStyle = opts?.color || "#f0f0f5";
  ctx.stroke();
}

var data, focusedDevice, focusedShape, formatTime, canvasZoom, zoomLevel = d3.zoomIdentity;

// Canvas shapes do not have browser-managed bounds, so the timeline owns this conversion.
const canvasRect = (s, pixelScale) => {
  const { e } = selectShape(s), t = data.tracks.get(s.split("-")[0]);
  const x = pixelScale(e.x), w = pixelScale(e.x+e.width)-x, y = t.offsetY+e.y;
  return {x0:x, x1:x+w, y0:y, y1:y+e.height};
};

const canvasDims = () => {
  const sideRect = rect("#device-list");
  return [Math.round(document.querySelector("#profiler").clientWidth-sideRect.width), Math.round(sideRect.height)];
}

function selectShape(key) {
  if (key == null) return {};
  const [t, idx] = key.split("-");
  const track = data.tracks.get(t);
  return { eventType:track?.eventType, e:track?.shapes[idx] };
}

// scaling function for time to pixels
const timelineScale = () => d3.scaleLinear().domain([data.first, data.dur]).range([0, canvasDims()[0]]);

function timeAtCycle(clk) {
  if (clk < data.instSt || clk > data.instEt || data.tracks.get("Shader Clock") == null) return "-";
  let cur = data.instSt, ns = 0, freq = null;
  // walk through all frequency changes and accumulate time in nanoseconds
  for (const [s, v] of data.tracks.get("Shader Clock").valueMap) {
    if (freq != null && freq > 0 && cur < s) {
      const et = Math.min(clk, s);
      ns += (et - cur) * 1e9 / freq;
      cur = et;
      if (cur === clk) break;
    }
    freq = v;
  }
  // ending cycles use the last known frequency
  if (cur < clk) ns += (clk - cur) * 1e9 / freq;
  const remNs = Math.round(ns % 1000);
  return ns/1000>1 ? formatMicroseconds(ns / 1000, true) + (remNs ? ` ${remNs}ns` : "") : Math.round(ns)+"ns";
}

function getZoomIdentity() {
  // for packets, set zoom to the full range of instruction events
  if (data.instSt != null) {
    const k = (data.dur - data.first) / (data.instEt - data.instSt), xscale = timelineScale();
    return d3.zoomIdentity.translate(-xscale(data.instSt) * k, 0).scale(k);
  }
  return d3.zoomIdentity;
}

const Modes = {0:'read', 1:'write', 2:'write+read'};

function setFocus(key) {
  if (key !== focusedShape) {
    browser.saveToHistory({ shape:focusedShape });
    // adjust zoom if the entire shape is off screen
    const { eventType, e } = selectShape(key);
    if (e != null) {
      const xscale = timelineScale();
      const [x0, x1] = eventType === EventTypes.EXEC ? [e.x, e.x+e.width] : [e.x[0], e.x.at(-1)];
      const [st, et] = xscale.range().map(zoomLevel.invertX, zoomLevel).map(xscale.invert, xscale);
      if (x1 < st || x0 > et) zoomLevel = d3.zoomIdentity.translate(-xscale((x0+x1)/2-(et-st)/2)*zoomLevel.k, 0).scale(zoomLevel.k);
    }
    const link = e?.arg.link ?? data.links.get(key);
    data.link = link == null ? null : [key, link];
    focusedShape = key; d3.select("#timeline").call(canvasZoom.transform, zoomLevel);
    const tooltip = document.getElementById("tooltip");
    if (tooltip.dataset.key !== key) tooltip.style.display = "none";
  }
  const { eventType, e } = selectShape(key);
  if (metadata.querySelector(".info") == null) d3.select(metadata).html("").append("div").classed("info", true);
  const html = d3.select(".info").html("");
  if (eventType === EventTypes.EXEC) {
    const [n, _, ...rest] = e.arg.tooltipText.split("\n");
    const tableData = [["Name", colored(e.arg.label)], ["Duration", formatTime(e.width)]];
    if (data.instSt != null) {
      const p = d3.create("p");
      p.append("span").text(timeAtCycle(e.x));
      p.append("span").style("margin-left", "8px").style("color", "#f0f0f566").text(formatTime(e.x));
      tableData.push(["Cycle", formatTime(e.x-data.instSt)], ["Time", p.node()]);
    } else tableData.push(["Start Time", formatTime(e.x)]);
    if (data.link != null) tableData.push(["Delay", `${formatTime(Math.abs(selectShape(data.link[0]).e.x - selectShape(data.link[1]).e.x))} Cycles`]);
    html.append(() => tabulate(tableData));
    let group = html.append("div").classed("args", true);
    for (const r of rest) group.append("p").text(r);
    group = html.append("div").classed("args", true);
    for (const b of e.arg.bufs.sort((a, b) => a.num - b.num)) {
      group.append("p").text(`${Modes[b.mode]}@data${b.num} ${formatUnit(b.nbytes, 'B')}`).style("cursor", "pointer").on("click", () => {
        const row = document.getElementById(b.k); if (!isExpanded(row)) { row.click(); }
        setFocus(b.key);
      });
    }
    if (e.arg.ctx != null) {
      const i = e.arg.ctx, s = e.arg.step;
      html.append("a").text(browser.getContexts()[i+1].steps[s].name).on("click", () => browser.switchCtx(i, s));
      const prgSrc = browser.getContexts()[i+1].steps.findIndex(s => s.name === "View Source");
      if (prgSrc !== -1) html.append("a").text("View Source").on("click", () => browser.switchCtx(i, prgSrc));
    }
    if (e.arg.trace != null) html.append(() => traceBlock(e.arg.trace.slice(1).reverse()));
  }
  if (eventType === EventTypes.BUF) {
    const [dtype, sz, nbytes, dur] = e.arg.tooltipText.split("\n");
    const rows = [["DType", dtype], ["Len", sz], ["Size", nbytes], ["Lifetime", dur]];
    if (e.arg.users != null) rows.push(["Users", e.arg.users.length]);
    html.append(() => tabulate(rows));
    const kernels = html.append("div").classed("args", true);
    for (let u=0; u<e.arg.users?.length; u++) {
      const { repr, num, mode, shape } = e.arg.users[u];
      const p = kernels.append("p").append(() => colored(`[${u}] ${repr} ${Modes[mode]}@data${num}`));
      const shapeInfo = selectShape(shape).e?.arg?.tooltipText?.split("\n");
      if (shapeInfo?.length > 5) p.append("span").text(" "+shapeInfo[5]);
      if (shape != null) p.style("cursor", "pointer").on("click", () => setFocus(shape));
    }
  }
  // instructions list renderer
  let instList = document.getElementById("insts");
  if (data.pcMap == null) return d3.select(instList?.parentElement).html("");
  if (instList == null) {
    let contents = "";
    for (let [pc, label] of Object.entries(data.pcMap)) {
      pc = parseInt(pc);
      const pcHex = pc.toString(16);
      contents += `<div class="line"><span class="left" id="inst-${pc}"><span class="pc">${"0x"+pcHex.padStart(Math.max(4, Math.ceil(pcHex.length/4)*4), 0)}</span><span class="label">${label}</span></span></div>`;
    }
    instList = d3.create("pre").append("code").classed("hljs", true).style("margin-top", "20px").attr("id", "insts").html(contents).node();
    metadata.insertBefore(instList.parentElement, html.node());
  }
  d3.select(instList).selectAll("span").classed("highlight", false);
  let instLine = document.getElementById(`inst-${e?.arg.pc}`);
  if (instLine == null && data.link != null) instLine = document.getElementById(`inst-${selectShape(data.link[1]).e.arg.pc}`);
  if (instLine != null) {
    instLine.classList.add("highlight");
    const r = rect(instLine), c = rect(instList);
    if (Math.max(c.top-r.bottom, r.top-c.bottom)>=-30) instList.scrollTop = instLine.offsetTop-instList.clientHeight/2+instLine.clientHeight/2;
  }
}

const EventTypes = { EXEC:0, BUF:1 };

export async function renderProfiler(path, opts) {
  displaySelection("#profiler");
  // support non realtime x axis units
  formatTime = opts.unit === "ms" ? formatMicroseconds : formatCycles;
  if (data?.path !== path) { data = {tracks:new Map(), axes:{}, path, first:null, links:new Map()}; focusedDevice = null; focusedShape = null; }
  setFocus(focusedShape);
  // layout once!
  if (data.tracks.size !== 0) return updateProgress(Status.COMPLETE);
  const profiler = d3.select("#profiler").html("");
  const buf = cache[path] ?? await fetchValue(path);
  const view = new DataView(buf);
  let offset = 0;
  const u8 = () => { const ret = view.getUint8(offset); offset += 1; return ret; }
  const u32 = () => { const ret = view.getUint32(offset, true); offset += 4; return ret; }
  const u64 = () => { const ret = new Number(view.getBigUint64(offset, true)); offset += 8; return ret; }
  const f32 = () => { const ret = view.getFloat32(offset, true); offset += 4; return ret; }
  const optional = (i) => i === 0 ? null : i-1;
  const dur = u32(), tracePeak = u64(), indexLen = u32(), layoutsLen = u32(); data.dur = dur;
  const textDecoder = new TextDecoder("utf-8");
  const { strings, dtypeSize, markers, ...extData } = JSON.parse(textDecoder.decode(new Uint8Array(buf, offset, indexLen))); offset += indexLen;
  for (const [k,v] of Object.entries(extData)) data[k] = v;
  // place devices on the y axis and set vertical positions
  const [tickSize, padding, baseOffset] = [5, 8, markers.length ? 14 : 0];
  const secondaryTick = opts.unit == "clk" ? timeAtCycle : null;
  const axisHeight = secondaryTick != null ? tickSize*2+(padding*2) : tickSize;
  const deviceList = profiler.append("div").attr("id", "device-list").style("padding-top", axisHeight+padding+baseOffset+"px");
  const canvas = profiler.append("canvas").attr("id", "timeline").node();
  // NOTE: scrolling via mouse can only zoom the graph
  canvas.addEventListener("wheel", e => (e.stopPropagation(), e.preventDefault()), { passive:false });
  const ctx = canvas.getContext("2d");
  const canvasTop = rect(canvas).top;
  // map event name to shape and label colors
  const colorMap = new Map(), coloredNames = new Map();
  // map shapes by event key
  const shapeMap = new Map();
  const maxheight = 100, heightScale = d3.scaleLinear().domain([0, tracePeak]).range([4, maxheight]);
  for (let i=0; i<layoutsLen; i++) {
    const nameLen = view.getUint8(offset, true); offset += 1;
    const k = textDecoder.decode(new Uint8Array(buf, offset, nameLen)); offset += nameLen;
    const div = deviceList.append("div").attr("id", k).text(k).style("padding", padding+"px").style("width", opts.width);
    const { y:baseY, height:baseHeight } = rect(div.node());
    const [dname, dnum] = k.split(":", 2);
    const colors = colorScheme[dname] ?? colorScheme.DEFAULT;
    const offsetY = baseY-canvasTop+padding/2;
    const shapes = [], visible = [];
    const eventType = u8(), eventsLen = u32();
    const [pcolor, scolor] = path.includes("sqtt") ? ["#00c72f", "#858b9d"] : ["#9ea2ad", null];
    // last row doesn't get a border
    const rowBorderColor = i<layoutsLen-1 ? "#22232a" : null;
    if (rowBorderColor != null) div.style("border-bottom", `1px solid ${rowBorderColor}`);
    if (eventType === EventTypes.EXEC) {
      const levelHeight = (baseHeight-padding)*(opts.heightScale ?? 1);
      const levels = [];
      data.tracks.set(k, { shapes, eventType, visible, offsetY, scolor, pcolor, rowBorderColor });
      let colorKey, ref;
      for (let j=0; j<eventsLen; j++) {
        const e = {name:strings[u32()], ref:optional(u32()), key:optional(u32()), st:u32(), dur:f32(), fmt:JSON.parse(strings[u32()])};
        // find a free level to put the event
        let depth = levels.findIndex(levelEt => e.st >= levelEt);
        const et = e.st+Math.trunc(e.dur);
        if (depth === -1) {
          depth = levels.length;
          levels.push(et);
        } else levels[depth] = et;
        if (depth === 0 || opts.colorByName) colorKey = e.name.split(" ")[0];
        if (!colorMap.has(colorKey)) {
          const color = typeof colors === "function" ? colors(colorKey)
                      : colors instanceof Map ? (colors.get(colorKey) || colors.get("DEFAULT")) : cycleColors(colors, colorMap.size);
          colorMap.set(colorKey, d3.rgb(color));
        }
        const fillColor = colorMap.get(colorKey).brighter(0.3*depth).toString();
        let label = coloredNames.get(e.name);
        if (label == null) {
          label = parseColors(e.name).flatMap(({ color, st }) => {
            const parts = [];
            for (let i=0; i<st.length; i+=4) { const part = st.slice(i, i+4); parts.push({ color, st:part, width:ctx.measureText(part).width }); }
            return parts;
          }); coloredNames.set(e.name, label);
        }
        let shapeRef = e.ref;
        if (shapeRef != null) { ref = {ctx:e.ref, step:0}; shapeRef = ref; }
        else if (ref != null) {
          const start = ref.step>0 ? ref.step+1 : 0;
          const steps = browser.getContexts()[ref.ctx+1].steps;
          for (let si=start; si<steps.length; si++) {
            if (steps[si].name == e.name) { ref.step = si; shapeRef = ref; break; }
          }
        } else {
          const steps = browser.getContexts()[browser.state.currentCtx].steps;
          for (let i=browser.state.currentStep+1; i<steps.length; i++) {
            const loc = steps[i].loc;
            if (loc == null) break;
            if (loc === e.name) { shapeRef = {ctx:browser.state.currentCtx-1, step:i}; break; }
          }
        }
        // tiny device events go straight to the rewrite rule
        const key = k.startsWith("TINY") ? null : `${k}-${j}`;
        const trace = e.fmt.tb, pc = e.fmt.pc, link = e.fmt.link;
        if (link != null) data.links.set(link, key);
        const arg = { tooltipText:[" N:"+shapes.length, formatTime(e.dur), ...formatData(e.fmt)].join("\n"), label, pc, trace, link, bufs:[], key, ctx:shapeRef?.ctx, step:shapeRef?.step };
        if (e.key != null) shapeMap.set(e.key, key);
        // offset y by depth
        shapes.push({x:e.st, y:levelHeight*depth, width:e.dur, height:levelHeight, arg, label:opts.hideLabels ? null : label, fillColor });
        if (j === 0) data.first = data.first == null ? e.st : Math.min(data.first, e.st);
      }
      div.style("height", levelHeight*levels.length+padding+"px").style("pointerEvents", "none");
    } else {
      const linear = u8(), peak = u64();
      const timestamps = [], valueMap = new Map();
      // start by unpacking the raw events
      const memEvents = [];
      let x = 0, y = 0, shapeIdx = 0;
      const allocs = new Map();
      for (let j=0; j<eventsLen; j++) {
        if (linear) { const ts = u32(), value = u64(); timestamps.push(ts); valueMap.set(ts, value); continue; }
        const alloc = u8(), ts = u32(), key = u32();
        if (alloc) {
          const dtype = strings[u32()], sz = u64(), nbytes = dtypeSize[dtype]*sz;
          allocs.set(key, {nbytes, shapeKey:`${k}-${shapeIdx++}`});
          memEvents.push({alloc, key, dtype, sz, nbytes});
          timestamps.push(ts);
          x += 1; y += nbytes; valueMap.set(ts, y);
        } else {
          const users = Array.from({ length: u32() }, () => ({shape:shapeMap.get(u32()), repr:strings[u32()], num:u32(), mode:u8()}));
          const {nbytes, shapeKey} = allocs.get(key); allocs.delete(key);
          users?.forEach((u) => selectShape(u.shape).e?.arg.bufs.push({ key:shapeKey, nbytes, num:u.num, mode:u.mode, k }));
          memEvents.push({alloc, key, users, nbytes});
          timestamps.push(ts); valueMap.set(ts, y);
          x += 1; y -= nbytes;
        }
      }
      timestamps.push(dur);
      const height = linear ? (baseHeight-padding)*(opts.heightScale ?? 1)*2 : heightScale(peak);
      const yscale = d3.scaleLinear().domain([0, peak]).range([height, 0]);
      // generic polygon merger
      const base0 = yscale(0);
      const sum = {x:[], y0:[], y1:[], fillColor:linear ? null : "#2b1b72"};
      for (let i=0; i<timestamps.length-1; i++) {
        const yv = yscale(valueMap.get(timestamps[i]));
        sum.x.push(timestamps[i], timestamps[i+1]); sum.y1.push(yv, yv); sum.y0.push(base0, base0);
      }
      // build individual buffer shapes when user clicks to expand, this detailed layout is n²
      let bufShapes = null;
      const buildBufShapes = () => {
        if (bufShapes != null) return bufShapes;
        bufShapes = [];
        const buf_shapes = new Map(), temp = new Map();
        let x = 0, y = 0;
        for (const e of memEvents) {
          if (e.alloc) {
            const shape = {x:[x], y:[y], dtype:e.dtype, sz:e.sz, nbytes:e.nbytes, key:e.key};
            buf_shapes.set(e.key, shape); temp.set(e.key, shape);
            x += 1; y += e.nbytes;
          } else {
            const free = buf_shapes.get(e.key);
            free.users = e.users;
            x += 1; y -= free.nbytes;
            free.x.push(x); free.y.push(free.y.at(-1));
            temp.delete(e.key);
            for (const [k, v] of temp) {
              if (k <= e.key) continue;
              v.x.push(x, x);
              v.y.push(v.y.at(-1), v.y.at(-1)-free.nbytes);
            }
          }
        }
        for (const [num, {dtype, sz, nbytes, y, x:steps, users}] of buf_shapes) {
          const x = steps.map(s => timestamps[s]);
          const dur = x.at(-1)-x[0];
          const arg = { tooltipText:`${dtype}\n${formatUnit(sz)}\n${formatUnit(nbytes, 'B')}\n${formatTime(dur)}`, users, key:`${k}-${bufShapes.length}` };
          bufShapes.push({ x, y0:y.map(yscale), y1:y.map(y0 => yscale(y0+nbytes)), arg, fillColor:cycleColors(colorScheme.BUFFER, bufShapes.length) });
        }
        return bufShapes;
      };
      if (timestamps.length > 0) data.first = data.first == null ? timestamps[0] : Math.min(data.first, timestamps[0]);
      data.tracks.set(k, { shapes:[sum], eventType, linear, visible, offsetY, pcolor:linear ? "#4fa3cc" : "#c9a8ff", height, peak, scaleFactor:maxheight*4/height,
                           get views() { return [[sum], linear ? null : buildBufShapes()]; }, valueMap, rowBorderColor, unit:linear ? "Hz" : "B" });
      div.style("height", height+padding+"px").style("cursor", "pointer").on("click", (e) => {
        if (linear) return;
        const newFocus = e.currentTarget.id === focusedDevice ? null : e.currentTarget.id;
        let offset = 0;
        for (const [tid, track] of data.tracks) {
          track.offsetY += offset;
          if (tid === newFocus) { track.shapes = track.views[1]; offset += rescaleTrack(track, tid, track.scaleFactor); }
          else if (tid === focusedDevice) { track.shapes = track.views[0]; offset += rescaleTrack(track, tid, 1/track.scaleFactor); }
        }
        const focusedTrack = data.tracks.get(newFocus);
        data.axes.y = newFocus != null ? {domain:[0, focusedTrack.peak], range:[focusedTrack.offsetY+focusedTrack.height, focusedTrack.offsetY], fmt:focusedTrack.unit} : null;
        toggleCls(document.getElementById(focusedDevice), document.getElementById(newFocus), "expanded");
        focusedDevice = newFocus;
        return resize();
      });
    }
  }
  for (const m of markers) m.label = m.name.split(/(\s+)/).map(st => ({ st, color:m.color, width:ctx.measureText(st).width }));
  if (data.pcMap != null) setFocus(focusedShape);
  // secondary axis mapping
  let instRange = null;
  for (const [k, { shapes }] of data.tracks) if (!k.includes("Clock") && path.includes("sqtt")) {
    const first = shapes[0].x, last = shapes.at(-1).x+shapes.at(-1).width;
    instRange = instRange == null ? [first, last] : [Math.min(first, instRange[0]), Math.max(last, instRange[1])];
  }
  if (instRange != null) [data.instSt, data.instEt] = instRange;
  updateProgress(Status.COMPLETE);
  // draw events on a timeline
  const dpr = window.devicePixelRatio || 1;
  const ellipsisWidth = ctx.measureText("...").width;
  const drawText = (ctx, label, lx, ly, maxWidth) => {
    let lw = 0;
    for (let li=0; li<label?.length; li++) {
      if (lw+label[li].width+(li===label.length-1 ? 0 : ellipsisWidth)+2 > maxWidth) {
        if (lw>0) ctx.fillText("...", lx+lw, ly);
        break;
      }
      ctx.fillStyle = label[li].color;
      ctx.fillText(label[li].st, lx+lw, ly);
      lw += label[li].width;
    }
  }
  function render(transform) {
    zoomLevel = transform;
    const canvasWidth = canvas.clientWidth;
    ctx.clearRect(0, 0, canvasWidth, canvas.clientHeight);
    // rescale to match current zoom
    const xscale = timelineScale();
    const visibleX = xscale.range().map(zoomLevel.invertX, zoomLevel).map(xscale.invert, xscale);
    const st = visibleX[0], et = visibleX[1];
    xscale.domain([st, et]);
    const profilerEl = profiler.node();
    const visibleYStart = profilerEl.scrollTop-canvasTop + rect(profilerEl).top, visibleYEnd = visibleYStart+profilerEl.clientHeight;
    ctx.textBaseline = "middle";
    // draw shapes
    for (const [k, { shapes, eventType, linear, visible, offsetY, valueMap, pcolor, scolor, unit, rowBorderColor }] of data.tracks) {
      visible.length = 0;
      const trackHeight = rect(document.getElementById(k)).height;
      if (offsetY+trackHeight < visibleYStart || offsetY > visibleYEnd) continue;
      const link0 = data.link?.[0]; const link1 = data.link?.[1], highlightRect = focusedShape != null || data.link != null, splitRects = scolor != null;
      if (eventType === EventTypes.BUF) { // generic polygon
        for (const e of shapes) {
          if (e.x[0]>et || e.x.at(-1)<st) continue;
          ctx.beginPath();
          const x = e.x.map(xscale);
          ctx.moveTo(x[0], offsetY+e.y1[0]);
          for (let i=1; i<x.length; i++) {
            ctx.lineTo(x[i], offsetY+e.y1[i]);
            let arg = e.arg;
            if (arg == null && valueMap != null) arg = {tooltipText: formatUnit(valueMap.get(e.x[i-1]), unit)}
            visible.push({ x0:x[i-1], x1:x[i], y0:offsetY+e.y1[i-1], y1:offsetY+e.y0[i], arg });
          }
          if (linear) { ctx.strokeStyle = pcolor; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
          // walk the path back and fill the complete shape
          else { for (let i=x.length-1; i>=0; i--) ctx.lineTo(x[i], offsetY+e.y0[i]); ctx.closePath(); ctx.fillStyle = e.fillColor; ctx.fill(); }
          if (focusedShape != null && e.arg?.key === focusedShape) { ctx.strokeStyle = pcolor; ctx.stroke(); }
        }
      } else { // contiguous rect
        for (const e of shapes) {
          if (e.x>et || e.x+e.width<st) continue;
          const x = xscale(e.x);
          const y = offsetY+e.y;
          const width = xscale(e.x+e.width)-x;
          visible.push({ y0:y, y1:y+e.height, x0:x, x1:x+width, arg:e.arg });
          ctx.fillStyle = e.fillColor;
          ctx.fillRect(x, y, width, e.height);
          // add label
          drawText(ctx, e.label, x+2, y+e.height/2, width);
          // draw highlights
          if (highlightRect) {
            const key = e.arg.key; if (key === focusedShape || key === link0 || key === link1) { ctx.strokeStyle = pcolor; ctx.strokeRect(x, y, width, e.height); continue; }
          }
          if (splitRects && width > 10) { ctx.strokeStyle = scolor; ctx.strokeRect(x, y, width, e.height); }
        }
      }
      // draw row line
      if (rowBorderColor != null) {
        const y = offsetY+trackHeight-padding/2 - 0.5;
        drawLine(ctx, [0, canvasWidth], [y, y], { color:rowBorderColor });
      }
    }
    // draw the link
    if (data.link != null) {
      const [a, b] = [canvasRect(data.link[0], xscale), canvasRect(data.link[1], xscale)];
      const [left, right] = a.x0 <= b.x0 ? [a, b] : [b, a];
      const startX = left.x1, endX = right.x0;
      const leftY = (left.y0+left.y1)/2, rightY = (right.y0+right.y1)/2;
      const dx = endX-startX, bend = Math.max(12, Math.min(40, dx/2));
      ctx.beginPath(); ctx.moveTo(startX, leftY); ctx.bezierCurveTo(startX+bend, leftY, endX-bend, rightY, endX, rightY); ctx.strokeStyle = "#858b9d"; ctx.stroke();
    }
    // draw axes
    ctx.translate(0, baseOffset);
    const y = secondaryTick != null ? tickSize+padding : 0;
    drawLine(ctx, xscale.range(), [y, y]);
    let lastLabelEnd = -Infinity;
    for (const tick of xscale.ticks()) {
      if (!Number.isInteger(tick)) continue;
      const x = xscale(tick);
      drawLine(ctx, [x, x], [y, y+tickSize]);
      const labelX = x+ctx.lineWidth+2;
      if (labelX <= lastLabelEnd) continue;
      const label = formatTime(tick, et-st <= 1e3);
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, y+tickSize);
      lastLabelEnd = labelX + ctx.measureText(label).width + 4;
      if (secondaryTick != null) {
        drawLine(ctx, [x, x], [y, y-tickSize]);
        const label = secondaryTick(tick, st, et); ctx.fillText(label, labelX, 0);
        lastLabelEnd = Math.max(lastLabelEnd, labelX + ctx.measureText(label).width + 4);
      }
    }
    if (data.axes.y != null) {
      drawLine(ctx, [0, 0], data.axes.y.range);
      const yscale = d3.scaleLinear().domain(data.axes.y.domain).range(data.axes.y.range);
      for (const tick of yscale.ticks()) {
        const y = yscale(tick);
        drawLine(ctx, [0, tickSize], [y, y]);
        ctx.textBaseline = "middle";
        ctx.fillText(formatUnit(tick, data.axes.y.fmt), tickSize+2, y);
      }
    }
    // draw markers
    ctx.translate(0, -baseOffset);
    ctx.textBaseline = "top";
    let prevX = null;
    for (let i=0; i<markers.length; i++) {
      const m = markers[i];
      const x = xscale(m.ts), tx = x+2;
      if (tx-prevX < 2) continue;
      prevX = tx;
      drawLine(ctx, [x, x], [0, canvas.clientHeight], { color:m.color });
      let maxWidth = canvasWidth-(tx);
      const nextMark = markers[i+1]?.ts;
      if (nextMark != null) maxWidth = Math.min(maxWidth, xscale(nextMark)-tx);
      if (maxWidth <= 0) continue;
      drawText(ctx, m.label, tx, 1, maxWidth);
    }
  }

  function resize() {
    const [width, height] = canvasDims();
    if (canvas.width === width*dpr && canvas.height === height*dpr) return;
    canvas.width = width*dpr;
    canvas.height = height*dpr;
    canvas.style.height = `${height}px`;
    canvas.style.width = `${width}px`;
    ctx.scale(dpr, dpr);
    d3.select(canvas).call(canvasZoom.transform, zoomLevel);
  }

  zoomLevel = getZoomIdentity();
  canvasZoom = d3.zoom().filter(vizZoomFilter).on("zoom", e => render(e.transform));
  d3.select(canvas).call(canvasZoom);
  document.addEventListener("contextmenu", e => e.ctrlKey && e.preventDefault());

  new ResizeObserver(([e]) => e.contentRect.width > 0 && resize()).observe(profiler.node());
  profiler.on("scroll", () => render(zoomLevel));

  function findRectAtPosition(x, y) {
    let track = null;
    for (const k of data.tracks.keys()) {
      const r = rect(document.getElementById(k));
      if (y >= r.y && y <= r.y+r.height) { track = data.tracks.get(k); break; }
    }
    if (track == null) return;
    const R = rect(canvas);
    const X = ((x-R.left) * (canvas.width/R.width))/dpr;
    const Y = ((y-R.top) * (canvas.height/R.height))/dpr;
    for (const r of track.visible) {
      if (Y>=r.y0 && Y<=r.y1 && X>=r.x0 && X<=r.x1) return r.arg;
    }
  }

  const clickShape = (e) => {
    e.preventDefault();
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.step != null && (foundRect?.key == null || e.type == "dblclick")) { return browser.switchCtx(foundRect.ctx, foundRect.step); }
    if (foundRect?.key != focusedShape) { setFocus(foundRect?.key); }
  }
  canvas.addEventListener("click", clickShape);
  canvas.addEventListener("dblclick", clickShape);

  canvas.addEventListener("mousemove", e => {
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    const tooltip = document.getElementById("tooltip");
    if (foundRect?.tooltipText != null) {
      tooltip.replaceChildren(colored(foundRect.label||[]), document.createTextNode(foundRect.tooltipText));
      tooltip.style.display = "block";
      tooltip.style.left = (e.pageX+10)+"px";
      tooltip.style.top = (e.pageY)+"px";
      tooltip.dataset.key = foundRect.key ?? "";
    } else tooltip.style.display = "none";
  });
  canvas.addEventListener("mouseleave", () => document.getElementById("tooltip").style.display = "none");
}

export const profilerVisible = () => document.getElementById("profiler").style.display !== "none";
export const restoreProfilerFocus = (shape) => setFocus(shape);
export function moveProfilerFocus(direction) {
  if (focusedShape == null) return false;
  const [t, idx] = focusedShape.split("-");
  const i = parseInt(idx), last = data.tracks.get(t).shapes.length-1;
  setFocus(`${t}-${direction < 0 ? Math.max(0, i-1) : Math.min(last, i+1)}`);
  return true;
}
export function fitProfiler() {
  const canvas = d3.select("#timeline");
  if (canvas.empty() || rect(canvas.node()).width === 0) return false;
  canvas.call(canvasZoom.transform, getZoomIdentity());
  return true;
}
