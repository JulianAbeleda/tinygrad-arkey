import {Status, displaySelection, parseColors, darkenHex, updateProgress, codeBlock} from "./ui.js";

let browser = null;
export const configureGraph = callbacks => browser = callbacks;

function intersectRect(r1, r2) {
  const dx = r2.x-r1.x;
  const dy = r2.y-r1.y;
  if (dx === 0 && dy === 0) throw new Error("Invalid node coordinates, rects must not overlap");
  const scaleX = dx !== 0 ? (r1.width/2)/Math.abs(dx) : Infinity;
  const scaleY = dy !== 0 ? (r1.height/2)/Math.abs(dy) : Infinity;
  const scale = Math.min(scaleX, scaleY);
  return {x:r1.x+dx*scale, y:r1.y+dy*scale};
}

function addTags(root, path) {
  root.selectAll("circle").data(d => d.rect ? [] : [d]).join("circle").attr("r", 5).style("fill", d => d.fill ?? null).style("stroke", d => d.stroke ?? null);
  root.selectAll("rect").data(d => d.rect ? [d] : []).join("rect").attr("x", d => -d.width/2).attr("y", d => -d.height/2)
    .attr("width", d => d.width).attr("height", d => d.height).style("fill", d => d.fill ?? null).style("stroke", d => d.stroke ?? null);
  if (path != null) root.selectAll("path").data(d => [d]).join("path").attr("d", path);
  else root.selectAll("text").data(d => [d]).join("text").text(d => d.text).attr("dy", "0.35em");
}

let anchor = null;
const drawGraph = (data) => {
  const g = dagre.graphlib.json.read(data);
  // draw nodes
  d3.select("#graph-svg").on("click", () => d3.selectAll(".highlight").classed("highlight", false));
  const callCount = g.graph().callCount;
  const nodes = d3.select("#nodes").selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g").attr("class", d => d.className ?? "node")
    .attr("transform", d => `translate(${d.x},${d.y})`).on("click", (e,d) => {
      const parents = g.predecessors(d.id);
      const children = g.successors(d.id);
      if (parents == null && children == null) return;
      const src = [...parents, ...children, d.id];
      nodes.classed("highlight", n => src.includes(n.id)).classed("child", n => children.includes(n.id));
      if (!e.target.classList.contains("token")) labels.selectAll("rect.bg").classed("highlight", false);
      const matchEdge = (v, w) => (v===d.id && children.includes(w)) ? "highlight child " : (parents.includes(v) && w===d.id) ? "highlight " : "";
      d3.select("#edges").selectAll("path.edgePath").attr("class", e => matchEdge(e.v, e.w)+"edgePath");
      d3.select("#edge-labels").selectAll("g.port").attr("class",  (_, i, n) => matchEdge(...n[i].id.split("-"))+"port");
      e.stopPropagation();
    });
  nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
    .attr("x", d => -d.width/2).attr("y", d => -d.height/2).classed("node", true);
  const STROKE_WIDTH = 1.4, textSpace = g.graph().textSpace;
  const labels = nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label");
  labels.attr("transform", d => `translate(${d.labelX-d.labelWidth/2}, -${d.labelHeight/2+STROKE_WIDTH*2})`);
  const rectGroup = labels.selectAll("g.rect-group").data(d => [d]).join("g").attr("class", "rect-group");
  const tokens = labels.selectAll("g.text-group").data(d => [d]).join("g").attr("class", "text-group").selectAll("text").data(d => {
    if (Array.isArray(d.label)) return [d.label];
    const ret = [[]];
    for (const s of parseColors(d.label, "initial")) {
      const color = darkenHex(s.color, 25);
      const lines = s.st.split("\n");
      ret.at(-1).push({ st:lines[0], color });
      for (let i=1; i<lines.length; i++) ret.push([{ st:lines[i], color }]);
    }
    return [ret];
  }).join("text").style("font-family", g.graph().font).selectAll("tspan").data(d => d).join("tspan").attr("x", "0").attr("dy", g.graph().lh)
    .selectAll("tspan").data(d => d).join("tspan").attr("dx", (d, i) => i > 0 && d.st !== "," ? textSpace: 0).text(d => d.st).classed("token", true)
    .attr("xml:space", "preserve").attr("fill", d => d.color);
  const tokensBg = rectGroup.selectAll("rect.bg").data((d, i, nodes) => {
    const ret = [];
    d3.select(nodes[i].parentElement).select("g.text-group").selectAll("tspan.token").each((d, i, nodes) => {
      if (!d.keys?.length) return;
      const b = nodes[i].getBBox(); ret.push({ keys:d.keys, x:b.x, y:b.y, width:b.width, height:b.height });
    });
    return ret;
  }).join("rect").attr("class", "bg").attr("x", d => d.x).attr("y", d => d.y).attr("width", d => d.width).attr("height", d => d.height);
  tokens.on("click", (e, { keys }) => {
    tokensBg.classed("highlight", (d, i, nodes) => !nodes[i].classList.contains("highlight") && d.keys.some(k => keys?.includes(k)));
  });
  addTags(nodes.selectAll("g.tag").data(d => d.tag != null ? [d] : []).join("g").attr("class", "tag")
    .attr("transform", d => `translate(${-d.width/2+8}, ${-d.height/2+8})`).datum(e => ({ text:e.tag })));
  addTags(nodes.selectAll("g.addrspace").data(d => d.addrspace != null ? [d] : []).join("g").attr("class", "tag addrspace")
    .attr("transform", d => `translate(${d.width/2-8}, ${-d.height/2+8})`).datum(e => ({ rect:true, width:10, height:10, fill:e.addrspace, stroke:"none" })));
  const CALL_TAG_WIDTH = 14;
  addTags(nodes.selectAll("g.type").data(d => d.collapsible ? [d] : []).join("g").attr("class", d => `tag clickable ${d.collapsed ? 'collapsed' : 'expanded'}`)
    .attr("transform", d => d.callNode ? `translate(${CALL_TAG_WIDTH/2-d.width/2}, ${0})` : `translate(${-d.width/2}, ${0})`)
    .datum(d => ({ ...d, text:d.collapsed ? "+" : "−", fill:d.callNode ? null : d.color,
      ...(d.callNode && { rect:true, width:CALL_TAG_WIDTH }) })).on("click", (e,d) => {
      e.stopPropagation();
      const t = d3.zoomTransform(document.getElementById("graph-svg"));
      const [x, y] = t.apply([d.x, d.y]);
      anchor = {id:d.id, x, y, k:t.k};
      if (d.callNode) {
        if (browser.state.callSrcMask.has(d.id)) browser.state.callSrcMask.delete(d.id); else browser.state.callSrcMask.add(d.id);
        if (browser.state.callSrcMask.size >= callCount) { browser.showCallSrc.toggle.checked = !browser.showCallSrc.toggle.checked; browser.state.callSrcMask.clear(); }
      } else { if (browser.state.expandedNodes.has(d.id)) browser.state.expandedNodes.delete(d.id); else browser.state.expandedNodes.add(d.id); }
      return browser.setState({});
    }));
  addTags(nodes.selectAll("g.ref").data(d => d.ref != null ? [d] : []).join("g").attr("class", "tag ref")
    .attr("transform", d => `translate(${d.width/2-2}, ${-d.height/2+2})`).on("click", (e,d) => { e.stopPropagation(); browser.switchCtx(d.ref); }).datum(d => ({ref:d.ref})),
    "M-1.7 1.7 L1.7 -1.7 M-0.55 -1.7 H1.7 V0.55");
  // draw edges
  const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis), edges = g.edges();
  d3.select("#edges").selectAll("path.edgePath").data(edges).join("path").attr("class", "edgePath").attr("d", (e) => {
    const edge = g.edge(e);
    const points = edge.points.slice(1, edge.points.length-1);
    points.unshift(intersectRect(g.node(e.v), points[0]));
    points.push(intersectRect(g.node(e.w), points[points.length-1]));
    return line(points);
  }).attr("marker-end", "url(#arrowhead)").attr("stroke", e => g.edge(e).color || "#4a4b57");
  return g;
}

// ** UOp graph

let workerUrl = null, worker = null;
export const graphWorkerReady = () => workerUrl != null;
export async function initWorker() {
  const resp = await Promise.all(["/assets/dagrejs.github.io/project/dagre/latest/dagre.min.js","/js/worker.js"].map(u => fetch(u)));
  workerUrl = URL.createObjectURL(new Blob([(await Promise.all(resp.map((r) => r.text()))).join("\n")], { type: "application/javascript" }));
}

export function renderDag(layoutSpec, { recenter }) {
  // start calculating the new layout (non-blocking)
  updateProgress(Status.STARTED, "Rendering new graph...");
  if (worker != null) worker.terminate();
  worker = new Worker(workerUrl);
  worker.postMessage(layoutSpec);
  worker.onmessage = (e) => {
    if (e.data.error) {
      updateProgress(Status.ERR, "Error in graph layout:\n"+e.data.error);
      return;
    }
    const data = e.data.result;
    displaySelection("#graph");
    updateProgress(Status.COMPLETE);
    const g = drawGraph(data);
    addTags(d3.select("#edge-labels").selectAll("g").data(data.edges).join("g").attr("transform", (e) => {
      // get a point near the end
      const [p1, p2] = e.value.points.slice(-2);
      const dx = p2.x-p1.x;
      const dy = p2.y-p1.y;
      // normalize to the unit vector
      const len = Math.sqrt(dx*dx + dy*dy);
      const ux = dx / len;
      const uy = dy / len;
      // avoid overlap with the arrowhead
      const offset = 17;
      const x = p2.x - ux * offset;
      const y = p2.y - uy * offset;
      return `translate(${x}, ${y})`
    }).attr("class", e => e.value.label.type).attr("id", e => `${e.v}-${e.w}`).datum(e => ({ text:e.value.label.text })));
    if (anchor != null) {
      const n = g.node(anchor.id);
      d3.select("#graph-svg").call(browser.svgZoom.transform, d3.zoomIdentity.translate(anchor.x-n.x*anchor.k, anchor.y-n.y*anchor.k).scale(anchor.k));
    } else if (recenter) document.getElementById("zoom-to-fit-btn").click();
    anchor = null;
  };
  worker.onerror = (e) => {
    e.preventDefault();
    updateProgress(Status.ERR, "Error in graph layout:\n"+e.message);
  }
}
