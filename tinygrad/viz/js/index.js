import {rect, vizZoomFilter} from "./ui.js";
import {fitProfiler} from "./profiler.js";
import {startRewriteBrowser} from "./rewrite.js";

const svgZoom = d3.zoom().filter(vizZoomFilter).on("zoom", (e) => d3.select("#render").attr("transform", e.transform));
d3.select("#graph-svg").call(svgZoom);

document.getElementById("zoom-to-fit-btn").addEventListener("click", () => {
  if (fitProfiler()) return;
  const svg = d3.select("#graph-svg");
  svg.call(svgZoom.transform, d3.zoomIdentity);
  const mainRect = rect(".main-container");
  const x0 = rect(".ctx-list-parent").right;
  const x1 = rect(".metadata-parent").left;
  const pad = 16;
  const target = {x:x0+pad, y:mainRect.top+pad, width:(x1 > 0 ? x1-x0 : mainRect.width)-2*pad, height:mainRect.height-2*pad};
  const rendered = rect("#render");
  if (rendered.width === 0) return;
  const scale = Math.min(target.width/rendered.width, target.height/rendered.height);
  const tx = target.x+(target.width-rendered.width*scale)/2-rendered.left*scale;
  const ty = target.y+(target.height-rendered.height*scale)/2;
  svg.call(svgZoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
});

startRewriteBrowser(svgZoom);
