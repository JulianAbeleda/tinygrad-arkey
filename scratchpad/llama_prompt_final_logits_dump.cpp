// Diagnostic-only shared-token prompt-final logits dumper for the pinned CUDA llama.cpp build.
// Input tokens are deliberately supplied as integer IDs; this never tokenizes text itself.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> csv(const char * path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open token file");
  std::vector<int32_t> out; std::string line, cell;
  std::getline(in, line); std::stringstream ss(line);
  while (std::getline(ss, cell, ',')) out.push_back(std::stoi(cell));
  return out;
}

int main(int argc, char ** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: %s MODEL TOKENS_CSV OUT_JSON\n", argv[0]); return 2; }
  auto tokens = csv(argv[2]);
  if (tokens.empty()) throw std::runtime_error("empty prompt");
  ggml_backend_load_all();
  llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
  llama_model * model = llama_model_load_from_file(argv[1], mp);
  if (!model) throw std::runtime_error("llama model load failed");
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = std::max<int>(1024, tokens.size()+1); cp.n_batch = tokens.size(); cp.n_ubatch = tokens.size(); cp.no_perf = true;
  llama_context * ctx = llama_init_from_model(model, cp);
  if (!ctx) throw std::runtime_error("llama context init failed");
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
  if (llama_decode(ctx, batch)) throw std::runtime_error("llama_decode failed");
  const float * logits = llama_get_logits_ith(ctx, -1);
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  int argmax = 0; for (int i=1; i<vocab; i++) if (logits[i] > logits[argmax]) argmax=i;
  int runner_up = argmax == 0 ? 1 : 0;
  for (int i=0; i<vocab; i++) if (i != argmax && logits[i] > logits[runner_up]) runner_up=i;
  // Fixed distributed sample makes a compact, stable comparison; include the winner too.
  const int ids[] = {0,1,2,3,10,42,151,256,512,1024,2048,4096,8192,16384,32768,49152,65536,81920,98304,114688,131072,147456,151935};
  std::ofstream out(argv[3]);
  out.precision(9);
  out << "{\"engine\":\"llama.cpp CUDA\",\"prompt_tokens\":" << tokens.size()
      << ",\"vocab\":" << vocab << ",\"argmax\":" << argmax << ",\"runner_up\":" << runner_up
      << ",\"argmax_logit\":" << logits[argmax] << ",\"runner_up_logit\":" << logits[runner_up]
      << ",\"selected\":{\"" << argmax << "\":" << logits[argmax];
  for (int id: ids) if (id < vocab && id != argmax) out << ",\"" << id << "\":" << logits[id];
  out << "}}\n";
  // Full-vocabulary raw f32 sidecar is intentionally binary, not a giant JSON
  // payload. It permits aggregate and top-k comparison without sampling bias.
  std::ofstream raw(std::string(argv[3]) + ".f32", std::ios::binary);
  raw.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
  llama_free(ctx); llama_model_free(model);
  return 0;
}
