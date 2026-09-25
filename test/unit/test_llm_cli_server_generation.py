from types import SimpleNamespace
import threading

from tinygrad.llm.cli import Handler


class MutatingModel:
  def get_start_pos(self, tokens): return 0
  def generate(self, tokens, **kwargs):
    tokens.append(9)
    yield 9


class Tokenizer:
  def stream_decoder(self): return lambda token_id=None: "x" if token_id is not None else ""
  def is_end(self, token_id): return False


def test_stream_generation_does_not_mutate_prompt_usage():
  metrics = {"last_prompt_tokens": None, "last_completion_tokens": None, "last_cached_prefix_tokens": None,
             "last_prefill_tok_s": None, "last_decode_tok_s": None, "last_finish_reason": None}
  state = SimpleNamespace(model=MutatingModel(), tok=Tokenizer(), prefill_chunk_size=64, metrics=metrics,
                          remote_metrics=False, cancel_event=threading.Event())
  handler = Handler.__new__(Handler)
  handler.server, handler.path = SimpleNamespace(state=state), "/v1/chat/completions"
  prompt = [1, 2, 3]
  assert list(handler._stream_tokens(prompt, 1, 0.0)) == [("delta", "x"), ("finish", "length")]
  assert prompt == [1, 2, 3]
  assert metrics["last_prompt_tokens"] == 3
  assert metrics["last_completion_tokens"] == 1

# Native template fixture copied verbatim from Qwen/Qwen3-0.6B-GGUF Q8_0 metadata.
# This tests the real tokenizer/template and HTTP boundary; only model token generation is scripted.
import json
from pathlib import Path
import urllib.request
import urllib.error
import pytest
from tinygrad.llm.chat import NativeChat, ChatError
from tinygrad.llm.cli import LLMServer
from tinygrad.llm.runtime_state import SimpleTokenizer
from extra.llm.bench.qwen_lora_smoke_train import build_completion_examples

TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {
  'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
MESSAGES = [{'role': 'user', 'content': 'Read café.txt.'}]
CALL = '<tool_call>\n{"name":"read_file","arguments":{"path":"café.txt"}}\n</tool_call>'


def native_tokenizer():
  bs = [*range(33,127), *range(161,173), *range(174,256)]
  mapping = {b:chr(b) for b in bs} | {b:chr(256+i) for i,b in enumerate(b for b in range(256) if b not in bs)}
  return SimpleTokenizer({mapping[b]:b for b in range(256)}, {'<|im_start|>':256, '<|im_end|>':257},
    preset='qwen2', bos_id=None, eos_id=257, architecture='qwen3',
    chat_template=(Path(__file__).parents[1]/'fixtures/llm/qwen3_chat.jinja').read_text())


def test_native_training_uses_exact_serving_prefix_and_eos_with_observation():
  tok = native_tokenizer(); protocol = NativeChat(tok)
  call = protocol.parse(CALL, TOOLS, 'stop')
  history = MESSAGES + [call, {'role':'tool', 'tool_call_id':call['tool_calls'][0]['id'], 'content':'{"text":"雪"}'}]
  for messages in (MESSAGES + [call], history + [{'role':'assistant','content':'The file says 雪.'}]):
    prefix = protocol.prompt({'messages':messages[:-1], 'tools':TOOLS})
    row = {'id':'a','source_id':'verified-a','messages':messages,'tools':TOOLS}
    examples = build_completion_examples([row], tok, completion_scope='all')
    assert examples[0]['tokens'] == prefix
    targets = [e['target'] for e in examples]
    assert targets[-1] == tok.eos_id
    assert examples[-1]['tokens'] == prefix + targets[:-1]
    assert 'café.txt' in tok.decode(prefix)
    if len(messages) > 2: assert '<tool_response>\n{"text":"雪"}' in tok.decode(prefix)


@pytest.mark.parametrize('raw,finish', [(CALL,'length'), (CALL+' text','stop'), ('text '+CALL,'stop'),
  (CALL+CALL,'stop'), (CALL.replace('read_file','unknown'),'stop'), (CALL[:-3],'stop'),
  ('<tool_call>{"name":"read_file","arguments":[]}</tool_call>','stop'),
  ('<tool_call>{"name":"read_file","arguments":{"path":NaN}}</tool_call>','stop'),
  ('<tool_call>{"name":"read_file","name":"other","arguments":{}}</tool_call>','stop')])
def test_native_rejects_non_executable_output(raw, finish):
  with pytest.raises(ChatError): NativeChat(native_tokenizer()).parse(raw, TOOLS, finish)


def test_native_rejects_bad_history_and_unsupported_controls():
  protocol = NativeChat(native_tokenizer())
  call = protocol.parse(CALL, TOOLS, 'stop')
  invalid = [MESSAGES+[call], MESSAGES+[{'role':'tool','tool_call_id':'absent','content':'x'}],
             MESSAGES+[call, {'role':'tool','tool_call_id':'wrong','content':'x'}]]
  for messages in invalid:
    with pytest.raises(ChatError): protocol.prompt({'messages':messages,'tools':TOOLS})
  for option in ({'tool_choice':'required'}, {'tool_choice':'none'}, {'parallel_tool_calls':True}):
    with pytest.raises(ChatError): protocol.prompt({'messages':MESSAGES,'tools':TOOLS, **option})
  tok = native_tokenizer(); tok.architecture='llama'
  with pytest.raises(ChatError, match='unsupported'): NativeChat(tok)
  tok.chat_template=None
  with pytest.raises(ChatError, match='no native'): NativeChat(tok)


class ScriptedTokens:
  def __init__(self, tok, text): self.tok, self.text, self.prompts = tok, text, []
  def get_start_pos(self, ids): return 0
  def generate(self, ids, **kwargs):
    self.prompts.append(list(ids))
    yield from self.tok.encode(self.text)
    yield self.tok.eos_id


@pytest.fixture
def native_server():
  tok = native_tokenizer()
  state = SimpleNamespace(model=ScriptedTokens(tok,CALL), tok=tok, prefill_chunk_size=64, metrics={}, remote_metrics=False,
    cancel_event=threading.Event(), loaded=True, gen_lock=threading.Lock(), request_count=0, max_context=8192,
    default_max_tokens=512, model_id='fixture')
  server = LLMServer(('127.0.0.1',0),state)
  thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
  def request(body):
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req,timeout=5).read().decode()
  try: yield state,request
  finally: server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_native_http_stream_nonstream_and_real_observation_continuation(native_server):
  state, request = native_server
  body = {'messages':MESSAGES,'tools':TOOLS,'temperature':0,'parallel_tool_calls':False}
  result = json.loads(request(body)); choice=result['choices'][0]
  assert choice['finish_reason']=='tool_calls'
  call=choice['message']['tool_calls'][0]
  assert json.loads(call['function']['arguments'])=={'path':'café.txt'}
  chunks = [json.loads(line[6:]) for line in request(dict(body,stream=True)).splitlines() if line.startswith('data: ') and '[DONE]' not in line]
  assert chunks[0]['choices'][0]['delta']['tool_calls'][0]['index']==0
  assert chunks[-1]['choices'][0]['finish_reason']=='tool_calls'
  state.model.text='Observed 雪.'
  continuation = dict(body,messages=MESSAGES+[choice['message'],{'role':'tool','tool_call_id':call['id'],'content':'雪'}])
  result=json.loads(request(continuation))
  assert result['choices'][0]['message']['content']=='Observed 雪.'
  assert '<tool_response>\n雪\n</tool_response>' in state.tok.decode(state.model.prompts[-1])
  state.model.text=CALL[:-5]
  with pytest.raises(urllib.error.HTTPError) as failure: request(body)
  assert failure.value.code==502
  assert json.loads(failure.value.read())['error']['type']=='invalid_model_output'
  with pytest.raises(urllib.error.HTTPError) as failure: request(dict(body,tool_choice='required'))
  assert failure.value.code==400
  assert not state.gen_lock.locked()


def test_native_sft_loader_requires_explicit_mode_and_complete_targets(tmp_path):
  from extra.llm.bench.sft_smoke_train import load_sft_rows
  tok = native_tokenizer(); protocol = NativeChat(tok)
  row = {'id':'v1', 'source_id':'independently-verified', 'tools':TOOLS,
         'messages':MESSAGES+[protocol.parse(CALL,TOOLS,'stop')]}
  path=tmp_path/'rows.jsonl'; path.write_text(json.dumps(row)+'\n')
  with pytest.raises(ValueError): load_sft_rows(path)
  loaded=load_sft_rows(path,native=True)
  assert loaded==[row]
  with pytest.raises(ValueError, match='completion_scope=all'): build_completion_examples(loaded,tok)
  with pytest.raises(ValueError, match='own their system'): build_completion_examples(loaded,tok,completion_scope='all',system_prompt='override')
  row['messages'][-1]['tool_calls'][0]['function']['arguments']='{broken'
  path.write_text(json.dumps(row)+'\n')
  with pytest.raises(ValueError): load_sft_rows(path,native=True)


def test_native_http_legacy_text_remains_available_and_truncation_rejected(native_server):
  state, request = native_server
  state.model.text='Ordinary text.'
  assert json.loads(request({'messages':MESSAGES}))['choices'][0]['message']['content']=='Ordinary text.'
  state.model.text=CALL
  with pytest.raises(urllib.error.HTTPError) as failure: request({'messages':MESSAGES,'tools':TOOLS,'max_tokens':16})
  assert failure.value.code==502
  assert len(state.model.prompts)==2
  for invalid in ({'tool_choice':'required'}, {'tools':None}, {'tools':TOOLS,'messages':[None]}):
    with pytest.raises(urllib.error.HTTPError) as failure: request({'messages':MESSAGES,**invalid})
    assert failure.value.code==400
  assert len(state.model.prompts)==2  # Validation failures never enter generation.


def test_native_truncated_plain_text_is_never_executable(native_server):
  state, request = native_server
  state.model.text='Plain answer.'
  result=json.loads(request({'messages':MESSAGES,'tools':TOOLS,'max_tokens':5}))['choices'][0]
  assert result['finish_reason']=='length'
  assert result['message']=={'role':'assistant','content':'Plain'}
  with pytest.raises(ChatError): NativeChat(state.tok).training([None],TOOLS)


def test_native_missing_optional_dependency_is_explicit(monkeypatch):
  import builtins
  original=builtins.__import__
  def importing(name,*args,**kwargs):
    if name.startswith('jinja2'): raise ImportError('absent')
    return original(name,*args,**kwargs)
  monkeypatch.setattr(builtins,'__import__',importing)
  with pytest.raises(ChatError,match='optional dependency'): NativeChat(native_tokenizer())


def test_native_history_rejects_duplicate_ids_results_and_nonjson_arguments():
  protocol=NativeChat(native_tokenizer()); call=protocol.parse(CALL,TOOLS,'stop')
  result={'role':'tool','tool_call_id':call['tool_calls'][0]['id'],'content':'value'}
  for history in (MESSAGES+[call,result,result], MESSAGES+[call,result,call,result]):
    with pytest.raises(ChatError): protocol.prompt({'messages':history,'tools':TOOLS})
  call['tool_calls'][0]['function']['arguments']='[]'
  with pytest.raises(ChatError): protocol.prompt({'messages':MESSAGES+[call,result],'tools':TOOLS})
