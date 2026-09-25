"""Native chat serialization shared by serving and SFT; no tool execution or argument repair."""
from __future__ import annotations
import copy, hashlib, json, re, uuid


class ChatError(ValueError): pass


def native_request(body:dict) -> bool:
  return any(key in body for key in ('tools', 'tool_choice', 'parallel_tool_calls')) or any(m.get('role') == 'tool' or 'tool_calls' in m for m in (body.get('messages') or []) if isinstance(m, dict))


def _object(text:str) -> dict:
  def pairs(items):
    result = {}
    for key, value in items:
      if key in result: raise ChatError('duplicate JSON key')
      result[key] = value
    return result
  def constant(value): raise ChatError(f'invalid JSON constant {value}')
  try: result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
  except (ValueError, TypeError) as exc: raise ChatError(f'invalid JSON object: {exc}') from exc
  if not isinstance(result, dict): raise ChatError('arguments must be a JSON object')
  return result


def _name(value):
  if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value): raise ChatError('invalid function name')
  return value


def validate_messages(messages:list, tools:list, *, target=False) -> list:
  if not isinstance(tools, list): raise ChatError('tools must be a list')
  names = set()
  for tool in tools:
    if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
      raise ChatError('only function tools are supported')
    fn = tool['function']; name = _name(fn.get('name'))
    if name in names: raise ChatError('duplicate tool name')
    names.add(name)
    if not isinstance(fn.get('parameters'), dict) or fn['parameters'].get('type') != 'object':
      raise ChatError('function parameters must declare an object schema')
  if not isinstance(messages, list) or not messages: raise ChatError('messages must be a nonempty list')
  result, pending, seen = [], set(), set()
  for original in messages:
    if not isinstance(original, dict): raise ChatError('message must be an object')
    msg = copy.deepcopy(original); role = msg.get('role')
    if role not in ('system', 'user', 'assistant', 'tool'): raise ChatError('unsupported message role')
    content = msg.get('content')
    if content is None and role == 'assistant' and msg.get('tool_calls'): content = ''
    if not isinstance(content, str): raise ChatError('native chat supports string content only')
    msg['content'] = content
    if role == 'tool':
      cid = msg.get('tool_call_id')
      if not isinstance(cid, str) or cid not in pending: raise ChatError('orphan or duplicate tool result')
      pending.remove(cid)
    elif pending: raise ChatError('all tool calls require results before the next message')
    if 'tool_calls' in msg:
      if role != 'assistant' or not isinstance(msg['tool_calls'], list) or not msg['tool_calls']:
        raise ChatError('tool_calls requires a nonempty assistant call list')
      for call in msg['tool_calls']:
        if not isinstance(call, dict): raise ChatError('tool call must be an object')
        cid = call.get('id')
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', cid) or cid in seen:
          raise ChatError('invalid or reused tool call id')
        seen.add(cid); pending.add(cid)
        fn = call.get('function')
        if call.get('type') != 'function' or not isinstance(fn, dict) or _name(fn.get('name')) not in names:
          raise ChatError('unknown tool call function')
        if not isinstance(fn.get('arguments'), str): raise ChatError('wire tool arguments must be a JSON string')
        fn['arguments'] = _object(fn['arguments'])  # Native templates consume objects, wire clients consume strings.
    if role == 'assistant': msg.setdefault('tool_calls', [])
    result.append(msg)
  if pending and not target: raise ChatError('missing tool results')
  return result


class NativeChat:
  """Template comes from the loaded GGUF. Only the Qwen XML-call dialect is currently qualified."""
  def __init__(self, tok):
    self.tok = tok
    self.template = getattr(tok, 'chat_template', None)
    if not isinstance(self.template, str) or not self.template.strip(): raise ChatError('GGUF has no native chat template')
    # Architecture and actual template protocol, never a filename or tokenizer-preset guess.
    self.dialect = 'qwen-xml' if getattr(tok, 'architecture', None) in ('qwen2', 'qwen3') and all(
      marker in self.template for marker in ('<tool_call>', '</tool_call>', 'tool_calls', 'tools')) else None
    if self.dialect is None: raise ChatError('unsupported native tool dialect for this GGUF architecture/template')
    try:
      from jinja2.sandbox import ImmutableSandboxedEnvironment
      from jinja2 import StrictUndefined
    except ImportError as exc: raise ChatError('native tools require the optional dependency: pip install tinygrad[llm]') from exc
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    def fail(message): raise ChatError(str(message))
    env.globals['raise_exception'] = fail
    try: self.compiled = env.from_string(self.template)
    except Exception as exc: raise ChatError(f'invalid GGUF chat template: {exc}') from exc
    self.identity = {'dialect': self.dialect, 'template_sha256': hashlib.sha256(self.template.encode()).hexdigest(),
                     'enable_thinking': False}

  def render(self, messages:list, tools:list, *, generation=True, target=False) -> str:
    normalized = validate_messages(messages, tools, target=target)
    try:
      return self.compiled.render(messages=normalized, tools=tools, add_generation_prompt=generation, enable_thinking=False,
        bos_token=self.tok.decode([self.tok.bos_id]) if self.tok.bos_id is not None else '', eos_token=self.tok.decode([self.tok.eos_id]))
    except ChatError: raise
    except Exception as exc: raise ChatError(f'native template rendering failed: {exc}') from exc

  def prompt(self, body:dict) -> list[int]:
    if body.get('tool_choice', 'auto') != 'auto': raise ChatError('only tool_choice=auto is supported')
    if 'parallel_tool_calls' in body and body['parallel_tool_calls'] is not False:
      raise ChatError('parallel_tool_calls=true is unsupported')
    messages, tools = body.get('messages'), body.get('tools', [])
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get('role') == 'assistant':
      raise ChatError('native generation requires a user or tool result, not assistant prefill')
    return self.tok.encode(self.render(messages, tools))

  def training(self, messages:list, tools:list) -> tuple[list[int], list[int]]:
    validate_messages(messages, tools, target=True)
    if messages[-1]['role'] != 'assistant': raise ChatError('training requires a final assistant target')
    prefix = self.prompt({'messages': messages[:-1], 'tools': tools})
    whole = self.tok.encode(self.render(messages, tools, generation=False, target=True))
    if whole[:len(prefix)] != prefix: raise ChatError('template does not preserve generation prefix for assistant target')
    completion = whole[len(prefix):]
    # Stop where serving stops, including the actual template's EOS/EOT. Do not append an invented delimiter.
    ends = [i for i, token in enumerate(completion) if self.tok.is_end(token)]
    if not ends: raise ChatError('native training target lacks an end-of-turn token')
    completion = completion[:ends[0]+1]
    if len(completion) < 2: raise ChatError('empty assistant target')
    raw = self.tok.decode(completion[:-1])
    self.parse(raw, tools, 'stop')  # Invalid corrections cannot silently enter the target stream.
    return prefix, completion

  def parse(self, raw:str, tools:list, finish:str) -> dict:
    if '<tool_call' not in raw and '</tool_call' not in raw:
      return {'role': 'assistant', 'content': raw}
    if finish != 'stop': raise ChatError('incomplete tool generation cannot emit an executable call')
    match = re.fullmatch(r'\s*<tool_call>\s*(.*?)\s*</tool_call>\s*', raw, re.S)
    if match is None: raise ChatError('malformed, mixed, or parallel native tool calls')
    value = _object(match[1])
    if set(value) != {'name', 'arguments'} or not isinstance(value['arguments'], dict): raise ChatError('invalid native tool call shape')
    name = _name(value['name'])
    if name not in {tool['function']['name'] for tool in tools}: raise ChatError('model selected an unknown tool')
    return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_'+uuid.uuid4().hex[:24], 'type': 'function',
      'function': {'name': name, 'arguments': json.dumps(value['arguments'], ensure_ascii=False, allow_nan=False)}}]}
