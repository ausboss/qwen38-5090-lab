#!/usr/bin/env python3
"""Configure Open WebUI for this machine, in one pass.

Run with Open WebUI STOPPED. It caches config in memory and rewrites rows on
change, so a write while it is running gets silently clobbered.

    pkill -f "open_webui|open-webui serve"; sleep 5
    python3 bin/webui-configure.py
    open-webui serve --port 8080 &

Every value here was verified against the installed package source or measured
on this machine. Notes explain the non-obvious ones.
"""
import json
import sqlite3
import sys
import time

DB = ('/home/ausboss/.local/share/uv/tools/open-webui/lib/python3.12/'
      'site-packages/open_webui/data/webui.db')
USER_ID = 'b9ab5b70-6fe7-43eb-b33b-66be62b52ad5'
LAB = '/home/ausboss/Documents/sglang-qwen3.8-lab'

c = sqlite3.connect(DB)
now = int(time.time())
applied = []


def put(k, v):
    s = json.dumps(v)
    if c.execute('select 1 from config where key=?', (k,)).fetchone():
        c.execute('update config set value=?,updated_at=? where key=?', (s, now, k))
    else:
        c.execute('insert into config(key,value,updated_at) values(?,?,?)', (k, s, now))
    applied.append(k)


def get(k, d=None):
    r = c.execute('select value from config where key=?', (k,)).fetchone()
    return json.loads(r[0]) if r else d


# ---------------------------------------------------------------- image gen --
# Verified by generating a real image through this exact graph.
# device="cpu" on the CLIP loader is the VRAM-critical bit: it moves the 7.5 GB
# Qwen3-4B text encoder off the card, leaving ~2.1 GB headroom instead of
# ~650 MB, for only ~1.7s more per image.
WORKFLOW = {
    '1': {'class_type': 'UNETLoader', 'inputs': {
        'unet_name': 'ZIMAGETURBO/dZITNUTZ_lightFixedFp8.safetensors',
        'weight_dtype': 'default'}},
    '2': {'class_type': 'CLIPLoader', 'inputs': {
        'clip_name': 'qwen_3_4b.safetensors', 'type': 'qwen_image', 'device': 'cpu'}},
    '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'z-image-ae.safetensors'}},
    '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': ''}},
    '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': ''}},
    '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {
        'width': 1024, 'height': 1024, 'batch_size': 1}},
    '7': {'class_type': 'KSampler', 'inputs': {
        'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0],
        'latent_image': ['6', 0], 'seed': 0, 'steps': 10, 'cfg': 1.0,
        'sampler_name': 'dpmpp_sde', 'scheduler': 'beta', 'denoise': 1.0}},
    '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
    '9': {'class_type': 'SaveImage', 'inputs': {
        'images': ['8', 0], 'filename_prefix': 'open-webui'}},
}
# Schema: utils/images/comfyui.py:_apply_workflow_nodes.
# TRAP: types model/seed/image index inputs[node.key] with NO fallback, and
# ComfyUINodeInput.key defaults to "text" — so omitting key on those silently
# writes into an input named "text" and ComfyUI rejects the graph.
put('image_generation.comfyui.workflow', json.dumps(WORKFLOW))
put('image_generation.comfyui.nodes', [
    {'type': 'model', 'node_ids': ['1'], 'key': 'unet_name'},
    {'type': 'prompt', 'node_ids': ['4'], 'key': 'text'},
    {'type': 'negative_prompt', 'node_ids': ['5'], 'key': 'text'},
    {'type': 'width', 'node_ids': ['6'], 'key': 'width'},
    {'type': 'height', 'node_ids': ['6'], 'key': 'height'},
    {'type': 'n', 'node_ids': ['6'], 'key': 'batch_size'},
    {'type': 'steps', 'node_ids': ['7'], 'key': 'steps'},
    {'type': 'seed', 'node_ids': ['7'], 'key': 'seed'},
])
put('image_generation.engine', 'comfyui')
put('image_generation.enable', True)
put('image_generation.comfyui.base_url', 'http://127.0.0.1:8188')
put('image_generation.model', 'ZIMAGETURBO/dZITNUTZ_lightFixedFp8.safetensors')
put('image_generation.size', '1024x1024')
put('image_generation.steps', 10)
put('image_generation.prompt.enable', True)
# Z-Image is trained on flowing prose, not booru tags. The local LLM rewrites the
# prompt before it reaches ComfyUI, which is nearly free at 153 tok/s.
put('task.image.prompt_template', """### Task:
Write a single vivid image-generation prompt based on the conversation below.
This model responds to flowing natural-language description, not comma-separated tags.

### Guidelines:
- One paragraph, 40-70 words.
- Subject first, then setting, then lighting, then lens or medium.
- Be concrete about colour and material. Words like "masterpiece", "8k" and
  "best quality" do nothing on this model — omit them.

### Output:
Strictly return in JSON format:
{
    "prompt": "Your prompt here."
}

### Chat History:
<chat_history>
{{MESSAGES:END:6}}
</chat_history>""")

# --------------------------------------------------------------- task model --
# get_task_model_id (utils/task.py:16-27) reads task.model.EXTERNAL here, not
# .default, because openai.api_configs entries carry no connection_type and
# routers/openai.py:584 defaults them to 'external'. Setting .default alone is
# a no-op — an easy and invisible mistake.
put('task.model.external', 'qwen38-task')
put('task.model.default', 'qwen38-task')
# The stock templates already demand bare JSON and the model obeys them.
# Overriding them risks breaking the JSON contract the parser depends on.
for k in ('task.title.prompt_template', 'task.tags.prompt_template',
          'task.follow_up.prompt_template', 'task.query.prompt_template',
          'task.autocomplete.prompt_template'):
    put(k, '')
put('task.autocomplete.enable', False)   # llama.cpp has 1 slot; typing would stutter

# ------------------------------------------------------- built-in tool suite --
# The single highest-leverage change. Native function calling is the default in
# this build, which ships search_web, fetch_url, execute_code, generate_image,
# memory, notes, calendar and automations as real tools — but each is gated on a
# per-chat feature flag seeded from the model preset. Without this they all sit
# dormant behind icons. models.default_metadata merges into EVERY model.
put('models.default_metadata', {
    'defaultFeatureIds': ['web_search', 'code_interpreter', 'image_generation'],
    'capabilities': {'web_search': True, 'code_interpreter': True,
                     'image_generation': True, 'memory': True, 'vision': True},
})

# -------------------------------------------------------------- web search ---
put('web.search.enable', True)
put('web.search.engine', 'duckduckgo')   # the only genuinely keyless engine
put('web.search.result_count', 5)
put('web.search.concurrent_requests', 4)

# --------------------------------------------------------------------- RAG ---
# chunk_size is measured in CHARACTERS while rag.text_splitter is "" (the
# default). all-MiniLM-L6-v2 truncates at 256 TOKENS, so 1000 chars was already
# borderline. 900 keeps it safely inside.
put('rag.chunk_size', 900)
put('rag.chunk_overlap', 150)
put('rag.enable_hybrid_search', True)    # BM25 + vector; rank_bm25 already installed
put('rag.top_k', 6)
put('rag.top_k_reranker', 6)
put('rag.embedding_batch_size', 32)      # was 1 — the ingest bottleneck
put('rag.pdf_extract_images', True)      # rapidocr ships its own weights, offline

# ------------------------------------------------------- context compaction --
# 131K context with no compaction means a long chat eventually falls off a cliff.
put('chat.context_compaction.enable', True)
put('chat.context_compaction.model', 'qwen38-task')
put('chat.context_compaction.token_threshold', 100000)

# ----------------------------------------------------------------- cleanup ---
# With arena enabled and an EMPTY model list, utils/models.py injects a fake
# "Arena Model" into the picker that cannot work with one backend model.
put('evaluation.arena.enable', False)
put('channels.enable', False)            # Slack-style multi-user; adds 4 dead tools
put('memories.enable', True)
put('memories.background_review.enable', False)  # uses the CHAT model, not the task one
put('subagents.enable', False)           # 1 llama.cpp slot; they would serialize
put('auth.enable_api_keys', True)        # script against the local endpoint

# ------------------------------------------------------------------ UI ------
# SHAPE TRAP: these two look alike but are not. DEFAULT_MODELS is read as a
# plain env string (config.py:1633) and the frontend calls .split(',') on it —
# storing a JSON array here crashes the chat page with
# "default_models.split is not a function" and nothing renders.
# MODEL_ORDER_LIST really is json.loads'd (config.py:1676), so it IS an array.
put('ui.default_models', 'qwen38-fast')
put('ui.model_order_list', ['qwen38-fast', 'qwen38-deep', 'qwen38-code',
                            'qwen38-vision', 'qwen38-longdoc', 'qwen38-writer'])
put('ui.enable_community_sharing', False)
put('ui.prompt_suggestions', [
    {'title': ['Explain this screenshot', 'what am I looking at?'],
     'content': "I'm attaching a screenshot. Explain what's happening and what I should do next."},
    {'title': ['Generate an image', 'photoreal, ~10 seconds'],
     'content': 'Generate an image: a rain-soaked neon alley in Osaka at night, shallow depth of field, 35mm'},
    {'title': ['Read a long document', 'paste once, ask many times'],
     'content': "I'm going to paste a long document. Read it, then wait for my questions. Don't summarise it yet."},
    {'title': ['Debug this error', 'root cause, not symptom'],
     'content': "Here's an error I'm getting. Work out the root cause, not just the symptom:\n\n"},
    {'title': ['Review this code', 'edge cases and failure modes'],
     'content': 'Review this code. Focus on correctness and edge cases rather than style:\n\n'},
    {'title': ['Can my GPU run this?', 'VRAM math'],
     'content': "I have an RTX 5090 (32GB) and I'm already running a 22GB local model. Can I also run "},
    {'title': ['Search the web', 'then summarise'],
     'content': 'Search the web and give me a short, sourced summary of: '},
    {'title': ['Plan a ComfyUI workflow', 'node graph'],
     'content': 'Help me design a ComfyUI workflow that '},
])
put('ui.banners', [{
    'id': 'vram', 'type': 'info', 'title': 'One GPU, shared',
    'content': ('Qwen3.8-27B (~22GB) and ComfyUI image generation share one 32GB '
                'card. If images fail, something else is holding VRAM — check '
                'with `qwen status`. ComfyUI must be running for image generation.'),
    'dismissible': True, 'timestamp': now}])

# ------------------------------------------------- task preset (no thinking) --
# Measured: 283 -> 17 completion tokens for the same title task. custom_params
# reaches the request body via utils/payload.py:111-124. Deliberately NO system
# prompt: routers/openai.py applies a preset's system prompt to task calls too,
# so a chatty one would leak into generated titles.
c.execute('delete from model where id=?', ('qwen38-task',))
c.execute(
    'insert into model(id,user_id,base_model_id,name,params,meta,updated_at,created_at,is_active)'
    ' values(?,?,?,?,?,?,?,?,?)',
    ('qwen38-task', USER_ID, 'qwen38-fast', 'Qwen3.8 · Task',
     json.dumps({'temperature': 0.2, 'top_p': 0.9, 'top_k': 20, 'max_tokens': 512,
                 'custom_params': {'chat_template_kwargs': {'enable_thinking': False}}}),
     json.dumps({'description': 'Internal: titles, tags, follow-ups. Not for chat.',
                 'capabilities': {'vision': False, 'usage': False}}),
     now, now, 1))

# ------------------------------------------------------------- prompts -------
PROMPTS = [
    ('/rootcause', 'Root cause this',
     'Work out the ROOT CAUSE of the problem below, not just the surface symptom. '
     'State what you verified vs what you are assuming. If you need to see a file '
     'or output to be sure, say exactly which one.\n\n'),
    ('/review', 'Review this code',
     'Review the code below for correctness, edge cases and failure modes. Ignore '
     'style unless it causes bugs. Point out what breaks under bad input, '
     'concurrency, or scale. Be specific about the failing case.\n\n'),
    ('/vram', 'Will this fit on my GPU?',
     'I have an RTX 5090 with 32GB VRAM. The desktop compositor holds 2-4GB. Work '
     'out whether the following will fit, and roughly how fast it will run. Show '
     'the arithmetic. Remember decode speed is bandwidth-bound: '
     'tok/s ~= 1150-1200 GB/s divided by the streamed weight bytes.\n\n'),
    ('/comfy', 'ComfyUI workflow help',
     'Help me with a ComfyUI workflow. My install is at '
     '~/ComfyUI-Easy-Install-Linux/ComfyUI-Easy-Install/ComfyUI on 127.0.0.1:8188, '
     'launched with --use-sage-attention. Give API-format JSON with real node '
     'class_types, and tell me which model files it needs.\n\n'),
    ('/shorter', 'Cut this down',
     'Rewrite the text below to be significantly shorter and plainer, keeping every '
     'point of substance. No filler opening, no summary paragraph at the end.\n\n'),
    ('/eli5', 'Explain simply',
     'Explain the following in plain language, assuming I am smart but unfamiliar '
     'with the jargon. Use a concrete analogy. Skip the history lesson.\n\n'),
    ('/steelman', 'Argue the other side',
     'Steelman the strongest case AGAINST what I just said. Do not hedge or '
     'both-sides it — make the best argument you genuinely can, then tell me which '
     'parts of it you find convincing.\n\n'),
    ('/checkit', 'Fact-check with sources',
     'Search the web and check the claims below. For each: true, false, or '
     'unclear, with a source link. Say plainly when you could not find support '
     'either way rather than guessing.\n\n'),
    ('/bash', 'Explain this command',
     'Explain exactly what this shell command does, flag by flag, and call out '
     'anything destructive or surprising BEFORE I run it.\n\n'),
    ('/summarise', 'Summarise the document',
     'Summarise the document I have provided. Lead with the single most important '
     'takeaway, then the supporting points. Quote directly where precision matters. '
     'Say explicitly if something I would expect to be covered is absent.\n\n'),
]
for cmd, name, content in PROMPTS:
    c.execute('delete from prompt where command=?', (cmd,))
    c.execute(
        'insert into prompt(id,command,user_id,name,content,data,meta,is_active,'
        'tags,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?)',
        (cmd.lstrip('/'), cmd, USER_ID, name, content,
         json.dumps({}), json.dumps({}), 1, json.dumps([]), now, now))

c.commit()

print(f'config keys written : {len(applied)}')
print(f'prompts installed   : {len(PROMPTS)}')
print(f'task preset         : qwen38-task (thinking disabled)')
print()
print('spot-check:')
for k in ('image_generation.engine', 'task.model.external', 'web.search.engine',
          'rag.enable_hybrid_search', 'chat.context_compaction.enable',
          'evaluation.arena.enable', 'models.default_metadata'):
    print(f'  {k:38} {json.dumps(get(k))[:60]}')
print(f"  models: {[r[0] for r in c.execute('select id from model')]}")
print(f"  prompts: {[r[0] for r in c.execute('select command from prompt')]}")
c.close()
