#!/usr/bin/env python3
"""Type-check every Open WebUI config key we wrote against the app's own default.

A key whose stored value has the wrong SHAPE is the nastiest failure mode here:
the backend often tolerates it and the frontend crashes with something opaque
like "default_models.split is not a function", leaving a blank page.

Run with Open WebUI's interpreter so open_webui.config imports:
  /home/ausboss/.local/share/uv/tools/open-webui/bin/python bin/webui-typecheck.py
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(
    0, '/home/ausboss/.local/share/uv/tools/open-webui/lib/python3.12/site-packages')

from open_webui.config import DEFAULT_CONFIG  # noqa: E402

SRC = ('/home/ausboss/.local/share/uv/tools/open-webui/lib/python3.12/'
       'site-packages/open_webui/data/webui.db')

KEYS = [
    'image_generation.engine', 'image_generation.enable', 'image_generation.model',
    'image_generation.size', 'image_generation.steps', 'image_generation.comfyui.nodes',
    'image_generation.comfyui.workflow', 'image_generation.comfyui.base_url',
    'image_generation.prompt.enable',
    'task.model.external', 'task.model.default', 'task.autocomplete.enable',
    'task.image.prompt_template', 'task.title.prompt_template', 'task.tags.prompt_template',
    'models.default_metadata',
    'web.search.enable', 'web.search.engine', 'web.search.result_count',
    'web.search.concurrent_requests',
    'rag.chunk_size', 'rag.chunk_overlap', 'rag.enable_hybrid_search', 'rag.top_k',
    'rag.top_k_reranker', 'rag.embedding_batch_size', 'rag.pdf_extract_images',
    'chat.context_compaction.enable', 'chat.context_compaction.model',
    'chat.context_compaction.token_threshold',
    'evaluation.arena.enable', 'channels.enable', 'memories.enable',
    'memories.background_review.enable', 'subagents.enable', 'auth.enable_api_keys',
    'ui.default_models', 'ui.model_order_list', 'ui.prompt_suggestions', 'ui.banners',
    'ui.enable_community_sharing', 'ui.default_user_role',
]


def main():
    tmp = tempfile.mkdtemp()
    for ext in ('', '-wal', '-shm'):
        if os.path.exists(SRC + ext):
            shutil.copy2(SRC + ext, os.path.join(tmp, 'w.db' + ext))
    c = sqlite3.connect(os.path.join(tmp, 'w.db'))

    problems = 0
    for k in KEYS:
        row = c.execute('select value from config where key=?', (k,)).fetchone()
        if not row:
            print(f'  MISSING       {k}')
            problems += 1
            continue
        # Values are usually JSON text, but sqlite stores some natively as
        # INTEGER/REAL (e.g. rag.top_k) — json.loads would choke on those.
        raw = row[0]
        mine = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        if k not in DEFAULT_CONFIG:
            print(f'  UNKNOWN KEY   {k}  (app never reads this — silently ignored)')
            problems += 1
            continue
        dflt = DEFAULT_CONFIG[k]
        if dflt is None:
            continue  # a None default means the shape is unconstrained
        same_number = isinstance(mine, (int, float)) and isinstance(dflt, (int, float))
        if type(mine) is not type(dflt) and not same_number:
            print(f'  TYPE MISMATCH {k}: stored={type(mine).__name__} '
                  f'expected={type(dflt).__name__}')
            problems += 1

    print(f'\nchecked {len(KEYS)} keys — {problems} problem(s)')
    c.close()
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
