# RAG From Scratch

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning is one way to mitigate this, but is often [not well-suited for factual recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).
Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learning. 
These notebooks accompany a [video playlist](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared) that builds up an understanding of RAG from scratch, starting with the basics of indexing, retrieval, and generation. 
![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)
 
[Video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)

## Environment

The notebooks load credentials from a local `.env` file using `python-dotenv`.
Use `.env.example` as the template and set `GOOGLE_API_KEY` before running the Gemini examples.
Set `OPENROUTER_API_KEY` only if you set `RAG_CHAT_PROVIDER=openrouter`, set `COHERE_API_KEY` only if you run the Cohere reranking sections, and set the LangSmith variables only if you want tracing.

For low-cost OpenRouter experiments, the default is `OPENROUTER_MODEL=openrouter/free`.
Use `OPENROUTER_MODEL=inclusionai/ling-2.6-flash` for a specific ultra-cheap paid model, or `OPENROUTER_MODEL=qwen/qwen3-235b-a22b-2507` when you want a stronger low-cost RAG answer model.
All notebooks use the shared `chat_model()` helper for generation cells that can run through either `RAG_CHAT_PROVIDER=openrouter` or `RAG_CHAT_PROVIDER=gemini`.
Retrieval examples default to `RAG_EMBEDDING_PROVIDER=local`, an API-free TF-IDF helper; set `RAG_EMBEDDING_PROVIDER=gemini` to use the original Gemini embeddings.

Install notebook dependencies with uv:

```bash
uv sync
```

Start a local Chroma server:

```bash
docker compose up -d chroma
```

Check it:

```bash
curl http://localhost:8000/api/v2/heartbeat
```

Start Chroma with the optional admin UI:

```bash
docker compose up -d
```

Open the admin UI at `http://localhost:3001`. In the setup screen, use:

- Connection string: `http://chroma:8000`
- Tenant: `default_tenant`
- Database: `default_database`

Use `http://localhost:8000` only from your host machine. From inside Docker Compose, the admin UI must connect to the Chroma service by name: `http://chroma:8000`.

## Notebook Map

Classic RAG path:

- `rag_from_scratch_1_to_4.ipynb`: overview, indexing, retrieval, generation.
- `rag_from_scratch_5_to_9.ipynb`: query transformations, RAG-Fusion, decomposition, step-back prompting, HyDE.
- `rag_from_scratch_10_and_11.ipynb`: logical routing, semantic routing, and query structuring.
- `rag_from_scratch_12_to_14.ipynb`: multi-representation indexing, RAPTOR references, ColBERT.
- `rag_from_scratch_15_to_18.ipynb`: reranking, CRAG/Self-RAG references, and long-context impact.

Modern RAG extensions:

- `rag_from_scratch_19_eval_rag.ipynb`: retrieval and generation evaluation loops.
- `rag_from_scratch_20_hybrid_search_and_reranking.ipynb`: BM25, dense retrieval, RRF, and optional reranking.
- `rag_from_scratch_21_long_context_vs_rag.ipynb`: route between long-context prompting and retrieval.
- `rag_from_scratch_22_graphrag_and_lightrag.ipynb`: graph indexing, local graph search, and global community context.
- `rag_from_scratch_23_agentic_rag_langgraph.ipynb`: retrieve-grade-rewrite-generate control loop with LangGraph.
- `rag_from_scratch_24_multimodal_document_rag.ipynb`: page-level document RAG and multimodal embedding sketches.
- `rag_from_scratch_25_managed_rag_file_search.ipynb`: OpenAI and Gemini managed file-search patterns.
- `rag_from_scratch_26_production_rag_freshness_security.ipynb`: versioning, freshness, ACL filtering, citations, and operational controls.

The modern notebooks are intentionally small and local-first. Cells that require hosted APIs are marked as optional and can be uncommented after setting the relevant API key.
