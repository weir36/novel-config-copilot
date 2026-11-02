PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install install-full build build-rule serve test eval eval-all up down demo

help:
	@echo "网文作品资源配置助手 —— 常用命令"
	@echo ""
	@echo "  make demo          零依赖跑通：装依赖 + 建库 + 跑全部离线评测（不需要 docker / API key）"
	@echo "  make install       安装核心依赖（纯 Python）"
	@echo "  make install-full  额外装 bge / faiss / neo4j / redis / es 客户端"
	@echo "  make build         离线建库（hybrid 抽取，需要 LLM_API_KEY）"
	@echo "  make build-rule    离线建库（纯规则抽取，完全离线）"
	@echo "  make serve         启动服务，浏览器打开 http://localhost:8000"
	@echo "  make test          单元测试"
	@echo "  make eval          离线可跑的评测：意图 + 检索消融 + 图谱抽取 + 追问识别"
	@echo "  make eval-all      全部评测，含端到端和纯大模型对照（需要 LLM_API_KEY）"
	@echo "  make up / down     启停 Neo4j + Redis + Elasticsearch"

venv:
	@test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "核心依赖安装完成"

install-full: install
	$(PIP) install -q sentence-transformers torch faiss-cpu neo4j redis "elasticsearch>=8.13,<9"
	@echo "完整依赖安装完成（首次使用 bge 会自动下载约 100MB 模型）"

build:
	$(PY) -m app.indexing.build --extractor hybrid

build-rule:
	$(PY) -m app.indexing.build --extractor rule

serve:
	$(PY) -m uvicorn app.server.main:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest tests -q

eval:
	@echo "\n=== 意图识别（测试集）===" && $(PY) -m eval.run_intent_eval
	@echo "\n=== 检索消融 ==="         && $(PY) -m eval.run_retrieval_eval
	@echo "\n=== 图谱抽取（规则）+ 实体（词表）===" && $(PY) -m eval.run_graph_eval --extractors rule --entities vocab
	@echo "\n=== 追问识别（测试集）===" && $(PY) -m eval.run_followup_eval
	@$(PY) -m eval.report

eval-all: eval
	@echo "\n=== 图谱抽取（三方对比）+ 实体（词表 vs 大模型）===" && $(PY) -m eval.run_graph_eval --extractors rule,llm,hybrid --entities vocab,llm
	@echo "\n=== 端到端（Agent on）==="   && $(PY) -m eval.run_e2e_eval
	@echo "\n=== 端到端（Agent off）==="  && $(PY) -m eval.run_e2e_eval --no-agent
	@echo "\n=== 端到端（纯大模型对照）===" && $(PY) -m eval.run_e2e_eval --baseline
	@$(PY) -m eval.report

demo: install build-rule test eval
	@echo ""
	@echo "全部跑通。执行 make serve 后打开 http://localhost:8000"
	@echo "注意：未配置 LLM_API_KEY 时生成环节走 MockLLM，检索与图谱链路是真实的。"

up:
	docker compose up -d
	@echo "Neo4j  http://localhost:7474 （neo4j / livecopilot）"
	@echo "Redis  localhost:6379     Elasticsearch http://localhost:9200"

down:
	docker compose down
