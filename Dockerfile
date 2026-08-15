# One image serves the API, the workers and the MCP server. They share the whole
# of src/ — the graph client, the scope module, the ontology loader — so building
# three images would mean three chances for them to drift apart.
#
# Which process runs is chosen by APP_MODULE at container start. AgentCore
# Runtime cannot override a container's command, only its environment, so the
# switch has to live in the environment rather than in the CMD.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml ./
COPY src/ src/

# src/ontology/loader.py resolves the pack directory relative to the repo root,
# so the packs must sit beside src/ rather than inside it.
COPY ontologies/ ontologies/

# The example metric pack and the sample legal documents. Both are read from disk at
# runtime by the seed and sample-data endpoints, so leaving them out of the image makes
# those endpoints fail with "could not be read" on a deployment that otherwise looks fine.
COPY sample/ sample/

RUN pip install --no-cache-dir .

# 8000 for both the API and the MCP server: AgentCore's MCP protocol contract
# fixes the MCP port at 8000, so matching it keeps local and deployed identical.
ENV APP_MODULE=src.api.app:app \
    PORT=8000

EXPOSE 8000

# Shell form so $APP_MODULE and $PORT expand at runtime rather than build time.
CMD ["sh", "-c", "exec uvicorn $APP_MODULE --host 0.0.0.0 --port $PORT"]
