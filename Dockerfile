FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-simulated.txt ./
RUN pip install --no-cache-dir -r requirements-simulated.txt
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/data /app/storage /app/results /app/.cache /app/notebooks && chown -R rag:rag /app

FROM dependencies AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY --chown=rag:rag . .
USER rag
CMD ["python", "rerank_rag.py", "simular"]

FROM dependencies AS simulation
COPY --chown=rag:rag . .
USER rag
CMD ["python", "rerank_rag.py", "simular"]
