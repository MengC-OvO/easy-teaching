FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system easyteaching && adduser --system --ingroup easyteaching easyteaching

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data/local /app/data/exports && chown -R easyteaching:easyteaching /app

USER easyteaching

CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
