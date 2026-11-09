
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pandas \
    numpy \
    pyarrow \
    scikit-learn \
    xgboost \
    joblib \
    requests \
    streamlit

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
