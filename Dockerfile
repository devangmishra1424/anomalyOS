FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY src/ ./src/
COPY api/ ./api/
COPY app.py .
COPY start.sh .

RUN mkdir -p logs reports data

EXPOSE 7860

RUN chmod +x start.sh
CMD ["./start.sh"]