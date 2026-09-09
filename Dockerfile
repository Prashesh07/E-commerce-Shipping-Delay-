#base image
FROM python:3.12-slim
#working directory
WORKDIR /app
#copy requirements file and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#copy application code and models
COPY api/ api/
COPY models/ models/
#run command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
