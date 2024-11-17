FROM python:3.10-slim

ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy files
COPY packages ./packages
COPY api ./api

# Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --directory api
RUN python -m nltk.downloader stopwords

EXPOSE 8000

# Run
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]