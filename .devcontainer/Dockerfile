# Python 3.12 matches Databricks Runtime 18 LTS ML (see runtime.toml)
FROM mcr.microsoft.com/devcontainers/python:3.12

# default-jre-headless: Java 17+ for local Spark tests
RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg2 default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
