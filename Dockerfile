FROM jupyter/datascience-notebook:python-3.9.10
COPY requirements.txt .
RUN pip install -r requirements.txt