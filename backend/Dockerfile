FROM python:3.9
USER root
RUN pip install -U pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install python-igraph
RUN pip install loguru

RUN pip install requests
RUN pip install torch

COPY . .
EXPOSE 80
CMD ["python", "app/app.py"]
