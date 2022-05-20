FROM cr.msk.sbercloud.ru/aicloud-jupyter/jupyter-cuda11.1-pt1.9.1-gpu:0.0.80
USER root
RUN pip install -U pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
RUN pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install pandas
RUN pip install sklearn
RUN pip install python-igraph
RUN pip install loguru
RUN pip install lmdb
RUN pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
RUN pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
RUN pip install torch-geometric==1.7.2
RUN pip install tensorboardX==2.4.1
RUN pip install ogb==1.3.2
RUN pip install rdkit-pypi==2021.9.3
RUN pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
RUN pip install loguru
RUN pip install igraph
RUN pip install fairseq
COPY app app/
RUN chmod +x app/install.sh
RUN app/install.sh
RUN pip install setuptools==59.5.0

#RUN python app/app.py
# Expose the port uvicorn is running on
EXPOSE 3006

CMD ["python", "app/app.py"]
