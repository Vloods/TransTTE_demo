# pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# pip install fastapi
# pip install pydantic
# pip install uvicorn
# pip install pandas
# pip install sklearn
# pip install python-igraph
# pip install loguru
# pip install lmdb
# pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
# pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
# pip install torch-geometric==1.7.2
# pip install tensorboardX==2.4.1
# pip install ogb==1.3.2
# pip install rdkit-pypi==2021.9.3
# pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
# pip install loguru
# pip install igraph
# pip install fairseq
# pip install setuptools==59.5.0



# cd graphormer_repo
# git clone https://github.com/pytorch/fairseq
# cd fairseq
# # if fairseq submodule has not been checkouted, run:
# # git submodule update --init --recursive
# # pip install . --use-feature=in-tree-build
# # python setup.py build_ext --inplace

# pip install --editable ./

# # cd ..
# # git clone https://github.com/NVIDIA/apex
# # cd apex
# # pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
# #   --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
# #   --global-option="--fast_multihead_attn" ./


#####################

cd graphormer_repo
git clone https://github.com/pytorch/fairseq
cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
pip install . --use-feature=in-tree-build
python setup.py build_ext --inplace

# cd ..
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#   --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#   --global-option="--fast_multihead_attn" ./
