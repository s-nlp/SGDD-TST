wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz

gdown https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
gzip -d GoogleNews-vectors-negative300.bin.gz

python -m spacy download en_core_web_md
pip install -r requirements.txt