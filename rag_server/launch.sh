#tmux new -s rag_server
#conda activate searchr1
#bash retrieval_launch.sh
#tmux detach
#bash train_ppo.sh

nvcc --version

file_path=/home/xueml3/DATA/retrieval_corpus
index_file=$file_path/e5_flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=/home/xueml3/pretrained_weights/e5-base-v2

export CUDA_VISIBLE_DEVICES=5,6,7
python3 rag_server/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever &
sleep 1200000000
