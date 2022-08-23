# source conda
source ~/anaconda3/etc/profile.d/conda.sh 
conda activate easyrl # easyrl here can be changed to another name of conda env that you have created
python task0.py --env_name CartPole-v1 --train_eps 500 --epsilon_decay 1000 --memory_capacity 200000 --batch_size 128 --device cuda