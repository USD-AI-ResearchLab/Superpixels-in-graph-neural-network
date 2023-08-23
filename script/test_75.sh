seed0=41
seed1=95
seed2=12
seed3=35
code=main_superpixel_lungs_classification.py
tmux new -s benchmark_lungs -d
dataset=LUNGS_75
tmux send-keys "
echo "RUNNING"
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MLP_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GCN_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GraphSage_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GatedGCN_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GAT_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MoNet_LUNGS_75.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GIN_LUNGS_75.json' &
wait" C-m
echo "FINISH"