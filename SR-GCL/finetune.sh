input_model_file="./models_graphtrans/graphtrans-sr-gcl.pth"
for dataset in sider hiv clintox tox21 toxcast bace bbbp muv
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune_graphtrans.py --runseed $runseed --dataset $dataset --input_model_file $input_model_file --device 0
done
done
