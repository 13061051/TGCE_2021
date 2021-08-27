# TGCE_2021
codes for TGCE
#step 1  
run the code in the GBUL(https://github.com/minhcp/GBUL/) to preprocess the original dataset  
#step2:  
change the code in the line 135 in the https://github.com/minhcp/GBUL/blob/master/candidate_generation.py as the following:  
"for path_lv in range(4):"  
to use the user behavior logs at 4 levels.  
#step3  
change the code in the line 94 in the https://github.com/minhcp/GBUL/blob/master/xgb.py as the following:  
"x,y,train_pairs,test_pairs,xt,xv,yv"  
to both save the train_pairs and test_pairs for my model.  

#  
mkdir data dir 'data', then put your process data in the 'data' dir 
put the 'candidates', 'original' and 'tmp' in GBUL dir to the TGCE 'data' dir  

run the following codes to obtain the final results:  
#before train, apply python gensim with skip-gram method to generate each items's embedding.  
#step4:  
applies the data from the GBUL work for my model.  
There are 3 steps for my codes:  
##step 4A:  
run the main_generate_graph.py to generate the long-range dependency paths for each user log  
##step 4B  
run the following codes to get result only based on the log's sematic embedding features:  
python3 main_gnn_match.py --gnn_function GRU_POSGAT --dim 64 --model_path GRU_POSGAT_d64_p10_m2 --deep_path 10 --dropout_fc 0  
the threshold are selected as 0.5 for the final result as common setting  
##step 4C:  
run the following codes to get result only based on the log's sematic embedding features and the statistic features from GBUL:  
python3 train_gnn_xgb.py --dim 64 --model_path gnn_xgb_m5 --drop_out_fc 0.3 --valid_ratio 0.05 --epochs 20  
and  
python3 eva_gnn_xgb.py --dim 64 --model_path gnn_xgb_m5 --drop_out_fc 0.3 --valid_ratio 0.05  
to obtain the results following GBUL's best threshold setting  
 

