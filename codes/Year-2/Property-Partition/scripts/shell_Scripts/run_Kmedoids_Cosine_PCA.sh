#Step 1: Activate the deep-gate2 environment from soumik-experiment-env
#source ~/workspace_Soumik/soumik-experiment-env/bin/activate #--source not found error
#Param 1: Design Name (with out aig)
#------THIS SCRIPT FOR KMEDOIDS AND PCA ------
designName=$1
output_struct_filename=$designName"_outputS-cosine_Kmedoids_PCA.txt"
output_funct_filename=$designName"_outputF-cosine_Kmedoids_PCA.txt"
python3 /home/prateek/soumikSep13/clustering_v1.py --input_dir ./$designName/reduced_structPkl --method kmedoids --distance_metric cosine > /home/prateek/soumikSep13/cluster_Info/$output_struct_filename
echo "Struct Cosine similarity printed on file "
python3 /home/prateek/soumikSep13/clustering_v1.py  --input_dir ./$designName/reduced_funcPkl --method kmedoids --distance_metric cosine > /home/prateek/soumikSep13/cluster_Info/$output_funct_filename
echo "Funct Cosine similarity printed on file "
echo "Open the the files  ./cluster_Info/$output_struct_filename and  /cluster_Info/$output_funct_filename"



#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric euclidean > outputS-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric euclidean > outputF-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric manhattan > outputS-manhattan.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric manhattan > outputF-manhattan.txt
exit 0
