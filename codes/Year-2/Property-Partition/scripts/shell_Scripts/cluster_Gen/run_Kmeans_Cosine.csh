#Step 1: Activate the deep-gate2 environment from soumik-experiment-env
#source ~/workspace_Soumik/soumik-experiment-env/bin/activate #--source not found error
#Param 1: Design Name (with out aig)
#------THIS SCRIPT FOR KMEANS AND NON PCA ------
set designName=$1
set property_count=$2
if ( $#argv  !=  2 )then
        echo "Input missing ..."
        echo "Param 1: Design name (6s199)| Param 2: Nomber of properties"
        exit 1
endif
set pickle_File_Location="../../../output/Pickle_File"
set struct_Pickle_Folder=$pickle_File_Location"/"$designName"/structPkl"
set funct_Pickle_Folder=$pickle_File_Location"/"$designName"/functPkl"
set cluster_Info_Location='../../../output/cluster_output/'$designName
echo $pickle_File_Location

set output_struct_filename=$designName"_outputS-cosine_Kmeans.txt"
set output_funct_filename=$designName"_outputF-cosine_Kmeans.txt"
set struct_cluster_filename=$cluster_Info_Location"/"$output_struct_filename
set funct_cluster_filename=$cluster_Info_Location"/"$output_funct_filename
echo "Struct Cluster File Name:"$struct_cluster_filename
echo "Funct Cluster File Name:"$funct_cluster_filename
echo "Struct Pickle Files Loc: $struct_Pickle_Folder"
echo "Funct Pickle Files loc: $funct_Pickle_Folder"
@ clusterCount = $property_count - 1

python3 ../../python_Scripts/clusterGen/clustering_v1.py --n $clusterCount  --input_dir $struct_Pickle_Folder --method kmeans --distance_metric cosine > $struct_cluster_filename
echo "Struct Cosine similarity printed on file "
python3 ../../python_Scripts/clusterGen/clustering_v1.py --n $clusterCount  --input_dir $funct_Pickle_Folder  --method kmeans --distance_metric cosine > $funct_cluster_filename
echo "Funct Cosine similarity printed on file "
echo "Open the the files  $struct_cluster_filename and  $funct_cluster_filename"



#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric euclidean > outputS-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric euclidean > outputF-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric manhattan > outputS-manhattan.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric manhattan > outputF-manhattan.txt
exit 0
