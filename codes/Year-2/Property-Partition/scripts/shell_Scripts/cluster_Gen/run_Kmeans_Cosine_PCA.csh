#Step 1: Activate the deep-gate2 environment from soumik-experiment-env
#source ~/workspace_Soumik/soumik-experiment-env/bin/activate #--source not found error
#-----THIS SCRIPT FOR KMEANS AND PCA------
set designName=$1
set property_count=$2
if ( $#argv  !=  2 )then
        echo "Input missing ..."
        echo "Param 1: Design name (6s199)| Param 2: Nomber of properties"
        exit 1
endif
set pickle_File_Location='../../../output/Pickle_File'
set struct_Pickle_Folder=$pickle_File_Location"/"$designName'/reduced_structPkl'
set funct_Pickle_Folder=$pickle_File_Location"/"$designName'/reduced_funcPkl'
set cluster_Info_Location='../../../output/cluster_output/'$designName
echo "Cluser Info Loc: $cluster_Info_Location"
mkdir $cluster_Info_Location

set output_struct_filename=$designName"_outputS-cosine_Kmeans_PCA.txt"
set output_funct_filename=$designName"_outputF-cosine_Kmeans_PCA.txt"

set struct_cluster_filename=$cluster_Info_Location'/'$output_struct_filename
set funct_cluster_filename=$cluster_Info_Location'/'$output_funct_filename
@ clusterCount = $property_count - 1
echo "No of property: $property_count|Cluster no:$clusterCount"

python3 ../../python_Scripts/clusterGen/clustering_v1.py  --n $clusterCount --input_dir $struct_Pickle_Folder  --method kmeans --distance_metric cosine > $struct_cluster_filename
echo "(PCA)Struct Cosine similarity printed on file "
python3 ../../python_Scripts/clusterGen/clustering_v1.py  --n $clusterCount  --input_dir $funct_Pickle_Folder --method kmeans --distance_metric cosine > $funct_cluster_filename
echo "(PCA)Funct Cosine similarity printed on file "
echo "Open the the files  $struct_cluster_filename and $funct_cluster_filename"



#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric euclidean > outputS-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric euclidean > outputF-euclidean.txt
#python3 clustering.py --input_dir pickles/prop1Structural --method kmedoids --distance_metric manhattan > outputS-manhattan.txt
#python3 clustering.py --input_dir pickles/prop1Functional --method kmedoids --distance_metric manhattan > outputF-manhattan.txt
exit 0
