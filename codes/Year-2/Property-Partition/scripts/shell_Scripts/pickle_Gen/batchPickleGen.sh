#This script generates both NON PCA and PCA Pickles
#Param 1: File containg all aig file names
	csh ./generateAllfiles.csh $1 && echo "Non PCA Pickle Gen Done "
	csh  ./generateAllfiles_Reduced_Tensor.csh $1 && echo "PCA Pickle Gen Done" &&
	echo "All Pickle Gen Done"
exit 0
