#!/bin/tcsh -f
#THIS SCRIPT GENERATES PICKLES(NON PCA) FOR A GIVEN AIG
#---After the discussion on fold ----
#Param 1: design name : e.g   6s199.aig
set startTime=`date +%H-%M-%S`" :Hrs"
set pickle_Holder_Folder='../../../output/Pickle_File'
#set designNameList=$1
set designPath='../../../multi'
#foreach fileName (`cat $designNameList `)
set fileName=$1
if  ($#argv != 1 )then
	echo "File name missing ..."
	exit 1
endif
set fileLocName=$designPath'/'$fileName
if ( ! -f  $fileLocName )then
	echo "File does not exists.."
	exit 2
endif
	
	#Extract the number of outputs -- note number of outputs is same as the number of properties
	#set NoOfOutputs = `abc -c "read $fileName;ps" | grep "i/o" | cut -d "/" -f3 | sed 's@\([0-9]\+\).*@\1@'`
	
	set NoOfOutputs = `abc -c "read $fileLocName ;fold;ps" | tr -s "  (" "|" | cut -d "|" -f 6 | tail -1` #Modification by SGR -- Note the fold
	
	set designName = `echo $fileName | cut -d  "." -f1`
	echo " Design Path:$designPath | Design name: $designName | NoOfOutputs: $NoOfOutputs | Pickle Holder Location: $pickle_Holder_Folder | FileLocName: $fileLocName"
	#exit 0

	set j = 0
	
	#Create folder with a design name
	mkdir $pickle_Holder_Folder/$designName

	rm -rf $pickle_Holder_Folder/$designName/funcPkl
	rm -rf $pickle_Holder_Folder/$designName/structPkl
	rm -rf $pickle_Holder_Folder/$designName/cone_prop_Dir
	mkdir -p $pickle_Holder_Folder/$designName/funcPkl
	mkdir -p  $pickle_Holder_Folder/$designName/structPkl
	mkdir -p $pickle_Holder_Folder/$designName/cone_prop_Dir
	
	echo  "\n 3 Folders created ..."
	# exit 0
	echo "Start Time: $startTime"
 	while ($j < $NoOfOutputs)
		#echo "\nCreating the cone for property $j"
		echo "Property Status [ $j / $NoOfOutputs ]"
		#Create COI of each property and save as aig file
		set propFileName = $designName"_p"$j.aig
		set coneFileLocName=$pickle_Holder_Folder'/'$designName'/'cone_prop_Dir'/'$propFileName
		#echo "Cone file name loc: $coneFileLocName" 
		set r1 = `abc -c "read $fileLocName;fold; cone -s -R 1 -O $j;scl;write $coneFileLocName"` # Note the fold ----

		#Create inductive unfolding of each property cone  file upto depth 1
		set propName = `echo $propFileName | cut -d "." -f1`
		set propFileNameInd = $propName"_ind1".aig
		#echo "Inductive name: $propFileNameInd"
		set inductiveFileLocName=$pickle_Holder_Folder'/'$designName'/'cone_prop_Dir'/'$propFileNameInd
		
		#echo "Inductive file name: $inductiveFileLocName"

		set r2 = `abc -c "read $coneFileLocName;&get;&frames -F 1 -s -a;&write $inductiveFileLocName"`

		#Create structural and functional embeddings for each inductive unfolding
		set structPklFileName = $propName"_ind1_s".pkl
		set funcPklFileName = $propName"_ind1_f".pkl
		~/workspace_Soumik/multiproperty/scripts/python_Scripts/rowavg_embedding.py -c $pickle_Holder_Folder/$designName/cone_prop_Dir/$propFileNameInd -o1 $pickle_Holder_Folder/$designName/structPkl/$structPklFileName -o2 $pickle_Holder_Folder/$designName/funcPkl/$funcPklFileName
		#mv *f.pkl ./funcPkl
		#mv *s.pkl ./structPkl
		#rm -rf *p*.aig #Removing the cone aig file

   		@ j++
 	end #--End of for each property 
#end #--End of foreach loop
endTime=`date +%s`
#@ timeDiff= (endTime - startTime)
#echo "\nTime Diff: $timeDiff"
echo "$fileName:$startTime:$endTime">> $pickle_Holder_Folder/pickle_Gen_Time.txt
