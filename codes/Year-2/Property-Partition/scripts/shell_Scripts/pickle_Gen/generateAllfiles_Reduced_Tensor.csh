#!/bin/tcsh -f
#----Generate s the  Pickles(PCA) for a given aig file
#---After the discussion on fold ----
set startTime=`date +%H-%M-%S`" :Hrs"
set pickle_Holder_Folder='../../../output/Pickle_File'
#set designNameList=$1
set designPath='../../../multi'

#foreach fileName (`cat $designNameList`)
if ( $#argv != 1 )then
	echo "File name missing ..."
	exit 1
endif
set fileName=$1
set fileLocName=$designPath'/'$fileName
if ( ! -f $fileLocName )then
	echo "File does not exists ... "
	exit 2
endif


	#Extract the number of outputs -- note number of outputs is same as the number of properties
	#set NoOfOutputs = `abc -c "read $fileName;ps" | grep "i/o" | cut -d "/" -f3 | sed 's@\([0-9]\+\).*@\1@'`
	
	#set NoOfOutputs = `abc -c "read $fileName;fold;ps" | tr -s "  (" "|" | cut -d "|" -f 6 | tail -1` #Modification by SGR -- Note the fold---
	set NoOfOutputs = `abc -c "read $fileLocName ;fold;ps" | tr -s "  (" "|" | cut -d "|" -f 5 | tail -1` #Modification by SGR -- Note the fold
	
	set designName = `echo $fileName | cut -d  "." -f1`
	echo " Design Path:$designPath | Design name: $designName | NoOfOutputs: $NoOfOutputs | Pickle Holder Location: $pickle_Holder_Folder | FileLocName: $fileLocName"
	#exit 0

	set j = 0
	mkdir $pickle_Holder_Folder/$designName
	rm -rf $pickle_Holder_Folder/$designName/reduced_funcPkl
	rm -rf $pickle_Holder_Folder/$designName/reduced_structPkl
	rm -rf $pickle_Holder_Folder/$designName/cone_prop_Dir
	mkdir -p $pickle_Holder_Folder/$designName/reduced_funcPkl
	mkdir -p $pickle_Holder_Folder/$designName/reduced_structPkl
	mkdir -p $pickle_Holder_Folder/$designName/cone_prop_Dir

	echo  "\n 3 Folders created ..."
	echo "Start Time: $startTime"
 	while ($j < $NoOfOutputs)
		#echo "\nCreating the cone for property $j"
		echo "Property Status [ $j / $NoOfOutputs ]"
		#Create COI of each property and save as aig file
		set propFileName = $designName"_p"$j.aig
		set coneFileLocName=$pickle_Holder_Folder'/'$designName'/'cone_prop_Dir'/'$propFileName
		#set r1 = `abc -c "read $fileName;fold;cone -s -R 1 -O $j;scl;write ./cone_prop_Dir/$propFileName"` #--Note the fold----
		set r1 = `abc -c "read $fileLocName;fold; cone -s -R 1 -O $j;scl;write $coneFileLocName"` # Note the fold ----
		#Create inductive unfolding of each property file upto depth 1
		set propName = `echo $propFileName | cut -d "." -f1`
		set propFileNameInd = $propName"_ind1".aig
		set inductiveFileLocName=$pickle_Holder_Folder'/'$designName'/'cone_prop_Dir'/'$propFileNameInd
		#echo "\nInductive name: $propFileNameInd"
		#set r2 = `abc -c "read ./cone_prop_Dir/$propFileName;&get;&frames -F 1 -s -a;&write ./cone_prop_Dir/$propFileNameInd"`
		set r2 = `abc -c "read $coneFileLocName;&get;&frames -F 1 -s -a;&write $inductiveFileLocName"`
		#Create structural and functional embeddings for each inductive unfolding
		set structPklFileName = $propName"_ind1_s".pkl
		set funcPklFileName = $propName"_ind1_f".pkl
		#~/workspace_Soumik/multiproperty/scripts/python_Scripts/reduced_PCA_embedding_gen.py -c ./cone_prop_Dir/$propFileNameInd -o1 ./reduced_structPkl/$structPklFileName -o2 ./reduced_funcPkl/$funcPklFileName
		~/workspace_Soumik/multiproperty/scripts/python_Scripts/reduced_PCA_embedding_gen.py -c $pickle_Holder_Folder/$designName/cone_prop_Dir/$propFileNameInd -o1 $pickle_Holder_Folder/$designName/reduced_structPkl/$structPklFileName -o2 $pickle_Holder_Folder/$designName/reduced_funcPkl/$funcPklFileName
		#mv *f.pkl ./funcPkl
		#mv *s.pkl ./structPkl
		#rm -rf *p*.aig #Removing the cone aig file

   		@ j++
 	end #--End of for each proprty loop
#end #--End of the for each design loop
#endTime=`date +%s`
#@ timeDiff= (endTime - startTime)
#echo "\nTime Diff: $timeDiff"
