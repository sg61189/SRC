#!/bin/tcsh -f
set startTime=`date +%H-%M-%S`" :Hrs"
foreach fileName (`ls *.aig`)

	#Extract the number of outputs -- note number of outputs is same as the number of properties
	#set NoOfOutputs = `abc -c "read $fileName;ps" | grep "i/o" | cut -d "/" -f3 | sed 's@\([0-9]\+\).*@\1@'`
	set NoOfOutputs = `abc -c "read $fileName;ps" | tr -s "  (" "|" | cut -d "|" -f 6 | tail -1` #Modification by SGR
	
	set designName = `echo $fileName | cut -d  "." -f1`
	echo "\n Design name: $designName \t NoOfOutputs: $NoOfOutputs"
	#exit 0

	set j = 0

	rm -rf ./reduced_funcPkl
	rm -rf ./reduced_structPkl
	rm -rf ./cone_prop_Dir
	mkdir -p ./reduced_funcPkl
	mkdir -p ./reduced_structPkl
	mkdir -p ./cone_prop_Dir

	echo  "\n 3 Folders created ..."
	echo "Start Time: $startTime"
 	while ($j < $NoOfOutputs)
		#echo "\nCreating the cone for property $j"
		echo "Property Status [ $j / $NoOfOutputs ]"
		#Create COI of each property and save as aig file
		set propFileName = $designName"_p"$j.aig
		set r1 = `abc -c "read $fileName; cone -s -R 1 -O $j;scl;write ./cone_prop_Dir/$propFileName"` ###fold must be there

		#Create inductive unfolding of each property file upto depth 1
		set propName = `echo $propFileName | cut -d "." -f1`
		set propFileNameInd = $propName"_ind1".aig
		echo "\nInductiva name: $propFileNameInd"
		set r2 = `abc -c "read ./cone_prop_Dir/$propFileName;&get;&frames -F 1 -s -a;&write ./cone_prop_Dir/$propFileNameInd"`

		#Create structural and functional embeddings for each inductive unfolding
		set structPklFileName = $propName"_ind1_s".pkl
		set funcPklFileName = $propName"_ind1_f".pkl
		../../reduced_PCA_embedding_gen.py -c ./cone_prop_Dir/$propFileNameInd -o1 ./reduced_structPkl/$structPklFileName -o2 ./reduced_funcPkl/$funcPklFileName

		#mv *f.pkl ./funcPkl
		#mv *s.pkl ./structPkl
		#rm -rf *p*.aig #Removing the cone aig file

   		@ j++
 	end
end
#endTime=`date +%s`
#@ timeDiff= (endTime - startTime)
#echo "\nTime Diff: $timeDiff"
