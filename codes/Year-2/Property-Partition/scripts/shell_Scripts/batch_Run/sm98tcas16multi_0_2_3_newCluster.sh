#This script will run each command in this script
# Check # properties 
# Check Cluster type
# Check total time 1hr * #Properties
# Check the location of design and also the output log file
designName="../sm98tcas16multi.aig"
designBaseName=`basename $designName .aig`
(
prop="_0_2_3"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;cone -s  -O 1 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName"
);

(
	echo "All Done ...$designName"
)
exit 0
