#This script will run each command in this script
# Check # properties 
# Check Cluster type
# Check total time 1hr * #Properties
# Check the location of design and also the output log file
designName="../bob12m05m.aig"
designBaseName=`basename $designName .aig`
echo "Design:$designName"
(
prop="_0_4_9"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 3;swappos -N 9;swappos -N 5;cone -s  -O 3 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);

(
	echo "All Done ...$designName"
)
exit 0
