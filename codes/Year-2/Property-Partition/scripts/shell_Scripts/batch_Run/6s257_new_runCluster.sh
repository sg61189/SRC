#This script will run each command in this script
# Check # properties 
# Check Cluster type
# Check total time 1hr * #Properties
# Check the location of design and also the output log file
designName="../6s257.aig"
designBaseName=`basename $designName .aig`
echo "Design:$designName"
(
prop="_1_3"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;swappos -N 2;cone -s  -O 2 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_2_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 2;swappos -N 3;cone -s  -O 3 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_2_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 2;swappos -N 3;cone -s  -O 2 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 3;cone -s  -O 3 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_3_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 2;cone -s  -O 2 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_1_2"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;cone -s  -O 1 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_1_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;swappos -N 3;cone -s  -O 3 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_1_3_4"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;swappos -N 2;cone -s  -O 2 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_2"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;cone -s  -O 1 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName"
);
(
prop="_0_1"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;cone -s  -O 0 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName"
);
(
prop="_0_1_3"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;swappos -N 2;cone -s  -O 1 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName"
);
(
	echo "All Done ...$designName"
)
exit 0
