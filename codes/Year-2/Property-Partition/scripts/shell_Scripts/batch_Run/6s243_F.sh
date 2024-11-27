#This script will run each command in this script
# Check # properties 
# Check Cluster type
# Check total time 1hr * #Properties
# Check the location of design and also the output log file
designName="../6s243.aig"
designBaseName=`basename $designName .aig`
echo "Design:$designName"
(
prop="_0_1_2_6"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 3;swappos -N 6;swappos -N 3;cone -s  -O 0 -R 4 ;scl;bmc3 -g -a -v -T 14400" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_4_7"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 4;swappos -N 6;cone -s  -O 6 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_1_2_3_6"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 4;swappos -N 6;swappos -N 4;cone -s  -O 0 -R 5 ;scl;bmc3 -g -a -v -T 18000" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_1_2_3_5_6"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 5;swappos -N 4;swappos -N 6;swappos -N 5;cone -s  -O 0 -R 6 ;scl;bmc3 -g -a -v -T 21600" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_5"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 4;cone -s  -O 4 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
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
prop="_3_6"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 3;swappos -N 5;cone -s  -O 5 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_0_2_5"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 1;swappos -N 5;swappos -N 3;cone -s  -O 1 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);
(
prop="_1_3_6"
outputfileName=$designBaseName$prop".txt"
echo -e "Properties:$prop\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read $designName;ps;fold;ps;swappos -N 3;swappos -N 2;swappos -N 6;swappos -N 3;cone -s  -O 1 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/$outputfileName &&
echo "Done\t Open the file : ./abc_output_Cluster/$outputfileName" 
);

(
	echo "All Done ...$designName"
)
exit 0
