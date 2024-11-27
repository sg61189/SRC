#This script will run each command in this script
# Check # properties 
# Check Cluster type
# Check total time 1hr * #Properties
# Check the location of design and also the output log file
echo "Design:6s267"
(
echo -e "Properties:0,2\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read ../6s267.aig;ps;fold;ps;swappos -N 1;cone -s  -O 1 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/6s267_0_2.txt &&
echo "Done\t Open the file : ./abc_output_Cluster/6s267_0_2.txt" 
);
(
echo -e "Properties:1,3\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read ../6s267.aig;ps;fold;ps;swappos -N 3;swappos -N 1;cone -s  -O 0 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/6s267_1_3.txt &&
echo "Done\t Open the file : ./abc_output_Cluster/6s267_1_3.txt"
);
(
echo -e "Properties:0,1\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read ../6s267.aig;ps;fold;ps;cone -s  -O 0 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/6s267_0_1.txt &&
echo "Done\t Open the file : ./abc_output_Cluster/6s267_0_1.txt"
);
(
echo -e "Properties:2,3\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read ../6s267.aig;ps;fold;ps;cone -s  -O 2 -R 2 ;scl;bmc3 -g -a -v -T 7200" >../abc_output_Cluster/6s267_2_3.txt &&
echo "Done\t Open the file : ./abc_output_Cluster/6s267_2_3.txt"
);
(
echo -e "Properties:0,2,3\t Start: `date +%H-%M-%S`: Hrs" &&
/home/ecsu/installed_Softwares/abc/abc -c  "read ../6s267.aig;ps;fold;ps;swappos -N 1;cone -s  -O 1 -R 3 ;scl;bmc3 -g -a -v -T 10800" >../abc_output_Cluster/6s267_0_2_3.txt &&
echo "Done\t Open the file : ./abc_output_Cluster/6s267_0_2_3.txt"
);
(
	echo "All Done ...6s267.aig"
)
exit 0
