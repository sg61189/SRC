rm ./design_maxDepth_Engine_info.csv
echo "Design|Max-Depth|Engine" > ./design_maxDepth_Engine_info.csv
for design_outer in `cat ./used_circuit_Name.txt`
do
    #echo $design
    maxDepth=-99
    Flag_File_Exists=1
    for engine in `cat ./BMC_Engine_List.txt`
    do
    #echo $engine
        for design_inner in `cat ./used_circuit_Name.txt`    
        do

            echo "Compare $design_outer and $design_inner"
            path="./train_data_csv/"$engine"/"
            fileName=$path$design_inner
            if [ -e  $fileName ] 
            then
                if [ $design_outer = $design_inner ] 
                then
                    #echo "Found..."
                    tempdepth=`cat $fileName | cut -d ","  -f 1 | tail -1`
                    if [ $tempdepth -gt $maxDepth ] 
                    then
                        maxDepth=$tempdepth
                        maxDepthEngine=$engine
                    fi    
                fi
            else
                #File does not exists
                Flag_File_Exists=0
                
                break
            fi
        done #End of iiner design
        if [ $Flag_File_Exists -eq 0 ]
        then
            echo "$design_outer does not exists in $engine "
            break
        fi
    done #End of engine loop
    if [ $maxDepth -gt  0 ]
    then
        echo "$design_outer|$maxDepth|$maxDepthEngine" >> ./design_maxDepth_Engine_info.csv
    fi
    
done #End of the outer design
echo "Open the file ./design_maxDepth_Engine_info.csv"
exit 0