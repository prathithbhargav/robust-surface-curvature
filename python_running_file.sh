# this is to run the python file
plist=(10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0 14.5 15.0)
for i in ${plist[@]}; do
#     echo "submit a simulation with this parameter:"
#     echo "$i"
#     echo 'prot' 'output_1' "$i"
    python3 shape_complimentarity.py 'Protein-inhibitor' 'output_protein-ligand' "$i" >> run_detailsw.txt
done

echo "PID of this script: $$" >> run_details.txt
