# this file is written for the generation of sh files
folder_name='210_pdbs'
cd $folder_name
for file in *.pdb
do 
filename=$(basename -- "$file")
extension="${filename##*.}"
filename="${filename%.*}"
echo $filename # only file's name without extension
echo $file # full file name with extension
output_name=$filename'.dms'
echo $output_name
dms $file -a -d 3.0 -n -w 1.5 -o $output_name
echo "done with"$file
done

# for i in $FILE
# 	do
# 		echo $i
#         # echo "done with one file"
# 	done

