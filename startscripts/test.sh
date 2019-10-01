#!/bin/bash
for i in {6000..18000..1000}
do
   echo "testing $i-weights"
   /home/birgit/ML/myDarknet/darknet detector map cfg/person.data cfg/yolo-person.cfg backup/yolo-person_${i}.weights > results/person_${i}_map.txt &
   wait $!
   echo "$i-weights" >> results/analysis.txt
   sed -n '7p' < results/person_${i}_map.txt >> results/analysis.txt
   echo "finished $i test"
done

