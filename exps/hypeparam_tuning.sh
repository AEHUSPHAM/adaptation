
for x in adam
do
    for m1 in 0.3 0.6
    do
        for m2 in 0.6 0.9
        do
            for s in 0.1 2 
            do
                bash exps/run_glue.sh $m1 $m2 $s sequential $x
            done 
        done
    done
done