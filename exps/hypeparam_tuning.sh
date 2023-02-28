
for x in bidirectional
do
    for m1 in 0.3
    do
        for m2 in 0.2
        do
            for s in 1
            do
                bash exps/run_glue.sh $m1 $m2 $s lora $x
            done 
        done
    done
done