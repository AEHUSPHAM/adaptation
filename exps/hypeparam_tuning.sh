for x in origin threshold slided
do
    for option in sequential
    do
        for i in 0.6
        do
            for j in 0.8
            do
                bash exps/run_glue.sh $i $j $option $x
            done 
        done
    done
done