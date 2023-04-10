START=$(ps -p $1 -o vsz= );
MAX=$START;
echo "0" > $2;
while sleep 0.5; do
    VAR=$(ps -p $1 -o vsz= );
    if [ -z "$VAR" ]; then
        exit 0;
    fi;
    if [ $VAR -gt $MAX ]; then 
        MAX=$VAR;
        echo "$(($MAX-$START))" > $2;
    fi;
done;