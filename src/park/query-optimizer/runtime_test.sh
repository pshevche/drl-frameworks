
#query=simple-queries/0.sql
#queries=("simple-queries/0.sql" "join-order-benchmark/10a.sql")
#declare -a queries=("simple-queries/0.sql" "join-order-benchmark/10a.sql")
declare -a queries=("join-order-benchmark/10a.sql"
"join-order-benchmark/17a.sql" "join-order-benchmark/20a.sql")
for i in {1..10}
do
  ## now loop through the above array
  for query in "${queries[@]}"
  do
  echo $query
  ./drop_cache.sh
  echo "run $i"
  sleep 3
  START_TIME=$SECONDS
  psql -d imdb < $query
  ELAPSED_TIME=$(($SECONDS - $START_TIME))
  echo "query: $query, elapsed time: $ELAPSED_TIME"
  done
done
