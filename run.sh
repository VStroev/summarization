
if [ $1 == "test" ]
then
  python3 src/scripts/summarization.py --embedding_model resources/model.model --model_path resources/model_best.pt --data_path resources/test_split.json.gz
else
  if [ $1 == "demo" ]
  then
    python3 src/scripts/demo.py --embedding_model resources/model.model --model_path resources/model_best.pt
  else
    echo 'uncknown command'
  fi
fi