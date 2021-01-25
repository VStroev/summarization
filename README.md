
# This is simple project for Diaolg-2019 conference news summarization task

There is a simple approach for extractive summarization: selecting best sentences from text. 
I use simple classification using perceptron with mean w2v vectors of sentence words.   

## Demonstration

First: download resources 

`https://drive.google.com/file/d/1L0ujNRNDnbFAMUFWOlrkxt6FN0K5d3j4/view?usp=sharing` 

Archive contains trained model, w2v embeddings and held out 10K test samples

Build docker image

`docker build . -t summary`

To test model on 10K held out examples run

`docker run summary test`

To run webpage demo:

`docker run -p <PORT>:5000 summary demo`

Open to `localhost:<PORT>` to see it