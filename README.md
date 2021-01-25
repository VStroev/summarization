
# This is simple project for Diaolg-2019 conference news summarization task

There is a simple approach for extractive summarization: selecting best sentences from text. 
I use simple classification using perceptron with mean w2v vectors of sentence words.   

## Installation
Build docker image

`docker build . -t summary`

To test model on 10K held out examples run

`docker run summary test`

To run webpage demo:

`docker run -p <PORT>:5000 summary demo`

Go to `localhost:<PORT>` to see it