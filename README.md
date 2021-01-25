
# This is simple project for Diaolg-2019 conference news summarization task

There is a simple approach for extractive summarization: selecting best sentences from text. 
I use simple classification using perceptron with mean w2v vectors of sentence words.   

On held out test set model has following results:
```
====================================
| metric | precision | recall | f1 |
====================================
| rouge-1 | 0.1878293991156369 | 0.6420486830922776 | 0.280730641181189 |
| rouge-2 | 0.08961856806317192 | 0.3105037179892602 | 0.13436467474660363 |
| rouge-l | 0.17876295584133 | 0.5642170115264853 | 0.263445101324686 |
====================================
```

## Demonstration

First: download resources 

`https://drive.google.com/file/d/1L0ujNRNDnbFAMUFWOlrkxt6FN0K5d3j4/view?usp=sharing` 

Archive contains trained model, w2v embeddings and held out 10K test samples

Extract it's contents in project root

```
cd summarization/
tar -xvf resources.tar.gz
```

Build docker image

`docker build . -t summary`

To test model on 10K held out examples run

`docker run summary test`

To run webpage demo:

`docker run -p <PORT>:5000 summary demo`

Open to `localhost:<PORT>` to see it
