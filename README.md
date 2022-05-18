# serve-tensorflow-model

## based on this idea
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

## Setup

Install tensorflow serving
* run `./setup.sh` 
* run `./serve.sh` and this will serve out model in directory named 1

### Tips

* Can serve model out in 60 seconds via docker:  https://github.com/tensorflow/serving#serve-a-tensorflow-model-in-60-seconds

```
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# Returns => { "predictions": [2.5, 3.0, 4.5] }
```

### What is next?

* Figure out how to serve out pre-trained models via tensorflow serve and tfhub



## References

* [Watch on O'Reilly](https://learning.oreilly.com/videos/52-weeks-live/051312022VIDEOPAIML/051312022VIDEOPAIML-c1_s0/)
* [Watch on YouTube](https://www.youtube.com/watch?v=KjLmN6MNKms)
