ML
---

## Requirements

- `nvidia-docker`

## Getting Started


- build docker image
  ```
  make build
  ```

- start ml server (`0.0.0.0:5000`)

  ```
  make start
  ```

- detect human pose
  
  Send http POST request to `/infer`. Please include the input image data directly in the request body.

## TODO:
  - move classifier to other repository and install it via git in dockerfile


## (survey) how to estimate human pose

  - posenet
    - [Pose Detection in the Browser: PoseNet Model](https://github.com/tensorflow/tfjs-models/tree/master/posenet)

  - openpose(cmu)

    - [ildoonet/tf-pose-estimation: Deep Pose Estimation implemented using Tensorflow with Custom Architectures for fast inference.](https://github.com/ildoonet/tf-pose-estimation)

    - [【TensorFlow版】MacBookで行うOpenPose （osx Mojave対応） - Qiita](https://qiita.com/mdo4nt6n/items/d9523aff14dd9fb70c37)


### FYI

  - [hellock/icrawler: A multi-thread crawler framework with many builtin image crawlers provided.](https://github.com/hellock/icrawler)
  - [Big Sky :: golang で tensorflow のススメ](https://mattn.kaoriya.net/software/lang/go/20180825013735.htm)
