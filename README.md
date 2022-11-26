# AIM406: Tune performance and optimize ML inference using Amazon SageMaker

In this repository we will explore the testing and tuning process of maximizing performance of a TensorFlow model on a SageMaker Real-Time Endpoint. In this repository the following code samples and artifats are shared:

- <b>Notebook</b>
  - TensorFlow U-Net 50 SageMaker Real-Time Endpoint Creation & Inference
  - TensorFlow Serving Container Environment Variable Tuning
  - Neo Compilation
  - AutoScaling
  
- <b>Distributed Locust Scripts</b>: The load test we will be running live for the demo. For this example we will run this on an EC2 instance (c6i.32xlarge), but you can run on your own client side setup as long as it has the necessary compute power to handle the load you want to test at. We will be using Locust for this example as our third party testing tool.

  - locust_script.py: Contains the stress testing code with your sample inference/invoke, you can alter this to invoke multiple different payloads if you want.
  - distributed.sh: Sets up distributed locust, you can tune workers and users to further increase the transactions per second from the test.

NOTE: This repository assumes understanding of AWS and SageMaker Inference fundamentals. While the steps to deploy a SageMaker endpoint are shown, if you would like a deeper understanding of deploying models on SageMaker reference this [blog](https://aws.amazon.com/blogs/machine-learning/getting-started-with-deploying-real-time-models-on-amazon-sagemaker/).


## Steps

