# Colocated Python

## Purpose

This package provides the Colocated Python Sidecar implementation. It describes how to build the sidecar container image with custom Python dependencies. This sidecar container runs on the TPU workers and facilitates remote Python code execution, enabling seamless integration between the user code and Python-based tasks on the TPU worker.

**Why use Colocated Python?**

Colocated Python enables users to execute code that runs explicitly on a specified set of TPU VMs using simple annotations and sharding information. This increases throughput on data or I/O intensive tasks like data loading (as implemented in [MaxText's RemoteIterator class](https://github.com/AI-Hypercomputer/maxtext/blob/391a5a788d85cae8942334b042fdabdbd549af51/MaxText/multihost_dataloading.py#L175)).

## Examples

### Simple (No User Dependencies)

The following small example is modified from [JAX](https://github.com/jax-ml/jax/blob/f4c727abb3989048f49e3d9a4bf2e4052969974b/tests/colocated_python_test.py#L78-L89) with no additional user dependencies installed. It shows how you can use the JAX Colocated Python API to create a file on the specified TPU worker.

```python
import jax
from jax.experimental import colocated_python
from jax.experimental.colocated_python import serialization

@colocated_python.colocated_python
def create_a_file(dummy):
  """
  Creates a simple file on the TPU worker.
  """
  filename = "my_new_file.txt"
  content_to_write = f"This is written on TPU worker {jax.process_id}"

  try:
    with open(filename, 'w', encoding='utf-8') as file:
      file.write(content_to_write)
      print(f"Content written to '{filename}'.")

    print(f"File '{filename}' created and closed.")
  except IOError as e:
    print(f"An error occurred: {e}")

  return dummy

devices = jax.devices()
dummy_array = np.array(1)
dummy_array = jax.device_put(dummy_array, devices[0])

out = create_a_file(dummy_array)
```

### Medium (With User Dependencies)

What if you want to add your own dependencies to do more advanced logic?

The following is a simple line chart of the first 5 primes in matplotlib that is saved locally to the TPU worker.

```python
import jax
import numpy as np
from jax.experimental import colocated_python
from jax.experimental.colocated_python import serialization

# User added dependency
import matplotlib.pyplot as plt

@colocated_python.colocated_python
def create_and_save_primes_plot(dummy):
  """
  Creates a simple matplotlib line plot and saves it as a PNG image
  on the TPU worker.
  """
  worker_id = jax.process_id()
  plot_filename = f"simple_line_plot_worker_{worker_id}.png"

  # Sample data for the plot
  x_data = np.array([1, 2, 3, 4, 5])
  y_data = np.array([2, 3, 5, 7, 11])

  try:
    # Create the line plot
    plt.figure(figsize=(6, 4))
    plt.plot(x_data, y_data, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel("Nth Prime")
    plt.ylabel("Primes")
    plt.title(f"Simple Plot from TPU Worker {worker_id}")
    plt.grid(True)

    # Save the plot to the specified file
    plt.savefig(plot_filename)
    print(f"Plot successfully saved to '{plot_filename}' on worker {worker_id}.")

    plt.close()
  except Exception as e:
    print(f"An error occurred on worker {worker_id} while creating/saving the plot: {e}")

  return dummy

devices = jax.devices()
dummy_np_array = np.array(1, dtype=np.float32)
dummy_device_array = jax.device_put(dummy_np_array, devices[0])
out = create_and_save_plot(dummy_device_array)
```

### Advanced Usage (With User Dependencies and Control Flow Logic)

For more advanced usage (such as data loading), check out [MaxText's RemoteIterator class](https://github.com/AI-Hypercomputer/maxtext/blob/391a5a788d85cae8942334b042fdabdbd549af51/MaxText/multihost_dataloading.py#L175).

### Verification

To verify files were created, SSH into one of the TPU workers using the following command and check that the file was created.

`kubectl exec -it <pod_name> -- /bin/sh -c "cat my_new_file.txt"`

Logs can also be verified by tailing the pod.

`kubectl logs -f <pod_name>`

## Installation and Usage

Follow these steps to set up, build, and deploy your application with the Colocated Python sidecar.

**Prerequisites**

Ensure [Docker](https://docs.docker.com/engine/install/) is installed on your system along with [gcloud](https://cloud.google.com/sdk/docs/install). Ensure you are authenticated into gcloud.

**1. Clone the Repository**

Get the necessary code and scripts.

```bash
git clone https://github.com/AI-Hypercomputer/pathways-utils.git
cd pathways-utils
```

**2. Prepare Sidecar Dependencies**

Update the file named `requirements.txt`. List all the additional Python packages you need specifically for the sidecar environment, one package per line.

These dependencies may be the same as your main workload's dependencies.

```
# Example requirements.txt
jax>=0.5.1
tensorflow-datasets
tiktoken
grain-nightly>=0.0.10
sentencepiece==0.1.97
```

**3. Build the Colocated Python Sidecar Image and upload it to Artifact Registry**

Use the provided Dockerfile to create the sidecar image. This image will contain the required dependencies in your `requirements.txt`. Also specify the image location to upload to in Artifact Registry

```bash
export PROJECT_ID=<your_project_id>
export IMAGE_LOCATION=us-docker.pkg.dev/${PROJECT_ID}/colocated-python:latest

docker build -t ${IMAGE_LOCATION} .
```

**4. Update Deployment Configuration**

Modify your Kubernetes deployment YAML file to use your colocated python sidecar image. This assumes you are using the [pathways-job](https://github.com/google/pathways-job) api.

For example.

```yaml
...
spec:
  maxRestarts: 0
  customComponents:
  - componentType: colocated_python_sidecar
    image: us-docker.pkg.dev/<your_project_id>/colocated-python:latest
...
```

For a full sample Yaml, please refer to [pathways-job](https://github.com/google/pathways-job/blob/main/config/samples/colocated_python_example_pathwaysjob.yaml).

**5. (Optional) Turn on Data Loading Optimization in MaxText**

If using MaxText, to turn on the data loading optimization that uses Colocated Python feature.

```python
colocated_python_data_input = True
```

**6. Deploy the Application**

Apply the updated deployment configuration to your Kubernetes cluster:

```bash
kubectl apply -f path/to/your/deployment.yaml
```

This will create the necessary pods with your application, pathways head, and the Colocated Python sidecar containers.

## The Sharp Bits 🔪

**User Dependency Conflicts**

Colocated Python relies on specific internal dependencies, including JAX. Refer to the provided `server_requirements.txt` for the complete list of required dependencies. Using a different dependency version than the one provided in `server_requirements.txt` will cause the remote Python image build to fail.

