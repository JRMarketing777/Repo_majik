

## Google Cloud Run Cheatsheet for Deploying Containers

### **Overview**

Google Cloud Run is a managed compute platform that allows you to run containers directly on Google's scalable infrastructure. It abstracts away all infrastructure management, enabling you to focus on building and deploying applications quickly and efficiently.

### **1. Prerequisites**

Before deploying to Google Cloud Run, ensure you have the following:
- **Google Cloud Account**: Sign up at [Google Cloud](https://cloud.google.com/).
- **Google Cloud Project**: Create a project in the Google Cloud Console.
- **Billing Enabled**: Ensure billing is enabled for your project.
- **Docker Installed**: Install Docker from the [Docker website](https://www.docker.com/products/docker-desktop).
- **Google Cloud SDK Installed**: Install the Google Cloud SDK from the [Google Cloud SDK documentation](https://cloud.google.com/sdk/docs/install).

### **2. Setting Up Your Environment**

#### **Install Google Cloud SDK**
Follow the installation guide for your operating system from the [Google Cloud SDK documentation](https://cloud.google.com/sdk/docs/install).

#### **Initialize Google Cloud SDK**
```sh
gcloud init
```
This command will guide you through setting up your Google Cloud project and authentication.

### **3. Writing Your Application Code**

Create a simple Python web server using Flask:

**app.py**
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
```

### **4. Creating a Dockerfile**

A Dockerfile is a script that contains a series of instructions on how to build a Docker image, which is a lightweight, standalone, and executable package that includes everything needed to run a piece of software.

**Dockerfile**
```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
```

### **5. Building and Pushing the Docker Image**

#### **Build the Docker Image**
```sh
docker build -t gcr.io/your-project-id/my-python-app .
```
- **docker build**: Command to build a Docker image.
- **-t**: Tags the image with a name.
- **gcr.io/your-project-id/my-python-app**: The name of the image, including the Google Container Registry (GCR) URL and project ID.

#### **Push the Docker Image to Google Container Registry**
```sh
gcloud auth configure-docker
docker push gcr.io/your-project-id/my-python-app
```
- **gcloud auth configure-docker**: Configures Docker to use gcloud as a credential helper.
- **docker push**: Pushes the Docker image to the specified registry.

### **6. Deploying to Google Cloud Run**

#### **Using Google Cloud Console**

1. **Go to Cloud Run**: In the Google Cloud Console, navigate to Cloud Run.
2. **Create Service**: Click on "Create Service".
3. **Select Container Image**: Choose "Deploy one revision from an existing container image".
4. **Specify Image URL**: Enter the image URL (e.g., `gcr.io/your-project-id/my-python-app`).
5. **Service Name**: Enter a service name.
6. **Region**: Select the region where you want to deploy your container.
7. **Authentication**: Select "Allow unauthenticated invocations" if you want your service to be publicly accessible.
8. **Deploy**: Click "Create" and wait for the deployment to complete.

#### **Using gcloud CLI**

1. **Deploy the Container Image**
```sh
gcloud run deploy my-python-app \
  --image gcr.io/your-project-id/my-python-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
- **gcloud run deploy**: Command to deploy a container to Cloud Run.
- **--image**: Specifies the container image to deploy.
- **--platform managed**: Deploys to the fully managed Cloud Run platform.
- **--region**: Specifies the region for deployment.
- **--allow-unauthenticated**: Allows public access to the service.

### **7. Configuring Cloud Run Services**

#### **Setting Environment Variables**
Environment variables are used to configure your application without changing code.

**Using gcloud CLI**
```sh
gcloud run services update my-python-app \
  --update-env-vars NAME=World
```
- **--update-env-vars**: Updates environment variables for the service.

#### **Setting CPU and Memory Limits**
You can configure resource limits to ensure your service runs efficiently.

**Using gcloud CLI**
```sh
gcloud run services update my-python-app \
  --memory 512Mi \
  --cpu 1
```
- **--memory**: Sets the memory limit.
- **--cpu**: Sets the CPU limit.

### **8. Monitoring and Logging**

#### **Viewing Logs**
Logs are automatically ingested by Cloud Logging. You can view them in the Google Cloud Console.

**Using gcloud CLI**
```sh
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=my-python-app"
```
- **gcloud logging read**: Command to read logs from Cloud Logging.

#### **Monitoring Performance**
You can monitor the performance of your Cloud Run services using Google Cloud Console's monitoring tools.

### **9. Continuous Deployment**

You can set up continuous deployment using Cloud Build and GitHub.

#### **Example Cloud Build Configuration**

**cloudbuild.yaml**
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/my-python-app', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-python-app']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'my-python-app', '--image', 'gcr.io/$PROJECT_ID/my-python-app', '--region', 'us-central1', '--platform', 'managed', '--allow-unauthenticated']
```
- **steps**: Defines the steps to build, push, and deploy the Docker image.
- **gcr.io/cloud-builders/docker**: Uses the Docker builder to build and push images.
- **gcr.io/cloud-builders/gcloud**: Uses the gcloud builder to deploy the image.

#### **Triggering Builds**
You can configure Cloud Build triggers to automatically build and deploy your container whenever you push changes to your repository.

### **10. Best Practices**

- **Use Multi-Stage Builds**: Optimize your Dockerfile to reduce image size.
- **Security**: Regularly update your base images and scan for vulnerabilities.
- **Resource Management**: Set appropriate CPU and memory limits to optimize performance and cost.
- **Logging and Monitoring**: Regularly check logs and monitor performance to ensure your service is running smoothly.

### **Conclusion**

By following this cheatsheet, you can quickly and efficiently deploy your applications using Google Cloud Run. This guide covers the essential steps and best practices to ensure a smooth deployment process, from writing your application code to monitoring and maintaining your services.

