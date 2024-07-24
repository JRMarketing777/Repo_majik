

#Here's an in-depth cheat sheet for a freelance software developer to learn Docker concepts and deploy software quickly. This guide will walk you through the process, teaching the basic concepts and providing practical examples.

## Docker Cheat Sheet for Freelance Software Developers

### **1. Basic Docker Concepts**

#### **Containers**
- **Definition**: Containers are lightweight, standalone, and executable packages that include everything needed to run a piece of software, including the code, runtime, system tools, libraries, and settings.
- **Commands**:
  - List running containers: `docker ps`
  - List all containers (including stopped): `docker ps -a`
  - Start a container: `docker start <container_name>`
  - Stop a container: `docker stop <container_name>`
  - Remove a container: `docker rm <container_name>`

#### **Images**
- **Definition**: Docker images are read-only templates used to create containers. They include the application and its dependencies.
- **Commands**:
  - List images: `docker images`
  - Build an image: `docker build -t <image_name> .`
  - Remove an image: `docker rmi <image_name>`
  - Pull an image from Docker Hub: `docker pull <image_name>`
  - Push an image to Docker Hub: `docker push <username>/<image_name>`

### **2. Dockerfile**

#### **Definition**
A Dockerfile is a text file that contains a series of instructions on how to build a Docker image.

#### **Example Dockerfile**
```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

#### **Commands**
- Build an image: `docker build -t my-python-app .`
- Run a container: `docker run -p 4000:80 my-python-app`

### **3. Docker Compose**

#### **Definition**
Docker Compose is a tool for defining and running multi-container Docker applications. It uses a YAML file to configure the application’s services.

#### **Example docker-compose.yml**
```yaml
version: '3'
services:
  web:
    build: .
    ports:
      - "4000:80"
  redis:
    image: "redis:alpine"
```

#### **Commands**
- Start services: `docker-compose up`
- Stop services: `docker-compose down`
- View logs: `docker-compose logs`

### **4. Deployment**

#### **Push to Docker Hub**
1. **Tag the Image**:
   ```sh
   docker tag my-python-app:latest your-dockerhub-username/my-python-app:latest
   ```
2. **Login to Docker Hub**:
   ```sh
   docker login
   ```
3. **Push the Image**:
   ```sh
   docker push your-dockerhub-username/my-python-app:latest
   ```

#### **Deploy to a Cloud Service**
1. **Google Cloud Run**:
   - **Deploy using gcloud CLI**:
     ```sh
     gcloud run deploy --image gcr.io/your-project-id/my-python-app --platform managed
     ```

2. **AWS ECS**:
   - **Deploy using AWS CLI**:
     ```sh
     aws ecs create-cluster --cluster-name my-cluster
     aws ecs create-service --cluster my-cluster --service-name my-service --task-definition my-task
     ```

### **5. Monitoring and Maintenance**

#### **Monitor Containers**
- View resource usage stats: `docker stats`
- View container logs: `docker logs <container_name>`

#### **Update Images**
- Pull the latest image: `docker pull <image_name>`
- Rebuild the image: `docker build --no-cache -t my-python-app .`

### **6. Practical Example: Building and Running a Python Web App**

#### **Step-by-Step Guide**

1. **Write Your Application Code**
   - Create a simple Python web server using Flask.

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

2. **Create a Dockerfile**
   - Place the Dockerfile in the same directory as your `app.py`.

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

   # Define environment variable
   ENV NAME World

   # Run app.py when the container launches
   CMD ["python", "app.py"]
   ```

3. **Build the Docker Image**
   ```sh
   docker build -t my-python-app .
   ```

4. **Run the Docker Container**
   ```sh
   docker run -p 4000:80 my-python-app
   ```

5. **Push the Docker Image to Docker Hub**
   ```sh
   docker tag my-python-app:latest your-dockerhub-username/my-python-app:latest
   docker login
   docker push your-dockerhub-username/my-python-app:latest
   ```

6. **Deploy to Google Cloud Run**
   ```sh
   gcloud run deploy --image gcr.io/your-project-id/my-python-app --platform managed
   ```

By following this cheat sheet, you can effectively learn and use Docker to develop, test, and deploy your software quickly and efficiently. This guide covers the essential concepts and provides practical examples to help you get started with Docker in your freelance software development work.

Citations:
[1] Docker Compose Quickstart https://docs.docker.com/compose/gettingstarted/
[2] [PDF] CLI Cheat Sheet - Docker Docs https://docs.docker.com/get-started/docker_cheatsheet.pdf
[3] Containerize an application - Docker Docs https://docs.docker.com/guides/workshop/02_our_app/
[4] Docker Compose overview https://docs.docker.com/compose/
[5] Deploying containers on VMs and MIGs | Compute Engine ... https://cloud.google.com/compute/docs/containers/deploying-containers
