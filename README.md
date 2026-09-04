# Rakuten — MLOps platform

The repository contains the main components that are used to run a stack of connected Docker containers. The runtime services provide a streamlit app that run on your local machine and reachable at your local host IP address on port 8501. After being connected, you may try to run predictions on new data in the dedicated page. The repository with its DockerHub linked images also provide a simulation of the system using the locust library.

## First steps

Start by cloning the repository and go to the root of the project :

```bash
git clone "https://github.com/fibonaccos/rakuten-ops.git"

cd rakuten-ops
```

Then, run the following commands to create the required folders and download the dataset (link : [Rakuten Challenge](https://)) :

```bash
mkdir data data/raw data/clean data/features data/train data/prod data/batches


```

## Run the platform

There are 2 ways to run the platform on your local machine. Both of them require Docker to be installed in your computer.

### Rebuild images

1. Build the Docker images.

    ```bash
    docker build -f docker/Dockerfile.app -t fibonaccos/rakuten-app:1.0.0 .
    docker build -f docker/Dockerfile.api -t fibonaccos/rakuten-api:1.0.0 .
    docker build -f docker/Dockerfile.database -t fibonaccos/rakuten-database:1.0.0 .
    docker build -f docker/Dockerfile.inference -t fibonaccos/rakuten-inference:1.0.0 .
    docker build -f docker/Dockerfile.locust -t fibonaccos/rakuten-locust:1.0.0 .
    ```

2. Set all the environment variables according to the [.env.example](.env.example) template file.

3. Run the compose command (with or without optional locust profile) to pull the remaining images and launch the services. Note that this may take several minutes depending on your connection and your computer abilities.

    ```bash
    docker compose --project-name rakuten up -d  # locust simulation disabled
    docker compose --project-name rakuten --profile locust up -d  # locust simulation enabled
    ```

4. Go to `http://localhost:8501` and start using the platform.

### Using DockerHub

Note that this method supposes that your computer architecture is one of `amd64` or `arm64` (i.e. supports Linux, macOS and Windows).

1. Pull the platform images from DockerHub.

    ```bash
    docker pull fibonaccos/rakuten-app:1.0.0
    docker pull fibonaccos/rakuten-api:1.0.0
    docker pull fibonaccos/rakuten-database:1.0.0
    docker pull fibonaccos/rakuten-inference:1.0.0
    docker pull fibonaccos/rakuten-locust:1.0.0
    ```

2. Set all the environment variables according to the [.env.example](.env.example) template file.

3. Run the compose command (with or without optional locust profile) to pull the remaining images and launch the services. Note that this may take several minutes depending on your connection and your computer abilities.

    ```bash
    docker compose --project-name rakuten up -d  # locust simulation disabled
    docker compose --project-name rakuten --profile locust up -d  # locust simulation enabled
    ```

4. Go to `http://localhost:8501` and start using the platform.

**Note :**

- The `locust` image is optional and is only needed if you want to simulate the consumption of the services. Make sure that your computer can support the load of the services.
- The allowed `(username, password)` pairs registered in the database and used to login can be found in the file [passwords.json](./benchmark/config/passwords.json). ***These are placeholders and do not grant access to any kind of real world secured service.***

## About this project

You can find some documentation about the underlying architecture in the [docs](./docs/).

This project has been developed by the following people :

- Rizlène Banat, GitHub : [rbanat](https://github.com/rbanat)
- Romain Mazoyer, GitHub : [Romain057](https://github.com/Romain057)
- Steve Trincal, GitHub : [SteeveGitHub](https://github.com/SteeveGitHub)
- Bryan KHAN MAHMOOD, GitHub : [fibonaccos](https://github.com/fibonaccos)
