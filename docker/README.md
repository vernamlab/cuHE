
#Develop on cuHE within Docker!

Since we find that the overall environment to develop and run applications written with cuHE requires a bit of effort in setup we find usefull to write a protected and easly replicable environment with Docker containers.

##INSTALLATION

You will need of course an internet access all the time.

Install on your machine [NVIDIA-Docker][https://github.com/NVIDIA/nvidia-docker].

Run from ```./docker``` the command ```./manage.sh build```

**NOTE** This will take several time, up to hours.

You are done!

##RUN

First start your Docker image with
```./manage.sh run```

Then you can go into ```/home/sources```
and then to run an example:
```
cd cuhe
./clean_cmake.sh
./docker_cmake.sh 50 #your GPU capability

cd ../examples/DHS
./clean_cmake.sh
./docker_cmake.sh 50 #your GPU capability
./simpleDHS
```
