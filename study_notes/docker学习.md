# 获取镜像

```python
docker pull 镜像名
```



# 查看系统现有的镜像

```python
docker images
```



# 查看系统现有的容器

```
docker ps -a
```

镜像就像种子，可以由镜像创建出无穷多的容器，而这些容器都是可能环境被配置好的，因此可以直接使用其他好心人配置好的环境，所以使用docker在配置环境时非常方便。

# 创建容器

```
docker run -it --name 容器名 镜像名 /bin/bash
```

## 指定端口创建容器

```
docker run -it -d --name 容器名 -p 主机端口号:容器端口号 镜像名
```

若想要使用pycharm连接docker容器，容器端口号必须指定为22，因为SFTP默认使用22端口。



# 进入容器

Step1:

```
docker start 容器名/ID
```



Step2:

```
docker attach 容器名/ID
```





# 停止容器

```
docker stop 容器名/ID
```



# 退出容器

```
exit
```



# 删除容器

```
docker rm 容器名
```



# 删除镜像

```
docker rmi 镜像名
```



# docker run

## --link

- 源容器

```
docker run -d --name selenium_hub selenium/hub
```

- 接收容器

```
docker run -d --name node --link selenium_hub:hub selenium/node-chrome-debug
```

创建一个名字为node的容器，并将该容器与selenium_hub链接起来，hub是该容器在link下的别名（alias）。

通俗的说，selenium_hub和hub都是源容器的名字，并且作为容器的hostname，node用这2个名字中的任何一个都可以访问到源容器并与之通信（docker通过DNS自动解析）



WSL配置docker

https://www.cnblogs.com/yg0070/articles/13841932.html
