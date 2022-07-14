---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Bash
    language: bash
    name: bash
---

# Learn HDFS

In the following exercises you will learn to obtain data from a remote site, and import it in our Hadoop's cluster HDFS.

This is an important step, because HDFS is the default storage for your Hadoop applications and a first step toward your data lake.

They are many ways you can interact with the HDFS file systems, including using the python client for HDFS. For the following exercises you will be using the `hdfs dfs` command line (or `hadoop fs`), which comes by default with the Hadoop distribution. The command line is a client, which sends command from your notebook running in a docker container to a remote HDFS service running on the Hadoop big data cluster.

In the following exercise, we will use the term _notebook container_ to refer to the machine (docker container, laptop or other) on which this notebook is running, and _hdfs server_ to refer to the HDFS storage on the remote Hadoop cluster.

This notebook must be run with the bash kernel, and the hdfs client must be installed. The computing environement of this project has been properly configured for that. The most curious can find the recipe in the [Dockerfile](../Dockerfile), and try to understand how it is done. Under normal circumstances, editing this Dockerfile allows you to customize your computing environment for this git repository. Note however that the Docker image (the computing environment) of this exercise is pinned to a fixed version in [renku.ini](../.renku/renku.ini), which takes precedence over any changes you may have made to the [Dockerfile](../Dockerfile). You must remember to remove the line `image = ...` from the renku.ini file if you want to configure your computing environment. 

> _Important Note:_ for the sake of simplicity, the cluster is configured for basic security only. The settings in your notebook environment are such that they should prevent you from accidentally impersonating other users and causing any damage to our distributed data storage. However, they can easily be bypassed - **do not attempt it**, there is nothing to be proven, and you will have to face the consequences when things go avry.

Note that the answers in this notebook get your gaspar ID from the `${RENKU_USERNAME}` environment variable. If you are not spawning this notebook on the Renku service, you must make sure that this variable is properly set before running the notebook.


----

## Sanity check

Before going further, make sure that the following command returns your gaspar ID

```bash
echo ${RENKU_USERNAME}
```

----
## First steps with the HDFS command line

Execute the following `hdfs dfs` command.

The output is the list of HDFS file system actions available via the hdfs command line. Notice how most of the commands behave like the familiar Linux file system commands.

You can find more details about that on the [Hadoop file system shell](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-common/FileSystemShell.html) manual.

```bash
hdfs dfs
```

<!-- #region -->
-------------------
## Exploring the HDFS folders structure


**Q1:** As a first exercise, you will explore the content of the cluster's HDFS file system using the `hdfs dfs` command.

* We have created a home folder on HDFS for each of you, can you find yours?
  - If not, you must talk to us and we will create one for you. You will not be able to do the exercises without it.
* Create a sub-folder `work1` in your HDFS home folder.
* Copy the SBB data published by the [opentransportdata.ch](https://opentransportdata.swiss/en/dataset/istdaten/resource/84be0ee4-337e-4731-a539-4c5db69dfc8e) to your `work1` sub-folder (use the [_Download URL_](https://opentransportdata.swiss/dataset/0edc74a3-ad4d-486e-8657-f8f3b34a0979/resource/84be0ee4-337e-4731-a539-4c5db69dfc8e/download/2022-02-27_istdaten.csv))
* Change the owner rights of your HDFS `work1` sub-folder and its content hdfs to `-rwxrwxr--` (this will be needed if you want to access this data from Hive).

Hints:
1. Use the linux [curl](https://www.man7.org/linux/man-pages/man1/curl.1.html) or [wget](https://www.man7.org/linux/man-pages/man1/wget.1.html) command line in order to copy the data to `/tmp/` of your notebook container. Copying this data to `/tmp/` outside your git workspace is a good practice if you don't want it to be accidentally commited to your git repo.
2. Use the `hdfs dfs` commands to: (a) find your HDFS folder, (b) create a *work1* folder in it, and (3) copy the file from your notebook container to work1
3. Verify that the file is in your HDFS work1 directory
4. The `${RENKU_USERNAME}` environment variable is set to your gaspar name. We recommend you to take advantage of this variable in your code when you initialize the paths under your HDFS home directory instead of hardcoding your gaspar ID. This will make your code more portable, and easier for others to try it.
5. Note: HDFS does not like spaces in filenames.

<!-- #endregion -->

```bash

```

```bash

```

```bash

```

```bash

```

```bash

```

-----------
In the next notebook we will be using Hive to query our data. Before you move to this notebook, you must prepare an HDFS folder where you will be hosting your Hive metastore.

If you do not provide this folder, Hive will default to folders under hdfs `/warehouse/tablespace/managed/hive` and `/warehouse/tablespace/external/hive`. However, access to these folders are restricted and you may not be able to complete all the Hive exercices if you do not have sufficient permissions to write in those default files locations. In particular, you may have issues creating tables _managed_ by Hive (more about that later).

You will most likely get an arcane error of the kind:
> SemanticException 0:0 Error creating temporary folder on: maprfs:/user/.../hive. Error encountered near token 'TOK_TMP_FILE':28:27", 

If instead of the defaults folders we provide our own HDFS folder to Hive under our HDFS home folders, we must then allow Hive's user id `hive` to write to this folder.  We could use the simple POSIX permission model `hdfs dfs -chmod 776 ...`, but this is not something we want to do, because it is an almost all-or-nothing decision that forces us to grant access to other users ids than `hive`. A better approach is to use HDFS's Access Control List model ([ACL](https://hadoop.apache.org/docs/r3.1.0/hadoop-project-dist/hadoop-hdfs/HdfsPermissionsGuide.html)) permission model, which provides a fine-grained alternative to set different permissions for specific named users or named groups, not only the file’s owner and the file’s group.

You can find more info about ACL in the HDFS [File System Shell](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-common/FileSystemShell.html#setfacl) manual.

**Q2:** The commands required to complete the task are listed below. You must run them. Can you figure out what each of them does?

#### Remark

As noted above, using ACL is a better practice in a production HDFS environment. Note however that in our current development environment, our Hadoop infrastructure is configured to use the basic permission settings, meaning that those restrictions are easily bypassed, and they only offer a false sense of security.


```bash
hdfs dfs -mkdir -p                                /user/${RENKU_USERNAME}/hive
#hdfs dfs -setfacl -R -b                           /user/${RENKU_USERNAME}
hdfs dfs -setfacl -m    user:hive:r-x             /user/${RENKU_USERNAME}
hdfs dfs -setfacl -R -m group::r-x                /user/${RENKU_USERNAME}
hdfs dfs -setfacl -R -m other::---                /user/${RENKU_USERNAME}
hdfs dfs -setfacl -R -m default:group::r-x        /user/${RENKU_USERNAME}
hdfs dfs -setfacl -R -m default:other::---        /user/${RENKU_USERNAME}
hdfs dfs -setfacl -m    user:hive:rwx             /user/${RENKU_USERNAME}/hive
hdfs dfs -setfacl -m    default:user:hive:rwx     /user/${RENKU_USERNAME}/hive
hdfs dfs -getfacl -R                              /user/${RENKU_USERNAME}
```

**Q3:** Modify your *work1* directory and its content to give **rwx** access rights to hive.


```bash

```

----
## Explore the data stored in HDFS

We have already copied the SBB _istdaten_ (real time data) from 2018 to 2021 on HDFS using different storage format: CSV, Bz2 compressed CSV, ORC and parquet. The SBB data are in sub-folders under `/data/` on HDFS.

**Q4:** Focus on the CSV and Bz2 data for now (we will see ORC and parquet in later exercises). Do you understand how the folders are structured?

Note: you can use this opportunity to explore the content of a file using the `hdfs dfs -cat  /data/.../file|head` command.

```bash

```


**Q5:** For each HDFS storage format (CSV, Bz2, ORC, parquet), use the `hdfs dfs` command to print a human readable summary of the total HDFS size footprint of the SBB data for the years 2018 to 2021.

Note: exactly the same information is encoded in each format.

```bash

```

```bash

```

```bash

```

```bash

```

----------
**That's all folks!**

You can now move to the next set of exercises [exercises-2_epd.py](./exercises-2_epd.py)

When you are done with the next exercises, you may delete your `work1` directory on HDFS.

You have now mastered the very basics of HDFS. There is more, and we encourage you to read the litterature about HDFS. In particular the alternatives to HDFS, and the different optimization tricks such as optimum HDFS block sizes. Also familiarize yourself with the tradeoffs of data storage formats (ORC, parquet). You should also learn about the dos and don'ts of HDFS, such as optimum compression schemes. Remember don't gzip your files before importing them, because it is not splittable into blocks, bz2 is okay though however see the next exercises. What you save on storage size and network performance with compression, must be paid with CPU cost, and it is not always a winner.


```bash

```

----
#### Cleaning up

Uncomment the hdfs command below and run the cell if you wish to clean up your HDFS home folder.

But not until you have completed all the exercises from the other notebook.

```bash
#hdfs dfs -rm -r -skipTrash /user/${RENKU_USERNAME:-nouser}/work1
```

```bash

```
