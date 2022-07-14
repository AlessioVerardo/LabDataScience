---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Bash
    language: bash
    name: bash
---

-----
## 1. First steps with git

Note: this is a BASH notebook, all the commands are bash script command lines. They are not python code.

Replace the _TODO_ with the approprate code.

All exercises performed with git version 2.17.1.

#### 1.1 Pull the most recent updates

**Solution:**

`git pull`, or `git fetch`+`git checkout`

```bash
git pull
```

#### 1.2 Display the status of this repository

**Solution:**

`git status`

After a git pull, and if you have not created a new file or modified an existing one you should get:
> On branch master\
Your branch is up to date with 'origin/master'.
>
> nothing to commit, working tree clean

Otherwise if you have changed any files, e.g. saved this file, you should get a list of the currently new and modified files, and files in the staged area waiting to be committed, such as:
> On branch master\
Your branch is up to date with 'origin/master'.\
>\
> Changes not staged for commit:\
>  (use "git add <file>..." to update what will be committed)\
>  (use "git checkout -- <file>..." to discard changes in working directory)\
>\
>modified:   DSLab_week2-2.ipynb\
>\
> no changes added to commit (use "git add" and/or "git commit -a")


```bash
git status
```

#### 1.3 Print the commit log

Print the last 5 commits

```bash
git log -n 5
```

#### 1.4 List all the branches

**Solution:**

`git branch -a`

You should get:
```
* master
  remotes/origin/development
  remotes/origin/master
```

Notice the two branches, `master` and `development`.

The `*` indicates the checked out branch. It means that you are currently at the tip of master.

```bash
git branch -a
```

#### 1.4 Display the commit log of the full tree

* Graph mode showing the commits on all the branches
* Pretty (one line)
* Commits abbreviated to 7char

References:
- [git log](https://www.git-scm.com/docs/git-log) documentation

**Solution:**

* use the command `git log` with the options:

```
--graph:         Graph mode
--pretty=short:  pretty output, abbreviated headers
--abbrev-commit: commits abbreviated commit ids (7char default)
--all:           show all the references
```

* Note the commits on the `origin/development` branch, you will need them later.
* The log provide information about the position of your current environment in the tree:
    - `HEAD -> {branch-name}`: the parent commit of your current work environment. It is the tip of your working branch. It (_branch-name_) should be `master` unless you are checked-out to another branch. The status of the files in your work environment (new files, modified files) are relative to this commit point.
    - `origin/{branch-name}`: the tip of _branch-name_ at the origin when you last fetched from the remote repositry (others may have pushed new commits to it since then). Your HEAD will be ahead of this tip on the current branch if you have new commits in your local repository.




```bash
git log --all --graph --pretty=short --abbrev-commit -n 5
```

-----
## 2. Data versioning with git lfs

In this exercise we ask you to experiment with git large file systems commands.

First you must set the environment variables pointing to the file we are going to use, and verify that you are in the notebooks folder.

```bash
export DATA=../data/carbosense/CarboSense-October2017_curated.csv
pwd
```

#### 2.1 List the first 5 lines of the carbosense data

```bash
head -5 ${DATA}
```

If you have not run the git lfs pull command yet, the expected output should look like this:

> version https://git-lfs.github.com/spec/v1 \
oid sha256:736531dfad47edd51b7d2d7d6222acce0264427201b94bd877871fcb79dfee85 \
size 7564471

Otherwise, it will be the first 5 lines of the Carbosense data.



#### 2.2 Fetch the data

Use _two_ git lfs commands, one that fetches the data to your local repository, and one that copies it into your workspace. Check the content of the data file between the commands.

**Solution**:

Either a `git lfs fetch`+`git lfs checkout`, or a `git lfs pull`.

The command `git lfs fetch`[[doc]](https://www.mankier.com/1/git-lfs-fetch) downloads the data content (git objects at the given checked-out git reference, i.e. commit) locally. But note that the data in your work environment is still the tiny pointer file. The command `git lfs checkout`[[doc]](https://www.mankier.com/1/git-lfs-checkout) will update your work environment, and replace the tiny pointer file with the actual data. The `git lfs pull`[[doc]](https://www.mankier.com/1/git-lfs-pull), it is a shortcut for the other two commands.

```bash
git lfs fetch
```

```bash
head -5 ${DATA}
```

```bash
git lfs checkout
```

```bash
head -5 ${DATA}
```

Alternatively, the command `git lfs pull` would have done the same as the commands above


#### 2.3 Mark data for git lfs

Use the command `git lfs track {file pattern}` in order to automatically put all the files that match the specified pattern into LFS.

For instance `git lfs track *.png` will automatically put all PNG files into LFS.

```bash
git lfs track '*.png'
```

No files is in LFS yet (unless you have added a png file to this repository), however from now on, any file that matches this pattern will be put in LFS when you commit them and push them to your remote repository.

You can run the command `git lfs track` without argument in order to list the files currently tracked that way.

```bash
git lfs track
```

In order to better understand how git lfs track works, run the following two commands

```bash
git status
```

```bash
cat .gitattributes
```

First, notice that `git lfs track` has created a file [.gitattributes](./.gitattributes) in the current folder.
This file is used by git to specify [what to do](https://www.git-scm.com/docs/gitattributes) with files that match the given patterns.
Its use is not limited to git lfs. In this case it will invoke LFS plugin extensions that will transform the tracked files when certain git commands are invoked them, such as _clean_ and _smudge_ during checkout (filter event).


#### 2.4 List all files currently in LFS

```bash
git lfs ls-files
```

#### 2.5 Understanding how git LFS works

Compute the sha256 hash value of the ../data/CarboSense-October2017_curated.csv file, and compare with other hash values encountered above. What do you observe?

```bash
shasum -a 256 ${DATA}
```

-----
## 3 Easter Egg Hunting


In this interactive environment we have installed a visualization package called _plotly_. The method we used guarantees that if you update the version of this package, your collaborators will automatically have the same version the next time they start their interactive environment.

Can you find out and explain the mechanism used to install this package, and point out the files involved? Most of the clues can be found in this folder and the gitlab configuration of this project. Other bits of the method are automatically handled by the platform RENKU, and you will have to guess them.

Note that if you want to experiment with the method for yourself, you will need to [configure](https://renku.readthedocs.io/en/stable/user/templates.html?highlight=renku.ini#renku) the .renku/renku.ini file.


#### Answer




The hints are found in the three files hidden in this repository:
* [.gitlab-ci.yml](../.gitlab-ci.yml) (mainly lines 16)
* [Dockerfile](../Dockerfile#L22-L4)  (lines 21-24)
* [environment.yml](../environment.yml) (line 9)

The .gitlab-ci.yml file contains the instructions that tell gitlab to execute a script each time a specific events  happen, in this case a push[[doc]](https://docs.gitlab.com/ee/ci/). Line 16 of this files tells gitlab to execute a docker build operation on this repository after each push using the command: `docker build --tag ${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA_7} .`

The `--tag` options is used to specify the image registry where the resulting image will be stored, and the name of the image. They are respecively the gitlab registry of this renku project, and the abbreviated (7 characters) commit ID of the last push. The resulting tag ressemble something like `dslab2021-registry.epfl.ch/com490-pub/w2-exercises:b006f6e`. The final `.` is also important, it tells the docker build command to use the `Dockerfile` in the working directory, that is, the root directory of this git repository.

This `Dockerfile` contains the commands used to create the image, and note that it first copies the files `environment.yml` and `requirements.txt` inside the image (`COPY` directive) and then execute a `conda` command on the first and a `pip` command on the second (`RUN` directive). Both commands are package managers, and are used to download and install python packages (and more with `conda`) on your local machine, or more precisely in our case the docker image being built by gitlab. Their most important feature is that they will automatically download and install the missing dependencies of your packages, and update their versions as needed to ensure that they are all compatible with each other. The `conda` utility is more heavyweight but more powerful than the lighter and simplier to use `pip` command. It also provide additional functionalities not available in `pip`, such as the ability to manage multiple environments, allowing such things as hosting multiple environment versions (e.g. python 2.7 and 3.8) on the same machine without conflicts. Most python packages can be installed with either one, unfortunately this is not always the case, and both package managers are often needed, even though they are somewhat redundant. Both commands install the packages listed in their respective `environment.yml` and `requirements.txt` file. Note that the second is empty, meaning that `pip` is a no-op, and all our dependencies are installed with `conda`. A closer look in `environment.yml` confirm that this plotly is indeed listed there with other packages used in this exercise. Note also post-installation command `jupyter labextension install jupyterlab-plotly` in the Dockerfile, which is neeeded to complete the setup.

The rest is a bit of black magic and requires a bit of guessing about the operations of RENKU and its gitlab sidekick which are not all apparent in this git repository. Once the image is built it is saved in the image registry to the address pointed by `CI_REGISTRY` under its given tag `${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA_7}` (see docker login and docker push in .gitlab-ci.yml). You can see this image under the _Gitlab View / Packages / Container Registry_ tab. And you can see the logs of the whole operation described above under _Gitlab View / CI-CD / Pipelines_. And we can finally close the loop; when you or your colleague start a new docker container (interactive environment), RENKU knows which image to use, since it is tagged (indexed) with that commit ID (`CI_COMMIT_SHA_7`), and voilà le tour est joué: your code, and data inputs that are nicely kept in synch with their runtime environments; you can go back in time and be sure that for each version of code and data, you are using the corresponding version of the environment.

Are you now worried about creating gigbytes of images on each push? Then don't. If you do not change the Dockerfile or other files copied in the image during the build, then no new image is created, and the old image is reused with a new tag (commit ID) for indexing purpose, which is tiny. Even if you modify the image, Docker knows how to reuse existing layers from other image and will only create the new layers that it needs - in a Dockerfile, the first modified COPY or RUN directive and everything below is a new layer, and everything above is reused (including FROM directive of the base image). Finally each layer is identified by its hash sum, and is stored in an object store. If you or your colleagues happen to use the exact same layer multiple times (e.g. all the layers of the base image), only one copy of it needs to be stored in the object store.

There is a catch however: everything mentioned above is no longer applicable if you have an `image = ` configuration in your [.renku/renku.ini](../.renku/renku.ini) file. In that case your environment is pinned to the image version specified by this configuration. That is, even if you modify your Dockerfile and environment.yml, it will always start a new docker container using that version of the image until you remove it from your renku.ini file. Image pinning can dramatically speeds-up the starting time of the container. However, enabling it and forgetting about it may cause you or your peers to fail to understand why the image is no longer being updated with the Dockerfile. 




---


## 4. Fix me!!


The repository contains a very simple project for computing the Fibonacci’s sequence, following test driven development. But we did some mistakes and didn’t leave the repository in a working shape.

Your mission is to find the code and recover all the files (using only git commands), merge them and push the changes back to your forked repository. To check if you recovered all the files, you can run the test file, all 3 tests must pass on the main branch.

Use the code by calling:

>   python main.py {x}

where {x} is the max of the series that will be returned.

You can run the tests by:

>   python -m unittest test_fibonacci

**Solution:**

The answer was partially revealed in exercise 1 when we noticed the presence of the `development` branch with revealing commit message.

Many of the git actions performed here can also be done from the jupyterlab git interface. We will only present the command line based solution.

Note if the notebook _gets in the way_, i.e. it creates a modified version which causes you issue to execute some of the git commands that requires a _clean_ repos, you can reset the changes (use the jupyterlab git view), and execute the following commands in a terminal instead.

1. First checkout the commits on the `development` branch in reverse chronological until you succeed the unit tests. Or use the commit message as hint, and go directly go to the commit before things seem to go wrong.
1. Note that you will get a `detached HEAD` state message. What it means is that you can checkout a commit before the tip of a branch, however you cannot commit new changes starting from that commit, unless you start a new branch, which git asks you to do explicitely. This is because you can only add new commit to the _tip_ of a branch. You can start a new branch with the `git checkout -b {new-branch-name} commit-id`, or you can create a new branch with `git branch`. But before we create this new branch we must find a good commit starting point that passes the unittest.
1. **Warning:** all changes, including autosave form other notebooks or change via other views of the jupyterlab interface will now happen in this new workspace, until you checkout on another branch.
1. You can verify the position of your current workspace in the commit log using the `git log --graph --all` command, and note that HEAD is no longer pointing to a branch name (i.e. it is missing a -> {branch-name}), unless you have created one during checkout.
1. Notice that the jupyterlab file manager is updated with the version of the code you just checked out. Also you are in the notebooks folder, and the code is apparently located in the parent folder. It would be easier to `cd` to the same folder before runnint the unit tests.
1. Once you find a functional commit, you have several options - you can do a `git revert {working-commit}`[[doc]](https://git-scm.com/docs/git-revert) from the development branch, or you can check it out on a new branch (we call it `fix-dev`), and checkout master, and `git merge fix-dev`[[doc]](https://git-scm.com/docs/git-merge) that branch on `master` (it should not cause any conflict because the branches modified different files). There are other more _destructive_ options, such as rebase. However, they modify your project history instead of adding new commit that reverts previous commits, which could cause a store of other issues, and they are not thus recommended.
1. Finally remember to set the working directory to ./notebooks and checkout the master branch, otherwise you may not be able to replay this notebook.

```bash

```

```bash
git checkout f950a28
```

```bash
git log --all --graph --pretty=short --abbrev-commit -n 5
```

```bash
cd ../fibo
pwd
```

```bash
python -m unittest test
```

**STOP and read!!** the two options below would normally require you to edit a message in an editor, which you cannot easily do in a notebook environment. We prevent that behavior with the `--no-edit` option, however this is discouraged.

**Option 1:** revert the bad commits, or more specifically create new commits that do the inverse of the reverted commits. The history (including the bad commit) is unchanged. You can continue working on the development branch.

Note that it only work from a _clean_ repository, i.e. one that does not have modified or new files, including this notebook which may have been modified and autosaved (you must commit them first or stash them).

```bash
# Option 1
git checkout development
git revert --no-edit f950a28 ## <- we reverse up to that bad commit
git log --all --graph --pretty=short --abbrev-commit -n 8
```

**Option 2:** create a new branch starting at the last good commit, and merge it on master. Or you can continue working on it and merge it later, as long as you don't modify the same files on both branches (or you will have to resolve the merge conflict). You can then delete the development branch.

```bash
# Option 2
git checkout -B fix-dev 501e48c
git checkout master
git merge --no-edit fix-dev
git log --all --graph --pretty=short --abbrev-commit -n 8
```

```bash

```

```bash

```
