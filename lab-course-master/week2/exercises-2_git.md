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

Replace the TODO with the approprate code.

#### 1.1 Pull the most recent updates

```bash
# TODO
```

#### 1.2 Display the status of this repository

```bash
# TODO
```

#### 1.3 Print the commit log

Print the last 5 commits

```bash
# TODO
```

#### 1.4 List all the branches

```bash
# TODO
```

#### 1.4 Display the commit log of the full tree

* Graph mode showing the commits on all the branches
* Pretty (one line)
* Commits abbreviated to 7char

References:
- [git log](https://www.git-scm.com/docs/git-log) documentation

```bash
# TODO
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

```bash
# TODO fetch the content
```

```bash
head -5 ${DATA}
```

```bash
# TODO checkout the LFS content
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

* Use the git lfs command to list files already in git lfs (not the tracked files)

```bash
# TODO
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







---


## 4. Fix me!!


The repository contains a very simple project for computing the Fibonacci’s sequence, following test driven development. But we did some mistakes and didn’t leave the repository in a working shape.

Your mission is to find the code and recover all the files (using only git commands), merge them and push the changes back to your forked repository. To check if you recovered all the files, you can run the test file, all 3 tests must pass on the main branch.

Use the code by calling:

>   python main.py {x}

where {x} is the max of the series that will be returned.

You can run the tests by:

>   python -m unittest test
or
>   python test.py


```bash

```

```bash

```
