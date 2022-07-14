# EPFL COM490 Spring 2022 - Large-Scale Data Science for Real-World Data

## Introduction

This is the repository where you will find all the exercises for the COM490 course.

You are supposed to make a fork of this repository in order to work on it independently and to be able to _push_ your changes independently of others. Every week your project will be automatically updated with the new files that have been uploaded for the week.

The exercises are done online on the Renku platform. There are no special requirements to start working. You just need to connect to Renku in a web browser according to the instructions below at the end of the page.

## Calendrier des cours

* Module 1
  * 23.02.2022 - General introduction to Data Science
  * 02.03.2022 - Collaborative Data Science
* Module 2
  * 09.03.2022 - General introduction to Big Data
  * 16.03.2022 - Data Wrangling with Hadoop
* Module 3
  * 23.03.2022 - Introduction to Spark runtime architecture
  * 30.03.2022 - Spark Data Frames
  * 06.04.2022 - Advanced Spark, optimization and partitioning
* Module 4
  * 13.04.2022 - Introduction to data stream processing
  * 27.04.2022 - Advanced data stream processing
  * 04.05.2022 - Analytics on data at rest and data in motion
* Review
  * 11.05.2022 - Putting it all together
  * 18.05.2022 - Final project Q&A
  * 25.05.2022 - Final project Q&A

## Working with the project

The simplest way to start your project is right from the Renku
platform.

- Log in at <https://dslab2022-renku.epfl.ch> with your EPFL GASPAR ID.
- If you have not yet made a copy of this project for yourself:
   * Navigate to the project page, <https://dslab2022-renku.epfl.ch/projects/com490/lab-course>
   * Create a _fork_ of the project by clicking on **Fork** in the upper right corner. You are redirected to the URL of your copy of the files, probably as <dslab2022-renku.epfl.ch/projects/[your_username]/lab-course>. Make a note of this for later.
- Otherwise:
   * Navigate to your copy of the project page, probably as (<dslab2022-renku.epfl.ch/projects/[your_username]/lab-course> or navigate to it from the Renku home page)
- Navigate to the **Sessions** tab of the project.
- If you have a current running session:
   * Connect to it by clicking on **Open in new tab**
   * Click on the **Terminal** icon
   * Run this: `./post-init.sh`. This will automatically update your projects with the week's exercises.
- Otherwise:
   * Click **Start**, or **New session**, leaving the default settings. This creates your remote virtual machine and takes a while the first time; wait for its _status_ to change from **Pending** to **Running**, then click **Open in new tab**. This will open an interactive environment right in your browser.
- You are now in the JupyterLab interface, running on your virtual machine. On the left, you have the list of files and folders in the repository. Open the folder corresponding to the week you want and work on it.
- Once you have finished your work, save the files. They are now only saved on the virtual machine, which has a limited lifespan! If you quit your machine now, you may lose your changes.
- Switch to the left **Git** tab to see your changes. For each of the files in Staged or Untracked:
    * Confirm that the addition or change to this file is to be considered by clicking the **+**. The file will then move to the **Staged** section.
    * At the bottom of the window, enter in **Summary** a brief comment on the changes made, then click **Commit**.
    * At the top of the window, click on the **Push committed changes** icon. Your changes are now saved in your repository.

If, while working, you close your browser by mistake, this does not immediately terminate the session, and you can access it later from the Renku **Sessions** menu as described above. However, the session is automatically terminated if it remains disconnected from the browser for more than 24 hours.


To work with the project anywhere outside the Renku platform,
click the `Settings` tab where you will find the
git repo URLs - use `git` to clone the project on whichever machine you want.


### Changing interactive environment dependencies

This is a Renku project - basically a git repository with some
bells and whistles. You'll find we have already created some
useful things like `data` and `notebooks` directories and
a `Dockerfile`.

Initially we install a very minimal set of packages to keep the images small.
However, you can add python and conda packages in `requirements.txt` and
`environment.yml` to your heart's content. If you need more fine-grained
control over your environment, please see [the documentation](https://renku.readthedocs.io/en/latest/user/advanced_interfaces.html#dockerfile-modifications).

### Project configuration

Project options can be found in `.renku/renku.ini`. In this
project there is currently only one option, which specifies
the default type of environment to open, in this case `/lab` for
JupyterLab. You may also choose `/tree` to get to the "classic" Jupyter
interface.

Happy data wrangling!
