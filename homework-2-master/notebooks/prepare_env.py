# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Bash
#     language: bash
#     name: bash
# ---

# %% [markdown]
# # Prepare your environment
#
# You must prepare your environment __if you have not done so in the previous exercises__ in order to do this homework. Follow the steps below.
#
# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Note:</b> 
#     <br>
#     The following steps will clear your Hive database and your user HDFS home folders.
#     </br>
# </div>

# %% [markdown]
# -----
# 1. Run the next cell, and verify that you are logged as you, and not as someone else

# %%
echo "You are the boss, Mr ${RENKU_USERNAME:-nobody}"

# %% [markdown]
# -----
# 2. Run the code below and verify the existance of your database. Execute step 3 if you have a database.\
# Otherwise go to step 6, if it shows an empty list as shown below:
#
# ```
#     +-----------------+
#     |  database_name  |
#     +-----------------+
#     +-----------------+
# ```

# %%
beeline -e "show databases like '${RENKU_USERNAME:-nobody}'"

# %% [markdown]
# -----
# 3. Review the content of your database if you have one.

# %%
beeline -e "show tables in ${RENKU_USERNAME:-nobody};"

# %% [markdown]
# -----
# 4. Drop your database after having reviewed its content in step 3, and __you are ok losing its content__.

# %%
beeline -e "drop database if exists ${RENKU_USERNAME:-nobody} cascade;"

# %% [markdown]
# * Verify that you the database is gone

# %%
beeline -e "show databases like '${RENKU_USERNAME:-nobody}';"

# %% [markdown]
# -----
# 5. Run the remaining cells to reconstruct your hive folder and reconfigure ACL permissions on HDFS

# %%
hdfs dfs -ls /user/${RENKU_USERNAME:-nobody}

# %%
hdfs dfs -rm -r -f -skipTrash /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -rm -r -f -skipTrash /user/${RENKU_USERNAME:-nobody}/.Trash

# %%
hdfs dfs -mkdir -p                                /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl    -m group::r-x                /user/${RENKU_USERNAME:-nobody}
hdfs dfs -setfacl    -m other::---                /user/${RENKU_USERNAME:-nobody}
hdfs dfs -setfacl    -m default:group::r-x        /user/${RENKU_USERNAME:-nobody}
hdfs dfs -setfacl    -m default:other::---        /user/${RENKU_USERNAME:-nobody}
hdfs dfs -setfacl -R -m group::r-x                /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl -R -m other::---                /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl -R -m default:group::r-x        /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl -R -m default:other::---        /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl    -m user:hive:rwx             /user/${RENKU_USERNAME:-nobody}/hive
hdfs dfs -setfacl    -m default:user:hive:rwx     /user/${RENKU_USERNAME:-nobody}/hive

# %%
hdfs dfs -ls /data/sbb/part_orc/istdaten/year=2018

# %%
