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

-----
### **A.** Explore the sbb text (CSV) files

The SBB data of actual arrival/departure times (istdaten) in CSV storage format can be found under `/data/sbb/csv/istdaten`.

**Step 1**:

* Explore the HDFS subdirectory structure under `/data/sbb/csv/istdaten`. What do you observe?

```bash
hdfs dfs -ls /data/sbb/csv/istdaten/*
```

```bash
###
```

Note that it is up to use to decide how we want to organize this folder structure. We chose to organize it temporally by grouping together all events happening in the same time window. However we could also have organized this data spatially into regions, e.g. `/data/sbb/csv/isdaten/{canton}/{city}/`, public transport service providers, or not at all (shallow folder hiearchy).

Hive will happily allow you to create an external table on any folder (node) under this folder structure and consider all the files in the subfolders below the node as one big table. You do this using the location argument of the `create external table` DDL command. 

```
create external table namespace.tablename
    (...)
location '/path/to/exteranl/data/on/hdfs';
```

Starting lower in the folder hierarchy allows us to create smaller external table, resulting in faster query times. This is essentially a form of manual partitioning. Because in practice we are in control of the folder structure of _external_ tables, we optimize it to provide the best performances for the type of query we expect to do on this data. When inserting new data in a table, Hive decides of the folder structure of its data. Even so, it provides methods allowing us to specify [partitioning](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-PartitionedTables) preferences, and [bucketed (cluster)](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-BucketedSortedTables) preferences. 


----
**Step 2**:

* Verify the content of the data - the first 10 lines of December 01, 2020.

```bash
hdfs dfs -cat /data/sbb/csv/istdaten/2020/12/2020-12-01_istdaten.csv | head
```

-----
### B. Explore the twitter data

The twitter data is located under `/data/twitter/json`. It is in JSON format. Use `jq` or `json_pp`, which come preinstalled in this docker image, and pretty print the first line of the file.

**Step 1**:

* Explore the HDFS directory structure under `/data/twitter/json` and understand how it is organized.

```bash
hdfs dfs -ls /data/twitter/json/year=2020/month=09/day=30/01/
```

```bash
###
```

Note that the files are all bzip2-compressed (.bz2). Hive recognizes the extension and knows how to handle it.


----

**Step 2**:

* Pretty print the first line of one of the twitter file 

**Note**: the output will end with a _bzcat: Broken pipe_, this is expected because by design `head` abruptly closes the input from `bzcat` (`bunzip2`) after reading the specified number of lines. 

```bash
hdfs dfs -cat /data/twitter/json/2020/09/30/01/00.json.bz2 | bzcat -s | head -1 | jq
```

```bash

```
