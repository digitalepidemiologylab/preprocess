# Crowdbreaks preprocess

This repository can be integrated as a submodule to a Crowdbreaks project. It provides common pipeline tooling for data (such as tweets, images or annotation data) collected through Crowdbreaks.

# Install
This repository is usually added as a submodule to a project repository. But it can also be run as a standalone.

**Cloning the repository as standalone**
```bash
git clone git@github.com:crowdbreaks/preprocess.git --recursive
```

**Cloning as part of a project**
```bash
# cd into your project folder
git submodule add git@github.com:crowdbreaks/preprocess.git
git submodule update --init
```

Install dependencies using
```
cd preprocess
pip install -r requirements.txt
```

# Usage

This code only works with access to an S3 bucket containing the raw data (by default called `crowdbreaks-prd`). This bucket contains either raw data (collected using `crowdbreaks/crowdbreaks-streamer`) or annotation data (collected through `crowdbreaks/crowdbreaks`).

Make sure you have the AWS CLI tools installed (`pip install awscli`) and you can access the bucket using your default credentials (stored under `~/.aws/credentials`).

The code can be run with the following syntax:
```
python main.py <command> [<args>]

Available commands:
  init             Initialize project
  sync             Sync project data from S3
  parse            Preprocessing of data to generate `/data/1_parsed`
  sample           Sample cleaned data to generate `data/2_sampled`
  batch            Creates a new batch of tweets from a sampled file in `/data/2_sampled`
  clean_labels     Clean labels generated from Mturk (`data/3_labelled`) and merge/clean to generate `/data/4_cleaned_labels`
  stats            Output various stats about project
  split            Splits data into training and test data
  prepare_predict  Prepares parsed data for prediction with txcl

positional arguments:
  command     Subcommand to run
```

## Initialize project

Each project has a unique name (by convention identical to Elasticsearch index name). Initialize the project you want to work with using

```
python main.py init --project <project_name>
```

This will generate a config file called `project_info.json` specific to the project.

## Sync data

Sync streaming data, annotation data, and media data (images, gifs, ...) for the specific project.

```
python main.py sync
```

## Parse data

Parse streaming data (whatever is in `data/0_raw`) and generate a single output file in `data/1_parsed`.

```
python main.py parse
```

## Sample data

From the parsed data generate a sample file (used for selecting annotation data). Generates sample data in `data/2_sampled`.

Example:
```
python main.py sample -s 10000
```

## Batch data

From sample files generate batches (in `data/2_sampled/batch_{batch_id}/`).

```
python main.py batch
```
These individual batches can then be annotated with Crowdbreaks.

## Clean labels
First, make sure to use the `sync` command to pull the latest annotation data from Crowdbreaks. Clean and merge annotation data based on a consensus and multiple filter options (saved to `data/4_labels_cleaned`).

Example:
```
python main.py clean_labels -s unanimous
```

## Split

Split into training and test data for specific question tags.

```
python main.py split
```

## Stats

Show summary statistics of all data for project.

```
python main.py stats
```

