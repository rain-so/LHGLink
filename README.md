# LHGLink

This repository contains the official implementation of the paper: **"LHGLink: LLM-Enhanced Heterogeneous Graph Learning for Issue-Issue Link Prediction"**.

## Abstract

We present LHGLink, a hybrid framework that integrates Large language models with Heterogeneous Graph neural networks to  predict Links between software issues. 

our approach follows a three-stage workflow:

* First, we employ an LLM-based semantic enhancement module to enrich and encode issue descriptions, producing high-quality textual representations that mitigate noise, missing context, and inconsistency in raw issue reports. 
* Second, using issues, assignees, and components as distinct entity types, we construct a task-specific heterogeneous graph and define a set of semantically meaningful metapaths to characterize the social and structural relationships among issues. 
* Third, we perform context-aware representation learning, where metapath–specific encoders aggregate intra-path information and an attention-based fusion layer integrates heterogeneous knowledge into unified issue embeddings. 

Finally, the learned embeddings of issue pairs are combined through a nonlinear prediction layer to determine whether a link exists between them.


## Dataset

This repository **includes the complete dataset** used in our experiments. We collected issue data from **23 large-scale open-source projects** under the Apache Software Foundation (ASF) using the Jira REST API.

###  Construction & Filtering
* **Scale:** 314,293 issues across 23 projects.
* **Time Range:** From each project's creation up to January 1, 2025.

To ensure high-quality supervision signals, the dataset was rigorously filtered:
* **Status:** Only issues with **`Resolved`** or **`Closed`** status are retained to ensure complete resolution context.
* **Link Coverage:** Selected projects exhibit a link coverage rate of **≥ 20%** to avoid extreme sparsity.


###  Data Content
For each issue, we provide the following attributes necessary for constructing the heterogeneous graph:
* **Identity:** `Key`, `Type`
* **Textual Features:** `Summary`, `Description` 
* **Structural Nodes:** `Assignee` , `Components` 
* **Ground Truth:** `InwardIssueLinks`, `OutwardIssueLinks`

The full dataset is organized in the `dataset/` directory. You can use it directly to reproduce our results.

## Framework Overview

LHGLink is implemented by PyTorch over a server equipped with NVIDIA GTX 4090 GPU.

The LHGLink framework consists of three main stages:
1.  **Issue Text Enhancement:** Reorganizing raw issue text using `GPT-4o-mini`.
2.  **Feature Extraction:** Generating semantic embeddings using a locally deployed `Llama-3-8B`.
3.  **Heterogeneous Graph Learning:** Aggregating information via metapath-guided HGNNs.


