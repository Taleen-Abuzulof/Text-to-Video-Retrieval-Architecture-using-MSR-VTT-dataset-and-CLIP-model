# Text-to-Video-Retrieval-Architecture-using-MSR-VTT-dataset-and-CLIP-model


A deep learning system for **retrieving relevant videos from a natural language query** using a **CLIP-based multimodal embedding space** and efficient similarity search.

This project implements a **two-stage text-to-video retrieval pipeline** evaluated on the **MSR-VTT dataset**, combining efficient embedding-based retrieval with optional cross-encoder reranking for improved semantic precision.

This project was developed for **Computer Vision (ENCS5200)** at **Birzeit University**. for further design details, check out the PDF Report.



Repository Structure
====================

```
project/
│
├── V4_Vision_Project2_MSR_VTT.ipynb
│
├── Computer_Vision_Project_2.pdf
│
└── README.md
```

Project Overview
================

Text-to-video retrieval is a challenging multimodal task where the system must find videos that match a **natural language description**.

The main challenge comes from the **semantic gap between text and visual content**. To address this problem, the system uses **CLIP (Contrastive Language-Image Pretraining)** to map both modalities into a **shared embedding space**, allowing retrieval using similarity search.


The pipeline consists of:

1.  **Frame sampling from videos**
    
2.  **Visual encoding using CLIP**
    
3.  **Video embedding aggregation**
    
4.  **Text query encoding**
    
5.  **Similarity search using FAISS**
    
6.  **Optional cross-encoder reranking**
    

Dataset
=======

The system is evaluated on the **MSR-VTT (Microsoft Research Video to Text)** dataset.

Key characteristics:

*   **10,000 videos**
    
*   **~41 hours of video content**
    
*   **200,000 video–sentence pairs**
    
*   **20 semantic categories**
    
*   **~29,000 unique words**
    

Each video is annotated with multiple natural language descriptions, making it suitable for evaluating **video–language retrieval models**.


System Pipeline
===============

The overall retrieval system follows a **dual-encoder architecture**.

### Step 1 — Video Processing

Videos are converted into embeddings through the following process:

1.  Uniformly sample **8 frames per video**
    
2.  Encode each frame using the **CLIP image encoder**
    
3.  Aggregate frame embeddings using **mean pooling**
    
4.  Normalize the resulting embedding
    

This produces a **single fixed-length vector representation per video**.

Mathematically:

 v_i = f_img(x_i)v_video = (1/N) * Σ v_i   `

Where:

*   x\_i = sampled frame
    
*   f\_img = CLIP image encoder
    
*   N = number of frames
    

### Step 2 — Text Encoding

Text queries are encoded using the **CLIP text transformer**.

Process:

1.  Tokenize query using CLIP tokenizer
    
2.  Encode text using transformer encoder
    
3.  Normalize resulting embedding
    

This creates a **query embedding** that lives in the same semantic space as video embeddings.

### Step 3 — Shared Embedding Space

Both text and video embeddings are projected into a **shared multimodal space**, allowing similarity comparison.

Retrieval is performed using **cosine similarity**:

sim(q, v) = (q · v) / (||q|| ||v||)   `

Where:

*   q = query embedding
    
*   v = video embedding
    

Videos are ranked by similarity score.

### Step 4 — Efficient Retrieval with FAISS

To enable **fast large-scale retrieval**, all video embeddings are:

*   Precomputed
    
*   Stored in a **FAISS index**
    

This allows fast similarity search across thousands of videos.

Advantages:

*   Low latency retrieval
    
*   Scalable to large datasets
    
*   Efficient memory usage
    

Two-Stage Retrieval (Optional)
==============================

To improve ranking quality, a **two-stage retrieval pipeline** can be used.

### Stage 1 — Initial Retrieval

*   Query embedding compared with all video embeddings
    
*   Top-K candidate videos are retrieved using FAISS
    

### Stage 2 — Cross-Encoder Reranking

The top candidates are reranked using a **cross-encoder model** that jointly processes:

*   Query text
    
*   Candidate video representation
    

This allows **deeper semantic interaction between modalities** and improves early-rank accuracy.

However, reranking increases inference time.

Explored Design Variations
==========================

Several approaches were explored during development.

### 1\. Narration-Augmented Representation

An auxiliary caption was generated for each video frame and combined with visual embeddings.

Fusion equation:

v_fused = α v_video + (1 − α) v_narration   `

However, performance was sensitive to caption quality and often introduced noise.

### 2\. Text-Encoder Fine-Tuning

Instead of training the full model:

*   **CLIP vision encoder remained frozen**
    
*   Only the **text encoder was fine-tuned**
    

This allowed better alignment with video embeddings while keeping training efficient.

Evaluation Metrics
==================

The system is evaluated using standard text-video retrieval metrics:

MetricDescriptionRecall@KPercentage of queries with correct video in top K resultsMedRMedian rank of correct videoMeanRAverage rank of correct videomAPMean Average PrecisionRetrieval TimeQuery latency

Results
=======

The **CLIP visual baseline** achieved strong performance while remaining computationally efficient.

Key observations:

*   CLIP embeddings provide strong semantic alignment
    
*   Simple **frame sampling + mean pooling** works surprisingly well
    
*   Reranking improves early-rank metrics but increases latency
    
*   Narration-based augmentation did not consistently improve results
    

This confirms that **lightweight CLIP pipelines can perform competitively without heavy temporal modeling**.

Implementation
==============

The system is implemented using:

*   **Python**
    
*   **PyTorch**
    
*   **OpenCLIP**
    
*   **FAISS**
    

    

Limitations
===========

Several practical challenges were encountered:

*   Video processing is **GPU-intensive**
    
*   CLIP models require significant memory
    
*   Reranking increases inference latency
    
*   Large embedding files require significant storage
    

These constraints limited extensive hyperparameter exploration.


    

