---
layout: post
title: Streaming NLP Data Quality Engine
description: detecting bad reviews at ingestion with Kafka
---

The Goal & Progress
============

I noticed that a lot of my past projects were notebook-based, so I wanted to try building a more production-like system for real-time NLP data quality monitoring (Also, I wanted to practice what I learned in CS544 Big Data Systems lol). The goal was to create a pipeline that could ingest streaming text data, analyze it for quality issues, and flag problematic entries in real-time.

This matters because data quality problems compound. A batch pipeline might train on 10% garbage data without us knowing. A streaming pipeline can catch and route bad data immediately, giving us clean training sets and the ability to monitor quality degradation in real-time.

For now, I'm going with a Faust-powered Kafka streaming pipeline. Faust is a Python library for building streaming applications, and it integrates well with Kafka. The idea is to set up a Kafka topic for incoming text data, process each message with Faust to check for quality issues, and then output the results to another Kafka topic or a monitoring dashboard.

I'm planning on adding monitoring + sentiment analysis features at some point in the future.

All of these ^ aside, I also want to see the impact of this pipeline on downstream model performance. So I'll set up a simple ML model that trains on both clean and unclean data to compare results.

Later on, I might explore adding Yelp Fusion API connector (or something similar) to monitor incoming reviews from external sources in real-time.

The Data & Pipeline
============

I worked with the Yelp Open Dataset (6.9M reviews), which contains user reviews, business information, and user data. For this project, I focused on the review text to analyze quality issues.

I built a multi-level validation system that scores each review from 0 (horrible) to 1 (perfect). The structure looks something like:

  * Level 1: Schema validation using Pydantic
  * Level 2: NLP-specific validation
  * Level 3: Language detection using `langdetect` with confidence threshold
  * Level 4: Composite quality score

Following this, I encountered two technical challenges that I was somewhat unfamiliar with: implementing exactly-once semantics in a streaming context, and managing state for running statistics.

For exactly-once semantics – which means that we need to guarantee that each review gets processed exactly once even if the system crashes mid-processing – I addressed this with three layers: 

  * On the Kafka side, I configured the producer with `acks='all'` to ensure messages are replicated before acknowledging.
  * On the Faust side, I implemented stateful deduplication using a tumbling window table that tracks review IDs for one hour and expires after two.
  * On the database side, I used PostgreSQL's ON CONFLICT DO UPDATE to make writes idempotent.

This means if you process a review, write to the database, then crash before committing the Kafka offset, reprocessing on restart is safe because the database write is idempotent and the deduplication table prevents double-counting.

For state management, since the data is now infinite (meaning we can't load everything into memory like we would in a notebook), I had to rethink how to maintain running statistics. I did this by using RocksDB-backed tables in Faust.

The Results
============

Phase 1 (Faust + Kafka Pipeline):
------------

Processing 115,400 reviews through the pipeline yielded the following quality score distribution:

~~~
SELECT rating, COUNT(*), AVG(data_quality_score)
FROM cleaned_reviews
GROUP BY rating;

 rating | count | avg_quality
--------|-------|-------------
    1.0 | 12611 |       0.696
    2.0 |  9341 |       0.697
    3.0 | 13078 |       0.697
    4.0 | 29202 |       0.697
    5.0 | 51168 |       0.698
~~~

Phase 2 (Monitoring + Sentiment Analysis):
------------
More to come soon!
