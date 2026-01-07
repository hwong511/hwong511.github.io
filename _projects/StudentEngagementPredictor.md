---
layout: post
title: Student Engagement Predictor
description: predicting student engagement from digital logs with XGBoost
---

The Question
============

During my industry partnership with Carnegie Learning, I started this project with a simple question: Can we detect student engagement from digital learning logs? More specifically, can I predict what a trained human observer would see? Would they code a student as genuinely on-task, gaming the system, or completely checked out?

This matters because human observation, in this case following the [BROMP protocol](https://learninganalytics.upenn.edu/ryanbaker/bromp.html){:target="_blank"}, is expensive and doesn't scale. If digital logs could reliably proxy for human judgment, we could build systems that automatically detect and respond to disengagement.

The Data
============

I worked with multimodal time-series data from students using an adaptive video-based math platform. I had two data sources: behavioral observations coded by trained observers using the BROMP protocol, and digital event logs from the IMS Caliper Analytics standard capturing every interaction. 

The first problem that stopped me in my tracks: how do you even merge these? The observations came at one rate, the digital events at a completely different rate (sometimes multiple per second, sometimes nothing for minutes). After some consideration, I chose merging via time windows over nearest neighbor to capture more context and patterns.

To prevent temporal data leakage (using future information to predict current state), I used backward-looking windows only, meaning only events from 90 seconds before each observation.

~~~
d_merged <- d1 %>%
  fuzzy_left_join(d2, 
                  by = c("student_id" = "student_id", "time_utc" = "time_utc"),
                  match_fun = list(`==`, function(x, y) {
                    abs(difftime(x, y, units = "secs")) <= 90
                  })) %>%
  rename(bromp_time = time_utc.x, caliper_time = time_utc.y) %>%
  mutate(student_id = student_id.x,
         time_diff = as.numeric(difftime(caliper_time, bromp_time, units = "secs")))
~~~

Feature Engineering
============

For feature engineering, I started building event counts at different time windows (30, 60, 90 seconds). I figured different behaviors might have different time signatures. Maybe gaming shows up in quick 30-second bursts but confusion takes 90 seconds to detect.

I quickly realized that simple counts weren't good enough... I needed to measure deviation from each student's personal baseline. So for each student, I calculated their running baseline using only past observations `(events.shift(1).expanding().mean())` and then measured how much they deviated from that baseline. This eventually became one of my strongest features.

~~~
# Sort by student and time first!
df = df.sort_values(['student_id', 'bromp_time']).reset_index(drop=True)

lag_features = ['events_symmetric_90s', 'events_back_60s', 'pauses_back_60s', 
                'pauses_symmetric_60s', 'time_since_last_event', 'assessment_back_60s']

for col in lag_features:
    # Mean
    df[f'student_avg_{col}'] = df.groupby('student_id')[col].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # STD
    df[f'student_std_{col}'] = df.groupby('student_id')[col].transform(
        lambda x: x.shift(1).expanding().std()
    )
    
    # Rolling mean (last 5)
    df[f'student_recent_{col}'] = df.groupby('student_id')[col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

df['obs_number'] = df.groupby('student_id').cumcount() + 1
~~~

Modeling & Results
============

One thing I had to be very careful about was data leakage at the student level. Since I had multiple observations per student, if I randomly split the data into training and validation sets, I risked having data from the same student in both sets. This would lead to overly optimistic performance estimates because the model could just learn student-specific patterns, kind of like predicting someone's right half of the face with knowledge of their left half. To address this, I used GroupKFold cross-validation, ensuring that all observations from a student were contained within either the training or validation set for each fold, never both.

![Model Performance]({{site.baseurl}}/assets/images/CL_performance.png)

The model achieved statistically significant discrimination between engagement states
(Test AUC = 0.639). It is worth noting, however, that the effect is weak, as the AUC indicates
that the model correctly distinguishes between engaged and disengaged signals only 64% of the
time, compared to 50% for random guessing.

The gap between CV performance (0.677) and test performance (0.639) reflects the
challenge of generalizing engagement predictions to completely new students. This 0.04 AUC
drop is smaller than typical gaps, suggesting the proper temporal lag features helped the model
learn more generalizable patterns. The consistency between validation (0.632) and test (0.639)
performance further supports the stability of these estimates.

![Tableau Dashboard]({{site.baseurl}}/assets/images/CL_dashboard.png)

SHAP analysis showed what actually mattered: extended silence (time_since_last_event), recent activity levels (events_back_60s), assessment attempts, and personal baselines. 

Main Takeaways
============

I think I've figured out the fundamental problem with my model performance, and that is that digital logs capture exposure time, not cognitive engagement. In other words, a student can...

  * ...watch a video while completely zoned out (high digital activity, low engagement). 
  * ...pause to think deeply about a problem (low digital activity, high engagement). 
  * ...game the system rapidly (high digital activity, low engagement). 
  * ...sit confused but still try (low digital activity, ON TASK). 
  
The digital and the cognitive just don't align cleanly.

The results suggest that automated engagement detection from digital logs is not practical for deployment in high-stakes situations, as it would fail to correctly distinguish between engaged and disengaged behaviors ~36% of the time. That being said, it does provide value over naive baseline models, and I believe the model is good enough for low-stakes exploratory analysis or intervention screening (e.g. to flag students who might need help).

I'm actually pretty happy about the result! Finding the ceiling tells us something real about the limits of what's possible, and that the gap between behavior and cognition might be fundamental, not just a data problem waiting for more clever algorithms.

I'm proud about the methodological choices I made (namely, student-level CV and expanding-window lag features), as they provided pretty robust performance estimates. This is supported by the relatively small gap between CV (AUC = 0.677) and test (AUC = 0.639), suggesting good generalizability despite the model’s modest performance. Future work in this domain could adopt similar validationstrategies to ensure honest performance estimates.
