# AITA Dataset README

## Overview

This dataset contains raw posts and comments collected from the subreddit **r/AmItheAsshole (AITA)** between **January 1, 2021 and October 21, 2021**. If you are looking for specifically the subset of this data used in our experiments, please refer [here](https://github.com/nvshrao/social_gaze/tree/main/data).

On r/AITA, users post descriptions of interpersonal conflicts and ask the community to judge whether they behaved appropriately. Other users comment on the post and provide a **verdict**, which other readers can upvote or downvote. The verdict receiving the most community support typically becomes the **majority verdict** for the post.

This dataset aggregates posts, comments, verdict labels, and vote stats from that period.

### Possible Verdicts

| Verdict  | Meaning                |
| -------- | ---------------------- |
| **YTA**  | You're the Asshole     |
| **NTA**  | Not the Asshole        |
| **ESH**  | Everyone Sucks Here    |
| **NAH**  | No Assholes Here       |
| **INFO** | Not Enough Information |

---
## Link to Dataset

The complete raw data (without comment_authors) is posted [here](https://drive.google.com/drive/folders/1DpAan9JFKxcyXMG_uUqHoF9jxdeln5uG?usp=sharing)

---
## Usage

The dataset is stored in JSON format and can be easily loaded using pandas.

```python
import pandas as pd
df = pd.read_json("aita_complete_dataset.json")
df["utc"] = pd.to_datetime(df["utc"], unit="ms")
```

This loads the dataset as a pandas DataFrame where each row corresponds to a single Reddit post and its associated comments.

---

## Files

Two versions of the dataset are provided:

**1. `AITA_complete` (~46,000 posts)**
Contains all collected posts during the time range.

**2. `AITA_filtered` (~1,200 posts)**
A subset of posts where community disagreement is high. These posts were later highlighted on r/AITA_filtered. In most filtered cases, the majority verdict receives less than ~70% of total upvotes. Both files have the same column schema. For our experiments, we remove AITA_filtered IDs from AITA_complete to exclude posts with low consensus.

---

## Column Descriptions

To protect user privacy, comment author usernames are not included in the released dataset. If your research requires access to the `comment_authors` column, please contact me at anveshrao1@gmail.com with an explanation of your use case!

### `post_id`

Unique identifier for the Reddit post.
Corresponds to the ID used in the Reddit URL.

Example:

```
https://www.reddit.com/r/AmItheAsshole/comments/<post_id>/
```

---

### `title`

Title of the Reddit post.

---

### `utc`

Timestamp representing when the post was created (UTC time).

---

### `url`

Link to the original Reddit thread.

Note:
Some URLs may no longer be accessible due to deletion or moderation actions.

---

### `post`

The full text body of the Reddit post.

---

### `majority_verdict`

The verdict whose comments collectively receive the most upvotes.

This represents the **community's overall judgment**.

---

### `verdict_scores`

A dictionary representing the **distribution of upvotes across verdict categories**.

Example:

```
{'YTA': 100, 'NTA': 0, 'ESH': 0, 'NAH': 0, 'INFO': 0}
```

This means all upvotes across verdict comments support **YTA**.

---

### `comment_upvotes`

An ordered list of upvote counts for comments.

Comments are sorted from **highest upvotes to lowest**.

Note:
Upvotes may be **negative** if a comment receives more downvotes than upvotes.

Example:

```
[17, 14, 10, 1, -3]
```

---

### `comment_bodies`

The text of each comment corresponding to the `comment_upvotes` list.

The ordering is identical.

Example:

```
comment_upvotes: [17, 14, 10, 1, -3]
comment_bodies:  [comment1, comment2, comment3, comment4, comment5]
```

---

### `majority_verdict_score`

The score corresponding to the **majority verdict** from `verdict_scores`.

Interpretation:

* **100%** → All upvotes support the same verdict.
* **Lower values** → Greater disagreement among commenters.

This value reflects the **strength of consensus** within the discussion.


---

## Intended Use

This dataset may be useful for research in:

* Social judgment and moral reasoning
* Natural language understanding
* Argumentation analysis
* Online discourse and disagreement
* Consensus formation
* LLM evaluation and alignment studies

---

## Disclaimer

The dataset contains **user-generated content from Reddit** and may include:

* offensive language
* sensitive personal situations
* strong opinions

Researchers should handle the data responsibly.

---

## Contact

For questions, dataset issues, or requests for restricted fields (e.g., comment authors), please contact me!
